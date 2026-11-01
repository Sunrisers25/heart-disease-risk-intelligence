"""Model comparison framework.
Executes unified Stratified 5-Fold GridSearchCV across SVM, Random Forest, and XGBoost.
Evaluates candidates, selects the winning pipeline based on CV ROC-AUC,
evaluates on an untouched test set, and exports structured results.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_SPLITS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_pipeline.loader import load_and_clean_data
from src.models.evaluate import compute_metrics
from src.models.model_factory import get_model_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Hyperparameter search grids designed for a 300-row benchmark dataset
HYPERPARAMETER_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "SVM": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__gamma": ["scale", "auto"],
        "classifier__kernel": ["rbf"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.03, 0.1],
        "classifier__subsample": [0.8, 1.0],
    },
}


@dataclass
class CandidateResult:
    """Stores performance metrics and training metadata for an evaluated model."""
    model_name: str
    best_params: Dict[str, Any]
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    training_duration_seconds: float
    test_metrics: Dict[str, Any]
    best_estimator: Optional[Pipeline] = None


class ModelComparisonFramework:
    """Coordinates reproducible model benchmarking and model selection."""

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        cv_splits: int = CV_SPLITS,
        test_size: float = 0.2,
    ):
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.test_size = test_size
        self.results: Dict[str, CandidateResult] = {}
        self.best_model_name: Optional[str] = None
        self.best_pipeline: Optional[Pipeline] = None

    def prepare_data(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Loads and performs a leak-free stratified train/test split.

        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None:
            df = load_and_clean_data(save_processed=True)

        X = df[FEATURE_COLUMNS].copy()
        # Ensure target is binary {0, 1}
        y = (df[TARGET_COLUMN] == "Yes").astype(int).values

        logger.info(
            "Splitting data (total=%d, test_size=%.2f, random_state=%d, stratify=True)",
            len(df),
            self.test_size,
            self.random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Train set: %d samples (pos=%d, neg=%d) | Test set: %d samples (pos=%d, neg=%d)",
            len(y_train),
            int(np.sum(y_train)),
            len(y_train) - int(np.sum(y_train)),
            len(y_test),
            int(np.sum(y_test)),
            len(y_test) - int(np.sum(y_test)),
        )

        return X_train, X_test, y_train, y_test

    def run_comparison(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ) -> Dict[str, CandidateResult]:
        """Runs 5-fold Stratified GridSearchCV for each model candidate.

        Selects best parameters strictly based on Cross-Validation ROC-AUC.
        Evaluates the refitted candidate on the untouched test set.
        """
        grids = custom_grids or HYPERPARAMETER_GRIDS
        model_keys = ["SVM", "Random Forest", "XGBoost"]

        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        self.results = {}

        for model_name in model_keys:
            logger.info("=" * 60)
            logger.info("Initiating GridSearchCV for [%s]", model_name)
            logger.info("=" * 60)

            pipeline = get_model_pipeline(model_name, random_state=self.random_state)
            param_grid = grids.get(model_name, {})

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                verbose=0,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            duration = round(time.time() - start_time, 2)

            best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = float(grid_search.best_score_)

            # Compute CV standard deviation for best candidate
            best_idx = grid_search.best_index_
            cv_std = float(grid_search.cv_results_["std_test_score"][best_idx])

            # Evaluate untouched test set
            y_pred = best_pipeline.predict(X_test)
            classes = list(best_pipeline.classes_)
            pos_idx = classes.index(1) if 1 in classes else 1
            y_proba = best_pipeline.predict_proba(X_test)[:, pos_idx]

            test_metrics = compute_metrics(y_test, y_pred, y_proba)

            candidate_res = CandidateResult(
                model_name=model_name,
                best_params=best_params,
                cv_roc_auc_mean=round(best_score, 4),
                cv_roc_auc_std=round(cv_std, 4),
                training_duration_seconds=duration,
                test_metrics=test_metrics,
                best_estimator=best_pipeline,
            )

            self.results[model_name] = candidate_res

            logger.info(
                "[%s] Completed in %.2fs | Best CV ROC-AUC: %.4f (±%.4f) | Test ROC-AUC: %.4f | Test Acc: %.4f",
                model_name,
                duration,
                best_score,
                cv_std,
                test_metrics["roc_auc"],
                test_metrics["accuracy"],
            )

        # Select winner using Cross-Validation ROC-AUC only (not test metrics)
        self.best_model_name = max(
            self.results.keys(),
            key=lambda name: self.results[name].cv_roc_auc_mean,
        )
        self.best_pipeline = self.results[self.best_model_name].best_estimator

        logger.info("=" * 60)
        logger.info(
            "Selected Best Candidate: [%s] (Mean CV ROC-AUC: %.4f)",
            self.best_model_name,
            self.results[self.best_model_name].cv_roc_auc_mean,
        )
        logger.info("=" * 60)

        return self.results

    def generate_summary_table(self) -> str:
        """Generate human-readable console table comparing models."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("                  CARDIOVASCULAR MODEL COMPARISON BENCHMARK")
        lines.append("=" * 80)
        header = f"{'Model':<16} {'CV ROC-AUC':<16} {'Test ROC-AUC':<14} {'Accuracy':<10} {'F1':<8} {'Time (s)':<8}"
        lines.append(header)
        lines.append("-" * 80)

        for name, res in self.results.items():
            tm = res.test_metrics
            cv_str = f"{res.cv_roc_auc_mean:.4f} ± {res.cv_roc_auc_std:.4f}"
            row = (
                f"{name:<16} {cv_str:<16} {tm['roc_auc']:<14.4f} "
                f"{tm['accuracy']:<10.4f} {tm['f1_score']:<8.4f} {res.training_duration_seconds:<8.2f}"
            )
            lines.append(row)

        lines.append("-" * 80)
        if self.best_model_name:
            lines.append(
                f"SELECTED WINNER (CV ROC-AUC): {self.best_model_name} "
                f"({self.results[self.best_model_name].cv_roc_auc_mean:.4f})"
            )
        lines.append("=" * 80 + "\n")
        return "\n".join(lines)

    def to_export_dict(
        self,
        total_samples: int,
        train_samples: int,
        test_samples: int,
    ) -> Dict[str, Any]:
        """Convert results into structured JSON-serializable dictionary."""
        models_data = {}
        for name, res in self.results.items():
            models_data[name] = {
                "best_hyperparameters": res.best_params,
                "cv_roc_auc_mean": res.cv_roc_auc_mean,
                "cv_roc_auc_std": res.cv_roc_auc_std,
                "training_duration_seconds": res.training_duration_seconds,
                "test_metrics": res.test_metrics,
            }

        return {
            "dataset_size": total_samples,
            "training_sample_count": train_samples,
            "test_sample_count": test_samples,
            "random_seed": self.random_state,
            "cross_validation": {
                "method": "StratifiedKFold",
                "n_splits": self.cv_splits,
                "shuffle": True,
                "scoring": "roc_auc",
            },
            "selected_model_name": self.best_model_name,
            "selection_criterion": "highest_mean_cv_roc_auc",
            "models": models_data,
        }
