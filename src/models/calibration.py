"""Probability calibration framework for biomedical machine learning models.
Implements:
- Training-only calibration selection (Sigmoid vs Isotonic vs Uncalibrated).
- Cross-validated calibration wrapping complete pipelines to prevent data leakage.
- Independent nested sub-splitting for FrozenEstimator calibration to eliminate training leakage.
- Direct external calibration on uncalibrated estimators (decision_function) avoiding double-Platt calibration.
- Comprehensive reliability metrics: Expected Calibration Error (ECE), Brier Score, Log Loss.
- Calibration curve generation and artifact persistence.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CALIBRATED_MODEL_PATH,
    CALIBRATION_METADATA_PATH,
    CALIBRATION_METRICS_PATH,
    CV_SPLITS,
    RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Calculate Expected Calibration Error (ECE) with equal-width binning.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability for the positive class.
        n_bins: Number of probability bins (default: 5 for small sample cohorts).

    Returns:
        float: Expected Calibration Error (0.0 to 1.0).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(round(ece, 4))


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Compute reliability and discrimination metrics for predicted probabilities.

    Metrics: Brier Score, Log Loss, ROC-AUC, ECE, Accuracy, Precision, Recall, F1,
    Mean Predicted Probability, Observed Positive Frequency, and Calibration Curve coordinates.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    brier = float(brier_score_loss(y_true, y_prob))
    eps = 1e-15
    clipped_prob = np.clip(y_prob, eps, 1 - eps)
    ll = float(log_loss(y_true, clipped_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, TypeError):
        auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "expected_calibration_error": round(ece, 4),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "mean_predicted_prob": round(float(np.mean(y_prob)), 4),
        "observed_positive_rate": round(float(np.mean(y_true)), 4),
        "calibration_curve": {
            "fraction_of_positives": [round(float(v), 4) for v in prob_true],
            "mean_predicted_value": [round(float(v), 4) for v in prob_pred],
        },
    }


def prepare_uncalibrated_estimator(base_pipeline: Pipeline) -> Pipeline:
    """Ensure estimator does not internally pre-calibrate (e.g., SVC probability=False)
    so external CalibratedClassifierCV calibrates raw decision function scores without double-Platt stacking.
    """
    pipe = clone(base_pipeline)
    clf = pipe.named_steps["classifier"]
    if hasattr(clf, "probability") and clf.probability is True:
        logger.info("Setting SVC(probability=False) to calibrate raw decision_function externally.")
        clf.set_params(probability=False)
    return pipe


class ModelCalibrator:
    """Manages training-only calibration, selection, and evaluation.

    Protects against data leakage:
    1. Outer validation fold (X_val, y_val) is strictly held out.
    2. Outer training fold (X_tr, y_tr) is partitioned into an independent model training subset (75%)
       and an independent calibration subset (25%).
    3. The base pipeline is fit strictly on the model-training subset.
    4. The calibrator is fit strictly on the independent calibration subset.
    5. The final test set (X_test, y_test) is untouched during calibration fitting and selection.
    """

    def __init__(
        self,
        base_pipeline: Pipeline,
        cv_splits: int = CV_SPLITS,
        random_state: int = RANDOM_STATE,
        n_bins: int = 5,
    ):
        self.base_pipeline = base_pipeline
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_bins = n_bins

        self.cv_evaluations: Dict[str, Dict[str, Any]] = {}
        self.best_method: Optional[str] = None
        self.calibrated_pipeline: Optional[CalibratedClassifierCV] = None
        self.train_samples: int = 0
        self.test_samples: int = 0

    def evaluate_training_cv_calibration(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate Uncalibrated, Sigmoid, and Isotonic calibration strictly on training data
        using leak-free independent nested calibration splits across K folds.
        """
        self.train_samples = len(X_train)
        logger.info(
            "Evaluating probability calibration methods via %d-fold nested CV on %d training samples...",
            self.cv_splits,
            self.train_samples,
        )
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        oof_probs = {
            "uncalibrated": np.zeros(len(y_train)),
            "sigmoid": np.zeros(len(y_train)),
            "isotonic": np.zeros(len(y_train)),
        }

        uncal_pipe_template = prepare_uncalibrated_estimator(self.base_pipeline)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

            # 1. Evaluate uncalibrated baseline: fit on full fold-train, predict on fold-val
            base_fold = clone(self.base_pipeline)
            base_fold.fit(X_tr, y_tr)
            pos_idx = list(base_fold.classes_).index(1) if 1 in base_fold.classes_ else 1
            oof_probs["uncalibrated"][val_idx] = base_fold.predict_proba(X_val)[:, pos_idx]

            # 2. Independent Calibration Nested Sub-Split:
            # Partition X_tr into model training sub-split (75%) and calibration sub-split (25%)
            X_sub_tr, X_sub_cal, y_sub_tr, y_sub_cal = train_test_split(
                X_tr,
                y_tr,
                test_size=0.25,
                random_state=self.random_state + fold_idx,
                stratify=y_tr,
            )

            # Fit base pipeline ONLY on the model training sub-split
            model_sub = clone(uncal_pipe_template)
            model_sub.fit(X_sub_tr, y_sub_tr)

            # Fit Sigmoid and Isotonic Calibrators strictly on independent calibration sub-split
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                cal_sig = CalibratedClassifierCV(
                    estimator=model_sub,
                    method="sigmoid",
                    cv="prefit",
                )
                cal_sig.fit(X_sub_cal, y_sub_cal)
                oof_probs["sigmoid"][val_idx] = cal_sig.predict_proba(X_val)[:, pos_idx]

                cal_iso = CalibratedClassifierCV(
                    estimator=model_sub,
                    method="isotonic",
                    cv="prefit",
                )
                cal_iso.fit(X_sub_cal, y_sub_cal)
                oof_probs["isotonic"][val_idx] = cal_iso.predict_proba(X_val)[:, pos_idx]

        # Calculate Out-Of-Fold metrics for calibration selection
        self.cv_evaluations = {}
        for method, probs in oof_probs.items():
            metrics = compute_calibration_metrics(y_train, probs, n_bins=self.n_bins)
            self.cv_evaluations[method] = metrics
            logger.info(
                "Training OOF [%s] -> Brier Score: %.4f | Log Loss: %.4f | ECE: %.4f | ROC-AUC: %.4f",
                method,
                metrics["brier_score"],
                metrics["log_loss"],
                metrics["expected_calibration_error"],
                metrics["roc_auc"],
            )

        # Select method with lowest Out-Of-Fold Brier score on training data
        self.best_method = min(
            ["sigmoid", "isotonic", "uncalibrated"],
            key=lambda m: self.cv_evaluations[m]["brier_score"],
        )

        logger.info(
            "Selected Best Calibration Method (Training OOF Brier Score): [%s] (Brier=%.4f)",
            self.best_method,
            self.cv_evaluations[self.best_method]["brier_score"],
        )
        return self.cv_evaluations

    def fit_calibrated_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        method: Optional[str] = None,
    ) -> CalibratedClassifierCV:
        """Fit final CalibratedClassifierCV strictly on training dataset using internal Stratified CV
        and raw decision function scores to avoid double-Platt calibration.
        """
        chosen_method = method or self.best_method or "sigmoid"
        cal_method = chosen_method if chosen_method in {"sigmoid", "isotonic"} else "sigmoid"

        logger.info(
            "Fitting final CalibratedClassifierCV(method='%s', cv=%d) on %d training samples...",
            cal_method,
            self.cv_splits,
            len(X_train),
        )

        cv_inner = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        uncal_pipe = prepare_uncalibrated_estimator(self.base_pipeline)

        calibrated_clf = CalibratedClassifierCV(
            estimator=uncal_pipe,
            method=cal_method,
            cv=cv_inner,
        )

        calibrated_clf.fit(X_train, y_train)
        self.calibrated_pipeline = calibrated_clf
        return calibrated_clf

    def evaluate_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate uncalibrated pipeline and calibrated pipeline on the untouched test set."""
        if self.calibrated_pipeline is None:
            raise ValueError("Calibrated model is not fitted yet. Call fit_calibrated_model first.")

        self.test_samples = len(X_test)

        # 1. Uncalibrated test metrics
        pos_idx = list(self.base_pipeline.classes_).index(1) if 1 in self.base_pipeline.classes_ else 1
        y_prob_uncal = self.base_pipeline.predict_proba(X_test)[:, pos_idx]
        uncal_metrics = compute_calibration_metrics(y_test, y_prob_uncal, n_bins=self.n_bins)

        # 2. Calibrated test metrics
        cal_classes = list(self.calibrated_pipeline.classes_)
        cal_pos_idx = cal_classes.index(1) if 1 in cal_classes else 1
        y_prob_cal = self.calibrated_pipeline.predict_proba(X_test)[:, cal_pos_idx]
        cal_metrics = compute_calibration_metrics(y_test, y_prob_cal, n_bins=self.n_bins)

        return {
            "uncalibrated": uncal_metrics,
            "calibrated": cal_metrics,
            "calibration_method": self.best_method or "sigmoid",
        }

    def save_artifacts(
        self,
        test_comparison_results: Dict[str, Any],
        model_path: Path = CALIBRATED_MODEL_PATH,
        metrics_path: Path = CALIBRATION_METRICS_PATH,
        metadata_path: Path = CALIBRATION_METADATA_PATH,
    ) -> None:
        """Persist calibrated pipeline, calibration metrics, and rich metadata to artifacts/."""
        model_path = Path(model_path)
        metrics_path = Path(metrics_path)
        metadata_path = Path(metadata_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.calibrated_pipeline is not None:
            joblib.dump(self.calibrated_pipeline, model_path)
            logger.info("Saved calibrated model pipeline to %s", model_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_comparison_results, f, indent=4)
        logger.info("Saved calibration metrics to %s", metrics_path)

        metadata = {
            "calibrated_model_type": self.calibrated_pipeline.__class__.__name__,
            "base_estimator": self.base_pipeline.named_steps["classifier"].__class__.__name__,
            "calibration_method": self.best_method,
            "training_sample_count": self.train_samples,
            "test_sample_count": self.test_samples,
            "calibration_sample_strategy": "independent_nested_subsplit_25pct",
            "calibration_cv_splits": self.cv_splits,
            "ece_bin_count": self.n_bins,
            "random_seed": self.random_state,
            "svm_uses_external_calibration": True,
            "svc_internal_probability_disabled": True,
            "calibration_selection_methodology": "lowest_training_oof_brier_score",
            "training_cv_evaluations": self.cv_evaluations,
            "test_evaluation": test_comparison_results,
            "limitations_and_disclaimer": (
                "Experimental research prototype evaluated using synthetic benchmark data (N=300). "
                "Probabilities and feature explanations are not clinically validated and must not "
                "be used for patient diagnosis or clinical treatment decisions."
            ),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved calibration metadata to %s", metadata_path)
