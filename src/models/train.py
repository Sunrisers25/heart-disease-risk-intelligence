"""Model training and tuning module.
Builds the complete end-to-end Pipeline: Preprocessor -> PCA -> SVC.
Performs 5-Fold StratifiedKFold GridSearchCV, evaluates performance,
and serializes artifacts to disk.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (
    CV_SPLITS,
    FEATURE_COLUMNS,
    MODEL_PATH,
    PARAM_GRID,
    PCA_VARIANCE,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_pipeline.loader import load_and_clean_data
from src.data_pipeline.preprocessor import create_preprocessor
from src.models.evaluate import compute_metrics, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Construct full scikit-learn Pipeline: Preprocessor -> PCA -> SVC.

    Transforms and scales features, reduces dimensionality retaining 95% variance,
    and applies RBF Support Vector Classifier with balanced class weights.
    """
    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def train_model(save_artifact: bool = True) -> Pipeline:
    """Train, tune, evaluate and serialize the heart disease risk model.

    Returns:
        Pipeline: Best fitted pipeline.
    """
    logger.info("Loading cleaned dataset...")
    df = load_and_clean_data(save_processed=True)

    X = df[FEATURE_COLUMNS]
    # Map Yes -> 1, No -> 0
    y = (df[TARGET_COLUMN] == "Yes").astype(int).values

    logger.info("Dataset shape: %s | Target distribution: %s", X.shape, np.bincount(y))

    # Split train and test set with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    base_pipeline = build_pipeline()

    # 5-fold Stratified K-Fold cross validation
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    logger.info("Running GridSearchCV over hyperparameter grid...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best Cross-Validation ROC-AUC: %.4f", grid_search.best_score_)

    best_pipeline = grid_search.best_estimator_

    # Evaluate on held-out test set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = grid_search.best_params_
    metrics["cv_roc_auc_score"] = round(float(grid_search.best_score_), 4)

    logger.info("Test set performance metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    if save_artifact:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_PATH)
        logger.info("Model pipeline successfully serialized to %s", MODEL_PATH)
        save_metrics(metrics)

    return best_pipeline


if __name__ == "__main__":
    train_model()
