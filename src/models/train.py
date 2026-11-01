"""Model training and comparison orchestrator.
Trains and cross-validates SVM, Random Forest, and XGBoost.
Selects the best performing candidate based on CV ROC-AUC.
Calibrates probabilities strictly on training data using CalibratedClassifierCV.
Preserves existing SVM artifacts and creates Phase 2 model comparison & calibration artifacts.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
    CALIBRATED_MODEL_PATH,
    CALIBRATION_METADATA_PATH,
    CALIBRATION_METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
)
from src.data_pipeline.loader import load_and_clean_data
from src.models.calibration import ModelCalibrator
from src.models.model_comparison import ModelComparisonFramework

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(save_artifact: bool = True) -> Tuple[Pipeline, ModelComparisonFramework]:
    """Train, cross-validate, compare SVM, RF, XGBoost, and calibrate the winning model.

    Preserves `artifacts/svm_pca_pipeline.joblib` for backward compatibility.
    Saves:
      - `artifacts/svm_pca_pipeline.joblib` (baseline SVM pipeline)
      - `artifacts/best_model_pipeline.joblib` (selected winning pipeline)
      - `artifacts/calibrated_model_pipeline.joblib` (calibrated winning pipeline)
      - `artifacts/calibration_metrics.json` (calibration performance comparison)
      - `artifacts/calibration_metadata.json` (calibration metadata & parameters)
      - `artifacts/model_comparison.json` (benchmark metrics for all models)
      - `artifacts/best_model_metadata.json` (metadata and config of winning model)

    Returns:
        Tuple[Pipeline, ModelComparisonFramework]:
            - Winning Pipeline
            - Executed comparison framework with structured results
    """
    logger.info("Initiating Phase 2 Model Comparison Pipeline...")
    df = load_and_clean_data(save_processed=True)

    framework = ModelComparisonFramework()
    X_train, X_test, y_train, y_test = framework.prepare_data(df)

    # 1. Execute GridSearchCV across all candidates
    framework.run_comparison(X_train, X_test, y_train, y_test)

    # Print summary table
    summary_table = framework.generate_summary_table()
    print(summary_table)

    best_pipeline = framework.best_pipeline
    best_name = framework.best_model_name

    # 2. Probability Calibration on the selected best model
    logger.info("Initiating Probability Calibration on Selected Model [%s]...", best_name)
    calibrator = ModelCalibrator(base_pipeline=best_pipeline)

    # Evaluate calibration strictly on training data
    calibrator.evaluate_training_cv_calibration(X_train, y_train)

    # Fit final calibrated model on training data
    calibrator.fit_calibrated_model(X_train, y_train)

    # Evaluate calibration on untouched test set
    cal_test_results = calibrator.evaluate_test_set(X_test, y_test)

    logger.info("=" * 60)
    logger.info("CALIBRATION EVALUATION (UNTOUCHED TEST SET)")
    logger.info("=" * 60)
    logger.info(
        "Uncalibrated -> Brier: %.4f | Log Loss: %.4f | ECE: %.4f",
        cal_test_results["uncalibrated"]["brier_score"],
        cal_test_results["uncalibrated"]["log_loss"],
        cal_test_results["uncalibrated"]["expected_calibration_error"],
    )
    logger.info(
        "Calibrated   -> Brier: %.4f | Log Loss: %.4f | ECE: %.4f (Method: %s)",
        cal_test_results["calibrated"]["brier_score"],
        cal_test_results["calibrated"]["log_loss"],
        cal_test_results["calibrated"]["expected_calibration_error"],
        cal_test_results["calibration_method"],
    )
    logger.info("=" * 60)

    if save_artifact:
        ARTIFACTS_DIR = MODEL_PATH.parent
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Preserve/save baseline SVM artifact for full backward compatibility
        if "SVM" in framework.results and framework.results["SVM"].best_estimator is not None:
            svm_pipeline = framework.results["SVM"].best_estimator
            joblib.dump(svm_pipeline, MODEL_PATH)
            logger.info("Saved baseline SVM artifact to %s", MODEL_PATH)

        # 2. Save winning model pipeline
        if best_pipeline is not None:
            joblib.dump(best_pipeline, BEST_MODEL_PATH)
            logger.info("Saved best performing [%s] pipeline to %s", best_name, BEST_MODEL_PATH)

        # 3. Save calibrated model pipeline and calibration metrics
        calibrator.save_artifacts(test_comparison_results=cal_test_results)

        # 4. Save comparison benchmark JSON
        export_data = framework.to_export_dict(
            total_samples=len(df),
            train_samples=len(X_train),
            test_samples=len(X_test),
        )
        with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
        logger.info("Saved comparison metrics to %s", MODEL_COMPARISON_PATH)

        # 5. Save metadata of winning model
        winner_result = framework.results[best_name]
        metadata = {
            "model_name": best_name,
            "trained_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "selection_metric": "cv_roc_auc_mean",
            "cv_roc_auc_mean": winner_result.cv_roc_auc_mean,
            "cv_roc_auc_std": winner_result.cv_roc_auc_std,
            "best_hyperparameters": winner_result.best_params,
            "test_metrics": winner_result.test_metrics,
            "training_duration_seconds": winner_result.training_duration_seconds,
            "pipeline_steps": [name for name, _ in best_pipeline.steps],
        }
        with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved best model metadata to %s", BEST_MODEL_METADATA_PATH)

    return best_pipeline, framework


if __name__ == "__main__":
    train_model(save_artifact=True)
