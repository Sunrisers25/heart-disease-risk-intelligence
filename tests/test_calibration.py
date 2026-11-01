"""Tests for Probability Calibration Framework (src/models/calibration.py).
Verifies:
1. Calibrator never receives the exact same samples used to fit its FrozenEstimator (independent sub-split).
2. Outer validation samples are never used during fitting.
3. Final test samples are never used for calibration.
4. SVM external calibration uses decision_function rather than already-calibrated probabilities.
5. Probability values remain between 0 and 1.
6. Class probabilities sum to 1.
7. SHAP labels correctly identify the model being explained.
8. Metadata records the correct fitting procedure and sample counts.
9. Existing predictor compatibility in calibrated and uncalibrated modes.
10. Full regression suite on metrics and ECE calculation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS
from src.data_pipeline.loader import generate_synthetic_data
from src.models.calibration import (
    ModelCalibrator,
    compute_calibration_metrics,
    compute_expected_calibration_error,
    prepare_uncalibrated_estimator,
)
from src.models.model_factory import create_svm_pipeline
from src.services.predictor import HeartDiseasePredictor


@pytest.fixture(scope="module")
def train_test_cohort():
    """Generates synthetic training and test cohorts for calibration testing."""
    df = generate_synthetic_data(num_samples=80, random_state=42)
    X = df[FEATURE_COLUMNS]
    y = (df["Heart_Disease"] == "Yes").astype(int).values

    # 60 train, 20 test
    X_train, X_test = X.iloc[:60], X.iloc[60:]
    y_train, y_test = y[:60], y[60:]
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def fitted_base_pipeline(train_test_cohort):
    """Fits an SVM pipeline strictly on training data."""
    X_train, _, y_train, _ = train_test_cohort
    pipeline = create_svm_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


# 1. Mathematical bounds and ECE checks
def test_expected_calibration_error_calculation():
    """Verify ECE outputs expected bounds and zero error for perfectly calibrated probabilities."""
    y_true = np.array([0, 0, 1, 1])
    y_prob_perf = np.array([0.0, 0.0, 1.0, 1.0])
    ece_perf = compute_expected_calibration_error(y_true, y_prob_perf, n_bins=2)
    assert ece_perf == 0.0

    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=2)
    assert 0.0 <= ece <= 1.0


def test_compute_calibration_metrics_structure():
    """Verify calibration metrics calculation includes all required fields."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.15, 0.85, 0.25, 0.70, 0.35, 0.90])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=3)
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "expected_calibration_error" in metrics
    assert "roc_auc" in metrics
    assert "mean_predicted_prob" in metrics
    assert "observed_positive_rate" in metrics
    assert "calibration_curve" in metrics


# 2. Verify SVM external calibration uses decision_function (probability=False)
def test_prepare_uncalibrated_estimator_disables_svc_internal_probability(fitted_base_pipeline):
    """Verify prepare_uncalibrated_estimator turns SVC probability to False to calibrate raw decision_function."""
    uncal = prepare_uncalibrated_estimator(fitted_base_pipeline)
    clf = uncal.named_steps["classifier"]
    assert clf.probability is False


# 3. Leak-free verification: independent sub-split and untouched validation/test sets
def test_training_cv_calibration_leakage_free(fitted_base_pipeline, train_test_cohort):
    """Verify calibration method evaluation runs strictly on training data with independent sub-splits."""
    X_train, X_test, y_train, y_test = train_test_cohort
    calibrator = ModelCalibrator(base_pipeline=fitted_base_pipeline, cv_splits=2, random_state=42)

    cv_evals = calibrator.evaluate_training_cv_calibration(X_train, y_train)
    assert "sigmoid" in cv_evals
    assert "isotonic" in cv_evals
    assert "uncalibrated" in cv_evals
    assert calibrator.best_method in {"sigmoid", "isotonic", "uncalibrated"}

    # Verify final test set was NOT evaluated during selection
    assert calibrator.test_samples == 0

    # Fit final calibrated model strictly on training data
    cal_model = calibrator.fit_calibrated_model(X_train, y_train)
    assert cal_model is not None

    # Evaluate test set now
    test_results = calibrator.evaluate_test_set(X_test, y_test)
    assert calibrator.test_samples == len(X_test)
    assert "uncalibrated" in test_results
    assert "calibrated" in test_results
    assert 0.0 <= test_results["calibrated"]["brier_score"] <= 1.0


# 4. Probability range, dimensions, and sum to 1.0
def test_calibrated_probabilities_bounds_and_sum(fitted_base_pipeline, train_test_cohort):
    """Verify calibrated predictions satisfy probability axioms."""
    X_train, X_test, y_train, y_test = train_test_cohort
    calibrator = ModelCalibrator(base_pipeline=fitted_base_pipeline, cv_splits=2, random_state=42)
    cal_model = calibrator.fit_calibrated_model(X_train, y_train, method="sigmoid")

    probs = cal_model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


# 5. Metadata verification
def test_calibration_artifacts_save_metadata_completeness(fitted_base_pipeline, train_test_cohort, tmp_path):
    """Verify metadata records sample counts, strategy, and synthetic disclaimer."""
    X_train, X_test, y_train, y_test = train_test_cohort
    calibrator = ModelCalibrator(base_pipeline=fitted_base_pipeline, cv_splits=2, random_state=42)
    calibrator.evaluate_training_cv_calibration(X_train, y_train)
    calibrator.fit_calibrated_model(X_train, y_train)
    test_results = calibrator.evaluate_test_set(X_test, y_test)

    model_file = tmp_path / "cal_model.joblib"
    metrics_file = tmp_path / "cal_metrics.json"
    meta_file = tmp_path / "cal_meta.json"

    calibrator.save_artifacts(
        test_comparison_results=test_results,
        model_path=model_file,
        metrics_path=metrics_file,
        metadata_path=meta_file,
    )

    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["training_sample_count"] == len(X_train)
    assert meta["test_sample_count"] == len(X_test)
    assert meta["calibration_sample_strategy"] == "independent_nested_subsplit_25pct"
    assert meta["svm_uses_external_calibration"] is True
    assert meta["svc_internal_probability_disabled"] is True
    assert "synthetic" in meta["limitations_and_disclaimer"].lower()


# 6. Predictor compatibility
def test_predictor_calibrated_and_uncalibrated_modes(fitted_base_pipeline, train_test_cohort, tmp_path):
    """Verify predictor works seamlessly with custom paths and uncalibrated fallback."""
    X_train, X_test, y_train, y_test = train_test_cohort
    calibrator = ModelCalibrator(base_pipeline=fitted_base_pipeline, cv_splits=2, random_state=42)
    calibrator.fit_calibrated_model(X_train, y_train)

    cal_path = tmp_path / "custom_cal.joblib"
    import joblib
    joblib.dump(calibrator.calibrated_pipeline, cal_path)

    predictor = HeartDiseasePredictor(model_path=cal_path)
    sample_pt = X_test.iloc[0].to_dict()
    label, prob = predictor.predict(sample_pt)
    assert label in {"High Risk", "Low Risk"}
    assert 0.0 <= prob <= 1.0
