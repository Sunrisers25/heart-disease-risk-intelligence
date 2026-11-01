"""Unit and integration tests for Model Factory, Model Comparison Framework,
and multi-model pipelines (SVM, Random Forest, XGBoost).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS
from src.data_pipeline.loader import generate_synthetic_data
from src.models.evaluate import compute_metrics
from src.models.model_comparison import (
    CandidateResult,
    ModelComparisonFramework,
)
from src.models.model_factory import (
    create_rf_pipeline,
    create_svm_pipeline,
    create_xgb_pipeline,
    get_model_pipeline,
)


@pytest.fixture(scope="module")
def sample_data():
    """Generates a small synthetic fixture for fast unit testing."""
    df = generate_synthetic_data(num_samples=40, random_state=42)
    X = df[FEATURE_COLUMNS]
    y = (df["Heart_Disease"] == "Yes").astype(int).values
    return X, y


# 1. Pipeline construction tests
def test_svm_pipeline_construction():
    """Verify SVM pipeline has Preprocessor, PCA, and SVC steps."""
    pipe = create_svm_pipeline()
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocessor", "pca", "classifier"]


def test_random_forest_pipeline_construction():
    """Verify Random Forest pipeline has Preprocessor and RandomForestClassifier steps without PCA."""
    pipe = create_rf_pipeline()
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocessor", "classifier"]
    assert "pca" not in step_names


def test_xgboost_pipeline_construction():
    """Verify XGBoost pipeline has Preprocessor and XGBClassifier steps without PCA."""
    pipe = create_xgb_pipeline()
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocessor", "classifier"]
    assert "pca" not in step_names


def test_get_model_pipeline_helper():
    """Verify get_model_pipeline resolves valid model names and raises on invalid ones."""
    assert isinstance(get_model_pipeline("svm"), Pipeline)
    assert isinstance(get_model_pipeline("Random Forest"), Pipeline)
    assert isinstance(get_model_pipeline("xgboost"), Pipeline)
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_model_pipeline("invalid_model_type")


# 2. Input schema and inference checks
@pytest.mark.parametrize("model_name", ["svm", "random_forest", "xgboost"])
def test_pipelines_support_schema_and_predict_valid_classes(model_name, sample_data):
    """Ensure all models fit on training schema and output valid binary predictions."""
    X, y = sample_data
    pipe = get_model_pipeline(model_name)
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Check row-sum to 1.0
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


# 3. Evaluation metrics calculation and confusion matrix dimensions
def test_compute_metrics_accuracy_and_dimensions():
    """Verify compute_metrics calculates accuracy, F1, ROC-AUC, sensitivity, specificity, and confusion matrix."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.85])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)
    # Specific checks: 3 TN, 0 FP, 1 FN, 2 TP
    assert cm[0, 0] == 3
    assert cm[0, 1] == 0
    assert cm[1, 0] == 1
    assert cm[1, 1] == 2
    assert metrics["sensitivity"] == pytest.approx(2 / 3, abs=0.01)
    assert metrics["specificity"] == 1.0


# 4. Reproducible splits with same random seed
def test_train_test_split_reproducibility():
    """Verify identical random state reproduces exact same train/test partitions."""
    df = generate_synthetic_data(num_samples=50, random_state=42)
    framework1 = ModelComparisonFramework(random_state=42, test_size=0.2)
    X_train1, X_test1, y_train1, y_test1 = framework1.prepare_data(df)

    framework2 = ModelComparisonFramework(random_state=42, test_size=0.2)
    X_train2, X_test2, y_train2, y_test2 = framework2.prepare_data(df)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


# 5. Fast model comparison execution and CV-based winner selection
def test_model_comparison_execution_and_json_export(sample_data):
    """Run miniature grid search comparison to verify selection logic and JSON export validity."""
    X, y = sample_data
    # Use small subset for rapid unit test
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y[:30], y[30:]

    framework = ModelComparisonFramework(random_state=42, cv_splits=2)

    # Miniature hyperparameter grids for fast testing
    mini_grids = {
        "SVM": {"classifier__C": [1.0]},
        "Random Forest": {"classifier__n_estimators": [10]},
        "XGBoost": {"classifier__n_estimators": [10], "classifier__max_depth": [2]},
    }

    results = framework.run_comparison(X_train, X_test, y_train, y_test, custom_grids=mini_grids)

    assert len(results) == 3
    assert set(results.keys()) == {"SVM", "Random Forest", "XGBoost"}
    assert framework.best_model_name in {"SVM", "Random Forest", "XGBoost"}
    assert isinstance(framework.best_pipeline, Pipeline)

    # Verify selection is strictly based on mean CV ROC-AUC
    expected_winner = max(results.keys(), key=lambda name: results[name].cv_roc_auc_mean)
    assert framework.best_model_name == expected_winner

    # Verify export dictionary is JSON-serializable
    export_dict = framework.to_export_dict(len(X), len(X_train), len(X_test))
    json_str = json.dumps(export_dict)
    assert json_str is not None
    assert export_dict["selected_model_name"] == expected_winner
    assert "models" in export_dict
    assert len(export_dict["models"]) == 3
