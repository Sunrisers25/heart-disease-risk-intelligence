"""Tests for Model Explainability with SHAP (src/services/explainer.py).
Verifies:
- SHAP output dimensions.
- Feature name alignment with original clinical schema.
- Global importance generation.
- Individual prediction explanations.
- Missing values and input handling.
- Compatibility with SVM (with PCA) and Tree-based models (Random Forest, XGBoost).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS
from src.data_pipeline.loader import generate_synthetic_data
from src.models.model_factory import (
    create_rf_pipeline,
    create_svm_pipeline,
    create_xgb_pipeline,
)
from src.services.explainer import ModelExplainer


@pytest.fixture(scope="module")
def sample_dataset():
    """Generate small fixture dataset."""
    df = generate_synthetic_data(num_samples=35, random_state=42)
    X = df[FEATURE_COLUMNS]
    y = (df["Heart_Disease"] == "Yes").astype(int).values
    return X, y


@pytest.fixture(scope="module")
def fitted_svm(sample_dataset):
    X, y = sample_dataset
    pipe = create_svm_pipeline()
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def fitted_rf(sample_dataset):
    X, y = sample_dataset
    pipe = create_rf_pipeline()
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def fitted_xgb(sample_dataset):
    X, y = sample_dataset
    pipe = create_xgb_pipeline()
    pipe.fit(X, y)
    return pipe


def test_explainer_svm_instance_explanation(fitted_svm, sample_dataset):
    """Verify SVM with PCA explains predictions strictly on original 8 clinical features."""
    X, _ = sample_dataset
    patient = X.iloc[0].to_dict()

    explainer = ModelExplainer(pipeline=fitted_svm, background_data=X, background_size=15)
    explanation = explainer.explain_instance(patient, nsamples=30)

    assert "base_value" in explanation
    assert "patient_risk_prob" in explanation
    assert "feature_contributions" in explanation
    assert 0.0 <= explanation["patient_risk_prob"] <= 1.0

    contributions = explanation["feature_contributions"]
    assert len(contributions) == len(FEATURE_COLUMNS)

    # Verify all original clinical feature names are present
    explained_features = {c["feature"] for c in contributions}
    assert explained_features == set(FEATURE_COLUMNS)

    # Verify no PCA latent names appear
    for f in explained_features:
        assert not f.lower().startswith("pca")


def test_explainer_rf_instance_explanation(fitted_rf, sample_dataset):
    """Verify TreeExplainer works on Random Forest pipeline."""
    X, _ = sample_dataset
    patient = X.iloc[1].to_dict()

    explainer = ModelExplainer(pipeline=fitted_rf, background_data=X, background_size=15)
    explanation = explainer.explain_instance(patient)

    assert len(explanation["feature_contributions"]) == len(FEATURE_COLUMNS)
    assert 0.0 <= explanation["patient_risk_prob"] <= 1.0


def test_explainer_xgb_instance_explanation(fitted_xgb, sample_dataset):
    """Verify TreeExplainer works on XGBoost pipeline."""
    X, _ = sample_dataset
    patient = X.iloc[2].to_dict()

    explainer = ModelExplainer(pipeline=fitted_xgb, background_data=X, background_size=15)
    explanation = explainer.explain_instance(patient)

    assert len(explanation["feature_contributions"]) == len(FEATURE_COLUMNS)
    assert 0.0 <= explanation["patient_risk_prob"] <= 1.0


def test_compute_global_importance(fitted_svm, sample_dataset):
    """Verify global importance computes non-negative mean absolute SHAP values."""
    X, _ = sample_dataset
    explainer = ModelExplainer(pipeline=fitted_svm, background_data=X, background_size=15)
    imp_df = explainer.compute_global_importance(samples=X, max_samples=10, nsamples=25)

    assert isinstance(imp_df, pd.DataFrame)
    assert set(imp_df.columns) == {"Feature", "Mean_Abs_SHAP"}
    assert len(imp_df) == len(FEATURE_COLUMNS)
    assert (imp_df["Mean_Abs_SHAP"] >= 0).all()
    # Check that it is sorted descending
    assert imp_df["Mean_Abs_SHAP"].is_monotonic_decreasing


def test_explainer_input_dataframe_and_dict(fitted_svm, sample_dataset):
    """Ensure explain_instance accepts both DataFrame and dictionary inputs identically."""
    X, _ = sample_dataset
    explainer = ModelExplainer(pipeline=fitted_svm, background_data=X, background_size=15)

    res_dict = explainer.explain_instance(X.iloc[0].to_dict(), nsamples=20)
    res_df = explainer.explain_instance(X.iloc[[0]], nsamples=20)

    assert res_dict["patient_risk_prob"] == res_df["patient_risk_prob"]
    assert len(res_dict["feature_contributions"]) == len(res_df["feature_contributions"])
