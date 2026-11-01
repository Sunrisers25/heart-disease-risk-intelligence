"""Integration tests for model predictor and business logic services."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import train_model
from src.services.predictor import HeartDiseasePredictor
from src.services.recommender import generate_recommendations


@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    """Ensure a trained pipeline exists and initialize the predictor."""
    # Ensure trained model artifact is created
    train_model(save_artifact=True)
    return HeartDiseasePredictor()


def test_predictor_single_sample_range_and_probabilities(trained_predictor):
    """Verify predictor outputs valid classification and probability between 0 and 1."""
    patient = {
        "Age": 55,
        "Gender": "Male",
        "Blood_Pressure": 140,
        "Cholesterol": 240,
        "BMI": 28.5,
        "Blood_Sugar": 130,
        "Heart_Rate": 150,
        "ECG_Result": 1,
    }

    classification, probability = trained_predictor.predict(patient)

    assert classification in {"High Risk", "Low Risk"}
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_predictor_batch_probabilities_sum(trained_predictor):
    """Verify batch predict probabilities adhere to probability constraints."""
    patients = pd.DataFrame([
        {
            "Age": 45,
            "Gender": "Female",
            "Blood_Pressure": 115,
            "Cholesterol": 180,
            "BMI": 22.0,
            "Blood_Sugar": 90,
            "Heart_Rate": 160,
            "ECG_Result": 0,
        },
        {
            "Age": 68,
            "Gender": "Male",
            "Blood_Pressure": 165,
            "Cholesterol": 290,
            "BMI": 33.0,
            "Blood_Sugar": 180,
            "Heart_Rate": 120,
            "ECG_Result": 2,
        },
    ])

    labels, probs = trained_predictor.predict_batch(patients)
    assert len(labels) == 2
    assert len(probs) == 2
    for p in probs:
        assert 0.0 <= p <= 1.0

    # Test full pipeline underlying predict_proba row-wise sums to 1.0
    underlying_probs = trained_predictor.pipeline.predict_proba(patients)
    for row in underlying_probs:
        assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_recommender_guidelines():
    """Verify medical guidelines generate appropriate alerts for clinical thresholds."""
    critical_patient = {
        "Blood_Pressure": 150,
        "Cholesterol": 260,
        "Blood_Sugar": 140,
        "BMI": 31.0,
        "ECG_Result": 2,
    }
    recs = generate_recommendations(critical_patient, risk_classification="High Risk")
    categories = [r["category"] for r in recs]

    assert "Blood Pressure" in categories
    assert "Cholesterol" in categories
    assert "Blood Sugar" in categories
    assert "BMI & Weight" in categories
    assert "Cardiology Urgent Action" in categories
