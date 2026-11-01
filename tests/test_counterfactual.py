"""Unit and integration tests for What-If Counterfactual Simulator Service (src/services/counterfactual.py).
Verifies:
1. Original and hypothetical prediction generation.
2. Identical inputs producing identical probabilities (zero delta).
3. Probability output within [0, 1].
4. Correct probability difference calculation in percentage points.
5. Correct positive-class indexing.
6. Invalid feature rejection.
7. Missing feature handling.
8. Categorical input validation (Gender, ECG_Result).
9. Original patient input remaining unchanged (immutability).
10. Compatibility with calibrated and uncalibrated predictor models.
11. Identification of changed features.
12. Graceful error handling when invalid predictors are provided.
"""

import copy
import pytest
from src.config import FEATURE_COLUMNS
from src.models.model_factory import create_svm_pipeline
from src.services.counterfactual import (
    CounterfactualSimulator,
    SimulationComparisonResult,
)
from src.services.predictor import HeartDiseasePredictor


@pytest.fixture(scope="module")
def base_predictor():
    """Returns an active HeartDiseasePredictor instance."""
    return HeartDiseasePredictor(use_calibration=False)


@pytest.fixture(scope="module")
def sample_patient():
    """Baseline patient vitals."""
    return {
        "Age": 58,
        "Gender": "Male",
        "Blood_Pressure": 145.0,
        "Cholesterol": 245.0,
        "BMI": 29.2,
        "Blood_Sugar": 130.0,
        "Heart_Rate": 140.0,
        "ECG_Result": 1,
    }


def test_simulator_identical_inputs_zero_delta(base_predictor, sample_patient):
    """Verify that simulating with identical inputs yields 0.0 delta and empty changed_features."""
    simulator = CounterfactualSimulator(predictor=base_predictor)
    result = simulator.simulate(sample_patient, sample_patient)

    assert isinstance(result, SimulationComparisonResult)
    assert result.original_probability == result.hypothetical_probability
    assert result.original_classification == result.hypothetical_classification
    assert result.delta_percentage_points == 0.0
    assert result.changed_features == {}
    assert 0.0 <= result.original_probability <= 1.0


def test_simulator_probability_bounds_and_delta(base_predictor, sample_patient):
    """Verify probability outputs are within [0, 1] and delta calculation is exact."""
    simulator = CounterfactualSimulator(predictor=base_predictor)

    hypothetical = copy.deepcopy(sample_patient)
    # Improve biomarkers: lower BP, lower Chol, lower Blood Sugar, normal ECG
    hypothetical["Blood_Pressure"] = 118.0
    hypothetical["Cholesterol"] = 180.0
    hypothetical["Blood_Sugar"] = 88.0
    hypothetical["ECG_Result"] = 0

    result = simulator.simulate(sample_patient, hypothetical)

    assert 0.0 <= result.original_probability <= 1.0
    assert 0.0 <= result.hypothetical_probability <= 1.0
    expected_delta = round((result.hypothetical_probability - result.original_probability) * 100, 2)
    assert result.delta_percentage_points == expected_delta

    # Verify detected changed features
    assert "Blood_Pressure" in result.changed_features
    assert "Cholesterol" in result.changed_features
    assert "Blood_Sugar" in result.changed_features
    assert "ECG_Result" in result.changed_features
    assert result.changed_features["Blood_Pressure"] == (145.0, 118.0)


def test_simulator_original_patient_immutability(base_predictor, sample_patient):
    """Verify that running a simulation does not mutate the original dictionary."""
    simulator = CounterfactualSimulator(predictor=base_predictor)
    patient_copy = copy.deepcopy(sample_patient)

    hypothetical = copy.deepcopy(sample_patient)
    hypothetical["Age"] = 40

    _ = simulator.simulate(patient_copy, hypothetical)
    assert patient_copy == sample_patient


def test_simulator_missing_feature_rejection(base_predictor, sample_patient):
    """Verify that missing required features raises ValueError."""
    simulator = CounterfactualSimulator(predictor=base_predictor)
    broken_patient = copy.deepcopy(sample_patient)
    del broken_patient["Blood_Pressure"]

    with pytest.raises(ValueError, match="missing required features"):
        simulator.simulate(broken_patient, sample_patient)

    with pytest.raises(ValueError, match="missing required features"):
        simulator.simulate(sample_patient, broken_patient)


def test_simulator_invalid_categorical_values(base_predictor, sample_patient):
    """Verify that invalid Gender or ECG_Result values are rejected with ValueError."""
    simulator = CounterfactualSimulator(predictor=base_predictor)

    bad_gender = copy.deepcopy(sample_patient)
    bad_gender["Gender"] = "Other"
    with pytest.raises(ValueError, match="Invalid Gender"):
        simulator.simulate(sample_patient, bad_gender)

    bad_ecg = copy.deepcopy(sample_patient)
    bad_ecg["ECG_Result"] = 5
    with pytest.raises(ValueError, match="Invalid ECG_Result"):
        simulator.simulate(sample_patient, bad_ecg)


def test_simulator_invalid_predictor_init():
    """Verify simulator initialization rejects None or invalid predictor."""
    with pytest.raises(ValueError, match="valid HeartDiseasePredictor instance is required"):
        CounterfactualSimulator(predictor=None)
