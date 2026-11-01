"""Tests for Cardiovascular Risk Assessment FastAPI Endpoints (/api/v1/predict and /api/v1/predict/batch)."""

from unittest.mock import patch
import pytest
from fastapi import HTTPException, status
from starlette.testclient import TestClient

from backend.core.dependencies import ModelContainer
from backend.main import app


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def valid_patient_dict():
    """Synthetic patient biomarker payload."""
    return {
        "Age": 58,
        "Gender": "Male",
        "Blood_Pressure": 138.0,
        "Cholesterol": 242.0,
        "BMI": 28.4,
        "Blood_Sugar": 130.0,
        "Heart_Rate": 145.0,
        "ECG_Result": 1,
        "Patient_ID": "P-TEST-001",
    }


def test_predict_single_valid_uncalibrated(client, valid_patient_dict):
    """Test standard uncalibrated risk prediction for a single patient."""
    payload = {
        "patient": valid_patient_dict,
        "calibrate": False,
        "log_audit": True,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["patient_id"] == "P-TEST-001"
    assert data["risk_classification"] in {"High Risk", "Low Risk"}
    assert data["predicted_class"] in {0, 1}
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert 0.0 <= data["confidence_percentage"] <= 100.0
    assert data["is_calibrated"] is False
    assert "Research prototype" in data["disclaimer"]
    assert "model_metadata" in data
    assert len(data["recommendations"]) > 0
    assert data["audit_record_id"] is not None


def test_predict_single_valid_calibrated(client, valid_patient_dict):
    """Test Platt-calibrated risk prediction for a single patient."""
    payload = {
        "patient": valid_patient_dict,
        "calibrate": True,
        "log_audit": False,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["is_calibrated"] is True
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["audit_record_id"] is None


def test_predict_single_flat_payload_compatibility(client, valid_patient_dict):
    """Verify endpoint accepts flat biomarker payloads without nested 'patient' object."""
    flat_payload = valid_patient_dict.copy()
    flat_payload["calibrate"] = False
    flat_payload["log_audit"] = False

    response = client.post("/api/v1/predict", json=flat_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["patient_id"] == "P-TEST-001"
    assert data["predicted_class"] in {0, 1}


def test_predict_single_missing_required_fields(client, valid_patient_dict):
    """Reject request missing required biomarkers with HTTP 422 and structured error."""
    bad_payload = {"patient": valid_patient_dict.copy()}
    del bad_payload["patient"]["Blood_Pressure"]

    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert any("Blood_Pressure" in str(d["field"]) or "blood_pressure" in str(d["field"]) for d in data["details"])


def test_predict_single_invalid_data_types(client, valid_patient_dict):
    """Reject non-numeric input types for numeric biomarkers."""
    bad_payload = {"patient": valid_patient_dict.copy()}
    bad_payload["patient"]["Cholesterol"] = "very-high"

    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


def test_predict_single_invalid_categorical_gender(client, valid_patient_dict):
    """Reject invalid categorical values for Gender."""
    bad_payload = {"patient": valid_patient_dict.copy()}
    bad_payload["patient"]["Gender"] = "Unknown"

    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("field,bad_val", [
    ("Age", -2),
    ("Age", 180),
    ("Blood_Pressure", 20.0),
    ("Blood_Pressure", 400.0),
    ("Cholesterol", 20.0),
    ("Cholesterol", 1200.0),
    ("BMI", 2.0),
    ("BMI", 90.0),
    ("ECG_Result", 5),
])
def test_predict_single_out_of_range_values(client, valid_patient_dict, field, bad_val):
    """Enforce physiological boundaries on clinical inputs."""
    bad_payload = {"patient": valid_patient_dict.copy()}
    bad_payload["patient"][field] = bad_val

    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_batch_valid(client, valid_patient_dict):
    """Evaluate batch of valid patient records and ensure stable record identifiers."""
    rec1 = valid_patient_dict.copy()
    rec1["Patient_ID"] = "P-BATCH-1"

    rec2 = valid_patient_dict.copy()
    rec2["Patient_ID"] = None
    rec2["Age"] = 42
    rec2["Gender"] = "Female"

    batch_payload = {
        "records": [rec1, rec2],
        "calibrate": False,
        "log_audit": False,
    }

    response = client.post("/api/v1/predict/batch", json=batch_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2

    # Check stable identifiers
    assert data["predictions"][0]["record_id"] == "P-BATCH-1"
    assert data["predictions"][1]["record_id"] == "record-2"

    for item in data["predictions"]:
        assert item["risk_classification"] in {"High Risk", "Low Risk"}
        assert item["predicted_class"] in {0, 1}
        assert 0.0 <= item["risk_probability"] <= 1.0


def test_predict_batch_empty_rejected(client):
    """Reject empty batch requests."""
    response = client.post("/api/v1/predict/batch", json={"records": []})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_batch_oversized_rejected(client, valid_patient_dict):
    """Reject batch requests exceeding the 1000 record maximum threshold."""
    oversized = [valid_patient_dict] * 1001
    response = client.post("/api/v1/predict/batch", json={"records": oversized})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_batch_invalid_record_causes_atomic_failure(client, valid_patient_dict):
    """Ensure invalid record in a batch fails atomically without producing partial results."""
    bad_rec = valid_patient_dict.copy()
    del bad_rec["BMI"]

    batch_payload = {
        "records": [valid_patient_dict, bad_rec],
        "calibrate": False,
    }
    response = client.post("/api/v1/predict/batch", json=batch_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


def test_model_loading_failure_returns_503(client, valid_patient_dict):
    """Ensure model loading errors return HTTP 503 without revealing internal file paths."""
    with patch.object(
        ModelContainer,
        "get_predictor",
        side_effect=HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model service is currently unavailable.",
        ),
    ):
        payload = {"patient": valid_patient_dict}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "unavailable" in data["detail"]
        assert "C:\\" not in str(data)  # No internal Windows path leak
