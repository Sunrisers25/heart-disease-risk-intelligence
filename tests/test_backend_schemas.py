"""Tests for FastAPI backend schemas, validation constraints, and configuration."""

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from backend.core.config import Settings
from backend.main import app
from backend.schemas.common import APIErrorResponse, HealthResponse
from backend.schemas.patient import PatientInputSchema
from backend.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RecommendationItem,
    SinglePredictionRequest,
    SinglePredictionResponse,
)


@pytest.fixture
def valid_patient_payload():
    """Synthetic clinical patient vitals matching valid feature schema."""
    return {
        "Age": 58,
        "Gender": "Male",
        "Blood_Pressure": 138.0,
        "Cholesterol": 242.0,
        "BMI": 28.4,
        "Blood_Sugar": 130.0,
        "Heart_Rate": 145.0,
        "ECG_Result": 1,
        "Patient_ID": "P-SYNTH-01",
    }


def test_patient_schema_valid_pascal_case(valid_patient_payload):
    """Test schema instantiation with exact dataset PascalCase feature names."""
    patient = PatientInputSchema(**valid_patient_payload)
    assert patient.age == 58
    assert patient.gender == "Male"
    assert patient.blood_pressure == 138.0
    assert patient.patient_id == "P-SYNTH-01"

    feat_dict = patient.to_feature_dict()
    assert feat_dict["Age"] == 58
    assert feat_dict["Gender"] == "Male"
    assert feat_dict["Blood_Pressure"] == 138.0
    assert feat_dict["Patient_ID"] == "P-SYNTH-01"
    assert all(k in feat_dict for k in [
        "Age", "Gender", "Blood_Pressure", "Cholesterol",
        "BMI", "Blood_Sugar", "Heart_Rate", "ECG_Result"
    ])


def test_patient_schema_valid_snake_case():
    """Test schema instantiation with standard snake_case API identifiers."""
    data = {
        "age": 45,
        "gender": "Female",
        "blood_pressure": 120.0,
        "cholesterol": 190.0,
        "bmi": 23.5,
        "blood_sugar": 95.0,
        "heart_rate": 72.0,
        "ecg_result": 0,
        "patient_id": "P-SYNTH-02",
    }
    patient = PatientInputSchema(**data)
    assert patient.age == 45
    assert patient.gender == "Female"
    assert patient.blood_pressure == 120.0
    feat_dict = patient.to_feature_dict()
    assert feat_dict["Age"] == 45
    assert feat_dict["Gender"] == "Female"


def test_patient_schema_missing_required_fields(valid_patient_payload):
    """Reject patient payload when any mandatory clinical biomarker is missing."""
    invalid_data = valid_patient_payload.copy()
    del invalid_data["Blood_Pressure"]
    with pytest.raises(ValidationError) as exc_info:
        PatientInputSchema(**invalid_data)
    errors = exc_info.value.errors()
    assert any("Blood_Pressure" in str(e["loc"]) or "blood_pressure" in str(e["loc"]) for e in errors)


def test_patient_schema_invalid_data_types(valid_patient_payload):
    """Reject non-convertible types for numeric biomarkers."""
    invalid_data = valid_patient_payload.copy()
    invalid_data["Cholesterol"] = "not-a-number"
    with pytest.raises(ValidationError):
        PatientInputSchema(**invalid_data)


def test_patient_schema_invalid_categorical_values(valid_patient_payload):
    """Reject unsupported gender category."""
    invalid_data = valid_patient_payload.copy()
    invalid_data["Gender"] = "NonBinary"
    with pytest.raises(ValidationError) as exc_info:
        PatientInputSchema(**invalid_data)
    errors = exc_info.value.errors()
    assert any("Gender" in str(e["loc"]) or "gender" in str(e["loc"]) for e in errors)


@pytest.mark.parametrize("field,bad_val", [
    ("Age", 0),
    ("Age", 150),
    ("Blood_Pressure", 25.0),
    ("Blood_Pressure", 350.0),
    ("Cholesterol", 30.0),
    ("Cholesterol", 1200.0),
    ("BMI", 5.0),
    ("BMI", 95.0),
    ("ECG_Result", 3),
    ("ECG_Result", -1),
])
def test_patient_schema_physiological_bounds(valid_patient_payload, field, bad_val):
    """Enforce sane physiological validation boundaries."""
    invalid_data = valid_patient_payload.copy()
    invalid_data[field] = bad_val
    with pytest.raises(ValidationError):
        PatientInputSchema(**invalid_data)


def test_single_prediction_request_defaults(valid_patient_payload):
    """Verify single prediction request wrapping and defaults."""
    req = SinglePredictionRequest(patient=PatientInputSchema(**valid_patient_payload))
    assert req.calibrate is False
    assert req.log_audit is True


def test_batch_prediction_request_validation(valid_patient_payload):
    """Verify batch prediction request min and max limits."""
    # Valid batch of 2
    batch = BatchPredictionRequest(
        records=[
            PatientInputSchema(**valid_patient_payload),
            PatientInputSchema(**valid_patient_payload),
        ],
        calibrate=True,
    )
    assert len(batch.records) == 2
    assert batch.calibrate is True

    # Empty batch must be rejected
    with pytest.raises(ValidationError):
        BatchPredictionRequest(records=[])


def test_single_prediction_response_serialization():
    """Verify serialization and range checks for prediction response."""
    rec = RecommendationItem(
        category="Blood Pressure",
        status="Hypertension Alert",
        message="Consult cardiologist.",
        level="warning",
    )
    resp = SinglePredictionResponse(
        patient_id="P-01",
        risk_classification="High Risk",
        risk_probability=0.7421,
        confidence_percentage=74.21,
        is_calibrated=True,
        recommendations=[rec],
        audit_record_id=12,
    )
    assert resp.risk_classification == "High Risk"
    assert 0.0 <= resp.risk_probability <= 1.0
    dumped = resp.model_dump()
    assert dumped["risk_probability"] == 0.7421
    assert dumped["recommendations"][0]["level"] == "warning"

    # Out of range probability must fail
    with pytest.raises(ValidationError):
        SinglePredictionResponse(
            risk_classification="High Risk",
            risk_probability=1.5,
            confidence_percentage=150.0,
            is_calibrated=False,
        )


def test_batch_prediction_response_serialization():
    """Verify batch prediction response serialization."""
    item = BatchPredictionItem(
        patient_id="P-01",
        risk_classification="Low Risk",
        risk_probability=0.231,
        confidence_percentage=23.1,
    )
    batch_resp = BatchPredictionResponse(
        total_records=1,
        is_calibrated=False,
        predictions=[item],
    )
    assert batch_resp.total_records == 1
    assert batch_resp.predictions[0].risk_classification == "Low Risk"


def test_backend_settings_defaults():
    """Verify configuration defaults and CORS origin parsing."""
    settings = Settings()
    assert settings.PROJECT_NAME == "Heart Disease Risk Intelligence API"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.DEBUG is False
    assert any("8501" in origin for origin in settings.ALLOWED_ORIGINS)

    # Custom origin string parsing
    custom = Settings(ALLOWED_ORIGINS="http://site1.com, http://site2.com")
    assert "http://site1.com" in custom.ALLOWED_ORIGINS
    assert "http://site2.com" in custom.ALLOWED_ORIGINS


def test_health_check_endpoint():
    """Verify health endpoint via FastAPI TestClient."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "version" in data
    assert "model_loaded" in data

    # Verify v1 prefixed route
    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 200
    assert response_v1.json()["status"] == data["status"]

