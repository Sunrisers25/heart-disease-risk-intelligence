"""Tests for Patient Management and Prediction History API (/api/v1/patients and /api/v1/predict)."""

from datetime import datetime
import pytest
from fastapi import status
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from backend.core.dependencies import get_db
from backend.main import app
from src.database.models import Base, Patient, PredictionHistory


@pytest.fixture
def test_db_session():
    """Create an isolated, temporary in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(test_db_session):
    """FastAPI TestClient with overridden database session."""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_patient_payload():
    """Valid patient registration payload."""
    return {
        "patient_id": "P-TEST-REG-01",
        "age": 55,
        "gender": "Male",
    }


def test_register_patient_success(client, sample_patient_payload):
    """Register a new patient profile successfully."""
    response = client.post("/api/v1/patients", json=sample_patient_payload)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["patient_id"] == "P-TEST-REG-01"
    assert data["age"] == 55
    assert data["gender"] == "Male"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert data["total_predictions"] == 0


def test_register_patient_duplicate_conflict(client, sample_patient_payload):
    """Reject duplicate patient identifier with HTTP 409 Conflict."""
    first = client.post("/api/v1/patients", json=sample_patient_payload)
    assert first.status_code == status.HTTP_201_CREATED

    duplicate = client.post("/api/v1/patients", json=sample_patient_payload)
    assert duplicate.status_code == status.HTTP_409_CONFLICT
    assert "already exists" in duplicate.json()["detail"]


def test_register_patient_invalid_data(client):
    """Reject invalid patient demographic inputs with HTTP 422."""
    # Invalid gender
    bad_gender = {"patient_id": "P-BAD-1", "age": 40, "gender": "NonBinary"}
    resp1 = client.post("/api/v1/patients", json=bad_gender)
    assert resp1.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid age
    bad_age = {"patient_id": "P-BAD-2", "age": -5, "gender": "Female"}
    resp2 = client.post("/api/v1/patients", json=bad_age)
    assert resp2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_patient_by_id_success(client, sample_patient_payload):
    """Retrieve an existing patient profile."""
    reg = client.post("/api/v1/patients", json=sample_patient_payload)
    assert reg.status_code == status.HTTP_201_CREATED

    response = client.get(f"/api/v1/patients/{sample_patient_payload['patient_id']}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["patient_id"] == sample_patient_payload["patient_id"]
    assert data["age"] == 55


def test_get_patient_not_found(client):
    """Return HTTP 404 when requested patient identifier does not exist."""
    response = client.get("/api/v1/patients/NON-EXISTENT-ID")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"]


def test_update_patient_demographics(client, sample_patient_payload):
    """Partially update permitted demographics and verify immutable fields remain unchanged."""
    reg = client.post("/api/v1/patients", json=sample_patient_payload)
    orig_id = reg.json()["id"]

    patch_payload = {"age": 56, "gender": "Female"}
    response = client.patch(f"/api/v1/patients/{sample_patient_payload['patient_id']}", json=patch_payload)
    assert response.status_code == status.HTTP_200_OK

    updated = response.json()
    assert updated["id"] == orig_id
    assert updated["patient_id"] == sample_patient_payload["patient_id"]
    assert updated["age"] == 56
    assert updated["gender"] == "Female"


def test_update_patient_not_found(client):
    """Return 404 when updating a nonexistent patient."""
    response = client.patch("/api/v1/patients/DOES-NOT-EXIST", json={"age": 60})
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_list_patients_pagination_and_search(client):
    """Verify patient list pagination and search filtering."""
    for i in range(1, 6):
        client.post("/api/v1/patients", json={
            "patient_id": f"P-BATCH-SEARCH-{i:02d}",
            "age": 40 + i,
            "gender": "Male" if i % 2 == 0 else "Female",
        })

    # Page 1 with limit 2
    p1 = client.get("/api/v1/patients?page=1&limit=2")
    assert p1.status_code == status.HTTP_200_OK
    d1 = p1.json()
    assert d1["total"] == 5
    assert len(d1["items"]) == 2
    assert d1["total_pages"] == 3
    assert d1["has_next"] is True
    assert d1["has_prev"] is False

    # Search filter
    search_resp = client.get("/api/v1/patients?search=SEARCH-03")
    assert search_resp.status_code == status.HTTP_200_OK
    d_search = search_resp.json()
    assert d_search["total"] == 1
    assert d_search["items"][0]["patient_id"] == "P-BATCH-SEARCH-03"


def test_empty_prediction_history(client, sample_patient_payload):
    """New patient profile has an empty prediction history list."""
    client.post("/api/v1/patients", json=sample_patient_payload)
    response = client.get(f"/api/v1/patients/{sample_patient_payload['patient_id']}/predictions")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_prediction_association_with_registered_patient(client, sample_patient_payload):
    """Evaluating risk for a registered patient persists prediction history and updates record count."""
    client.post("/api/v1/patients", json=sample_patient_payload)

    vitals = {
        "patient": {
            "Patient_ID": sample_patient_payload["patient_id"],
            "Age": 55,
            "Gender": "Male",
            "Blood_Pressure": 140.0,
            "Cholesterol": 230.0,
            "BMI": 27.5,
            "Blood_Sugar": 110.0,
            "Heart_Rate": 80.0,
            "ECG_Result": 0,
        },
        "calibrate": False,
        "log_audit": True,
        "require_registered_patient": True,
    }

    pred_resp = client.post("/api/v1/predict", json=vitals)
    assert pred_resp.status_code == status.HTTP_200_OK
    pred_data = pred_resp.json()
    assert pred_data["prediction_history_id"] is not None

    # Fetch history
    history_resp = client.get(f"/api/v1/patients/{sample_patient_payload['patient_id']}/predictions")
    assert history_resp.status_code == status.HTTP_200_OK
    history_data = history_resp.json()
    assert history_data["total"] == 1
    assert history_data["items"][0]["id"] == pred_data["prediction_history_id"]
    assert history_data["items"][0]["risk_classification"] == pred_data["risk_classification"]


def test_rejection_of_unregistered_patient_when_enforced(client):
    """When require_registered_patient=True, unknown patient IDs must return HTTP 404."""
    vitals = {
        "patient": {
            "Patient_ID": "P-UNREGISTERED-999",
            "Age": 60,
            "Gender": "Female",
            "Blood_Pressure": 130.0,
            "Cholesterol": 210.0,
            "BMI": 26.0,
            "Blood_Sugar": 105.0,
            "Heart_Rate": 75.0,
            "ECG_Result": 0,
        },
        "require_registered_patient": True,
    }

    resp = client.post("/api/v1/predict", json=vitals)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "not registered" in resp.json()["detail"]


def test_prediction_history_ordering(client, test_db_session):
    """Verify that multiple historical predictions are returned in descending timestamp order."""
    patient = Patient(patient_id="P-ORDER-TEST", age=50, gender="Male")
    test_db_session.add(patient)
    test_db_session.commit()

    pred1 = PredictionHistory(
        patient_internal_id=patient.id,
        patient_id=patient.patient_id,
        predicted_class=0,
        risk_classification="Low Risk",
        risk_probability=0.25,
        prediction_timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    pred2 = PredictionHistory(
        patient_internal_id=patient.id,
        patient_id=patient.patient_id,
        predicted_class=1,
        risk_classification="High Risk",
        risk_probability=0.75,
        prediction_timestamp=datetime(2026, 1, 2, 10, 0, 0),
    )
    test_db_session.add_all([pred1, pred2])
    test_db_session.commit()

    response = client.get(f"/api/v1/patients/{patient.patient_id}/predictions")
    assert response.status_code == status.HTTP_200_OK
    items = response.json()["items"]
    assert len(items) == 2
    # Newer prediction must come first
    assert items[0]["predicted_class"] == 1
    assert items[1]["predicted_class"] == 0


def test_foreign_key_cascade_deletion(client, test_db_session):
    """Deleting a patient record cascades and removes all associated prediction history entries."""
    patient = Patient(patient_id="P-CASCADE-TEST", age=65, gender="Female")
    test_db_session.add(patient)
    test_db_session.commit()

    pred = PredictionHistory(
        patient_internal_id=patient.id,
        patient_id=patient.patient_id,
        predicted_class=1,
        risk_classification="High Risk",
        risk_probability=0.88,
    )
    test_db_session.add(pred)
    test_db_session.commit()

    assert test_db_session.query(PredictionHistory).filter(PredictionHistory.patient_id == patient.patient_id).count() == 1

    # Delete patient
    test_db_session.delete(patient)
    test_db_session.commit()

    # Prediction history must be deleted via cascade
    assert test_db_session.query(PredictionHistory).filter(PredictionHistory.patient_id == "P-CASCADE-TEST").count() == 0
