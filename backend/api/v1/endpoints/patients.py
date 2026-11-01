"""Patient management and historical risk assessment REST API endpoints."""

from datetime import datetime
import json
import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.core.dependencies import get_db
from backend.schemas.patient_management import (
    PaginatedPatientResponse,
    PaginatedPredictionHistoryResponse,
    PatientCreateRequest,
    PatientResponse,
    PatientUpdateRequest,
    PredictionHistoryItem,
)
from src.database.models import Patient, PredictionHistory

logger = logging.getLogger("backend.endpoints.patients")
router = APIRouter()


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new patient profile",
    description="Register a patient profile with baseline demographics. Rejects duplicate patient identifiers.",
)
def register_patient(
    request: PatientCreateRequest,
    db: Session = Depends(get_db),
) -> PatientResponse:
    """Register a new patient."""
    existing = db.query(Patient).filter(Patient.patient_id == request.patient_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient with identifier '{request.patient_id}' already exists.",
        )

    patient = Patient(
        patient_id=request.patient_id,
        age=request.age,
        gender=request.gender,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    try:
        db.add(patient)
        db.commit()
        db.refresh(patient)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient with identifier '{request.patient_id}' already exists.",
        )
    except Exception as exc:
        db.rollback()
        logger.error("Failed to register patient: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register patient record.",
        )

    logger.info("Registered new patient profile: %s", patient.patient_id)
    return PatientResponse(
        id=patient.id,
        patient_id=patient.patient_id,
        age=patient.age,
        gender=patient.gender,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        total_predictions=0,
    )


@router.get(
    "",
    response_model=PaginatedPatientResponse,
    status_code=status.HTTP_200_OK,
    summary="List patient profiles",
    description="Retrieve a paginated list of registered patient profiles with optional identifier search.",
)
def list_patients(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Records per page"),
    search: Optional[str] = Query(default=None, description="Search term for patient identifier"),
    db: Session = Depends(get_db),
) -> PaginatedPatientResponse:
    """List registered patients with pagination and optional search."""
    query = db.query(Patient)
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(Patient.patient_id.ilike(term))

    total = query.count()
    total_pages = math.ceil(total / limit) if total > 0 else 0
    offset = (page - 1) * limit

    patients = query.order_by(Patient.id.desc()).offset(offset).limit(limit).all()

    items = [
        PatientResponse(
            id=p.id,
            patient_id=p.patient_id,
            age=p.age,
            gender=p.gender,
            created_at=p.created_at,
            updated_at=p.updated_at,
            total_predictions=len(p.predictions),
        )
        for p in patients
    ]

    return PaginatedPatientResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1 and total_pages > 0,
    )


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a patient profile",
    description="Fetch a registered patient record by unique identifier.",
)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
) -> PatientResponse:
    """Retrieve an individual patient profile."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found.",
        )

    return PatientResponse(
        id=patient.id,
        patient_id=patient.patient_id,
        age=patient.age,
        gender=patient.gender,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        total_predictions=len(patient.predictions),
    )


@router.patch(
    "/{patient_id}",
    response_model=PatientResponse,
    status_code=status.HTTP_200_OK,
    summary="Update patient demographics",
    description="Update permitted demographic fields (age, gender) for an existing patient.",
)
def update_patient(
    patient_id: str,
    request: PatientUpdateRequest,
    db: Session = Depends(get_db),
) -> PatientResponse:
    """Partially update patient demographics."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found.",
        )

    updated = False
    if request.age is not None:
        patient.age = request.age
        updated = True
    if request.gender is not None:
        patient.gender = request.gender
        updated = True

    if updated:
        patient.updated_at = datetime.utcnow()
        try:
            db.commit()
            db.refresh(patient)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to update patient %s: %s", patient_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update patient record.",
            )

    logger.info("Updated patient profile: %s", patient.patient_id)
    return PatientResponse(
        id=patient.id,
        patient_id=patient.patient_id,
        age=patient.age,
        gender=patient.gender,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        total_predictions=len(patient.predictions),
    )


@router.get(
    "/{patient_id}/predictions",
    response_model=PaginatedPredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve patient prediction history",
    description="Fetch historical cardiovascular risk assessments for a patient in descending timestamp order.",
)
def get_patient_predictions(
    patient_id: str,
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Records per page"),
    db: Session = Depends(get_db),
) -> PaginatedPredictionHistoryResponse:
    """Retrieve historical prediction assessments for a patient."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found.",
        )

    query = db.query(PredictionHistory).filter(PredictionHistory.patient_id == patient_id)
    total = query.count()
    total_pages = math.ceil(total / limit) if total > 0 else 0
    offset = (page - 1) * limit

    predictions = (
        query.order_by(PredictionHistory.prediction_timestamp.desc(), PredictionHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = []
    for pred in predictions:
        meta_dict = None
        if pred.model_metadata:
            try:
                meta_dict = json.loads(pred.model_metadata)
            except Exception:
                meta_dict = {"raw": pred.model_metadata}

        items.append(
            PredictionHistoryItem(
                id=pred.id,
                patient_id=pred.patient_id,
                predicted_class=pred.predicted_class,
                risk_classification=pred.risk_classification,
                risk_probability=pred.risk_probability,
                confidence_percentage=round(pred.risk_probability * 100, 2),
                is_calibrated=pred.is_calibrated,
                model_metadata=meta_dict,
                prediction_timestamp=pred.prediction_timestamp,
                audit_record_id=pred.audit_record_id,
            )
        )

    return PaginatedPredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1 and total_pages > 0,
    )

