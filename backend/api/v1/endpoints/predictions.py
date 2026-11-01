"""Cardiovascular risk prediction API endpoints."""

import logging
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.core.dependencies import ModelContainer
from backend.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RecommendationItem,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.services.recommender import generate_recommendations

logger = logging.getLogger("backend.endpoints.predictions")
router = APIRouter()


@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate single patient cardiovascular risk",
    description=(
        "Assess cardiovascular risk probability and classification using the trained production "
        "SVM model or Platt-calibrated model. Returns actionable ACC/AHA clinical recommendations."
    ),
)
async def predict_single(request: SinglePredictionRequest) -> SinglePredictionResponse:
    """Run inference for a single patient profile."""
    if request.patient is None:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Patient clinical biomarker data is required.",
        )

    patient_dict = request.patient.to_feature_dict()
    patient_id = request.patient.patient_id

    try:
        predictor = ModelContainer.get_predictor(calibrate=request.calibrate)
        classification, risk_probability = predictor.predict(patient_dict)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during machine learning inference.",
        )

    # Class indexing: 1 = High Risk, 0 = Low Risk
    predicted_class = 1 if classification == "High Risk" else 0
    confidence_pct = round(risk_probability * 100, 2)

    # Generate evidence-based clinical recommendations
    raw_recs = generate_recommendations(patient_dict, risk_classification=classification)
    recommendations: List[RecommendationItem] = [
        RecommendationItem(
            category=r.get("category", "General"),
            status=r.get("status", ""),
            message=r.get("message", ""),
            level=r.get("level", "info"),
        )
        for r in raw_recs
    ]

    # Optional SQLite audit logging
    audit_id: Optional[int] = None
    if request.log_audit:
        try:
            db_handler = ModelContainer.get_db_handler()
            clinical_notes = [f"[{r.category}] {r.message}" for r in recommendations]
            audit_id = db_handler.log_patient_record(
                patient_data=patient_dict,
                risk_classification=classification,
                risk_probability=risk_probability,
                clinical_notes=clinical_notes,
            )
        except Exception as exc:
            logger.warning("Failed to record patient assessment in audit log: %s", exc)

    logger.info(
        "Evaluated patient %s: classification=%s, probability=%.4f, calibrated=%s",
        patient_id or "anonymous",
        classification,
        risk_probability,
        request.calibrate,
    )

    metadata = {
        "model_architecture": "Support Vector Machine with PCA (RBF Kernel)",
        "selection_metric": "cv_roc_auc_mean",
        "is_calibrated": getattr(predictor, "use_calibration", False),
        "calibration_technique": (
            "Platt Sigmoid (CalibratedClassifierCV)"
            if getattr(predictor, "use_calibration", False)
            else "None (Raw Model Output)"
        ),
    }

    return SinglePredictionResponse(
        patient_id=patient_id,
        predicted_class=predicted_class,
        risk_classification=classification,
        risk_probability=risk_probability,
        confidence_percentage=confidence_pct,
        is_calibrated=getattr(predictor, "use_calibration", False),
        model_metadata=metadata,
        recommendations=recommendations,
        audit_record_id=audit_id,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate batch of patient records",
    description="Vectorized batch inference across multiple patient records (1 to 1000 records).",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run vectorized batch inference for multiple patient records."""
    records_count = len(request.records)
    if records_count == 0:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Batch prediction request must contain at least one patient record.",
        )

    try:
        predictor = ModelContainer.get_predictor(calibrate=request.calibrate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to acquire model predictor for batch: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while preparing the inference engine.",
        )

    records_data = [rec.to_feature_dict() for rec in request.records]
    df_batch = pd.DataFrame(records_data)

    try:
        labels, probabilities = predictor.predict_batch(df_batch)
    except Exception as exc:
        logger.error("Batch prediction failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed during feature transformation or model scoring.",
        )

    db_handler = ModelContainer.get_db_handler() if request.log_audit else None
    results: List[BatchPredictionItem] = []

    for idx, rec in enumerate(request.records):
        prob = float(probabilities[idx])
        classification = str(labels[idx])
        predicted_class = 1 if classification == "High Risk" else 0
        pct = round(prob * 100, 2)
        patient_id = rec.patient_id
        record_id = patient_id if patient_id else f"record-{idx + 1}"

        if request.log_audit and db_handler:
            try:
                db_handler.log_patient_record(
                    patient_data=records_data[idx],
                    risk_classification=classification,
                    risk_probability=prob,
                    clinical_notes=["Batch prediction record"],
                )
            except Exception as exc:
                logger.warning("Failed to log batch record %s: %s", record_id, exc)

        results.append(
            BatchPredictionItem(
                record_id=record_id,
                patient_id=patient_id,
                predicted_class=predicted_class,
                risk_classification=classification,
                risk_probability=round(prob, 4),
                confidence_percentage=pct,
            )
        )

    logger.info("Successfully evaluated batch of %d records (calibrated=%s)", records_count, request.calibrate)

    return BatchPredictionResponse(
        total_records=records_count,
        is_calibrated=getattr(predictor, "use_calibration", False),
        predictions=results,
    )
