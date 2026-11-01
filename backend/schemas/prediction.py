"""Cardiovascular risk prediction request and response schemas."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.schemas.patient import PatientInputSchema


class SinglePredictionRequest(BaseModel):
    """Payload for requesting cardiovascular risk prediction on a single patient."""

    patient: PatientInputSchema = Field(
        ...,
        description="Patient clinical biomarker vitals",
    )
    calibrate: bool = Field(
        default=False,
        description="Whether to use Platt sigmoid probability calibration instead of raw model output",
    )
    log_audit: bool = Field(
        default=True,
        description="Whether to persist the clinical prediction and timestamp into the SQLite audit database",
    )

    model_config = ConfigDict(populate_by_name=True)


class BatchPredictionRequest(BaseModel):
    """Payload for high-throughput batch cardiovascular risk evaluation."""

    records: List[PatientInputSchema] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of patient vitals to evaluate (1 to 1000 patient records)",
    )
    calibrate: bool = Field(
        default=False,
        description="Whether to apply Platt probability calibration to all records in the batch",
    )
    log_audit: bool = Field(
        default=False,
        description="Whether to log every batch record to the audit database",
    )

    model_config = ConfigDict(populate_by_name=True)


class RecommendationItem(BaseModel):
    """Evidence-based clinical guideline recommendation."""

    category: str = Field(..., description="Biomarker category (e.g. Blood Pressure, Cholesterol, Blood Sugar)")
    status: str = Field(..., description="Brief clinical finding summary")
    message: str = Field(..., description="Actionable clinical guidance text")
    level: Literal["info", "warning", "critical"] = Field(..., description="Clinical alert level")


class SinglePredictionResponse(BaseModel):
    """Structured result of a single patient cardiovascular risk evaluation."""

    patient_id: Optional[str] = Field(default=None, description="Patient identifier if supplied")
    risk_classification: Literal["High Risk", "Low Risk"] = Field(
        ...,
        description="Binary clinical cardiovascular risk category",
    )
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of cardiovascular disease (0.0 to 1.0)",
    )
    confidence_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk probability expressed as a percentage (0.0% to 100.0%)",
    )
    is_calibrated: bool = Field(
        ...,
        description="Indicates whether probability calibration was applied",
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Clinical recommendations generated according to ACC/AHA guidelines",
    )
    audit_record_id: Optional[int] = Field(
        default=None,
        description="Database primary key ID in audit_logs table if persisted",
    )

    model_config = ConfigDict(populate_by_name=True)


class BatchPredictionItem(BaseModel):
    """Assessment item within a batch prediction result."""

    patient_id: Optional[str] = Field(default=None, description="Patient identifier")
    risk_classification: Literal["High Risk", "Low Risk"] = Field(..., description="Risk label")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability (0.0 to 1.0)")
    confidence_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage (0.0% to 100.0%)")


class BatchPredictionResponse(BaseModel):
    """Structured response for batch patient evaluation."""

    total_records: int = Field(..., ge=1, description="Count of evaluated patient records")
    is_calibrated: bool = Field(..., description="Whether probability calibration was applied")
    predictions: List[BatchPredictionItem] = Field(..., description="Individual risk assessments")

    model_config = ConfigDict(populate_by_name=True)

