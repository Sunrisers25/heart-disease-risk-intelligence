"""Cardiovascular risk prediction request and response schemas."""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.schemas.patient import PatientInputSchema


class SinglePredictionRequest(BaseModel):
    """Payload for requesting cardiovascular risk prediction on a single patient.

    Supports both nested format (`{"patient": {...}, "calibrate": false}`)
    and flat patient biomarker format (`{"Age": 58, "Gender": "Male", ...}`).
    """

    patient: Optional[PatientInputSchema] = Field(
        default=None,
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

    @model_validator(mode="before")
    @classmethod
    def assemble_patient(cls, data: Any) -> Any:
        """Handle flat patient payloads by grouping biomarkers under `patient`."""
        if isinstance(data, dict):
            if "patient" in data and data["patient"] is not None:
                return data
            biomarkers = {}
            top_level = {}
            for k, v in data.items():
                if k in {"calibrate", "log_audit"}:
                    top_level[k] = v
                else:
                    biomarkers[k] = v
            if biomarkers:
                top_level["patient"] = PatientInputSchema(**biomarkers)
                return top_level
        return data


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
    predicted_class: int = Field(
        default=0,
        description="Binary classification index: 1 (High Risk) or 0 (Low Risk)",
        examples=[1],
    )
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
    model_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata describing active model architecture and calibration technique",
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Clinical recommendations generated according to ACC/AHA guidelines",
    )
    audit_record_id: Optional[int] = Field(
        default=None,
        description="Database primary key ID in audit_logs table if persisted",
    )
    disclaimer: str = Field(
        default=(
            "Research prototype evaluated on synthetic data. "
            "Model estimates are statistical approximations and do not constitute clinical diagnosis."
        ),
        description="Mandatory clinical research prototype disclaimer",
    )

    model_config = ConfigDict(populate_by_name=True)


class BatchPredictionItem(BaseModel):
    """Assessment item within a batch prediction result."""

    record_id: Optional[Union[str, int]] = Field(
        default=None,
        description="Stable record identifier within batch",
        examples=["P-01", "record-1"],
    )
    patient_id: Optional[str] = Field(default=None, description="Patient identifier")
    predicted_class: int = Field(default=0, description="Binary classification index (1: High Risk, 0: Low Risk)")
    risk_classification: Literal["High Risk", "Low Risk"] = Field(..., description="Risk label")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability (0.0 to 1.0)")
    confidence_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage (0.0% to 100.0%)")


class BatchPredictionResponse(BaseModel):
    """Structured response for batch patient evaluation."""

    total_records: int = Field(..., ge=1, description="Count of evaluated patient records")
    is_calibrated: bool = Field(..., description="Whether probability calibration was applied")
    predictions: List[BatchPredictionItem] = Field(..., description="Individual risk assessments")
    disclaimer: str = Field(
        default=(
            "Research prototype evaluated on synthetic data. "
            "Model estimates are statistical approximations and do not constitute clinical diagnosis."
        ),
        description="Mandatory clinical research prototype disclaimer",
    )

    model_config = ConfigDict(populate_by_name=True)

