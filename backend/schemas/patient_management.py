"""Pydantic schemas for patient management and historical risk assessments."""

from datetime import datetime
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientCreateRequest(BaseModel):
    """Payload for registering a new patient record."""

    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique de-identified patient identifier",
        examples=["P-10492"],
    )
    age: int = Field(
        ...,
        ge=1,
        le=125,
        description="Patient chronological age in years",
        examples=[58],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Biological sex of the patient ('Male' or 'Female')",
        examples=["Male"],
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PatientUpdateRequest(BaseModel):
    """Payload for modifying permitted demographic attributes of a registered patient."""

    age: Optional[int] = Field(
        default=None,
        ge=1,
        le=125,
        description="Updated patient age in years",
        examples=[59],
    )
    gender: Optional[Literal["Male", "Female"]] = Field(
        default=None,
        description="Updated biological sex ('Male' or 'Female')",
        examples=["Male"],
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PatientResponse(BaseModel):
    """Structured patient profile representation."""

    id: int = Field(..., description="Internal database primary key ID")
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., description="Patient age in years")
    gender: str = Field(..., description="Biological sex")
    created_at: datetime = Field(..., description="Record creation UTC timestamp")
    updated_at: datetime = Field(..., description="Last profile update UTC timestamp")
    total_predictions: int = Field(default=0, description="Total count of historical risk predictions for this patient")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class PaginatedPatientResponse(BaseModel):
    """Paginated list of patient records."""

    items: List[PatientResponse] = Field(..., description="List of patient profiles for current page")
    total: int = Field(..., ge=0, description="Total matching patient records across all pages")
    page: int = Field(..., ge=1, description="Current page index")
    limit: int = Field(..., ge=1, description="Maximum items per page")
    total_pages: int = Field(..., ge=0, description="Total calculated page count")
    has_next: bool = Field(..., description="True if a subsequent page exists")
    has_prev: bool = Field(..., description="True if a previous page exists")

    model_config = ConfigDict(populate_by_name=True)


class PredictionHistoryItem(BaseModel):
    """Detailed record of an individual clinical risk prediction."""

    id: int = Field(..., description="Internal primary key of prediction record")
    patient_id: str = Field(..., description="Associated patient identifier")
    predicted_class: int = Field(..., description="Binary classification (1: High Risk, 0: Low Risk)")
    risk_classification: Literal["High Risk", "Low Risk"] = Field(..., description="Clinical risk category")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability (0.0 to 1.0)")
    confidence_percentage: float = Field(..., ge=0.0, le=100.0, description="Risk probability percentage")
    is_calibrated: bool = Field(..., description="Whether Platt probability calibration was applied")
    model_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata describing model pipeline")
    prediction_timestamp: datetime = Field(..., description="UTC timestamp when prediction was computed")
    audit_record_id: Optional[int] = Field(default=None, description="Corresponding SQLite audit log primary key")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class PaginatedPredictionHistoryResponse(BaseModel):
    """Paginated list of historical risk predictions."""

    items: List[PredictionHistoryItem] = Field(..., description="Prediction records for current page")
    total: int = Field(..., ge=0, description="Total prediction records across all pages")
    page: int = Field(..., ge=1, description="Current page index")
    limit: int = Field(..., ge=1, description="Maximum records per page")
    total_pages: int = Field(..., ge=0, description="Total calculated page count")
    has_next: bool = Field(..., description="True if a subsequent page exists")
    has_prev: bool = Field(..., description="True if a previous page exists")

    model_config = ConfigDict(populate_by_name=True)

