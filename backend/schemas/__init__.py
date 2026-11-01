"""Pydantic v2 validation schemas for API inputs and outputs."""

from backend.schemas.common import APIErrorResponse, ErrorDetail, HealthResponse
from backend.schemas.patient import PatientInputSchema
from backend.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RecommendationItem,
    SinglePredictionRequest,
    SinglePredictionResponse,
)

__all__ = [
    "HealthResponse",
    "APIErrorResponse",
    "ErrorDetail",
    "PatientInputSchema",
    "SinglePredictionRequest",
    "BatchPredictionRequest",
    "SinglePredictionResponse",
    "BatchPredictionResponse",
    "BatchPredictionItem",
    "RecommendationItem",
]

