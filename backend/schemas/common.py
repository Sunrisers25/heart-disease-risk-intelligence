"""Common and standardized API schemas: Health check and Error handling."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """System health and service status response."""

    status: str = Field(..., description="Overall health status ('healthy', 'degraded', or 'unhealthy')", examples=["healthy"])
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the health check")
    version: str = Field(..., description="Application version", examples=["1.0.0"])
    project_name: str = Field(..., description="Application display name")
    model_loaded: bool = Field(..., description="Indicates if the primary model artifact exists and is loadable")
    calibrated_model_available: bool = Field(..., description="Indicates if Platt-calibrated pipeline is available")
    database_connected: bool = Field(..., description="Indicates if SQLite audit database connection succeeds")

    model_config = ConfigDict(populate_by_name=True)


class ErrorDetail(BaseModel):
    """Detailed validation or processing failure description."""

    field: Optional[str] = Field(default=None, description="Request parameter or field causing the error")
    message: str = Field(..., description="Descriptive error explanation")
    type: Optional[str] = Field(default=None, description="Pydantic or system error type identifier")


class APIErrorResponse(BaseModel):
    """Standardized error envelope returned across all failed API interactions."""

    success: bool = Field(default=False, description="Always false for error responses")
    error_code: str = Field(..., description="Standardized HTTP or domain error code", examples=["VALIDATION_ERROR", "INTERNAL_SERVER_ERROR"])
    message: str = Field(..., description="High-level human-readable error summary")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="List of specific field validation errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of error event")

    model_config = ConfigDict(populate_by_name=True)

