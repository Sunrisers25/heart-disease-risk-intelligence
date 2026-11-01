"""FastAPI Application entry point for Heart Disease Risk Intelligence API."""

from datetime import datetime
import logging
from typing import List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.schemas.common import APIErrorResponse, ErrorDetail, HealthResponse
from src.config import BEST_MODEL_PATH, CALIBRATED_MODEL_PATH, DB_PATH, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backend.main")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Standardize Pydantic validation errors into structured APIErrorResponse."""
    details: List[ErrorDetail] = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        details.append(
            ErrorDetail(
                field=field_path,
                message=err.get("msg", "Invalid input value"),
                type=err.get("type", "value_error"),
            )
        )

    error_response = APIErrorResponse(
        success=False,
        error_code="VALIDATION_ERROR",
        message="Request payload failed clinical schema validation.",
        details=details,
        timestamp=datetime.utcnow(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Standardize unexpected internal exceptions into APIErrorResponse."""
    logger.exception("Unhandled backend error: %s", exc)
    error_response = APIErrorResponse(
        success=False,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred during request processing.",
        details=[ErrorDetail(message=str(exc))],
        timestamp=datetime.utcnow(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode="json"),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check verification",
)
@app.get(
    f"{settings.API_V1_PREFIX}/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check verification v1",
)
async def health_check() -> HealthResponse:
    """Check service availability, database connectivity, and artifact presence."""
    model_ready = BEST_MODEL_PATH.exists() or MODEL_PATH.exists()
    calibrated_ready = CALIBRATED_MODEL_PATH.exists()
    db_ready = DB_PATH.exists()

    status_str = "healthy" if (model_ready and db_ready) else "degraded"

    return HealthResponse(
        status=status_str,
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        project_name=settings.PROJECT_NAME,
        model_loaded=model_ready,
        calibrated_model_available=calibrated_ready,
        database_connected=db_ready,
    )

