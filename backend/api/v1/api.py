"""API v1 router aggregator."""

from fastapi import APIRouter
from backend.api.v1.endpoints import patients, predictions

api_router = APIRouter()
api_router.include_router(predictions.router, tags=["Cardiovascular Risk Prediction"])
api_router.include_router(patients.router, prefix="/patients", tags=["Patient Management"])
