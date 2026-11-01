"""API v1 router aggregator."""

from fastapi import APIRouter
from backend.api.v1.endpoints import predictions

api_router = APIRouter()
api_router.include_router(predictions.router, tags=["Cardiovascular Risk Prediction"])
