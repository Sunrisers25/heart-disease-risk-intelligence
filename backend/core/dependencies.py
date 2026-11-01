"""FastAPI dependency injection providers for ML models, database handlers, and singletons."""

import logging
from typing import Generator, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from src.config import BEST_MODEL_PATH, CALIBRATED_MODEL_PATH, MODEL_PATH
from src.database.db_handler import DatabaseHandler
from src.database.session import get_db_session
from src.services.predictor import HeartDiseasePredictor

logger = logging.getLogger("backend.dependencies")


class ModelContainer:
    """Thread-safe singleton container for loaded predictors and services.

    Caches pipelines in memory so model files are not reloaded on every HTTP request.
    """

    _raw_predictor: Optional[HeartDiseasePredictor] = None
    _calibrated_predictor: Optional[HeartDiseasePredictor] = None
    _db_handler: Optional[DatabaseHandler] = None

    @classmethod
    def get_raw_predictor(cls) -> HeartDiseasePredictor:
        """Retrieve or initialize the primary production model predictor."""
        if cls._raw_predictor is None:
            model_file = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
            if not model_file.exists():
                logger.error("Primary model artifact not found at %s or %s", BEST_MODEL_PATH, MODEL_PATH)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Primary machine learning model service is currently unavailable.",
                )
            try:
                cls._raw_predictor = HeartDiseasePredictor(model_path=model_file, use_calibration=False)
                logger.info("Initialized raw predictor from %s", model_file.name)
            except Exception as e:
                logger.error("Failed to load raw model: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to load machine learning model service.",
                )
        return cls._raw_predictor

    @classmethod
    def get_calibrated_predictor(cls) -> HeartDiseasePredictor:
        """Retrieve or initialize the Platt-calibrated model predictor."""
        if cls._calibrated_predictor is None:
            if not CALIBRATED_MODEL_PATH.exists():
                logger.warning("Calibrated model artifact not found, falling back to raw model")
                return cls.get_raw_predictor()
            try:
                cls._calibrated_predictor = HeartDiseasePredictor(use_calibration=True)
                logger.info("Initialized calibrated predictor from %s", CALIBRATED_MODEL_PATH.name)
            except Exception as e:
                logger.error("Failed to load calibrated model: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to load calibrated machine learning model service.",
                )
        return cls._calibrated_predictor

    @classmethod
    def get_predictor(cls, calibrate: bool = False) -> HeartDiseasePredictor:
        """Retrieve predictor according to requested calibration mode."""
        if calibrate:
            return cls.get_calibrated_predictor()
        return cls.get_raw_predictor()

    @classmethod
    def get_db_handler(cls) -> DatabaseHandler:
        """Retrieve or initialize the SQLite database handler."""
        if cls._db_handler is None:
            cls._db_handler = DatabaseHandler()
        return cls._db_handler

    @classmethod
    def reset(cls) -> None:
        """Reset cached singletons (useful for test isolation / mocks)."""
        cls._raw_predictor = None
        cls._calibrated_predictor = None
        cls._db_handler = None


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a transactional database session."""
    yield from get_db_session()
