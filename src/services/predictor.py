"""Inference service module.
Loads trained pipeline artifacts (baseline or best model) and runs predictions on single/batch records.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_PATH, FEATURE_COLUMNS, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """Encapsulates loading the model pipeline and running inference.

    Defaults to `MODEL_PATH` (baseline SVM) for backward compatibility,
    or can accept `BEST_MODEL_PATH` or custom pipeline path.
    """

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline = self._load_model()

    def _load_model(self) -> Pipeline:
        """Load serialized scikit-learn pipeline from joblib file."""
        if not self.model_path.exists():
            # Fallback to BEST_MODEL_PATH if default MODEL_PATH is missing, or vice-versa
            alt_path = BEST_MODEL_PATH if self.model_path == MODEL_PATH else MODEL_PATH
            if alt_path.exists():
                logger.warning(
                    "Model artifact not found at %s. Falling back to alternative at %s",
                    self.model_path,
                    alt_path,
                )
                self.model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Trained model artifact not found at {self.model_path}. "
                    "Please run `python -m src.models.train` first."
                )

        try:
            pipeline = joblib.load(self.model_path)
            logger.info("Successfully loaded model pipeline from %s", self.model_path)
            return pipeline
        except Exception as e:
            logger.error("Failed to load model from %s: %s", self.model_path, e)
            raise

    def _prepare_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Ensure input matches required feature columns format."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dictionary or pandas DataFrame.")

        # Check required feature columns
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required features: {missing_cols}")

        return df[FEATURE_COLUMNS]

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Tuple[str, float]:
        """Predict heart disease risk classification and confidence probability.

        Args:
            data: Patient clinical vitals (dict or single-row DataFrame).

        Returns:
            Tuple[str, float]:
                - classification: "High Risk" or "Low Risk"
                - risk_probability: float probability of heart disease (0.0 to 1.0)
        """
        df = self._prepare_input(data)

        # Predict probability for positive class (1 = High Risk)
        proba = self.pipeline.predict_proba(df)[0]
        classes = list(self.pipeline.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)
            risk_prob = float(proba[pos_idx])
        else:
            risk_prob = float(proba[0])

        classification = "High Risk" if risk_prob >= 0.5 else "Low Risk"
        return classification, round(risk_prob, 4)

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and probabilities for multiple patient records."""
        clean_df = self._prepare_input(df)
        classes = list(self.pipeline.classes_)
        pos_idx = classes.index(1) if 1 in classes else 1
        probabilities = self.pipeline.predict_proba(clean_df)[:, pos_idx]
        labels = np.where(probabilities >= 0.5, "High Risk", "Low Risk")
        return labels, np.round(probabilities, 4)
