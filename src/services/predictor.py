"""Inference service module.
Loads trained SVM-PCA pipeline artifact and runs predictions on single/batch records.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """Encapsulates loading the model pipeline and running inference."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline = self._load_model()

    def _load_model(self) -> Pipeline:
        """Load serialized scikit-learn pipeline from joblib file."""
        if not self.model_path.exists():
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
        # In SVC with classes_ [0, 1], index 1 corresponds to positive class
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
        probabilities = self.pipeline.predict_proba(clean_df)[:, 1]
        labels = np.where(probabilities >= 0.5, "High Risk", "Low Risk")
        return labels, np.round(probabilities, 4)
