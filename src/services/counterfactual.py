"""Interactive What-If Counterfactual Simulator Service.
Allows comparing baseline patient measurements against hypothetical biomarker modifications
without modifying model weights, mutating patient records, or retraining.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS
from src.services.predictor import HeartDiseasePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SimulationComparisonResult:
    """Structured result containing baseline vs hypothetical counterfactual simulation."""
    original_inputs: Dict[str, Any]
    hypothetical_inputs: Dict[str, Any]
    original_probability: float
    hypothetical_probability: float
    original_classification: str
    hypothetical_classification: str
    delta_percentage_points: float
    relative_change_percent: Optional[float]
    changed_features: Dict[str, Tuple[Any, Any]]
    is_calibrated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CounterfactualSimulator:
    """Computes and compares hypothetical what-if scenarios against baseline patient data."""

    def __init__(self, predictor: HeartDiseasePredictor):
        if predictor is None or predictor.pipeline is None:
            raise ValueError("A valid HeartDiseasePredictor instance is required.")
        self.predictor = predictor

    def _validate_input_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate presence, non-null status, and types for patient vitals."""
        if not isinstance(data, dict):
            raise TypeError("Patient input must be a dictionary.")

        cleaned = {}
        missing_fields = []
        for col in FEATURE_COLUMNS:
            if col not in data:
                missing_fields.append(col)

        if missing_fields:
            raise ValueError(f"Patient input is missing required features: {missing_fields}")

        # Validate Gender
        gender = str(data["Gender"]).strip()
        if gender not in {"Male", "Female"}:
            raise ValueError(f"Invalid Gender value '{gender}'. Must be 'Male' or 'Female'.")
        cleaned["Gender"] = gender

        # Numeric conversions and validations
        try:
            cleaned["Age"] = int(data["Age"])
            cleaned["Blood_Pressure"] = float(data["Blood_Pressure"])
            cleaned["Cholesterol"] = float(data["Cholesterol"])
            cleaned["BMI"] = float(data["BMI"])
            cleaned["Blood_Sugar"] = float(data["Blood_Sugar"])
            cleaned["Heart_Rate"] = float(data["Heart_Rate"])
            cleaned["ECG_Result"] = int(data["ECG_Result"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to parse numeric biomarker: {e}")

        # Validate ECG_Result domain
        if cleaned["ECG_Result"] not in {0, 1, 2}:
            raise ValueError(f"Invalid ECG_Result value {cleaned['ECG_Result']}. Must be 0, 1, or 2.")

        # Preserve Patient_ID if passed
        if "Patient_ID" in data:
            cleaned["Patient_ID"] = str(data["Patient_ID"])

        return cleaned

    def simulate(
        self,
        original_patient: Dict[str, Any],
        hypothetical_patient: Dict[str, Any],
    ) -> SimulationComparisonResult:
        """Simulate hypothetical biomarker changes against the original patient.

        Args:
            original_patient: Dictionary containing baseline patient biomarkers.
            hypothetical_patient: Dictionary containing adjusted hypothetical values.

        Returns:
            SimulationComparisonResult: Comparison metrics, probability delta, and feature deltas.
        """
        orig_clean = self._validate_input_dict(original_patient)
        hypo_clean = self._validate_input_dict(hypothetical_patient)

        # Run predictions through active predictor
        orig_class, orig_prob = self.predictor.predict(orig_clean)
        hypo_class, hypo_prob = self.predictor.predict(hypo_clean)

        # Delta in percentage points: (P_hypo - P_orig) * 100
        delta_pp = round((hypo_prob - orig_prob) * 100, 2)

        # Relative change percentage: only when orig_prob > 0
        if orig_prob > 0.0001:
            rel_change = round(((hypo_prob - orig_prob) / orig_prob) * 100, 2)
        else:
            rel_change = None

        # Identify modified features: {feature: (original_val, hypothetical_val)}
        changed_features = {}
        for col in FEATURE_COLUMNS:
            v_orig = orig_clean[col]
            v_hypo = hypo_clean[col]
            if v_orig != v_hypo:
                changed_features[col] = (v_orig, v_hypo)

        return SimulationComparisonResult(
            original_inputs=orig_clean,
            hypothetical_inputs=hypo_clean,
            original_probability=orig_prob,
            hypothetical_probability=hypo_prob,
            original_classification=orig_class,
            hypothetical_classification=hypo_class,
            delta_percentage_points=delta_pp,
            relative_change_percent=rel_change,
            changed_features=changed_features,
            is_calibrated=getattr(self.predictor, "use_calibration", False),
        )
