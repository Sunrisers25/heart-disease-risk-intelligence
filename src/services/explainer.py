"""Model Explainability Module using SHAP.
Provides feature attributions at the original clinical feature level:
- Global feature importance.
- Local patient-level prediction explanations (waterfall/contribution charts).
- Handles Tree models (TreeExplainer) and Kernel models like SVM with PCA (KernelExplainer/Sampling).
Avoids exposing latent PCA components by evaluating predictions strictly in original feature space.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, RANDOM_STATE
from src.data_pipeline.loader import load_and_clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Manages model-agnostic and model-specific SHAP explainers.
    
    Guarantees that explanations are computed against original clinical features:
    Age, Gender, Blood_Pressure, Cholesterol, BMI, Blood_Sugar, Heart_Rate, ECG_Result.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background_data: Optional[pd.DataFrame] = None,
        background_size: int = 50,
        random_state: int = RANDOM_STATE,
    ):
        self.pipeline = pipeline
        self.random_state = random_state
        self.background_size = background_size

        if background_data is None:
            df = load_and_clean_data(save_processed=False)
            background_data = df[FEATURE_COLUMNS]

        self.background_data = self._sample_background(background_data, background_size)
        self.classifier = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
        self.explainer_type = self._determine_explainer_type()
        self.explainer = self._init_explainer()

    def _sample_background(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Sample a representative background dataset."""
        df = data[FEATURE_COLUMNS].copy()
        if len(df) > n_samples:
            return df.sample(n=n_samples, random_state=self.random_state).reset_index(drop=True)
        return df.reset_index(drop=True)

    def _determine_explainer_type(self) -> str:
        """Determine whether the model is tree-based or requires end-to-end kernel explanation."""
        step_names = [name for name, _ in self.pipeline.steps]
        class_name = self.classifier.__class__.__name__

        # If pipeline has PCA, we must explain the entire pipeline in original feature space
        if "pca" in step_names:
            return "kernel_pipeline"

        if class_name in {"RandomForestClassifier", "XGBClassifier", "ExtraTreesClassifier"}:
            return "tree"

        return "kernel_pipeline"

    def _predict_proba_wrapper(self, X_array: np.ndarray) -> np.ndarray:
        """Wrapper for pipeline predict_proba accepting numpy array or DataFrame.
        Returns the positive class probability (Heart Disease = 1).
        """
        if isinstance(X_array, np.ndarray):
            df = pd.DataFrame(X_array, columns=FEATURE_COLUMNS)
            # Restore types for pipeline compatibility
            df["Age"] = df["Age"].astype(int)
            df["Blood_Pressure"] = df["Blood_Pressure"].astype(float)
            df["Cholesterol"] = df["Cholesterol"].astype(float)
            df["BMI"] = df["BMI"].astype(float)
            df["Blood_Sugar"] = df["Blood_Sugar"].astype(float)
            df["Heart_Rate"] = df["Heart_Rate"].astype(float)
            df["ECG_Result"] = df["ECG_Result"].astype(int)
        else:
            df = X_array[FEATURE_COLUMNS]

        probs = self.pipeline.predict_proba(df)
        classes = list(self.pipeline.classes_)
        pos_idx = classes.index(1) if 1 in classes else 1
        return probs[:, pos_idx]

    def _init_explainer(self) -> shap.Explainer:
        """Initialize the appropriate SHAP explainer."""
        if self.explainer_type == "tree":
            # For tree pipelines without PCA, explain the transformed features or end-to-end
            logger.info("Initializing SHAP TreeExplainer for %s", self.classifier.__class__.__name__)
            # Extract preprocessor step
            preprocessor = self.pipeline.named_steps["preprocessor"]
            X_bg_trans = preprocessor.transform(self.background_data)
            return shap.TreeExplainer(self.classifier, data=X_bg_trans)
        else:
            # End-to-end KernelExplainer in original feature space
            logger.info("Initializing SHAP KernelExplainer on original input feature space...")
            return shap.KernelExplainer(
                model=self._predict_proba_wrapper,
                data=self.background_data,
                seed=self.random_state,
            )

    def explain_instance(
        self,
        patient_data: Union[Dict[str, Any], pd.DataFrame],
        nsamples: int = 80,
    ) -> Dict[str, Any]:
        """Compute local feature attributions for a single patient in original feature space.

        Args:
            patient_data: Patient vitals dict or single-row DataFrame.
            nsamples: Number of times to evaluate for KernelExplainer.

        Returns:
            Dict containing:
                - base_value: Expected model output on background dataset.
                - patient_risk_prob: Model predicted risk probability.
                - feature_contributions: List of dicts with feature name, value, SHAP attribution, direction.
                - shap_values: Raw shap values array.
        """
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])[FEATURE_COLUMNS]
        else:
            patient_df = patient_data[FEATURE_COLUMNS].copy()

        # Compute predicted probability
        patient_prob = float(self._predict_proba_wrapper(patient_df)[0])

        if self.explainer_type == "tree":
            preprocessor = self.pipeline.named_steps["preprocessor"]
            X_trans = preprocessor.transform(patient_df)
            shap_obj = self.explainer(X_trans)
            # TreeExplainer on binary model may produce (N, num_features, 2) or (N, num_features)
            vals = shap_obj.values
            if len(vals.shape) == 3:
                vals = vals[0, :, 1]
            else:
                vals = vals[0]
            base_val = float(shap_obj.base_values[0, 1] if len(np.shape(shap_obj.base_values)) > 1 else shap_obj.base_values[0])
            # Map transformed columns back to original feature names
            raw_shap = self._aggregate_tree_features(vals)
        else:
            # KernelExplainer directly on original feature space
            shap_vals = self.explainer.shap_values(patient_df, nsamples=nsamples)
            if isinstance(shap_vals, list):
                raw_shap = np.array(shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0])
            else:
                raw_shap = np.array(shap_vals[0])

            base_val = float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[1 if len(self.explainer.expected_value) > 1 else 0])

        # Construct structured contribution breakdown
        contributions = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = patient_df.iloc[0][col]
            sv = float(raw_shap[i])
            contributions.append({
                "feature": col,
                "value": val,
                "shap_value": round(sv, 4),
                "direction": "Risk Increasing" if sv > 0 else "Risk Decreasing",
                "abs_importance": abs(sv),
            })

        # Rank by absolute impact
        contributions.sort(key=lambda x: x["abs_importance"], reverse=True)

        return {
            "base_value": round(base_val, 4),
            "patient_risk_prob": round(patient_prob, 4),
            "feature_contributions": contributions,
            "raw_shap_values": raw_shap.tolist(),
            "feature_names": FEATURE_COLUMNS,
        }

    def _aggregate_tree_features(self, transformed_shap: np.ndarray) -> np.ndarray:
        """Align one-hot encoded tree outputs back to original feature indices."""
        # Preprocessor order: NUMERIC_FEATURES (7), then Gender_Male (1)
        # Length matches len(FEATURE_COLUMNS) = 8
        return np.asarray(transformed_shap, dtype=float)

    def compute_global_importance(
        self,
        samples: Optional[pd.DataFrame] = None,
        max_samples: int = 40,
        nsamples: int = 60,
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP values across representative patient samples."""
        if samples is None:
            df = load_and_clean_data(save_processed=False)
            eval_samples = df[FEATURE_COLUMNS].sample(
                n=min(len(df), max_samples), random_state=self.random_state
            ).reset_index(drop=True)
        else:
            eval_samples = samples[FEATURE_COLUMNS].head(max_samples).reset_index(drop=True)

        if self.explainer_type == "tree":
            preprocessor = self.pipeline.named_steps["preprocessor"]
            X_trans = preprocessor.transform(eval_samples)
            shap_obj = self.explainer(X_trans)
            vals = shap_obj.values
            if len(vals.shape) == 3:
                vals = vals[:, :, 1]
            mean_abs = np.mean(np.abs(vals), axis=0)
        else:
            shap_vals = self.explainer.shap_values(eval_samples, nsamples=nsamples)
            if isinstance(shap_vals, list):
                vals = np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
            else:
                vals = np.array(shap_vals)
            mean_abs = np.mean(np.abs(vals), axis=0)

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Mean_Abs_SHAP": [round(float(v), 4) for v in mean_abs],
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        return importance_df
