"""Model factory module.
Creates scikit-learn Pipeline instances for SVM, Random Forest, and XGBoost classifiers.
Encapsulates all feature preprocessing (scaling, one-hot encoding, PCA) cleanly within Pipelines
to strictly avoid data leakage.
"""

import logging
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PCA_VARIANCE,
    RANDOM_STATE,
)
from src.data_pipeline.preprocessor import create_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_tree_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer tailored for tree-based models (Random Forest, XGBoost).

    - Numerical features: passthrough (preserving raw non-linear monotonic scales).
    - Categorical features: OneHotEncoder(drop='first', sparse_output=False).
    - Remainder: drop.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[["Female", "Male"]],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def create_svm_pipeline(
    pca_variance: float = PCA_VARIANCE,
    random_state: int = RANDOM_STATE,
    probability: bool = True,
) -> Pipeline:
    """Construct standard SVM Pipeline:

    Steps:
      1. Preprocessor: StandardScaler for numeric features + OneHotEncoder for categoricals.
      2. PCA: Reduces dimension retaining `pca_variance` (default: 0.95).
      3. Classifier: SVC(kernel='rbf', probability=probability, class_weight='balanced').

    Args:
        probability: If False, outputs raw decision_function scores, ideal for external
                     Platt calibration (prevents double-Platt calibration).
                     If True, outputs internal cross-validated Platt probabilities.
    """
    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=probability,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def create_rf_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Construct Random Forest Pipeline:

    Steps:
      1. Preprocessor: Tree preprocessor (OneHotEncoder for categoricals, passthrough numeric).
      2. Classifier: RandomForestClassifier(class_weight='balanced', random_state=42).

    Note: Tree-based models are invariant to monotonic scaling, so PCA and StandardScaler
    are not forced to preserve feature interpretability and tree splits.
    """
    preprocessor = create_tree_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def create_xgb_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Construct XGBoost Pipeline:

    Steps:
      1. Preprocessor: Tree preprocessor (OneHotEncoder for categoricals, passthrough numeric).
      2. Classifier: XGBClassifier(eval_metric='logloss', random_state=42).

    Avoids deprecated parameters and ensures reproducible binary classification.
    """
    preprocessor = create_tree_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb.XGBClassifier(
                    eval_metric="logloss",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def get_model_pipeline(
    model_name: str,
    random_state: int = RANDOM_STATE,
    probability: bool = True,
) -> Pipeline:
    """Retrieve an instantiated Pipeline for a specified model architecture.

    Supported model names: 'svm', 'random_forest', 'xgboost' (case and space insensitive).
    """
    name = model_name.lower().replace(" ", "_").replace("-", "_").strip()
    if name in {"svm", "svc"}:
        return create_svm_pipeline(random_state=random_state, probability=probability)
    elif name in {"random_forest", "rf"}:
        return create_rf_pipeline(random_state=random_state)
    elif name in {"xgboost", "xgb"}:
        return create_xgb_pipeline(random_state=random_state)
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Choose from 'svm', 'random_forest', 'xgboost'."
        )
