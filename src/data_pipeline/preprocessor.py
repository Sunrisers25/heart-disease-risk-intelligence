"""Data preprocessing module.
Constructs scikit-learn transformers and preprocessor pipeline to prevent data leakage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer that scales numeric features and one-hot encodes categoricals.

    - Numeric: StandardScaler
    - Categorical: OneHotEncoder(drop='first', handle_unknown='ignore')

    Returns:
        ColumnTransformer: Preprocessor pipeline ready for scikit-learn Pipeline inclusion.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    categories=[["Female", "Male"]],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
