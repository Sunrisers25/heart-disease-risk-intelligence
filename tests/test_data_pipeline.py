"""Unit tests for data pipeline loader and preprocessor."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_pipeline.loader import generate_synthetic_data, load_and_clean_data, validate_schema
from src.data_pipeline.preprocessor import create_preprocessor


def test_synthetic_data_generation():
    """Verify synthetic data generation has expected columns and non-empty rows."""
    df = generate_synthetic_data(num_samples=50, random_state=42)
    assert len(df) == 50
    for col in ALL_COLUMNS:
        assert col in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({"Yes", "No"})
    assert set(df["Gender"].unique()).issubset({"Male", "Female"})
    assert set(df["ECG_Result"].unique()).issubset({0, 1, 2})


def test_validate_schema_valid():
    """Ensure valid DataFrame passes schema validation."""
    df = generate_synthetic_data(num_samples=20)
    is_valid, errors = validate_schema(df, is_training=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_missing_column():
    """Ensure missing required column is caught."""
    df = generate_synthetic_data(num_samples=20).drop(columns=["Blood_Pressure"])
    is_valid, errors = validate_schema(df, is_training=True)
    assert is_valid is False
    assert any("Blood_Pressure" in err for err in errors)


def test_load_and_clean_data_no_nans(tmp_path):
    """Verify load_and_clean_data eliminates NaNs and saves clean data."""
    raw_path = tmp_path / "raw.csv"
    proc_path = tmp_path / "clean.csv"

    # Write a dataset with some NaNs to test imputation
    df = generate_synthetic_data(num_samples=30)
    df.loc[0, "Blood_Pressure"] = np.nan
    df.loc[1, "Cholesterol"] = np.nan
    df.to_csv(raw_path, index=False)

    df_clean = load_and_clean_data(raw_path=raw_path, processed_path=proc_path, save_processed=True)

    assert not df_clean.empty
    assert df_clean.isnull().sum().sum() == 0
    assert proc_path.exists()


def test_preprocessor_transformation():
    """Verify ColumnTransformer outputs numeric matrix without NaNs."""
    df = generate_synthetic_data(num_samples=25)[FEATURE_COLUMNS]
    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(df)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 25
    assert not np.isnan(transformed).any()
