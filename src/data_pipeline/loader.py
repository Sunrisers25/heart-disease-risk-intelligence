"""Raw data loader and validator module.
Handles loading, synthetic data generation if missing/empty, validation, and saving clean data.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_samples: int = 300, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate realistic synthetic benchmark data for heart disease analysis.

    Adheres to clinical distributions and correlation patterns:
    - Age: 30 to 80
    - Gender: Male (60%), Female (40%)
    - Blood Pressure: Resting BP 90 to 200 mmHg
    - Cholesterol: 130 to 400 mg/dL
    - BMI: 18.0 to 45.0
    - Blood Sugar: Fasting 70 to 220 mg/dL
    - Heart Rate: 60 to 190 bpm
    - ECG Result: 0 (Normal, 60%), 1 (ST-T wave abnormality, 25%), 2 (Left ventricular hypertrophy, 15%)
    - Heart_Disease: Yes / No influenced by clinical risk score
    """
    rng = np.random.RandomState(random_state)

    patient_ids = [f"P{i:04d}" for i in range(1, num_samples + 1)]
    age = rng.randint(30, 80, size=num_samples)
    gender = rng.choice(["Male", "Female"], size=num_samples, p=[0.60, 0.40])

    # Blood pressure: mean 132, std 18, bounded [90, 200]
    bp = np.clip(np.round(rng.normal(132, 18, size=num_samples)), 90, 200).astype(int)

    # Cholesterol: mean 245, std 48, bounded [130, 420]
    chol = np.clip(np.round(rng.normal(245, 48, size=num_samples)), 130, 420).astype(int)

    # BMI: mean 27.5, std 4.5, bounded [18.0, 45.0]
    bmi = np.clip(np.round(rng.normal(27.5, 4.5, size=num_samples), 1), 18.0, 45.0)

    # Fasting blood sugar: right skewed, mixture of normal and diabetic
    base_sugar = rng.normal(105, 20, size=num_samples)
    diabetic_boost = rng.choice([0, 50], size=num_samples, p=[0.8, 0.2])
    blood_sugar = np.clip(np.round(base_sugar + diabetic_boost), 70, 240).astype(int)

    # Max Heart rate: decreases with age, std 18
    mean_hr = 208 - (0.7 * age)
    heart_rate = np.clip(np.round(rng.normal(mean_hr, 16)), 60, 205).astype(int)

    # ECG Result: 0 (Normal), 1 (ST-T Abnormality), 2 (Probable LVH)
    ecg = rng.choice([0, 1, 2], size=num_samples, p=[0.60, 0.25, 0.15])

    # Clinical risk logit for disease probability
    # Higher age, BP, Chol, BMI, Blood Sugar, abnormal ECG, and Male sex increase risk
    gender_factor = np.where(gender == "Male", 0.6, -0.3)
    logit = (
        0.04 * (age - 54)
        + 0.03 * (bp - 130)
        + 0.015 * (chol - 240)
        + 0.08 * (bmi - 27)
        + 0.012 * (blood_sugar - 110)
        - 0.02 * (heart_rate - 140)
        + 0.7 * ecg
        + gender_factor
        - 0.5  # baseline intercept
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Sample target based on probability
    target_num = rng.binomial(1, prob)
    target = np.where(target_num == 1, "Yes", "No")

    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": age,
        "Gender": gender,
        "Blood_Pressure": bp,
        "Cholesterol": chol,
        "BMI": bmi,
        "Blood_Sugar": blood_sugar,
        "Heart_Rate": heart_rate,
        "ECG_Result": ecg,
        "Heart_Disease": target,
    })

    return df


def validate_schema(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, list]:
    """Validate DataFrame column presence, types, and value sanity."""
    errors = []
    required_cols = ALL_COLUMNS if is_training else FEATURE_COLUMNS

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # Check numeric feature types
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")

    # Check Gender values
    valid_genders = {"Male", "Female"}
    invalid_genders = set(df["Gender"].dropna().unique()) - valid_genders
    if invalid_genders:
        errors.append(f"Invalid Gender values found: {invalid_genders}")

    # Check ECG_Result values
    valid_ecg = {0, 1, 2}
    invalid_ecg = set(df["ECG_Result"].dropna().unique()) - valid_ecg
    if invalid_ecg:
        errors.append(f"Invalid ECG_Result values found: {invalid_ecg}")

    # Check Target values if training
    if is_training and TARGET_COLUMN in df.columns:
        valid_targets = {"Yes", "No"}
        invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - valid_targets
        if invalid_targets:
            errors.append(f"Invalid Target values found: {invalid_targets}")

    return len(errors) == 0, errors


def load_and_clean_data(
    raw_path: Optional[Path] = None,
    save_processed: bool = True,
    processed_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load raw dataset (generating benchmark if missing or empty), clean, and optionally save.

    Returns:
        pd.DataFrame: Cleaned dataset ready for pipeline training.
    """
    path = raw_path or RAW_DATA_PATH
    proc_path = processed_path or PROCESSED_DATA_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Raw dataset not found or empty at %s. Generating 300-row synthetic benchmark.", path)
        df_raw = generate_synthetic_data(num_samples=300)
        df_raw.to_csv(path, index=False)
        logger.info("Saved synthetic benchmark to %s", path)
    else:
        df_raw = pd.read_csv(path)
        if df_raw.empty:
            logger.warning("Raw file is empty. Generating synthetic benchmark.")
            df_raw = generate_synthetic_data(num_samples=300)
            df_raw.to_csv(path, index=False)

    # Validate raw schema
    is_valid, errors = validate_schema(df_raw, is_training=True)
    if not is_valid:
        raise ValueError(f"Raw dataset schema validation failed: {errors}")

    # Cleaning: handle duplicates and missing values
    df_clean = df_raw.copy()
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[ID_COLUMN])
    if len(df_clean) < initial_len:
        logger.info("Dropped %d duplicate Patient_IDs", initial_len - len(df_clean))

    # Impute missing values defensively if any exist in raw data
    for col in NUMERIC_FEATURES:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info("Imputed missing values in '%s' with median (%s)", col, median_val)

    if df_clean["Gender"].isnull().any():
        df_clean["Gender"] = df_clean["Gender"].fillna("Male")

    if TARGET_COLUMN in df_clean.columns and df_clean[TARGET_COLUMN].isnull().any():
        df_clean = df_clean.dropna(subset=[TARGET_COLUMN])

    # Convert types
    df_clean["Age"] = df_clean["Age"].astype(int)
    df_clean["Blood_Pressure"] = df_clean["Blood_Pressure"].astype(float)
    df_clean["Cholesterol"] = df_clean["Cholesterol"].astype(float)
    df_clean["BMI"] = df_clean["BMI"].astype(float)
    df_clean["Blood_Sugar"] = df_clean["Blood_Sugar"].astype(float)
    df_clean["Heart_Rate"] = df_clean["Heart_Rate"].astype(float)
    df_clean["ECG_Result"] = df_clean["ECG_Result"].astype(int)

    if save_processed:
        df_clean.to_csv(proc_path, index=False)
        logger.info("Successfully cleaned and saved dataset to %s (%d rows)", proc_path, len(df_clean))

    return df_clean


if __name__ == "__main__":
    load_and_clean_data()
