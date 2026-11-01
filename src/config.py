"""Configuration module for the Heart Disease Risk Analysis project.
Provides path constants, schema definitions, and clinical threshold values.
"""

from pathlib import Path

# Base project directory (assumes this config file is inside src/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories and files
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_disease_clean.csv"

# Artifacts (trained model pipelines, evaluation metrics, and comparisons)
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "svm_pca_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

# Phase 2 Model Comparison Artifacts
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
BEST_MODEL_METADATA_PATH = ARTIFACTS_DIR / "best_model_metadata.json"

# Database storage
DB_PATH = BASE_DIR / "src" / "database" / "patients.db"

# Schema definitions
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Blood_Pressure",
    "Cholesterol",
    "BMI",
    "Blood_Sugar",
    "Heart_Rate",
    "ECG_Result",
]

NUMERIC_FEATURES = [
    "Age",
    "Blood_Pressure",
    "Cholesterol",
    "BMI",
    "Blood_Sugar",
    "Heart_Rate",
    "ECG_Result",
]

CATEGORICAL_FEATURES = ["Gender"]

TARGET_COLUMN = "Heart_Disease"
ID_COLUMN = "Patient_ID"

ALL_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# Clinical thresholds for recommender
THRESHOLDS = {
    "blood_pressure": 130,  # mmHg (Systolic >= 130 indicates Stage 1 Hypertension)
    "cholesterol": 200,     # mg/dL (Total cholesterol >= 200 indicates Borderline/High)
    "blood_sugar": 126,     # mg/dL (Fasting blood sugar >= 126 indicates diabetes risk)
    "bmi": 25.0,            # kg/m^2 (BMI >= 25 indicates overweight status)
    "ecg_result": 0,        # ECG_Result > 0 indicates abnormal resting electrocardiogram
}

# Cross-Validation & Grid Search parameters
PARAM_GRID = {
    "classifier__C": [0.1, 1.0, 10.0, 100.0],
    "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
}

RANDOM_STATE = 42
CV_SPLITS = 5
PCA_VARIANCE = 0.95
