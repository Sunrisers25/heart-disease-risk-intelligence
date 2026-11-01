"""Patient biomarker and clinical input schemas."""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientInputSchema(BaseModel):
    """Validated patient clinical biomarkers for cardiovascular risk assessment.

    Supports intake using either exact dataset keys (PascalCase, e.g. 'Blood_Pressure')
    or standard API identifiers (snake_case, e.g. 'blood_pressure').
    """

    age: int = Field(
        ...,
        alias="Age",
        ge=1,
        le=125,
        description="Patient chronological age in years",
        examples=[58],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        alias="Gender",
        description="Biological sex ('Male' or 'Female')",
        examples=["Male"],
    )
    blood_pressure: float = Field(
        ...,
        alias="Blood_Pressure",
        ge=40.0,
        le=300.0,
        description="Resting systolic blood pressure in mmHg",
        examples=[138.0],
    )
    cholesterol: float = Field(
        ...,
        alias="Cholesterol",
        ge=50.0,
        le=800.0,
        description="Total serum cholesterol in mg/dL",
        examples=[242.0],
    )
    bmi: float = Field(
        ...,
        alias="BMI",
        ge=10.0,
        le=80.0,
        description="Body Mass Index in kg/m^2",
        examples=[28.4],
    )
    blood_sugar: float = Field(
        ...,
        alias="Blood_Sugar",
        ge=30.0,
        le=600.0,
        description="Fasting blood sugar level in mg/dL",
        examples=[130.0],
    )
    heart_rate: float = Field(
        ...,
        alias="Heart_Rate",
        ge=30.0,
        le=260.0,
        description="Resting heart rate in beats per minute (bpm)",
        examples=[145.0],
    )
    ecg_result: int = Field(
        ...,
        alias="ECG_Result",
        ge=0,
        le=2,
        description="Resting electrocardiogram result (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)",
        examples=[1],
    )
    patient_id: Optional[str] = Field(
        default=None,
        alias="Patient_ID",
        max_length=64,
        description="Optional unique or de-identified patient record identifier",
        examples=["P-10492"],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "Age": 58,
                "Gender": "Male",
                "Blood_Pressure": 138.0,
                "Cholesterol": 242.0,
                "BMI": 28.4,
                "Blood_Sugar": 130.0,
                "Heart_Rate": 145.0,
                "ECG_Result": 1,
                "Patient_ID": "P-10492",
            }
        },
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert validated schema into the exact dictionary required by ML pipelines."""
        data: Dict[str, Any] = {
            "Age": int(self.age),
            "Gender": str(self.gender),
            "Blood_Pressure": float(self.blood_pressure),
            "Cholesterol": float(self.cholesterol),
            "BMI": float(self.bmi),
            "Blood_Sugar": float(self.blood_sugar),
            "Heart_Rate": float(self.heart_rate),
            "ECG_Result": int(self.ecg_result),
        }
        if self.patient_id is not None:
            data["Patient_ID"] = str(self.patient_id)
        return data

