"""Clinical guidelines and recommendation engine.
Evaluates patient vitals against clinical thresholds and generates actionable medical advice.
"""

from typing import Any, Dict, List
from src.config import THRESHOLDS


def generate_recommendations(
    patient_data: Dict[str, Any],
    risk_classification: str = "Low Risk",
) -> List[Dict[str, str]]:
    """Evaluate patient inputs against standard clinical guidelines.

    Guidelines:
      - BP >= 130 mmHg -> Hypertension warning + DASH diet / sodium restriction.
      - Cholesterol >= 200 mg/dL -> Hyperlipidemia warning + lipid panel check.
      - Blood Sugar >= 126 mg/dL -> High fasting glucose warning + HbA1c screening.
      - BMI >= 25.0 -> Weight management & aerobic exercise guidance.
      - ECG_Result > 0 or High Risk -> Flag for immediate cardiologist consultation.

    Returns:
        List of recommendation objects with "category", "status", "message", and "level" (warning/critical/info).
    """
    recommendations: List[Dict[str, str]] = []

    bp = float(patient_data.get("Blood_Pressure", 0.0))
    chol = float(patient_data.get("Cholesterol", 0.0))
    blood_sugar = float(patient_data.get("Blood_Sugar", 0.0))
    bmi = float(patient_data.get("BMI", 0.0))
    ecg = int(patient_data.get("ECG_Result", 0))

    # 1. Blood Pressure Check
    if bp >= THRESHOLDS["blood_pressure"]:
        recommendations.append({
            "category": "Blood Pressure",
            "status": "Stage 1/2 Hypertension Detected",
            "message": (
                f"Resting BP is elevated at {bp:.0f} mmHg (threshold: {THRESHOLDS['blood_pressure']} mmHg). "
                "Initiate DASH diet (Dietary Approaches to Stop Hypertension), enforce sodium restriction (<2,300 mg/day), "
                "and schedule blood pressure re-evaluation."
            ),
            "level": "warning",
        })
    else:
        recommendations.append({
            "category": "Blood Pressure",
            "status": "Optimal Blood Pressure",
            "message": f"Resting BP is {bp:.0f} mmHg, within normal physiological range (<130 mmHg).",
            "level": "info",
        })

    # 2. Cholesterol Check
    if chol >= THRESHOLDS["cholesterol"]:
        recommendations.append({
            "category": "Cholesterol",
            "status": "Hyperlipidemia / Elevated Serum Lipids",
            "message": (
                f"Total serum cholesterol is {chol:.0f} mg/dL (threshold: {THRESHOLDS['cholesterol']} mg/dL). "
                "Recommend comprehensive fasting lipid panel (LDL, HDL, triglycerides), reduction in saturated fats, "
                "and consideration of statin therapy if atherosclerotic cardiovascular risk is elevated."
            ),
            "level": "warning",
        })
    else:
        recommendations.append({
            "category": "Cholesterol",
            "status": "Desirable Cholesterol",
            "message": f"Serum cholesterol is {chol:.0f} mg/dL, within the desirable range (<200 mg/dL).",
            "level": "info",
        })

    # 3. Blood Sugar Check
    if blood_sugar >= THRESHOLDS["blood_sugar"]:
        recommendations.append({
            "category": "Blood Sugar",
            "status": "Hyperglycemia Alert",
            "message": (
                f"Fasting blood glucose is {blood_sugar:.0f} mg/dL (threshold: {THRESHOLDS['blood_sugar']} mg/dL). "
                "Indicative of potential diabetes mellitus. Recommend confirmatory venous HbA1c screening, "
                "endocrinology consultation, and low glycemic index nutrition plan."
            ),
            "level": "warning",
        })
    else:
        recommendations.append({
            "category": "Blood Sugar",
            "status": "Normal Fasting Glucose",
            "message": f"Fasting blood glucose is {blood_sugar:.0f} mg/dL, within the normal fasting range (<126 mg/dL).",
            "level": "info",
        })

    # 4. BMI Check
    if bmi >= THRESHOLDS["bmi"]:
        recommendations.append({
            "category": "BMI & Weight",
            "status": "Overweight / Elevated Adiposity",
            "message": (
                f"Calculated BMI is {bmi:.1f} kg/m² (normal threshold: <25.0 kg/m²). "
                "Enroll in structured weight management, caloric titration, and at least 150 minutes/week "
                "of moderate-intensity aerobic cardiovascular exercise."
            ),
            "level": "warning",
        })
    else:
        recommendations.append({
            "category": "BMI & Weight",
            "status": "Normal Body Mass Index",
            "message": f"BMI is {bmi:.1f} kg/m², representing a healthy weight range.",
            "level": "info",
        })

    # 5. ECG and Overall High Risk Check
    ecg_flag = ecg > THRESHOLDS["ecg_result"]
    is_high_risk = risk_classification == "High Risk"

    if ecg_flag or is_high_risk:
        reasons = []
        if ecg_flag:
            ecg_desc = "ST-T wave abnormality" if ecg == 1 else "Left ventricular hypertrophy"
            reasons.append(f"Abnormal resting electrocardiogram (ECG Result Code {ecg}: {ecg_desc})")
        if is_high_risk:
            reasons.append("ML Risk Classifier indicated High Cardiovascular Risk")

        recommendations.append({
            "category": "Cardiology Urgent Action",
            "status": "Immediate Cardiologist Consultation Flagged",
            "message": (
                f"CRITICAL: {'; '.join(reasons)}. "
                "Immediate referral for 12-lead ECG review, echocardiogram assessment, "
                "and supervised exercise stress testing is strongly recommended."
            ),
            "level": "critical",
        })

    return recommendations
