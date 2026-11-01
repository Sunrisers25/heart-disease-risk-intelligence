"""Streamlit web dashboard for Heart Disease Risk Analysis.
Integrates patient data intake, model inference, probability calibration,
SHAP explainability, clinical recommendations, and SQLite audit trail.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.ui_widgets import (
    render_audit_trail,
    render_calibration_section,
    render_clinical_disclaimer,
    render_explainability_section,
    render_header,
    render_recommendations,
    render_risk_badge,
)
from src.config import CALIBRATION_METRICS_PATH
from src.database.db_handler import DatabaseHandler
from src.models.evaluate import load_metrics
from src.models.train import train_model
from src.services.explainer import ModelExplainer
from src.services.predictor import HeartDiseasePredictor
from src.services.recommender import generate_recommendations

# Page setup
st.set_page_config(
    page_title="Cardiovascular Risk Decision Support",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_db_handler() -> DatabaseHandler:
    return DatabaseHandler()


@st.cache_resource
def get_uncalibrated_predictor() -> HeartDiseasePredictor:
    try:
        return HeartDiseasePredictor(use_calibration=False)
    except FileNotFoundError:
        st.warning("Model artifact not found. Auto-training initial model pipeline...")
        train_model()
        return HeartDiseasePredictor(use_calibration=False)


@st.cache_resource
def get_calibrated_predictor() -> Optional[HeartDiseasePredictor]:
    try:
        return HeartDiseasePredictor(use_calibration=True)
    except Exception:
        return None


@st.cache_resource
def get_explainer() -> ModelExplainer:
    predictor = get_uncalibrated_predictor()
    return ModelExplainer(pipeline=predictor.pipeline, background_size=30)


@st.cache_data
def get_cached_global_importance() -> Any:
    explainer = get_explainer()
    return explainer.compute_global_importance(max_samples=25, nsamples=40)


def load_calibration_metrics() -> Optional[dict]:
    if CALIBRATION_METRICS_PATH.exists():
        with open(CALIBRATION_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    render_header()
    render_clinical_disclaimer()

    db = get_db_handler()
    uncal_predictor = get_uncalibrated_predictor()
    cal_predictor = get_calibrated_predictor()
    explainer = get_explainer()

    # Sidebar for Patient Vitals Entry
    with st.sidebar:
        st.header("👤 Patient Demographics & Vitals")
        patient_id = st.text_input("Patient Identifier", value="P0301")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=58, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)

        st.markdown("---")
        st.subheader("🩺 Clinical Biomarkers")

        blood_pressure = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=70,
            max_value=240,
            value=135,
            step=1,
            help="Systolic Blood Pressure at rest. Normal: <120, Hypertension: >=130",
        )

        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=100,
            max_value=500,
            value=238,
            step=1,
            help="Total serum cholesterol. Desirable: <200, High: >=240",
        )

        bmi = st.number_input(
            "Body Mass Index (BMI kg/m²)",
            min_value=12.0,
            max_value=60.0,
            value=28.4,
            step=0.1,
            help="Normal weight: 18.5 - 24.9, Overweight: >=25.0, Obese: >=30.0",
        )

        blood_sugar = st.number_input(
            "Fasting Blood Sugar (mg/dL)",
            min_value=50,
            max_value=400,
            value=128,
            step=1,
            help="Fasting blood glucose. Normal: 70-99, Prediabetes: 100-125, Diabetes: >=126",
        )

        heart_rate = st.number_input(
            "Maximum / Resting Heart Rate (bpm)",
            min_value=40,
            max_value=220,
            value=142,
            step=1,
            help="Beats per minute.",
        )

        ecg_options = {
            0: "0 - Normal ECG",
            1: "1 - ST-T Wave Abnormality (inversion / elevation)",
            2: "2 - Probable / Definite Left Ventricular Hypertrophy",
        }
        ecg_selection = st.selectbox(
            "Resting ECG Result",
            options=list(ecg_options.keys()),
            format_func=lambda x: ecg_options[x],
            index=1,
        )

        st.markdown("---")
        # Keep calibration toggle default OFF as specified in Fix 5
        use_calibrated_mode = st.toggle("Apply Probability Calibration (Platt)", value=False)

        analyze_button = st.button("🚀 Analyze Cardiovascular Risk", use_container_width=True, type="primary")

    # Main Area Layout
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    patient_payload = {
        "Patient_ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "BMI": bmi,
        "Blood_Sugar": blood_sugar,
        "Heart_Rate": heart_rate,
        "ECG_Result": ecg_selection,
    }

    if analyze_button:
        with st.spinner("Executing model inference & computing SHAP feature attributions..."):
            uncal_class, uncal_prob = uncal_predictor.predict(patient_payload)

            if use_calibrated_mode and cal_predictor is not None:
                final_class, final_prob = cal_predictor.predict(patient_payload)
            else:
                final_class, final_prob = uncal_class, uncal_prob

            recommendations = generate_recommendations(patient_payload, risk_classification=final_class)

            explanation = explainer.explain_instance(patient_payload, nsamples=60)
            global_importance_df = get_cached_global_importance()

            clinical_notes = [rec["message"] for rec in recommendations]
            record_id = db.log_patient_record(
                patient_data=patient_payload,
                risk_classification=final_class,
                risk_probability=final_prob,
                clinical_notes=clinical_notes,
            )

        with col_left:
            st.subheader("🎯 Inference Outcome")
            render_risk_badge(
                classification=final_class,
                probability=final_prob,
                uncalibrated_probability=uncal_prob if use_calibrated_mode else None,
                is_calibrated=use_calibrated_mode,
            )
            render_recommendations(recommendations)

        with col_right:
            st.subheader("📊 Patient Snapshot")
            st.markdown(
                f"""
                - **Patient ID:** `{patient_id}` (DB Audit Log #{record_id})
                - **Age / Sex:** {age} y/o {gender}
                - **Blood Pressure:** {blood_pressure} mmHg
                - **Cholesterol:** {cholesterol} mg/dL
                - **BMI:** {bmi:.1f} kg/m²
                - **Fasting Blood Sugar:** {blood_sugar} mg/dL
                - **Heart Rate:** {heart_rate} bpm
                - **ECG Reading:** {ecg_options[ecg_selection]}
                """
            )

            try:
                metrics = load_metrics()
                st.markdown("---")
                st.subheader("🤖 Model Performance Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.2f}")
                m2.metric("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.1f}%")
                m3.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.2f}")
            except Exception:
                pass

        # Dedicated Probability Calibration Section
        cal_data = load_calibration_metrics()
        render_calibration_section(cal_data)

        # Dedicated Explainability Section below with explicit transparency flag
        render_explainability_section(explanation, global_importance_df, is_calibrated_mode=use_calibrated_mode)

    else:
        with col_left:
            st.info("👈 Enter patient demographic and clinical parameters in the sidebar and click **Analyze Cardiovascular Risk**.")
        with col_right:
            try:
                metrics = load_metrics()
                st.subheader("🤖 Production Model Architecture")
                st.markdown(
                    f"""
                    - **Active Classifier:** `{type(uncal_predictor.pipeline.named_steps.get('classifier')).__name__}`
                    - **Test ROC-AUC:** `{metrics.get('roc_auc', 0.0):.4f}`
                    - **Test Accuracy:** `{metrics.get('accuracy', 0.0)*100:.1f}%`
                    - **Probability Calibration:** Supported (Sigmoid / Platt scaling via `CalibratedClassifierCV`)
                    - **Explainability:** SHAP Feature Attribution (Original Biomarker Level)
                    """
                )
            except Exception:
                pass

        cal_data = load_calibration_metrics()
        if cal_data:
            render_calibration_section(cal_data)

    # Audit Trail Section
    st.markdown("---")
    records_df = db.get_recent_records(limit=30)
    render_audit_trail(records_df)


if __name__ == "__main__":
    main()
