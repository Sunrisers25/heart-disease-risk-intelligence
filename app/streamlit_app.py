"""Streamlit web dashboard for Heart Disease Risk Analysis.
Integrates patient data intake, model inference, probability calibration,
SHAP explainability, What-If Counterfactual Simulator, clinical recommendations,
and SQLite audit trail.
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
    render_counterfactual_comparison_chart,
    render_explainability_section,
    render_header,
    render_recommendations,
    render_risk_badge,
)
from src.config import CALIBRATION_METRICS_PATH
from src.database.db_handler import DatabaseHandler
from src.models.evaluate import load_metrics
from src.models.train import train_model
from src.services.counterfactual import CounterfactualSimulator
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
        use_calibrated_mode = st.toggle("Apply Probability Calibration (Platt)", value=False)

        analyze_button = st.button("🚀 Analyze Cardiovascular Risk", use_container_width=True, type="primary")

    # Patient baseline dictionary
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

    # Select active predictor based on user calibration toggle
    active_predictor = cal_predictor if (use_calibrated_mode and cal_predictor is not None) else uncal_predictor

    # Track baseline assessment in session state for Counterfactual Simulator
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

            # Store in session state
            st.session_state["baseline_patient"] = patient_payload.copy()
            st.session_state["baseline_class"] = final_class
            st.session_state["baseline_prob"] = final_prob
            st.session_state["uncal_prob"] = uncal_prob
            st.session_state["record_id"] = record_id
            st.session_state["recommendations"] = recommendations
            st.session_state["explanation"] = explanation
            st.session_state["global_importance_df"] = global_importance_df
            st.session_state["has_analyzed"] = True

    # Main Area Layout
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    if st.session_state.get("has_analyzed", False):
        final_class = st.session_state["baseline_class"]
        final_prob = st.session_state["baseline_prob"]
        uncal_prob = st.session_state["uncal_prob"]
        record_id = st.session_state["record_id"]
        recommendations = st.session_state["recommendations"]
        explanation = st.session_state["explanation"]
        global_importance_df = st.session_state["global_importance_df"]
        baseline_patient = st.session_state["baseline_patient"]

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
                - **Patient ID:** `{baseline_patient['Patient_ID']}` (DB Audit Log #{record_id})
                - **Age / Sex:** {baseline_patient['Age']} y/o {baseline_patient['Gender']}
                - **Blood Pressure:** {baseline_patient['Blood_Pressure']} mmHg
                - **Cholesterol:** {baseline_patient['Cholesterol']} mg/dL
                - **BMI:** {baseline_patient['BMI']:.1f} kg/m²
                - **Fasting Blood Sugar:** {baseline_patient['Blood_Sugar']} mg/dL
                - **Heart Rate:** {baseline_patient['Heart_Rate']} bpm
                - **ECG Reading:** {ecg_options[baseline_patient['ECG_Result']]}
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

        # -------------------------------------------------------------
        # INTERACTIVE WHAT-IF RISK SIMULATOR SECTION
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔄 Interactive What-If Risk Simulator")
        st.info(
            "💡 **Simulator Notice**: This simulator demonstrates how hypothetical input changes affect the "
            "predictions of a machine learning model. It does not predict the actual effect of lifestyle changes, "
            "medications, or clinical interventions. The underlying dataset is synthetic, and the model has not "
            "been clinically validated. Results must not be used for diagnosis or treatment decisions."
        )

        # Initialize simulator state keys from baseline patient if not present or if reset clicked
        sim_key_prefix = f"sim_{patient_id}_"

        # Check if reset was requested
        if st.button("↺ Reset Scenario to Original Patient Vitals", key="btn_reset_scenario"):
            st.session_state[f"{sim_key_prefix}bp"] = float(baseline_patient["Blood_Pressure"])
            st.session_state[f"{sim_key_prefix}chol"] = float(baseline_patient["Cholesterol"])
            st.session_state[f"{sim_key_prefix}bmi"] = float(baseline_patient["BMI"])
            st.session_state[f"{sim_key_prefix}sugar"] = float(baseline_patient["Blood_Sugar"])
            st.session_state[f"{sim_key_prefix}hr"] = float(baseline_patient["Heart_Rate"])
            st.session_state[f"{sim_key_prefix}ecg"] = int(baseline_patient["ECG_Result"])
            st.session_state[f"{sim_key_prefix}age"] = int(baseline_patient["Age"])
            st.session_state[f"{sim_key_prefix}gender"] = str(baseline_patient["Gender"])
            st.rerun()

        # Hypothetical controls columns
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            st.markdown("**Blood Pressure & Glucose**")
            hypo_bp = st.slider(
                "Hypothetical BP (mmHg)",
                min_value=80.0,
                max_value=220.0,
                value=float(st.session_state.get(f"{sim_key_prefix}bp", baseline_patient["Blood_Pressure"])),
                step=1.0,
                key=f"{sim_key_prefix}bp",
            )
            hypo_sugar = st.slider(
                "Hypothetical Blood Sugar (mg/dL)",
                min_value=60.0,
                max_value=350.0,
                value=float(st.session_state.get(f"{sim_key_prefix}sugar", baseline_patient["Blood_Sugar"])),
                step=1.0,
                key=f"{sim_key_prefix}sugar",
            )

        with sc2:
            st.markdown("**Lipids & Body Mass**")
            hypo_chol = st.slider(
                "Hypothetical Cholesterol (mg/dL)",
                min_value=120.0,
                max_value=450.0,
                value=float(st.session_state.get(f"{sim_key_prefix}chol", baseline_patient["Cholesterol"])),
                step=1.0,
                key=f"{sim_key_prefix}chol",
            )
            hypo_bmi = st.slider(
                "Hypothetical BMI (kg/m²)",
                min_value=15.0,
                max_value=50.0,
                value=float(st.session_state.get(f"{sim_key_prefix}bmi", baseline_patient["BMI"])),
                step=0.1,
                key=f"{sim_key_prefix}bmi",
            )

        with sc3:
            st.markdown("**Heart Rate & Resting ECG**")
            hypo_hr = st.slider(
                "Hypothetical Heart Rate (bpm)",
                min_value=50.0,
                max_value=200.0,
                value=float(st.session_state.get(f"{sim_key_prefix}hr", baseline_patient["Heart_Rate"])),
                step=1.0,
                key=f"{sim_key_prefix}hr",
            )
            hypo_ecg = st.selectbox(
                "Hypothetical ECG Result",
                options=[0, 1, 2],
                index=[0, 1, 2].index(int(st.session_state.get(f"{sim_key_prefix}ecg", baseline_patient["ECG_Result"]))),
                format_func=lambda x: ecg_options[x],
                key=f"{sim_key_prefix}ecg",
            )

        # Build hypothetical payload
        hypo_payload = {
            "Patient_ID": baseline_patient["Patient_ID"],
            "Age": baseline_patient["Age"],
            "Gender": baseline_patient["Gender"],
            "Blood_Pressure": hypo_bp,
            "Cholesterol": hypo_chol,
            "BMI": hypo_bmi,
            "Blood_Sugar": hypo_sugar,
            "Heart_Rate": hypo_hr,
            "ECG_Result": hypo_ecg,
        }

        # Run Simulation
        simulator = CounterfactualSimulator(predictor=active_predictor)
        sim_result = simulator.simulate(baseline_patient, hypo_payload)

        # Counterfactual Results Layout
        r_col1, r_col2 = st.columns([1.1, 0.9])

        with r_col1:
            st.markdown("#### Scenario Comparison")
            k1, k2, k3 = st.columns(3)
            k1.metric("Original Baseline", f"{sim_result.original_probability * 100:.1f}%", sim_result.original_classification)
            k2.metric(
                "Hypothetical Outcome",
                f"{sim_result.hypothetical_probability * 100:.1f}%",
                sim_result.hypothetical_classification,
            )
            k3.metric(
                "Probability Delta",
                f"{sim_result.delta_percentage_points:+.2f} pp",
                delta=f"{sim_result.delta_percentage_points:+.2f} pp",
                delta_color="inverse",
            )

            # Scenario Summary Callout
            delta_dir = "decreased" if sim_result.delta_percentage_points < 0 else (
                "increased" if sim_result.delta_percentage_points > 0 else "did not change"
            )
            st.markdown(
                f"""
                <div style="background-color: #1e293b; padding: 0.9rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-top: 0.8rem;">
                    <strong>Scenario Summary:</strong> Changing the selected hypothetical inputs {delta_dir} the model's 
                    estimated probability from <strong>{sim_result.original_probability*100:.1f}%</strong> to 
                    <strong>{sim_result.hypothetical_probability*100:.1f}%</strong> 
                    ({abs(sim_result.delta_percentage_points):.2f} percentage points difference).
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Optional on-demand SHAP explanation for the hypothetical scenario
            st.markdown("##### 🔬 Explain Hypothetical Scenario")
            if st.button("Generate SHAP for Hypothetical Scenario", key="btn_explain_hypo"):
                with st.spinner("Calculating SHAP attributions for hypothetical scenario..."):
                    hypo_explanation = explainer.explain_instance(hypo_payload, nsamples=40)
                    st.write("**Hypothetical Scenario Feature Attributions (SHAP):**")
                    hypo_df = pd.DataFrame(hypo_explanation["feature_contributions"])
                    st.dataframe(
                        hypo_df[["feature", "value", "shap_value", "direction"]].rename(
                            columns={
                                "feature": "Biomarker",
                                "value": "Hypothetical Value",
                                "shap_value": "SHAP Attribution",
                                "direction": "Risk Direction",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        with r_col2:
            st.markdown("#### Probability Trajectory Chart")
            render_counterfactual_comparison_chart(
                orig_prob=sim_result.original_probability,
                hypo_prob=sim_result.hypothetical_probability,
                delta_pp=sim_result.delta_percentage_points,
            )

            if sim_result.changed_features:
                st.write("**Modified Clinical Parameters:**")
                diff_data = []
                for feat, (orig_val, new_val) in sim_result.changed_features.items():
                    diff_data.append({
                        "Biomarker": feat,
                        "Original": orig_val,
                        "Hypothetical": new_val,
                        "Change": round(new_val - orig_val, 2) if isinstance(orig_val, (int, float)) else "Modified",
                    })
                st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)
            else:
                st.caption("No biomarkers modified yet. Adjust sliders above to simulate a scenario.")

        # Dedicated Probability Calibration Section
        cal_data = load_calibration_metrics()
        render_calibration_section(cal_data)

        # Dedicated Explainability Section below
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
                    - **Interactive Simulator:** What-If Counterfactual Analysis with Real-Time Delta Metrics
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
