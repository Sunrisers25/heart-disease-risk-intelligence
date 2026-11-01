"""UI widgets and visualization components for Heart Disease Risk Streamlit app."""

import json
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def render_header() -> None:
    """Render application header and system introduction."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 1.8rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">
            <h1 style="color: #f8fafc; margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 0.6rem;">
                ❤️ Cardiovascular Disease Risk Intelligence Platform
            </h1>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                Production-grade Clinical Decision Support System powered by scikit-learn PCA-SVM Pipeline & ACC/AHA Evidence-Based Guidelines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clinical_disclaimer() -> None:
    """Render mandatory institutional clinical disclaimer banner."""
    st.warning(
        "⚠️ **Research Prototype Disclaimer**: This system is an experimental research prototype evaluated "
        "using synthetic data. Its predicted risk probabilities and SHAP feature explanations are **not clinically validated** "
        "and must not be used for medical diagnosis, prognosis, or clinical treatment planning without independent clinical trials."
    )


def render_risk_badge(
    classification: str,
    probability: float,
    uncalibrated_probability: Optional[float] = None,
    is_calibrated: bool = False,
) -> None:
    """Render risk prediction outcome badge and confidence indicator."""
    is_high = classification == "High Risk"
    badge_bg = "#dc2626" if is_high else "#16a34a"
    badge_text = "HIGH RISK OF HEART DISEASE" if is_high else "LOW RISK OF HEART DISEASE"
    icon = "⚠️" if is_high else "✅"

    pct = probability * 100
    cal_label = "Platt-Calibrated Probability" if is_calibrated else "Model Estimated Probability"

    st.markdown(
        f"""
        <div style="background-color: {badge_bg}; color: white; padding: 1.2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
            <div style="font-size: 1.6rem; font-weight: 700; letter-spacing: 0.05em;">
                {icon} {badge_text}
            </div>
            <div style="font-size: 1.1rem; margin-top: 0.4rem; opacity: 0.95;">
                {cal_label}: <strong>{pct:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar indicator
    st.write(f"**Confidence Gauge ({pct:.1f}% Probability)**")
    st.progress(float(probability))

    if uncalibrated_probability is not None and is_calibrated:
        uncal_pct = uncalibrated_probability * 100
        st.caption(
            f"ℹ️ Raw Base Model Output: **{uncal_pct:.1f}%** ➔ "
            f"Platt-Calibrated Probability: **{pct:.1f}%**"
        )


def render_recommendations(recommendations: List[Dict[str, str]]) -> None:
    """Display clinical recommendation alert cards."""
    st.subheader("📋 Evidence-Based Clinical Recommendations")

    for rec in recommendations:
        level = rec.get("level", "info")
        category = rec.get("category", "General")
        status = rec.get("status", "")
        msg = rec.get("message", "")

        if level == "critical":
            st.error(f"🚨 **[{category}] {status}**\n\n{msg}")
        elif level == "warning":
            st.warning(f"⚠️ **[{category}] {status}**\n\n{msg}")
        else:
            st.info(f"ℹ️ **[{category}] {status}**\n\n{msg}")


def render_calibration_section(cal_metrics_dict: Optional[Dict[str, Any]]) -> None:
    """Render Probability Calibration Reliability Diagrams & Brier score comparisons."""
    st.markdown("---")
    st.subheader("🎯 Probability Calibration & Statistical Reliability")
    st.caption(
        "Evaluation comparing Uncalibrated Model Probabilities vs. Sigmoid (Platt) Calibrated "
        "Probabilities evaluated strictly on the untouched test cohort."
    )

    if not cal_metrics_dict:
        st.info("Calibration benchmark metrics will appear after running training.")
        return

    uncal = cal_metrics_dict.get("uncalibrated", {})
    cal = cal_metrics_dict.get("calibrated", {})

    # Top level metrics comparison
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Brier Score (Lower is better)",
        f"{cal.get('brier_score', 0.0):.4f}",
        delta=f"{cal.get('brier_score', 0.0) - uncal.get('brier_score', 0.0):.4f}",
        delta_color="inverse",
    )
    c2.metric(
        "Expected Calib. Error (ECE)",
        f"{cal.get('expected_calibration_error', 0.0):.4f}",
        delta=f"{cal.get('expected_calibration_error', 0.0) - uncal.get('expected_calibration_error', 0.0):.4f}",
        delta_color="inverse",
    )
    c3.metric(
        "Log Loss",
        f"{cal.get('log_loss', 0.0):.4f}",
        delta=f"{cal.get('log_loss', 0.0) - uncal.get('log_loss', 0.0):.4f}",
        delta_color="inverse",
    )
    c4.metric(
        "ROC-AUC",
        f"{cal.get('roc_auc', 0.0):.4f}",
        delta=f"{cal.get('roc_auc', 0.0) - uncal.get('roc_auc', 0.0):.4f}",
    )

    # Reliability Diagram Plot
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.write("**Reliability Diagram (Calibration Curve)**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        # Perfectly calibrated diagonal
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", color="#94a3b8", linewidth=1.5)

        # Uncalibrated curve
        uncal_curve = uncal.get("calibration_curve", {})
        if uncal_curve:
            ax.plot(
                uncal_curve.get("mean_predicted_value", []),
                uncal_curve.get("fraction_of_positives", []),
                "s-",
                color="#ef4444",
                label=f"Uncalibrated (Brier={uncal.get('brier_score', 0.0):.3f})",
                linewidth=1.8,
            )

        # Calibrated curve
        cal_curve = cal.get("calibration_curve", {})
        if cal_curve:
            ax.plot(
                cal_curve.get("mean_predicted_value", []),
                cal_curve.get("fraction_of_positives", []),
                "o-",
                color="#10b981",
                label=f"Sigmoid Calibrated (Brier={cal.get('brier_score', 0.0):.3f})",
                linewidth=2.0,
            )

        ax.set_xlabel("Mean Predicted Probability", color="#94a3b8")
        ax.set_ylabel("Observed Fraction of Positives", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_plot2:
        st.write("**Calibration Loss Comparison**")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
        fig_bar.patch.set_facecolor("#0f172a")
        ax_bar.set_facecolor("#1e293b")

        labels = ["Brier Score", "Log Loss", "ECE"]
        uncal_vals = [uncal.get("brier_score", 0), uncal.get("log_loss", 0), uncal.get("expected_calibration_error", 0)]
        cal_vals = [cal.get("brier_score", 0), cal.get("log_loss", 0), cal.get("expected_calibration_error", 0)]

        x = np.arange(len(labels))
        width = 0.35

        ax_bar.bar(x - width/2, uncal_vals, width, label="Uncalibrated", color="#ef4444")
        ax_bar.bar(x + width/2, cal_vals, width, label="Calibrated", color="#10b981")

        ax_bar.set_ylabel("Loss / Error Value", color="#94a3b8")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, color="#f8fafc")
        ax_bar.tick_params(colors="#94a3b8")
        ax_bar.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["bottom"].set_color("#334155")
        ax_bar.spines["left"].set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)


def render_explainability_section(
    explanation_data: Dict[str, Any],
    global_importance_df: Optional[pd.DataFrame] = None,
    is_calibrated_mode: bool = False,
) -> None:
    """Render SHAP feature importance and patient-level explanation plots."""
    st.markdown("---")
    st.subheader("🔬 Model Explainability & Feature Attributions (SHAP)")
    
    # Transparency label for SHAP
    if is_calibrated_mode:
        st.info(
            "📌 **Model Transparency Notice**: SHAP attributions explain the underlying base SVM decision function, "
            "not the external Platt calibration transformation. Probability calibration rescales output probabilities "
            "monotonically but does not alter the relative importance of individual clinical biomarkers."
        )
    else:
        st.caption(
            "Attributions are computed directly against original clinical biomarkers. "
            "Positive values push risk higher; negative values protect/lower risk."
        )

    tab_local, tab_global = st.tabs(["👤 Patient Prediction Breakdown", "🌍 Population Feature Importance"])

    with tab_local:
        contributions = explanation_data.get("feature_contributions", [])
        if contributions:
            features = [c["feature"] for c in reversed(contributions)]
            values = [c["shap_value"] for c in reversed(contributions)]
            colors = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

            fig, ax = plt.subplots(figsize=(8, 4.2))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#1e293b")

            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color="#f8fafc", fontsize=10)
            ax.tick_params(colors="#94a3b8")
            ax.axvline(0, color="#64748b", linestyle="--", linewidth=0.8)
            ax.set_xlabel("SHAP Attribution (Impact on Risk Probability)", color="#94a3b8", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#334155")
            ax.spines["left"].set_color("#334155")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            contrib_df = pd.DataFrame(contributions)
            contrib_df = contrib_df[["feature", "value", "shap_value", "direction"]].rename(
                columns={
                    "feature": "Biomarker",
                    "value": "Patient Value",
                    "shap_value": "SHAP Impact",
                    "direction": "Risk Direction",
                }
            )
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    with tab_global:
        if global_importance_df is not None and not global_importance_df.empty:
            st.write("**Global Biomarker Importance Across Population (Mean |SHAP|)**")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            fig2.patch.set_facecolor("#0f172a")
            ax2.set_facecolor("#1e293b")

            sorted_df = global_importance_df.sort_values(by="Mean_Abs_SHAP", ascending=True)
            y_pos2 = np.arange(len(sorted_df))
            ax2.barh(y_pos2, sorted_df["Mean_Abs_SHAP"], color="#8b5cf6", edgecolor="none", height=0.6)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(sorted_df["Feature"], color="#f8fafc", fontsize=10)
            ax2.tick_params(colors="#94a3b8")
            ax2.set_xlabel("Mean Absolute SHAP Value", color="#94a3b8", fontsize=10)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_color("#334155")
            ax2.spines["left"].set_color("#334155")
            plt.tight_layout()

            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Global importance is calculated on a background cohort sample.")


def render_counterfactual_comparison_chart(
    orig_prob: float,
    hypo_prob: float,
    delta_pp: float,
) -> None:
    """Render a clean comparison bar chart of original vs hypothetical probability."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    labels = ["Original Patient", "Hypothetical Scenario"]
    probs = [orig_prob * 100, hypo_prob * 100]
    colors = ["#3b82f6", "#10b981" if hypo_prob <= orig_prob else "#ef4444"]

    bars = ax.bar(labels, probs, color=colors, width=0.45)
    ax.set_ylabel("Predicted Risk Probability (%)", color="#94a3b8")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")

    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax.annotate(
            f"{p:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_audit_trail(df_records: pd.DataFrame) -> None:
    """Render historical patient records fetched from SQLite."""
    with st.expander("🕒 View Patient Audit Trail (SQLite Logs)", expanded=False):
        if df_records.empty:
            st.info("No audit records found in SQLite database yet.")
            return

        display_df = df_records.copy()
        if "risk_probability" in display_df.columns:
            display_df["risk_probability"] = display_df["risk_probability"].apply(lambda p: f"{p*100:.1f}%")

        st.dataframe(
            display_df[[
                "id",
                "timestamp",
                "patient_id",
                "age",
                "gender",
                "blood_pressure",
                "cholesterol",
                "bmi",
                "blood_sugar",
                "heart_rate",
                "ecg_result",
                "risk_classification",
                "risk_probability",
            ]],
            use_container_width=True,
            hide_index=True,
        )
