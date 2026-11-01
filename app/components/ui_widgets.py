"""UI widgets and visualization components for Heart Disease Risk Streamlit app."""

import json
from typing import Any, Dict, List
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


def render_risk_badge(classification: str, probability: float) -> None:
    """Render risk prediction outcome badge and confidence indicator."""
    is_high = classification == "High Risk"
    badge_bg = "#dc2626" if is_high else "#16a34a"
    badge_text = "HIGH RISK OF HEART DISEASE" if is_high else "LOW RISK OF HEART DISEASE"
    icon = "⚠️" if is_high else "✅"

    pct = probability * 100

    st.markdown(
        f"""
        <div style="background-color: {badge_bg}; color: white; padding: 1.2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
            <div style="font-size: 1.6rem; font-weight: 700; letter-spacing: 0.05em;">
                {icon} {badge_text}
            </div>
            <div style="font-size: 1.1rem; margin-top: 0.4rem; opacity: 0.95;">
                Model Estimated Probability: <strong>{pct:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar indicator
    st.write(f"**Confidence Gauge ({pct:.1f}% Probability)**")
    st.progress(float(probability))


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


def render_explainability_section(explanation_data: Dict[str, Any], global_importance_df: Optional[pd.DataFrame] = None) -> None:
    """Render SHAP feature importance and patient-level explanation plots."""
    st.markdown("---")
    st.subheader("🔬 Model Explainability & Feature Attributions (SHAP)")
    st.caption(
        "Attributions are computed directly against original clinical biomarkers. "
        "Positive values push risk higher; negative values protect/lower risk."
    )

    tab_local, tab_global = st.tabs(["👤 Patient Prediction Breakdown", "🌍 Population Feature Importance"])

    with tab_local:
        contributions = explanation_data.get("feature_contributions", [])
        if contributions:
            # Display waterfall-style horizontal bar chart
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

            # Contribution table breakdown
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
