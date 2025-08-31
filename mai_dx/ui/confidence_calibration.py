"""Analyze and visualize diagnostic confidence."""

from typing import Dict
import numpy as np
import streamlit as st


class ConfidenceCalibration:
    """Analyze and visualize diagnostic confidence."""

    def show_confidence_analysis(self, session):
        """Display confidence and certainty metrics for a session."""
        st.markdown("### 📊 Confidence Analysis")

        if not session or not hasattr(session, "case_state") or not session.case_state:
            st.info("No active session to analyze")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            if hasattr(session.case_state, "get_max_confidence"):
                current_conf = session.case_state.get_max_confidence()
            else:
                current_conf = 0
                if (
                    hasattr(session.case_state, "differential_diagnosis")
                    and session.case_state.differential_diagnosis
                ):
                    current_conf = max(session.case_state.differential_diagnosis.values())
            st.metric(
                "Current Confidence",
                f"{current_conf:.1%}",
                delta=f"{current_conf - 0.5:.1%} from baseline",
            )
            if current_conf < 0.3:
                st.error("⚠️ Very low confidence - significant uncertainty")
            elif current_conf < 0.6:
                st.warning("⚡ Moderate confidence - additional information needed")
            elif current_conf < 0.8:
                st.info("✓ Good confidence - diagnosis taking shape")
            else:
                st.success("✅ High confidence - strong diagnostic hypothesis")

        with col2:
            if hasattr(session, "turns") and len(session.turns) > 1:
                if hasattr(session.case_state, "get_leading_diagnosis"):
                    leading_diagnosis = session.case_state.get_leading_diagnosis()
                else:
                    if (
                        hasattr(session.case_state, "differential_diagnosis")
                        and session.case_state.differential_diagnosis
                    ):
                        leading_diagnosis = max(
                            session.case_state.differential_diagnosis.items(),
                            key=lambda x: x[1],
                        )[0]
                    else:
                        leading_diagnosis = None
                if leading_diagnosis:
                    confidences = []
                    for t in session.turns:
                        if hasattr(t, "differential_at_turn") and t.differential_at_turn:
                            confidences.append(t.differential_at_turn.get(leading_diagnosis, 0))
                    volatility = np.std(confidences) if len(confidences) > 1 else 0
                    if volatility < 0.1:
                        stability_label = "Stable"
                        stability_color = "normal"
                    elif volatility < 0.2:
                        stability_label = "Variable"
                        stability_color = "normal"
                    else:
                        stability_label = "Volatile"
                        stability_color = "inverse"
                    st.metric(
                        "Confidence Stability",
                        stability_label,
                        delta=f"σ = {volatility:.3f}",
                        delta_color=stability_color,
                    )
                else:
                    st.metric(
                        "Confidence Stability",
                        "N/A",
                        help="No leading diagnosis identified",
                    )
            else:
                st.metric(
                    "Confidence Stability",
                    "N/A",
                    help="Need multiple turns to assess stability",
                )

        with col3:
            if (
                hasattr(session.case_state, "differential_diagnosis")
                and session.case_state.differential_diagnosis
            ):
                entropy = self._calculate_entropy(session.case_state.differential_diagnosis)
                max_entropy = np.log2(len(session.case_state.differential_diagnosis))
                certainty = (1 - entropy / max_entropy) * 100 if max_entropy > 0 else 0
                st.metric(
                    "Diagnostic Certainty",
                    f"{certainty:.0f}%",
                    help="Lower entropy means higher certainty",
                )
            else:
                st.metric(
                    "Diagnostic Certainty",
                    "N/A",
                    help="No differential diagnosis yet",
                )

    def _calculate_entropy(self, differential: Dict[str, float]) -> float:
        """Compute entropy for a differential diagnosis distribution."""
        if not differential:
            return 0
        probs = np.array(list(differential.values()))
        if probs.sum() == 0:
            return 0
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
