"""Tools for exploring alternative diagnostic scenarios."""

from copy import deepcopy
import pandas as pd
import streamlit as st


class InteractiveExplorer:
    """Enable exploration of alternative diagnostic scenarios."""

    def show_exploration_panel(self, session):
        """Provide tools for simulating alternative evidence or thresholds."""
        st.markdown("### 🔬 Exploration Mode")
        st.info("Explore 'what-if' scenarios without affecting the diagnostic session")

        tabs = st.tabs(["Test Different Evidence", "Sensitivity Analysis"])

        with tabs[0]:
            st.markdown("**What if the patient had different symptoms?**")
            alternative_evidence = st.text_area(
                "Enter alternative findings:",
                placeholder="What if the patient also had fever and night sweats?",
                key="alt_evidence",
            )

            if st.button("Simulate Alternative", type="secondary"):
                if (
                    alternative_evidence
                    and session
                    and hasattr(session, "case_state")
                    and hasattr(session, "_orchestrator")
                ):
                    with st.spinner("Running alternative scenario..."):
                        try:
                            shadow_state = deepcopy(session.case_state)
                            shadow_state.add_evidence(alternative_evidence)
                            action, deliberation = session._orchestrator._perform_turn(shadow_state)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original Decision:**")
                                if session.turns:
                                    st.info(session.turns[-1].action_request.content)
                                else:
                                    st.info("No decision yet")
                            with col2:
                                st.markdown("**Alternative Decision:**")
                                st.success(action.content)

                            st.markdown("**Impact on Differential Diagnosis:**")
                            original_diag = session.case_state.differential_diagnosis if session.case_state else {}
                            alternative_diag = shadow_state.differential_diagnosis

                            all_diagnoses = set(list(original_diag.keys()) + list(alternative_diag.keys()))
                            if all_diagnoses:
                                comparison_data = []
                                for diag in all_diagnoses:
                                    orig_val = original_diag.get(diag, 0) * 100
                                    alt_val = alternative_diag.get(diag, 0) * 100
                                    comparison_data.append(
                                        {
                                            "Diagnosis": diag,
                                            "Original %": orig_val,
                                            "Alternative %": alt_val,
                                            "Change": alt_val - orig_val,
                                        }
                                    )

                                comparison_df = pd.DataFrame(comparison_data)
                                comparison_df = comparison_df.sort_values("Change", ascending=False)
                                st.dataframe(
                                    comparison_df.style.format(
                                        {
                                            "Original %": "{:.1f}",
                                            "Alternative %": "{:.1f}",
                                            "Change": "{:+.1f}",
                                        }
                                    ).background_gradient(subset=["Change"], cmap="RdYlGn"),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                        except Exception as e:
                            st.error(f"Could not run alternative scenario: {str(e)}")
                else:
                    if not alternative_evidence:
                        st.warning("Please enter alternative evidence to simulate")
                    else:
                        st.warning("Session not ready for exploration")

        with tabs[1]:
            st.markdown("**How sensitive is the diagnosis to confidence thresholds?**")
            threshold = st.slider(
                "Diagnosis confidence threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="confidence_threshold",
            )

            if session and hasattr(session, "case_state") and session.case_state:
                current_differential = (
                    session.case_state.differential_diagnosis
                    if hasattr(session.case_state, "differential_diagnosis")
                    else {}
                )
                if current_differential:
                    above_threshold = {
                        k: v for k, v in current_differential.items() if v >= threshold
                    }
                    if above_threshold:
                        st.success(f"Diagnoses above {threshold:.0%} confidence:")
                        for diag, conf in sorted(
                            above_threshold.items(), key=lambda x: x[1], reverse=True
                        ):
                            st.markdown(f"• **{diag}**: {conf:.1%}")
                    else:
                        st.warning(f"No diagnoses meet the {threshold:.0%} threshold")

                    max_conf = max(current_differential.values()) if current_differential else 0
                    if max_conf < threshold:
                        st.info(
                            f"Current maximum confidence is {max_conf:.1%}. Consider lowering the threshold or gathering more evidence."
                        )
                else:
                    st.info("No differential diagnosis available yet")
            else:
                st.info("No diagnostic session active")
