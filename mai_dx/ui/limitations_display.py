"""Display system limitations and guidance."""

import streamlit as st


class LimitationsDisplay:
    """Present system capabilities, limitations, and best practices."""

    def show_limitations_panel(self):
        """Display guidance on appropriate system use."""
        st.markdown("### ⚠️ System Limitations & Appropriate Use")
        tabs = st.tabs(["Capabilities", "Limitations", "Risk Factors", "Best Practices"])

        with tabs[0]:
            st.success("✅ **What this system does well:**")
            capabilities = [
                "Systematically considers multiple diagnoses simultaneously",
                "Integrates new evidence using probabilistic reasoning",
                "Identifies when additional information is needed",
                "Balances diagnostic thoroughness with cost-effectiveness",
                "Provides transparent reasoning for each decision",
                "Catches potential cognitive biases through challenger agent",
            ]
            for cap in capabilities:
                st.markdown(f"• {cap}")

        with tabs[1]:
            st.warning("⚠️ **Known limitations:**")
            limitations = [
                "**Not a replacement for clinical judgment** - designed as a decision support tool",
                "**Probability estimates are approximations** - not validated against epidemiological data",
                "**May miss rare presentations** - trained on typical cases",
                "**Cannot perform physical examination** - relies on reported findings",
                "**No real-time integration** - doesn't connect to EMR or lab systems",
                "**Language model constraints** - may hallucinate or misinterpret complex medical terminology",
            ]
            for lim in limitations:
                st.markdown(f"• {lim}")

        with tabs[2]:
            st.error("🚨 **Risk factors to monitor:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Clinical Risks:**")
                clinical_risks = [
                    "Anchoring on initial diagnosis",
                    "Missing time-sensitive conditions",
                    "Over-testing due to AI suggestions",
                    "False confidence in AI output",
                ]
                for risk in clinical_risks:
                    st.markdown(f"• {risk}")
            with col2:
                st.markdown("**Technical Risks:**")
                technical_risks = [
                    "Model hallucinations",
                    "Outdated medical knowledge",
                    "Biases in training data",
                    "API failures or timeouts",
                ]
                for risk in technical_risks:
                    st.markdown(f"• {risk}")

        with tabs[3]:
            st.info("💡 **Best practices for use:**")
            practices = [
                "Always verify AI suggestions against clinical guidelines",
                "Use as a 'second opinion' rather than primary decision maker",
                "Document when AI recommendations differ from clinical judgment",
                "Regularly review AI performance against actual outcomes",
                "Maintain awareness of system confidence levels",
                "Question unexpected or counterintuitive recommendations",
            ]
            for practice in practices:
                st.markdown(f"• {practice}")
