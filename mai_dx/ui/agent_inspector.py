"""Utilities for inspecting agent configurations."""

import streamlit as st
from mai_dx.main import AgentRole


class AgentInspector:
    """Provide interactive insights into agent configuration."""

    def show_agent_card(self, agent, role):
        """Display configuration details for a given agent."""
        with st.expander(f"🔧 {role.value} Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Model",
                    (agent.model_name if hasattr(agent, "model_name") else "N/A"),
                )
            with col2:
                st.metric(
                    "Temperature",
                    (f"{agent.temperature:.2f}" if hasattr(agent, "temperature") else "N/A"),
                )
            with col3:
                st.metric(
                    "Max Tokens",
                    (agent.max_tokens if hasattr(agent, "max_tokens") else "N/A"),
                )

            st.markdown("#### 📝 System Prompt")
            if hasattr(agent, "system_prompt"):
                st.code(agent.system_prompt, language="markdown")
                st.info(self._explain_prompt_purpose(role))
            else:
                st.info("System prompt not available")

            if hasattr(agent, "tools_list_dictionary") and agent.tools_list_dictionary:
                st.markdown("#### 🛠️ Available Tools")
                st.json(agent.tools_list_dictionary)

            st.markdown("#### ⚡ Performance Profile")
            self._show_agent_characteristics(role)

    def _explain_prompt_purpose(self, role):
        """Explain the role-specific system prompt."""
        explanations = {
            AgentRole.HYPOTHESIS: "This agent maintains the differential diagnosis using Bayesian reasoning. It updates disease probabilities based on new evidence, similar to how a physician mentally tracks likely diagnoses.",
            AgentRole.TEST_CHOOSER: "Selects diagnostic tests that provide maximum information gain. Considers which tests would best distinguish between competing diagnoses, optimizing for diagnostic value per dollar spent.",
            AgentRole.CHALLENGER: "Acts as devil's advocate to prevent anchoring bias and premature closure. Questions assumptions, suggests alternative diagnoses, and ensures the panel considers contradictory evidence.",
            AgentRole.STEWARDSHIP: "Ensures cost-effective care by challenging expensive, low-yield tests. Suggests cheaper alternatives when diagnostically equivalent options exist.",
            AgentRole.CHECKLIST: "Quality control agent that checks for logical consistency, ensures test names are valid, and flags potential errors in reasoning.",
            AgentRole.CONSENSUS: "Synthesizes all panel input to make the final decision. Must balance different perspectives and choose a single action.",
            AgentRole.GATEKEEPER: "Clinical information oracle that provides objective findings when requested. Only reveals information explicitly asked for, simulating real clinical scenarios.",
            AgentRole.JUDGE: "Evaluates final diagnosis accuracy against ground truth using a 5-point scale. Provides structured assessment of diagnostic performance.",
        }
        return explanations.get(role, "This agent participates in the diagnostic process.")

    def _show_agent_characteristics(self, role):
        """Show strengths, limitations, and common error patterns."""
        characteristics = {
            AgentRole.HYPOTHESIS: {
                "strengths": [
                    "Systematic probability tracking",
                    "Evidence integration",
                    "Maintains multiple hypotheses",
                ],
                "limitations": [
                    "May over-index on rare diseases",
                    "Probability estimates are approximations",
                ],
                "typical_errors": ["Availability bias", "Base rate neglect"],
            },
            AgentRole.TEST_CHOOSER: {
                "strengths": [
                    "Information theory approach",
                    "Cost-benefit analysis",
                    "Strategic test selection",
                ],
                "limitations": [
                    "Limited by test database",
                    "May not account for test availability",
                ],
                "typical_errors": [
                    "Over-testing when uncertain",
                    "Missing simple bedside tests",
                ],
            },
            AgentRole.CHALLENGER: {
                "strengths": [
                    "Prevents groupthink",
                    "Catches edge cases",
                    "Identifies contradictions",
                ],
                "limitations": [
                    "Can slow decision-making",
                    "May introduce noise",
                ],
                "typical_errors": [
                    "Over-challenging obvious diagnoses",
                    "Analysis paralysis",
                ],
            },
            AgentRole.STEWARDSHIP: {
                "strengths": [
                    "Cost optimization",
                    "Resource awareness",
                    "Practical constraints",
                ],
                "limitations": [
                    "May delay necessary expensive tests",
                    "Cost database approximations",
                ],
                "typical_errors": [
                    "Over-emphasis on cost",
                    "Missing cost-effective combinations",
                ],
            },
            AgentRole.CHECKLIST: {
                "strengths": [
                    "Consistency checking",
                    "Error detection",
                    "Quality assurance",
                ],
                "limitations": [
                    "Rule-based approach",
                    "May miss subtle errors",
                ],
                "typical_errors": [
                    "False positives on valid variations",
                    "Missing semantic errors",
                ],
            },
            AgentRole.CONSENSUS: {
                "strengths": [
                    "Synthesis ability",
                    "Balanced decisions",
                    "Structured output",
                ],
                "limitations": [
                    "May average out important outliers",
                    "Dependent on panel quality",
                ],
                "typical_errors": [
                    "Premature consensus",
                    "Missing minority reports",
                ],
            },
            AgentRole.GATEKEEPER: {
                "strengths": [
                    "Objective information provision",
                    "Prevents information leakage",
                    "Simulates real scenarios",
                ],
                "limitations": [
                    "Cannot provide physical exam",
                    "Limited to text descriptions",
                ],
                "typical_errors": [
                    "Information ambiguity",
                    "Missing contextual cues",
                ],
            },
            AgentRole.JUDGE: {
                "strengths": [
                    "Objective scoring",
                    "Consistent evaluation",
                    "Clear criteria",
                ],
                "limitations": [
                    "Binary ground truth assumptions",
                    "May miss partial correctness",
                ],
                "typical_errors": [
                    "Over-penalizing synonyms",
                    "Missing nuanced accuracy",
                ],
            },
        }

        if role in characteristics:
            char = characteristics[role]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Strengths:**")
                for s in char["strengths"]:
                    st.markdown(f"• {s}")
            with col2:
                st.markdown("**⚠️ Limitations:**")
                for limitation in char["limitations"]:
                    st.markdown(f"• {limitation}")
            if "typical_errors" in char:
                st.warning(f"**Common Error Patterns:** {', '.join(char['typical_errors'])}")
