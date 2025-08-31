"""Streamlit utilities for visualizing MAI-DxO diagnostic sessions."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import re
import os
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from copy import deepcopy
from mai_dx.main import AgentRole


class AuditLogger:
    """Simple append-only audit logging with hash chaining for integrity."""

    def __init__(self, log_path: str = "audit_logs/audit.log"):
        """Initialize the logger and load the last entry hash."""
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Load the previous entry hash from disk."""
        if not os.path.exists(self.log_path):
            return "0" * 64
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = None
                for last_line in f:
                    pass
                if last_line:
                    data = json.loads(last_line)
                    return data.get("hash", "0" * 64)
        except (json.JSONDecodeError, OSError):
            pass
        return "0" * 64

    def _write_entry(self, entry: Dict[str, Any]):
        """Write an entry to the log, updating the hash chain."""
        entry["prev_hash"] = self.last_hash
        entry_bytes = json.dumps(entry, sort_keys=True).encode("utf-8")
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry["hash"] = entry_hash
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.last_hash = entry_hash

    def log_state_change(self, state: Any, description: str = ""):
        """Record a state change in the audit log."""
        try:
            state_dict = asdict(state)
        except Exception:
            state_dict = state
        entry = {
            "timestamp": time.time(),
            "type": "state_change",
            "description": description,
            "state": state_dict,
        }
        self._write_entry(entry)

    def log_decision(
        self, decision: str, details: Optional[Dict[str, Any]] = None
    ):
        """Record a decision made by the system."""
        entry = {
            "timestamp": time.time(),
            "type": "decision",
            "decision": decision,
            "details": details or {},
        }
        self._write_entry(entry)


class DiagnosticFlowVisualizer:
    """Generate visualizations of diagnostic flows."""

    def generate_sankey(self, case_state) -> go.Figure:
        """Create an interactive Sankey diagram of diagnostic flow."""
        nodes = ["Initial Vignette"]
        sources: List[int] = []
        targets: List[int] = []
        values: List[int] = []

        # Build nodes for evidence/test/question logs
        current_index = 0
        for entry in case_state.evidence_log:
            nodes.append(entry)
            sources.append(current_index)
            targets.append(len(nodes) - 1)
            values.append(1)
            current_index = len(nodes) - 1

        final_diag = getattr(
            case_state, "get_leading_diagnosis", lambda: "Diagnosis"
        )()
        nodes.append(f"Diagnosis: {final_diag}")
        sources.append(current_index)
        targets.append(len(nodes) - 1)
        values.append(1)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(pad=15, thickness=20, label=nodes),
                    link=dict(source=sources, target=targets, value=values),
                )
            ]
        )
        fig.update_layout(title_text="Diagnostic Flow", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
        return fig


class DeliberationTheater:
    """Display a live view of agent deliberations."""

    def __init__(self):
        """Initialize placeholders for panel deliberation."""
        self.deliberation_log = []
        self.current_speakers = {}

    def show_live_panel(self, case_state):
        """Render the panel discussion layout for the current turn."""
        st.markdown(
            "### 🎭 Live Panel Discussion - Turn {}".format(
                case_state.iteration
            )
        )

        cols = st.columns(5)

        agents_layout = [
            ("🧠 Dr. Hypothesis", "hypothesis_placeholder"),
            ("🔬 Dr. Test-Chooser", "test_placeholder"),
            ("😈 Dr. Challenger", "challenger_placeholder"),
            ("💰 Dr. Stewardship", "steward_placeholder"),
            ("✅ Dr. Checklist", "checklist_placeholder"),
        ]

        placeholders = {}
        for col, (agent_name, key) in zip(cols, agents_layout):
            with col:
                st.markdown(f"**{agent_name}**")
                placeholders[key] = st.empty()

        st.markdown("---")
        consensus_placeholder = st.empty()

        return placeholders, consensus_placeholder

    def stream_agent_response(
        self,
        placeholder,
        agent_name: str,
        response: str,
        thinking: bool = False,
    ):
        """Animate an agent's streaming response in the UI."""
        if thinking:
            placeholder.info("🤔 *thinking...*")
            time.sleep(0.5)

        displayed_text = ""
        preview_length = min(200, len(response))

        for i in range(0, preview_length, 5):
            displayed_text = response[: i + 5]
            placeholder.markdown(
                f'<div style="background-color: #f0f2f6; padding: 10px; '
                f'border-radius: 10px; min-height: 100px;">{displayed_text}...</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.01)

        if len(response) > 200:
            with placeholder.container():
                st.markdown(
                    f'<div style="background-color: #f0f2f6; padding: 10px; '
                    f'border-radius: 10px; min-height: 100px;">{response[:200]}...</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("See full response"):
                    st.markdown(response)
        else:
            placeholder.markdown(
                f'<div style="background-color: #f0f2f6; padding: 10px; '
                f'border-radius: 10px; min-height: 100px;">{response}</div>',
                unsafe_allow_html=True,
            )


class AgentInspector:
    """Provide interactive insights into agent configuration."""

    def show_agent_card(self, agent, role):
        """Display configuration details for a given agent."""
        with st.expander(f"🔧 {role.value} Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Model",
                    (
                        agent.model_name
                        if hasattr(agent, "model_name")
                        else "N/A"
                    ),
                )
            with col2:
                st.metric(
                    "Temperature",
                    (
                        f"{agent.temperature:.2f}"
                        if hasattr(agent, "temperature")
                        else "N/A"
                    ),
                )
            with col3:
                st.metric(
                    "Max Tokens",
                    (
                        agent.max_tokens
                        if hasattr(agent, "max_tokens")
                        else "N/A"
                    ),
                )

            st.markdown("#### 📝 System Prompt")
            if hasattr(agent, "system_prompt"):
                st.code(agent.system_prompt, language="markdown")
                st.info(self._explain_prompt_purpose(role))
            else:
                st.info("System prompt not available")

            if (
                hasattr(agent, "tools_list_dictionary")
                and agent.tools_list_dictionary
            ):
                st.markdown("#### 🛠️ Available Tools")
                st.json(agent.tools_list_dictionary)

            st.markdown("#### ⚡ Performance Profile")
            self._show_agent_characteristics(role)

    def _explain_prompt_purpose(self, role):
        """Explain the role-specific system prompt."""
        explanations = {
            AgentRole.HYPOTHESIS: "This agent maintains the differential diagnosis using Bayesian reasoning. "
            "It updates disease probabilities based on new evidence, similar to how "
            "a physician mentally tracks likely diagnoses.",
            AgentRole.TEST_CHOOSER: "Selects diagnostic tests that provide maximum information gain. "
            "Considers which tests would best distinguish between competing diagnoses, "
            "optimizing for diagnostic value per dollar spent.",
            AgentRole.CHALLENGER: "Acts as devil's advocate to prevent anchoring bias and premature closure. "
            "Questions assumptions, suggests alternative diagnoses, and ensures "
            "the panel considers contradictory evidence.",
            AgentRole.STEWARDSHIP: "Ensures cost-effective care by challenging expensive, low-yield tests. "
            "Suggests cheaper alternatives when diagnostically equivalent options exist.",
            AgentRole.CHECKLIST: "Quality control agent that checks for logical consistency, "
            "ensures test names are valid, and flags potential errors in reasoning.",
            AgentRole.CONSENSUS: "Synthesizes all panel input to make the final decision. "
            "Must balance different perspectives and choose a single action.",
            AgentRole.GATEKEEPER: "Clinical information oracle that provides objective findings when requested. "
            "Only reveals information explicitly asked for, simulating real clinical scenarios.",
            AgentRole.JUDGE: "Evaluates final diagnosis accuracy against ground truth using a 5-point scale. "
            "Provides structured assessment of diagnostic performance.",
        }

        return explanations.get(
            role, "This agent participates in the diagnostic process."
        )

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
                st.warning(
                    f"**Common Error Patterns:** {', '.join(char['typical_errors'])}"
                )


class ReasoningTraceViewer:
    """Visualize how agents reason through a diagnostic turn."""

    def show_reasoning_flow(self, turn):
        """Display structured reasoning for a given turn."""
        st.markdown("### 🧩 Reasoning Flow")

        steps = [
            {
                "agent": "Dr. Hypothesis",
                "input": "Previous evidence + new findings",
                "process": "Bayesian probability update",
                "output": (
                    turn.differential_at_turn
                    if hasattr(turn, "differential_at_turn")
                    else {}
                ),
                "raw_response": (
                    turn.deliberation.get("Dr. Hypothesis", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else ""
                ),
            },
            {
                "agent": "Dr. Test-Chooser",
                "input": "Current differential diagnosis",
                "process": "Information gain calculation",
                "output": "Recommended tests",
                "raw_response": (
                    turn.deliberation.get("Dr. Test-Chooser", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else ""
                ),
            },
            {
                "agent": "Dr. Challenger",
                "input": "Panel recommendations",
                "process": "Critical analysis & bias detection",
                "output": "Alternative hypotheses",
                "raw_response": (
                    turn.deliberation.get("Dr. Challenger", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else ""
                ),
            },
            {
                "agent": "Dr. Stewardship",
                "input": "Proposed tests & current costs",
                "process": "Cost-effectiveness analysis",
                "output": "Resource optimization",
                "raw_response": (
                    turn.deliberation.get("Dr. Stewardship", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else ""
                ),
            },
            {
                "agent": "Dr. Checklist",
                "input": "All panel outputs",
                "process": "Quality control checks",
                "output": "Validation results",
                "raw_response": (
                    turn.deliberation.get("Dr. Checklist", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else ""
                ),
            },
        ]

        turn_num = turn.turn_number if hasattr(turn, "turn_number") else 0

        for i, step in enumerate(steps, 1):
            with st.expander(f"Step {i}: {step['agent']}", expanded=(i == 1)):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("**Input:**")
                    st.info(step["input"])

                    st.markdown("**Process:**")
                    st.info(step["process"])

                    st.markdown("**Output:**")
                    if isinstance(step["output"], dict) and step["output"]:
                        formatted_output = {}
                        for k, v in step["output"].items():
                            if isinstance(v, float):
                                formatted_output[k] = f"{v:.1%}"
                            else:
                                formatted_output[k] = v
                        st.json(formatted_output)
                    else:
                        st.success(
                            step["output"]
                            if step["output"]
                            else "Processing..."
                        )

                with col2:
                    st.markdown("**Raw Agent Response:**")
                    response_text = (
                        str(step["raw_response"])
                        if step["raw_response"]
                        else ""
                    )
                    if response_text:
                        display_text = response_text[:500] + (
                            "..." if len(response_text) > 500 else ""
                        )
                        st.text_area(
                            label=f"raw_response_{step['agent']}",
			    label_visibility="collapsed",
                            value=display_text,
                            height=200,
                            key=f"raw_{step['agent'].replace(' ', '_').replace('.', '')}_{turn_num}_{i}",
                        )

                        self._show_reasoning_highlights(response_text)
                    else:
                        st.info("No response available")

    def _show_reasoning_highlights(self, text: str):
        """Highlight notable reasoning patterns in text."""
        patterns = {
            "evidence_integration": r"(based on|given that|considering)",
            "uncertainty": r"(possibly|might|unclear|uncertain)",
            "confidence": r"(clearly|definitely|certainly|highly likely)",
            "alternatives": r"(alternatively|however|but|although)",
        }

        highlights = []
        search_text = text[:500] if len(text) > 500 else text
        for pattern_type, pattern in patterns.items():
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                highlights.append(
                    {
                        "type": pattern_type,
                        "text": match.group(),
                        "position": match.start(),
                    }
                )

        if highlights:
            st.caption("**Reasoning Patterns Detected:**")
            pattern_counts = {}
            for h in highlights:
                pattern_counts[h["type"]] = (
                    pattern_counts.get(h["type"], 0) + 1
                )

            cols = st.columns(len(pattern_counts))
            for idx, (ptype, count) in enumerate(pattern_counts.items()):
                emoji = {
                    "evidence_integration": "🔗",
                    "uncertainty": "❓",
                    "confidence": "✅",
                    "alternatives": "🔄",
                }.get(ptype, "")
                with cols[idx]:
                    st.caption(
                        f"{emoji} {ptype.replace('_', ' ').title()}: {count}"
                    )


class LimitationsDisplay:
    """Present system capabilities, limitations, and best practices."""

    def show_limitations_panel(self):
        """Display guidance on appropriate system use."""
        st.markdown("### ⚠️ System Limitations & Appropriate Use")

        tabs = st.tabs(
            ["Capabilities", "Limitations", "Risk Factors", "Best Practices"]
        )

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


class InteractiveExplorer:
    """Enable exploration of alternative diagnostic scenarios."""

    def show_exploration_panel(self, session):
        """Provide tools for simulating alternative evidence or thresholds."""
        st.markdown("### 🔬 Exploration Mode")
        st.info(
            "Explore 'what-if' scenarios without affecting the diagnostic session"
        )

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

                            action, deliberation = (
                                session._orchestrator._perform_turn(
                                    shadow_state
                                )
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original Decision:**")
                                if session.turns:
                                    st.info(
                                        session.turns[
                                            -1
                                        ].action_request.content
                                    )
                                else:
                                    st.info("No decision yet")

                            with col2:
                                st.markdown("**Alternative Decision:**")
                                st.success(action.content)

                            st.markdown(
                                "**Impact on Differential Diagnosis:**"
                            )

                            original_diag = (
                                session.case_state.differential_diagnosis
                                if session.case_state
                                else {}
                            )
                            alternative_diag = (
                                shadow_state.differential_diagnosis
                            )

                            all_diagnoses = set(
                                list(original_diag.keys())
                                + list(alternative_diag.keys())
                            )

                            if all_diagnoses:
                                comparison_data = []
                                for diag in all_diagnoses:
                                    orig_val = original_diag.get(diag, 0) * 100
                                    alt_val = (
                                        alternative_diag.get(diag, 0) * 100
                                    )
                                    comparison_data.append(
                                        {
                                            "Diagnosis": diag,
                                            "Original %": orig_val,
                                            "Alternative %": alt_val,
                                            "Change": alt_val - orig_val,
                                        }
                                    )

                                comparison_df = pd.DataFrame(comparison_data)
                                comparison_df = comparison_df.sort_values(
                                    "Change", ascending=False
                                )

                                st.dataframe(
                                    comparison_df.style.format(
                                        {
                                            "Original %": "{:.1f}",
                                            "Alternative %": "{:.1f}",
                                            "Change": "{:+.1f}",
                                        }
                                    ).background_gradient(
                                        subset=["Change"], cmap="RdYlGn"
                                    ),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                        except Exception as e:
                            st.error(
                                f"Could not run alternative scenario: {str(e)}"
                            )
                else:
                    if not alternative_evidence:
                        st.warning(
                            "Please enter alternative evidence to simulate"
                        )
                    else:
                        st.warning("Session not ready for exploration")

        with tabs[1]:
            st.markdown(
                "**How sensitive is the diagnosis to confidence thresholds?**"
            )

            threshold = st.slider(
                "Diagnosis confidence threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="confidence_threshold",
            )

            if (
                session
                and hasattr(session, "case_state")
                and session.case_state
            ):
                current_differential = (
                    session.case_state.differential_diagnosis
                    if hasattr(session.case_state, "differential_diagnosis")
                    else {}
                )

                if current_differential:
                    above_threshold = {
                        k: v
                        for k, v in current_differential.items()
                        if v >= threshold
                    }

                    if above_threshold:
                        st.success(
                            f"Diagnoses above {threshold:.0%} confidence:"
                        )
                        for diag, conf in sorted(
                            above_threshold.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        ):
                            st.markdown(f"• **{diag}**: {conf:.1%}")
                    else:
                        st.warning(
                            f"No diagnoses meet the {threshold:.0%} threshold"
                        )

                    max_conf = (
                        max(current_differential.values())
                        if current_differential
                        else 0
                    )
                    if max_conf < threshold:
                        st.info(
                            f"Current maximum confidence is {max_conf:.1%}. Consider lowering the threshold or gathering more evidence."
                        )
                else:
                    st.info("No differential diagnosis available yet")
            else:
                st.info("No diagnostic session active")


class ConfidenceCalibration:
    """Analyze and visualize diagnostic confidence."""

    def show_confidence_analysis(self, session):
        """Display confidence and certainty metrics for a session."""
        st.markdown("### 📊 Confidence Analysis")

        if (
            not session
            or not hasattr(session, "case_state")
            or not session.case_state
        ):
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
                    current_conf = max(
                        session.case_state.differential_diagnosis.values()
                    )

            st.metric(
                "Current Confidence",
                f"{current_conf:.1%}",
                delta=f"{current_conf - 0.5:.1%} from baseline",
            )

            if current_conf < 0.3:
                st.error("⚠️ Very low confidence - significant uncertainty")
            elif current_conf < 0.6:
                st.warning(
                    "⚡ Moderate confidence - additional information needed"
                )
            elif current_conf < 0.8:
                st.info("✓ Good confidence - diagnosis taking shape")
            else:
                st.success("✅ High confidence - strong diagnostic hypothesis")

        with col2:
            if hasattr(session, "turns") and len(session.turns) > 1:
                if hasattr(session.case_state, "get_leading_diagnosis"):
                    leading_diagnosis = (
                        session.case_state.get_leading_diagnosis()
                    )
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
                        if (
                            hasattr(t, "differential_at_turn")
                            and t.differential_at_turn
                        ):
                            confidences.append(
                                t.differential_at_turn.get(
                                    leading_diagnosis, 0
                                )
                            )

                    if len(confidences) > 1:
                        volatility = np.std(confidences)
                    else:
                        volatility = 0

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
                entropy = self._calculate_entropy(
                    session.case_state.differential_diagnosis
                )
                max_entropy = np.log2(
                    len(session.case_state.differential_diagnosis)
                )
                certainty = (
                    (1 - entropy / max_entropy) * 100 if max_entropy > 0 else 0
                )

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


def create_diagnostic_journey_visualization(session):
    """Create visualizations summarizing the diagnostic journey."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Confidence Evolution",
            "Cost Accumulation",
            "Differential Diagnosis Timeline",
            "Action Distribution",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    if not session or not hasattr(session, "turns") or not session.turns:
        fig.add_annotation(
            text="No data available yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(height=700, showlegend=False)
        return fig

    turns = []
    confidences = []
    costs = []

    for t in session.turns:
        if hasattr(t, "turn_number"):
            turns.append(t.turn_number)
        else:
            turns.append(len(turns) + 1)

        if hasattr(t, "differential_at_turn") and t.differential_at_turn:
            confidences.append(max(t.differential_at_turn.values()))
        else:
            confidences.append(0)

        if hasattr(t, "cost_at_turn"):
            costs.append(t.cost_at_turn)
        else:
            costs.append(0)

    fig.add_trace(
        go.Scatter(
            x=turns,
            y=confidences,
            mode="lines+markers",
            name="Confidence",
            line=dict(width=3, color="blue"),
            marker=dict(size=8),
            hovertemplate="Turn %{x}<br>Confidence: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=turns,
            y=costs,
            mode="lines+markers",
            name="Cost",
            line=dict(width=3, color="red"),
            marker=dict(size=8),
            hovertemplate="Turn %{x}<br>Cost: $%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    all_diagnoses = {}
    for turn in session.turns:
        if hasattr(turn, "differential_at_turn") and turn.differential_at_turn:
            turn_num = (
                turn.turn_number
                if hasattr(turn, "turn_number")
                else len(all_diagnoses) + 1
            )
            for diag, prob in turn.differential_at_turn.items():
                if diag not in all_diagnoses:
                    all_diagnoses[diag] = []
                all_diagnoses[diag].append((turn_num, prob))

    for diag, points in all_diagnoses.items():
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=diag[:30],
                hovertemplate=f"{diag}<br>Turn %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    action_counts = {"ask": 0, "test": 0, "diagnose": 0}
    for turn in session.turns:
        if hasattr(turn, "action_request") and hasattr(
            turn.action_request, "action_type"
        ):
            action_type = turn.action_request.action_type
            if action_type in action_counts:
                action_counts[action_type] += 1

    if sum(action_counts.values()) > 0:
        fig.add_trace(
            go.Pie(
                labels=list(action_counts.keys()),
                values=list(action_counts.values()),
                marker=dict(colors=["#3498db", "#e74c3c", "#2ecc71"]),
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Turn", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Turn", row=1, col=2)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    fig.update_xaxes(title_text="Turn", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    fig.update_layout(
        height=700, showlegend=True, title_text="Diagnostic Journey Overview"
    )

    return fig


def create_confidence_timeline(session):
    """Plot confidence and cost trends across turns."""
    if not session or not hasattr(session, "turns") or not session.turns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(height=300, showlegend=False)
        return fig

    fig = go.Figure()

    for i, turn in enumerate(session.turns):
        colors = {"ask": "blue", "test": "orange", "diagnose": "green"}

        action_type = "unknown"
        action_content = "No action"
        if hasattr(turn, "action_request"):
            if hasattr(turn.action_request, "action_type"):
                action_type = turn.action_request.action_type
            if hasattr(turn.action_request, "content"):
                action_content = str(turn.action_request.content)[:100]

        leading_diag = "None"
        confidence = 0
        if hasattr(turn, "differential_at_turn") and turn.differential_at_turn:
            max_item = max(
                turn.differential_at_turn.items(), key=lambda x: x[1]
            )
            leading_diag = max_item[0]
            confidence = max_item[1]

        cost = turn.cost_at_turn if hasattr(turn, "cost_at_turn") else 0

        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=colors.get(action_type, "gray"),
                    symbol="circle",
                ),
                text=action_type.upper(),
                textposition="top center",
                name=f"Turn {i+1}",
                hovertemplate=(
                    f"<b>Turn {i+1}</b><br>"
                    f"Action: {action_type}<br>"
                    f"Request: {action_content}...<br>"
                    f"Cost: ${cost}<br>"
                    f"Leading Diagnosis: {leading_diag}<br>"
                    f"Confidence: {confidence:.1%}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Diagnostic Timeline",
        xaxis_title="Turn Number",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        height=300,
        hovermode="closest",
    )

    return fig
