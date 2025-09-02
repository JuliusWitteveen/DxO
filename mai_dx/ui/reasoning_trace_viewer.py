"""Visualization of agent reasoning traces."""

import re
import streamlit as st


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
                    turn.differential_at_turn if hasattr(turn, "differential_at_turn") else {}
                ),
                "raw_response": (
                    turn.deliberation.get("Dr. Hypothesis", "")
                    if hasattr(turn, "deliberation") and turn.deliberation
                    else "",
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
                    else "",
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
                    else "",
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
                    else "",
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
                    else "",
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
                            formatted_output[k] = f"{v:.1%}" if isinstance(v, float) else v
                        st.json(formatted_output)
                    else:
                        st.success(step["output"] if step["output"] else "Processing...")
                with col2:
                    st.markdown("**Raw Agent Response:**")
                    response_text = (
                        str(step["raw_response"]) if step["raw_response"] else ""
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
            for match in re.finditer(pattern, search_text, re.IGNORECASE):
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
                pattern_counts[h["type"]] = pattern_counts.get(h["type"], 0) + 1

            cols = st.columns(len(pattern_counts))
            for idx, (ptype, count) in enumerate(pattern_counts.items()):
                emoji = {
                    "evidence_integration": "🔗",
                    "uncertainty": "❓",
                    "confidence": "✅",
                    "alternatives": "🔄",
                }.get(ptype, "")
                with cols[idx]:
                    st.caption(f"{emoji} {ptype.replace('_', ' ').title()}: {count}")
