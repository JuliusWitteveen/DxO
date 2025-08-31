"""Components for displaying live agent deliberations."""

import time
import streamlit as st


class DeliberationTheater:
    """Display a live view of agent deliberations."""

    def __init__(self):
        """Initialize placeholders for panel deliberation."""
        self.deliberation_log = []
        self.current_speakers = {}

    def show_live_panel(self, case_state):
        """Render the panel discussion layout for the current turn."""
        st.markdown("### 🎭 Live Panel Discussion - Turn {}".format(case_state.iteration))

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

    def stream_agent_response(self, placeholder, agent_name: str, response: str, thinking: bool = False):
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
