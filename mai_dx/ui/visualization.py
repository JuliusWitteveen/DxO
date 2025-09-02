"""Display helpers for the Streamlit diagnostic UI."""

import streamlit as st


def display_current_request():
    """Display the current request from the AI panel."""
    if not st.session_state.session or not st.session_state.session.turns:
        return
    last_turn = st.session_state.session.turns[-1]
    if not st.session_state.session.is_complete:
        action = last_turn.action_request
        if action.action_type == "ask":
            icon = "💬"
            color = "blue"
        elif action.action_type == "test":
            icon = "🔬"
            color = "orange"
        else:
            icon = "🎯"
            color = "green"
        st.markdown(
            f"""
        <div style=\"background-color: #f0f9ff; border-left: 4px solid {color}; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;\">
            <h4 style=\"margin: 0; color: #333;\">{icon} Clinical Request (Turn {last_turn.turn_number})</h4>
            <p style=\"margin: 0.5rem 0 0 0; font-size: 1.1rem;\">{action.content}</p>
            <p style=\"margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;\"><em>Reasoning: {action.reasoning}</em></p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_differential_diagnosis():
    """Display current differential diagnosis with visual indicators."""
    if not st.session_state.session or not st.session_state.session.case_state:
        return
    differential = st.session_state.session.case_state.differential_diagnosis
    if not differential:
        st.info("No differential diagnosis formulated yet")
        return
    st.subheader("📊 Current Differential Diagnosis")
    sorted_diff = sorted(differential.items(), key=lambda x: x[1], reverse=True)
    for diagnosis, probability in sorted_diff[:5]:
        if probability > 0.5:
            color = "#2ecc71"
            icon = "🟢"
        elif probability > 0.2:
            color = "#f39c12"
            icon = "🟡"
        else:
            color = "#e74c3c"
            icon = "🔴"
        st.markdown(
            f"""
        <div style=\"margin-bottom: 1rem;\">
            <div style=\"display: flex; justify-content: space-between; align-items: center;\">
                <span style=\"font-weight: bold;\">{icon} {diagnosis}</span>
                <span style=\"color: {color}; font-weight: bold;\">{probability:.1%}</span>
            </div>
            <div style=\"background-color: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;\">
                <div style=\"background-color: {color}; width: {probability*100}%; height: 100%;\"></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_session_history():
    """Display session history with expandable details."""
    if not st.session_state.session or not st.session_state.session.turns:
        return
    st.subheader("📋 Session History")
    for turn in reversed(st.session_state.session.turns[-5:]):
        action = turn.action_request
        if action.action_type == "ask":
            icon = "💬"
            color = "#3498db"
        elif action.action_type == "test":
            icon = "🔬"
            color = "#e67e22"
        else:
            icon = "🎯"
            color = "#27ae60"
        with st.expander(f"{icon} Turn {turn.turn_number} - {action.action_type.upper()}", expanded=(turn == st.session_state.session.turns[-1])):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Request:** {action.content}")
                if turn.physician_input:
                    st.markdown(f"**Your Response:** {turn.physician_input}")
                st.markdown(f"**AI Reasoning:** {action.reasoning}")
            with col2:
                if turn.differential_at_turn:
                    st.markdown("**Top Diagnoses:**")
                    for name, prob in list(sorted(turn.differential_at_turn.items(), key=lambda x: x[1], reverse=True))[:3]:
                        st.markdown(f"- {name}: {prob:.1%}")
                if hasattr(turn, "cost_at_turn"):
                    st.metric("Total Cost", f"${turn.cost_at_turn}")
