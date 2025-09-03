"""Session management utilities for the Streamlit UI."""

import os
from datetime import datetime
from typing import Optional

import streamlit as st

from mai_dx.export import export_session_to_markdown
from mai_dx.interactive import InteractiveDxSession
from mai_dx.persistence import load_session, save_session
from ui_controls import get_settings


def is_api_key_set() -> bool:
    """Check if the API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def initialize_session(model_name: str, mode: str, case_details: str = "") -> bool:
    """Initialize a new diagnostic session using InteractiveDxSession."""
    try:
        # Get the full, current settings from the UI controls to ensure
        # all parameters like top_p and temperature are respected.
        ui_settings = get_settings()

        config = {
            "model_name": model_name,
            "mode": mode,
            "max_iterations": 15,
            "initial_budget": 10000,
            "physician_visit_cost": 300,
            # Pass all relevant parameters to the orchestrator
            "top_p": ui_settings.get("top_p"),
            "temperature": ui_settings.get("temperature"),
        }

        st.session_state.session = InteractiveDxSession(orchestrator_config=config)
        if case_details:
            st.session_state.session.start(case_details)
            return True
        return False
    except Exception as e:
        st.error(f"Failed to initialize session: {str(e)}")
        return False


def process_clinical_response(response: str, ui_turn_number: int):
    """Process the clinical response using InteractiveDxSession."""
    if not st.session_state.session:
        return None
    try:
        with st.spinner("Processing clinical response..."):
            st.session_state.session.step(response, ui_turn_number)
            if st.session_state.session.is_complete:
                last_turn = st.session_state.session.turns[-1]
                return {
                    "diagnosis_complete": True,
                    "final_diagnosis": last_turn.action_request.content,
                    "action": last_turn.action_request,
                }
            else:
                last_turn = st.session_state.session.turns[-1]
                return {
                    "diagnosis_complete": False,
                    "next_request": f"{last_turn.action_request.action_type.capitalize()}: {last_turn.action_request.content}",
                    "action": last_turn.action_request,
                }
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return None


def save_current_session() -> Optional[str]:
    """Save current session using the persistence module."""
    if not st.session_state.session:
        st.warning("No session data to save")
        return None
    try:
        session_data = st.session_state.session.to_dict()
        session_id = st.session_state.session.session_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data["saved_at"] = timestamp
        save_session(session_data, session_id)
        st.success(f"Session saved at {timestamp} (ID: {session_id[:8]}...)")
        return session_id
    except Exception as e:
        st.error(f"Failed to save session: {str(e)}")
        return None


def load_saved_session(session_id: str) -> bool:
    """Load a saved session using the persistence module."""
    try:
        session_data = load_session(session_id)
        st.session_state.session = InteractiveDxSession.from_dict(session_data)
        st.success("Session loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return False


def export_to_markdown() -> Optional[str]:
    """Export session to markdown format."""
    if not st.session_state.session:
        return None
    try:
        session_data = st.session_state.session.to_dict()
        return export_session_to_markdown(session_data)
    except Exception as e:
        st.error(f"Failed to export: {str(e)}")
        return None