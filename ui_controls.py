from __future__ import annotations
import streamlit as st
from config import OPENAI_MODELS, DEFAULT_UI_SETTINGS

def _init_session() -> None:
    if "dxo_settings" not in st.session_state:
        st.session_state.dxo_settings = DEFAULT_UI_SETTINGS.copy()

def get_settings() -> dict:
    _init_session()
    return dict(st.session_state.dxo_settings)

def render_settings_panel() -> dict:
    st.sidebar.markdown("## Settings (OpenAI)")

    _init_session()
    current = st.session_state.dxo_settings

    model_name = st.sidebar.selectbox("Model", OPENAI_MODELS,
                                      index=max(0, OPENAI_MODELS.index(current["model_name"]) if current["model_name"] in OPENAI_MODELS else 0))

    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, float(current["temperature"]), 0.05,
                                    help="Higher = more exploratory")
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, float(current["top_p"]), 0.01,
                              help="Nucleus sampling cap; 1.0 disables top-p")
    max_tokens = st.sidebar.number_input("Max output tokens", min_value=64, max_value=65536, value=int(current["max_tokens"]), step=64,
                                         help="For Responses API this maps to max_output_tokens; for Chat Completions to max_tokens.")

    # Conditionally show reasoning effort only for compatible models
    reasoning_effort = current.get("reasoning_effort", "medium")
    if not model_name.lower().startswith("gpt-4o"):
        reasoning_effort = st.sidebar.selectbox("Reasoning effort", ["low", "medium", "high"],
                                                index=["low", "medium", "high"].index(reasoning_effort),
                                                help="Used by reasoning models (GPT-5, o3/o4-mini, 4.1 family).")
    else:
        # For non-compatible models, we can set a default or None
        reasoning_effort = "medium" # Default for gpt-4o family if needed


    st.sidebar.markdown("### System prompt")
    system_prompt = st.sidebar.text_area("Edit system prompt", value=current["system_prompt"], height=200)

    save_with_session = st.sidebar.toggle("Save these settings inside session files", value=current.get("save_settings_with_session", True))

    current.update({
        "model_name": model_name,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "reasoning_effort": reasoning_effort,
        "system_prompt": system_prompt,
        "save_settings_with_session": bool(save_with_session),
    })
    st.session_state.dxo_settings = current

    with st.sidebar.expander("Show current settings JSON", expanded=False):
        st.json(current)

    return dict(current)