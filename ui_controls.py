from __future__ import annotations
import streamlit as st
from config import OPENAI_MODELS, DEFAULT_UI_SETTINGS

def _init_session() -> None:
    """Initialize session state with default settings if not already present."""
    if "dxo_settings" not in st.session_state:
        st.session_state.dxo_settings = DEFAULT_UI_SETTINGS.copy()

def get_settings() -> dict:
    """Get current settings from session state."""
    _init_session()
    return dict(st.session_state.dxo_settings)

def _model_supports_reasoning(model: str) -> bool:
    """Check if a model supports the reasoning_effort parameter."""
    m = model.lower()
    # GPT-4o family doesn't support reasoning_effort
    if m.startswith("gpt-4o") or m == "gpt-4" or m.startswith("gpt-3"):
        return False
    # All other modern models support reasoning
    return True

def render_settings_panel() -> dict:
    """
    Render the settings panel in the Streamlit sidebar.
    Returns the current settings dictionary.
    """
    st.sidebar.markdown("## Settings (OpenAI)")

    _init_session()
    current = st.session_state.dxo_settings

    # Model selection dropdown
    model_name = st.sidebar.selectbox(
        "Model", 
        OPENAI_MODELS,
        index=max(0, OPENAI_MODELS.index(current["model_name"]) if current["model_name"] in OPENAI_MODELS else 0),
        help="Select the OpenAI model. GPT-5 and o-series models support advanced reasoning."
    )

    # Temperature slider - controls randomness
    temperature = st.sidebar.slider(
        "Temperature", 
        0.0, 
        2.0, 
        float(current["temperature"]), 
        0.05,
        help="Higher values make the output more creative but less predictable. For medical diagnosis, lower values (0.3-0.7) are recommended."
    )
    
    # Top-p slider - nucleus sampling
    top_p = st.sidebar.slider(
        "Top-p", 
        0.0, 
        1.0, 
        float(current["top_p"]), 
        0.01,
        help="Controls diversity via nucleus sampling. 1.0 considers all options. Lower values focus on more likely tokens."
    )
    
    # Max tokens - output length limit
    # GPT-5 supports up to 128,000 output tokens
    max_limit = 128000 if model_name.startswith("gpt-5") else 32768
    max_tokens = st.sidebar.number_input(
        "Max output tokens", 
        min_value=64, 
        max_value=max_limit,
        value=min(int(current["max_tokens"]), max_limit), 
        step=64,
        help=f"Maximum tokens in response. {model_name} supports up to {max_limit:,} tokens."
    )

    # Reasoning effort - only show for compatible models
    reasoning_effort = current.get("reasoning_effort", "medium")
    if _model_supports_reasoning(model_name):
        reasoning_effort = st.sidebar.selectbox(
            "Reasoning effort", 
            ["minimal", "low", "medium", "high"],
            index=["minimal", "low", "medium", "high"].index(reasoning_effort),
            help="""
            Controls thinking depth for reasoning models:
            • minimal: Fastest response, minimal reasoning
            • low: Quick thinking for simple problems
            • medium: Balanced speed and accuracy (recommended)
            • high: Deep analysis for complex diagnoses
            """
        )
        
        # Show estimated thinking time
        thinking_times = {
            "minimal": "~0-1 seconds",
            "low": "~2-5 seconds",
            "medium": "~5-15 seconds",
            "high": "~15-60 seconds"
        }
        st.sidebar.info(f"⏱️ Estimated thinking time: {thinking_times[reasoning_effort]}")
    else:
        st.sidebar.info(f"ℹ️ {model_name} uses traditional generation without explicit reasoning steps")
    
    # System prompt configuration
    st.sidebar.markdown("### System prompt")
    system_prompt = st.sidebar.text_area(
        "Edit system prompt", 
        value=current["system_prompt"], 
        height=200,
        help="Instructions that guide the AI's behavior throughout the diagnostic session"
    )

    # Option to save settings with session
    save_with_session = st.sidebar.toggle(
        "Save these settings inside session files", 
        value=current.get("save_settings_with_session", True),
        help="When enabled, your model and parameter choices will be saved with the diagnostic session"
    )

    # Update the session state with current values
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

    # Model capabilities indicator
    with st.sidebar.expander("Model Capabilities", expanded=False):
        if model_name.startswith("gpt-5"):
            st.success("✅ Unified model with automatic routing")
            st.info("GPT-5 automatically decides when to think deeply vs respond quickly")
        elif model_name.startswith("o"):
            st.success("✅ Pure reasoning model")
            st.info("Explicit chain-of-thought for complex problems")
        elif model_name.startswith("gpt-4.1"):
            st.success("✅ Enhanced instruction following")
            st.info("1M token context window for extensive medical histories")
        else:
            st.info("Traditional generation model")

    # Debug view of current settings
    with st.sidebar.expander("Show current settings JSON", expanded=False):
        st.json(current)

    return dict(current)