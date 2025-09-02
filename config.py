"""
Centralized configuration for the MAI-DxO application.

This file contains default settings for models, prompts, costs, and UI controls.
These values can be overridden by environment variables or user interactions
where applicable.
"""

# Default system prompt used by the Consensus agent.
DEFAULT_PROMPT = """You are a careful, stepwise diagnostic assistant. Propose next actions, justify briefly, and update the differential as new data arrives."""

# List of available OpenAI models for the UI dropdown.
OPENAI_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "o3",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

# Default costs for common diagnostic tests.
DEFAULT_TEST_COSTS = {
    "default": 150,
    "cbc": 50,
    "complete blood count": 50,
    "chest x-ray": 200,
    "mri": 1500,
    "ct scan": 1200,
    "biopsy": 800,
    "immunohistochemistry": 400,
    "fish test": 500,
    "ultrasound": 300,
    "ecg": 100,
    "blood glucose": 30,
}

# Default settings for the Streamlit UI controls in the sidebar.
DEFAULT_UI_SETTINGS = {
    "model_name": OPENAI_MODELS[0],
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 2048,
    "reasoning_effort": "medium",  # low | medium | high
    "system_prompt": DEFAULT_PROMPT,
    "save_settings_with_session": True,
}