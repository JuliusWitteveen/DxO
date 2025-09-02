"""
Centralized configuration for the MAI-DxO application.

This file contains default settings for models, prompts, costs, and UI controls.
These values can be overridden by environment variables or user interactions
where applicable.
"""

# Default system prompt used by the Consensus agent.
DEFAULT_PROMPT = """You are a careful, stepwise diagnostic assistant. Propose next actions, justify briefly, and update the differential as new data arrives."""

# List of available OpenAI models for the UI dropdown.
# Updated with all models released through August 2025
OPENAI_MODELS = [
    # GPT-5 unified models (August 2025) - combines speed with reasoning
    "gpt-5",           # Most powerful unified model with automatic routing
    "gpt-5-mini",      # Smaller, faster GPT-5 variant
    "gpt-5-nano",      # Ultra-lightweight GPT-5 for simple tasks
    
    # O-series reasoning models - explicit chain-of-thought
    "o3",              # Advanced reasoning model
    "o3-mini",         # Lighter o3 variant (January 2025 update)
    "o4-mini",         # Compact reasoning model
    "o1-preview",      # Preview reasoning model
    "o1-mini",         # Lightweight reasoning variant
    
    # GPT-4.1 series - enhanced capabilities
    "gpt-4.1",         # 1M token context, enhanced instruction following
    "gpt-4.1-mini",    # Smaller GPT-4.1 variant
    
    # GPT-4o family - optimized variants
    "gpt-4o",          # Optimized GPT-4 variant
    "gpt-4o-mini",     # Smaller GPT-4o version
    
    # Legacy models (still supported)
    "gpt-4-turbo",     # Fast GPT-4 variant
    "gpt-4",           # Original GPT-4
    "gpt-3.5-turbo",   # Cost-effective model
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
    "model_name": OPENAI_MODELS[0],  # Default to GPT-5
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 2048,
    "reasoning_effort": "medium",  # Critical for GPT-5 and o-series models
    "system_prompt": DEFAULT_PROMPT,
    "save_settings_with_session": True,
}