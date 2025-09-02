"""Utilities for selecting the appropriate OpenAI API."""

from __future__ import annotations


def use_responses_api(model: str) -> bool:
    """Return ``True`` if the given model uses the Responses API.

    Args:
        model: The model identifier.

    The determination is based on simple prefix checks. Newer model families
    such as GPT-5, the o-series, and GPT-4.1 require the Responses API, while
    legacy GPT-4o and GPT-3 models continue to use Chat Completions.
    """

    assert model, "model name must be provided"

    model_lower = model.lower()

    # GPT-4o family uses traditional Chat Completions
    if (
        model_lower.startswith("gpt-4o")
        or model_lower.startswith("gpt-4-")
        or model_lower.startswith("gpt-3")
    ):
        return False

    # All newer models use Responses API with reasoning support
    return True


__all__ = ["use_responses_api"]

