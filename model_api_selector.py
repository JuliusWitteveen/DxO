# model_api_selector.py
# Complete, self-contained utility to decide which OpenAI API family to use
# for a given model string. Keep this file at repo root.

from __future__ import annotations

def use_responses_api(model: str) -> bool:
    """
    Return True if the given model should be called via the *Responses* API.
    Otherwise, return False to indicate *Chat Completions*.

    Heuristics (conservative):
      - Chat Completions: gpt-3.*, gpt-4-*, gpt-4o*
      - Responses API   : everything else (o-series, gpt-4.1*, gpt-5*, o3/o4, etc.)
    """
    if not model:
        raise ValueError("model name must be provided")
    m = model.lower().strip()

    if m.startswith("gpt-3") or m.startswith("gpt-4-") or m.startswith("gpt-4o"):
        return False
    return True

__all__ = ["use_responses_api"]
