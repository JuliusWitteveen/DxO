"""Wrapper around litellm.completion with Responses API awareness.

This module provides a thin wrapper that builds the parameters dictionary
for `litellm.completion` while omitting unsupported options for models that use
OpenAI's newer Responses API. In particular, `top_p` is not supported by these
models and will be removed from the request.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    import litellm  # type: ignore
except Exception:  # pragma: no cover
    litellm = None  # type: ignore

from llm_client_factory import _use_responses_api

def completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call :func:`litellm.completion` with sensible defaults.

    ``top_p`` is only included for Chat Completions models. Responses API
    models (e.g., ``gpt-5``, ``o`` series, ``gpt-4.1``) do not support it.
    """
    if litellm is None:  # pragma: no cover - runtime check
        raise RuntimeError("litellm library is required for this function")

    completion_params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if not _use_responses_api(model):
        completion_params["top_p"] = top_p
    # Responses API models omit top_p entirely as it is unsupported.

    completion_params.update(kwargs)
    return litellm.completion(**completion_params)
