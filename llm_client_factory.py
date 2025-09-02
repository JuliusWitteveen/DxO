from __future__ import annotations
from typing import Any, Dict, List, Tuple
from openai import OpenAI

def _use_responses_api(model: str) -> bool:
    """
    Use Responses API for GPT-5, gpt-4.1*, and o-series. Use Chat Completions for gpt-4o*.
    """
    m = (model or "").lower()
    if m.startswith("gpt-4o"):
        return False
    # gpt-5*, gpt-4.1*, o3, o4-mini, etc.
    return True

class LLMClient:
    """
    Normalized generate() for OpenAI models.
      - Responses API for GPT-5 / 4.1 / o-series (supports reasoning.effort, max_output_tokens).
      - Chat Completions for 4o family.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: str | None = "medium",
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI()
        self._use_responses = _use_responses_api(model)

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str | None = None,
    ) -> Tuple[str, Any]:
        """
        messages: list of {"role": "user"|"assistant", "content": str}
        returns: (text, raw_response)
        """
        return (self._gen_responses(messages, system_prompt)
                if self._use_responses else
                self._gen_chat(messages, system_prompt))

    # --- Responses API path (GPT-5 / 4.1 / o-series)
    def _gen_responses(
        self, messages: List[Dict[str, str]], system_prompt: str | None
    ) -> Tuple[str, Any]:
        # Map our chat transcript to Responses 'input' items.
        input_items = []
        if messages:
            for m in messages:
                role = m.get("role", "user")
                text = m.get("content", "")
                input_items.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": text}]
                })

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items if input_items else " ",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        rsp = self._client.responses.create(**kwargs)
        # Unified text extraction
        text = getattr(rsp, "output_text", None)
        if not text:
            # Fallback to first text block if present
            try:
                text = rsp.output[0].content[0].text
            except Exception:
                text = ""
        return text, rsp

    # --- Chat Completions path (4o family)
    def _gen_chat(
        self, messages: List[Dict[str, str]], system_prompt: str | None
    ) -> Tuple[str, Any]:
        chat_msgs = []
        if system_prompt:
            chat_msgs.append({"role": "system", "content": system_prompt})
        chat_msgs.extend(messages or [{"role": "user", "content": ""}])

        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=chat_msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        text = rsp.choices[0].message.content
        return text, rsp
