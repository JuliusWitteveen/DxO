from __future__ import annotations
from typing import Any, Dict, List, Tuple

from model_api_selector import use_responses_api as _use_responses_api


def _to_responses_format(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert chat-style messages to the Responses API format."""

    formatted: List[Dict[str, Any]] = []
    if messages:
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            formatted.append({
                "role": role,
                "content": [{"type": "input_text", "text": text}],
            })
    return formatted

class LLMClient:
    """
    Unified client for OpenAI models supporting both API paradigms.
    
    - Responses API: For GPT-5, o-series, and GPT-4.1 (supports reasoning effort)
    - Chat Completions API: For GPT-4o family and legacy models
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
        
        # Initialize OpenAI client with error handling
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
            print("Please ensure OPENAI_API_KEY is set in your environment or .env file")
            self._client = None
            
        self._use_responses = _use_responses_api(model)
        
        if self._use_responses:
            print(f"Using Responses API for {model} with reasoning_effort={reasoning_effort}")
        else:
            print(f"Using Chat Completions API for {model}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str | None = None,
    ) -> Tuple[str, Any]:
        """
        Generate a response using the appropriate API for the model.
        
        messages: list of {"role": "user"|"assistant", "content": str}
        returns: (text, raw_response)
        """
        if not self._client:
            return "Error: OpenAI client not initialized. Please check your API key.", None
            
        try:
            if self._use_responses:
                return self._gen_responses(messages, system_prompt)
            else:
                return self._gen_chat(messages, system_prompt)
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            print(f"Error in LLMClient.generate: {error_msg}")
            return error_msg, None

    def _gen_responses(
        self, messages: List[Dict[str, str]], system_prompt: str | None
    ) -> Tuple[str, Any]:
        """
        Use the Responses API for models with reasoning capabilities.
        This includes GPT-5, o-series, and GPT-4.1 models.
        """
        # Convert chat messages to Responses API format
        input_items = _to_responses_format(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items if input_items else " ",
            "temperature": self.temperature,
            # top_p is NOT supported by the Responses API, so it is omitted.
            "max_output_tokens": self.max_tokens,  # Note: different parameter name
        }
        
        if system_prompt:
            kwargs["instructions"] = system_prompt
            
        # Add reasoning configuration for supported models
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = self._client.responses.create(**kwargs)
            assert response is not None, "Responses API returned no data"

            text = getattr(response, "output_text", None)
            if not text:
                # Try alternative response structure
                try:
                    text = response.output[0].content[0].text
                except Exception:
                    text = str(response) if response else ""

            return text, response
        except AttributeError:
            # Fallback if Responses API is not available
            print(f"Warning: Responses API not available for {self.model}, falling back to Chat Completions")
            return self._gen_chat(messages, system_prompt)

    def _gen_chat(
        self, messages: List[Dict[str, str]], system_prompt: str | None
    ) -> Tuple[str, Any]:
        """
        Use the traditional Chat Completions API.
        This is for GPT-4o family and legacy models.
        """
        chat_msgs = []
        if system_prompt:
            chat_msgs.append({"role": "system", "content": system_prompt})
        chat_msgs.extend(messages or [{"role": "user", "content": ""}])

        response = self._client.chat.completions.create(
            model=self.model,
            messages=chat_msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        assert response.choices, "Chat API returned no choices"
        text = response.choices[0].message.content
        return text, response
