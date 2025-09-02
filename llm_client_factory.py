from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import sys

def _use_responses_api(model: str) -> bool:
    """
    Determine which API to use based on model name.
    
    Responses API is used for:
    - GPT-5 family (unified models with reasoning capabilities)
    - O-series models (o1, o3, o4 reasoning models)
    - GPT-4.1 family (enhanced instruction following)
    
    Chat Completions API is used for:
    - GPT-4o family (optimized but traditional models)
    - Legacy GPT-4 and GPT-3.5 models
    """
    m = (model or "").lower()
    
    # GPT-4o family uses traditional Chat Completions
    if m.startswith("gpt-4o") or m.startswith("gpt-4-") or m.startswith("gpt-3"):
        return False
    
    # All newer models use Responses API with reasoning support
    # This includes GPT-5, o-series, and GPT-4.1 family
    return True

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
        input_items = []
        if messages:
            for m in messages:
                role = m.get("role", "user")
                text = m.get("content", "")
                # Responses API uses a different message structure
                input_items.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": text}]
                })

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
            rsp = self._client.responses.create(**kwargs)
            
            # Extract text from response
            text = getattr(rsp, "output_text", None)
            if not text:
                # Try alternative response structure
                try:
                    text = rsp.output[0].content[0].text
                except Exception:
                    text = str(rsp) if rsp else ""
                    
            return text, rsp
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

        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=chat_msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        
        text = rsp.choices[0].message.content
        return text, rsp