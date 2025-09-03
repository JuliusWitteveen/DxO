from __future__ import annotations
from typing import Any, Dict

from model_api_selector import use_responses_api as _use_responses_api

try:
    import swarms
except ImportError:
    swarms = None


class MAI_LLM:
    """
    A wrapper for the swarms.LLM that correctly handles model-specific
    parameters like top_p before making the final API call. This is the
    definitive fix for the recurring "Unsupported parameter: 'top_p'" error.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the wrapper and the underlying swarms.LLM.

        Args:
            **kwargs: Arguments to be passed to the swarms.LLM constructor.
        """
        self.model_name = kwargs.get("model_name", "")
        
        # Create a clean set of kwargs for the underlying LLM,
        # as the swarms.LLM might not expect all our custom params.
        llm_kwargs = kwargs.copy()

        if swarms:
            self._llm = swarms.LLM(**llm_kwargs)
        else:
            raise RuntimeError("The 'swarms' library is not installed.")

    def run(self, **kwargs: Any) -> Any:
        """
        Executes the LLM call, but first removes unsupported parameters.

        This method intercepts the arguments, removes 'top_p' if the model
        doesn't support it, and then passes the cleaned arguments to the
        underlying swarms.LLM instance.
        """
        
        # Make a copy to modify
        run_kwargs = kwargs.copy()

        # The critical fix: Before making the API call, check if the model
        # uses the Responses API and, if so, remove the 'top_p' parameter
        # to prevent the BadRequestError from litellm/OpenAI.
        if _use_responses_api(self.model_name):
            run_kwargs.pop("top_p", None)

        return self._llm.run(**run_kwargs)


class LLMClient:
    """
    Unified client for OpenAI models supporting both API paradigms.
    This client is used by the UI but NOT by the core orchestrator.
    The orchestrator now uses the MAI_LLM wrapper.
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