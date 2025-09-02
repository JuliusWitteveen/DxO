"""
General utility functions for the MAI-DxO project.
"""
import json
import re
import ast
from typing import Any, Optional, Dict


def resilient_parser(malformed_string: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse a malformed JSON or Python literal string into a dictionary.

    This function tries to fix common LLM output errors, such as trailing commas,
    and uses multiple parsing strategies (json, ast) to maximize robustness.

    Args:
        malformed_string: The potentially broken string to parse.

    Returns:
        A dictionary if parsing is successful, otherwise None.
    """
    # Remove trailing commas that are invalid in JSON
    # e.g., '{"key": "value",}' -> '{"key": "value"}'
    fixed_string = re.sub(r",\s*([}\]])", r"\1", malformed_string)

    try:
        # First, try the standard, strict JSON parser
        return json.loads(fixed_string)
    except json.JSONDecodeError:
        try:
            # If that fails, try the more lenient Python literal evaluator
            return ast.literal_eval(fixed_string)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            # All parsing attempts have failed
            return None