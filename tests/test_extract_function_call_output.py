import importlib
import os
import sys
import types


import pytest


@pytest.fixture
def main_module(monkeypatch):
    """Import mai_dx.main with stubbed dependencies."""

    swarms_stub = types.ModuleType("swarms")

    class DummyAgent:  # pragma: no cover - minimal stub
        pass

    swarms_stub.Agent = DummyAgent
    swarms_stub.Conversation = DummyAgent
    monkeypatch.setitem(sys.modules, "swarms", swarms_stub)

    dotenv_stub = types.ModuleType("dotenv")

    def fake_load_dotenv(*args, **kwargs):  # pragma: no cover - minimal stub
        return None

    dotenv_stub.load_dotenv = fake_load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    main = importlib.import_module("mai_dx.main")
    importlib.reload(main)
    return main


def test_list_response_parsed(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    response = [
        {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}
    ]
    result = orchestrator._extract_function_call_output(response)
    assert result == response[0]


def test_single_quoted_string_parsed(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    response = (
        "{'action_type': 'ask', 'content': 'Age?', 'reasoning': 'Need age'}"
    )
    expected = {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}
    result = orchestrator._extract_function_call_output(response)
    assert result == expected


def test_function_dict_parsed(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    response = {
        "function": {
            "arguments": (
                '{"action_type": "ask", "content": "Age?", "reasoning": "Need age"}'
            )
        }
    }
    expected = {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}
    result = orchestrator._extract_function_call_output(response)
    assert result == expected


def test_missing_reasoning_returns_none(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    response = {"action_type": "ask", "content": "Age?"}
    result = orchestrator._extract_function_call_output(response)
    assert result is None


def test_trailing_comma_in_json_is_handled(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    # Note the trailing comma after "Need age", which is invalid in strict JSON
    response = (
        'Some text before... {"action_type": "ask", "content": "Age?", "reasoning": "Need age",} ... and after'
    )
    expected = {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}
    result = orchestrator._extract_function_call_output(response)
    assert result == expected


def test_unparseable_string_returns_none(main_module):
    orchestrator = main_module.MaiDxOrchestrator.__new__(
        main_module.MaiDxOrchestrator
    )
    response = "This is just a regular string without any dict or json."
    result = orchestrator._extract_function_call_output(response)
    assert result is None