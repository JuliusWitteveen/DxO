import importlib
import os
import sys
import types

import pytest


@pytest.fixture
def main_module(monkeypatch):
    """Import mai_dx.main with minimal dependency stubs."""

    swarms_stub = types.ModuleType("swarms")

    class DummyAgent:  # pragma: no cover - stub
        def __init__(self):
            self.agent_name = "dummy"

        def run(self, prompt):  # pragma: no cover - stub
            return "dummy"

    swarms_stub.Agent = DummyAgent
    swarms_stub.Conversation = DummyAgent
    monkeypatch.setitem(sys.modules, "swarms", swarms_stub)

    dotenv_stub = types.ModuleType("dotenv")

    def fake_load_dotenv(*args, **kwargs):  # pragma: no cover - stub
        return None

    dotenv_stub.load_dotenv = fake_load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    main = importlib.import_module("mai_dx.main")
    importlib.reload(main)
    return main


def test_interact_with_gatekeeper_uses_enum(monkeypatch, main_module):
    dummy_agent = main_module.Agent()

    def fake_init_agents(self):
        self.agents = {main_module.AgentRole.GATEKEEPER: dummy_agent}

    monkeypatch.setattr(main_module.MaiDxOrchestrator, "_init_agents", fake_init_agents)

    orchestrator = main_module.MaiDxOrchestrator(model_name="test", request_delay=0)

    captured = {}

    def fake_safe_agent_run(self, agent, prompt):
        captured['agent'] = agent
        return main_module.AgentResult(success=True, data="ok")

    monkeypatch.setattr(main_module.MaiDxOrchestrator, "_safe_agent_run", fake_safe_agent_run)

    action = main_module.Action(action_type="ask", content="question", reasoning="because")
    response = orchestrator._interact_with_gatekeeper(action, "details")

    assert captured['agent'] is dummy_agent
    assert response == "ok"
