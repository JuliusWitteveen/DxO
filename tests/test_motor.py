import importlib
import os
import sys
import types
import time
import random

import pytest

# Ensure repository root on path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mai_dx.structures import AgentRole, Action


@pytest.fixture
def main_module(monkeypatch):
    """Import mai_dx.main with stubbed dependencies."""
    swarms_stub = types.ModuleType("swarms")

    class DummyAgent:
        def __init__(self, *_, agent_name=""):
            self.agent_name = agent_name

        def run(self, prompt):  # pragma: no cover - basic stub
            return prompt

    class DummyConversation:
        def __init__(self, *_, **__):
            self.history = []

        def add(self, speaker, message):
            self.history.append((speaker, message))

        def get_str(self):
            return "\n".join(f"{s}: {m}" for s, m in self.history)

    swarms_stub.Agent = DummyAgent
    swarms_stub.Conversation = DummyConversation
    monkeypatch.setitem(sys.modules, "swarms", swarms_stub)

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    main = importlib.import_module("mai_dx.main")
    importlib.reload(main)
    return main


@pytest.fixture
def orchestrator(main_module):
    orch = main_module.MaiDxOrchestrator.__new__(main_module.MaiDxOrchestrator)
    orch.request_delay = 0
    orch.max_iterations = 3
    orch.initial_budget = 10000
    orch.physician_visit_cost = 0
    orch.mode = "no_budget"
    orch.test_cost_db = {}
    orch.prompt_overrides = {}
    orch.agents = {}
    return orch


class TestSafeAgentRun:
    def test_safe_agent_run_success(self, orchestrator, main_module):
        class SuccessAgent:
            agent_name = "success"

            def run(self, prompt):
                return "ok"

        agent = SuccessAgent()
        result = orchestrator._safe_agent_run(agent, "hi")
        assert result.success and result.data == "ok"

    def test_safe_agent_run_failure(self, orchestrator, main_module):
        class FailAgent:
            agent_name = "fail"

            def run(self, prompt):
                raise RuntimeError("boom")

        agent = FailAgent()
        result = orchestrator._safe_agent_run(agent, "hi", timeout=0.05, retries=0)
        assert not result.success and "boom" in result.error

    def test_safe_agent_run_retries_on_timeout(self, orchestrator, main_module, monkeypatch):
        calls = {"count": 0}

        class SlowAgent:
            agent_name = "slow"

            def run(self, prompt):
                calls["count"] += 1
                if calls["count"] == 1:
                    time.sleep(0.1)  # exceed timeout
                return "ok"

        agent = SlowAgent()
        monkeypatch.setattr(random, "uniform", lambda a, b: 0)
        result = orchestrator._safe_agent_run(agent, "hi", timeout=0.05, retries=1)
        assert result.success and calls["count"] == 2

    def test_safe_agent_run_timeout_failure(self, orchestrator, main_module, monkeypatch):
        class HangingAgent:
            agent_name = "hang"

            def run(self, prompt):
                time.sleep(0.1)

        agent = HangingAgent()
        monkeypatch.setattr(random, "uniform", lambda a, b: 0)
        result = orchestrator._safe_agent_run(agent, "hi", timeout=0.05, retries=1)
        assert not result.success and "timeout" in result.error.lower()


class TestValidateAndCorrectAction:
    def test_question_only_converts_test_to_question(self, orchestrator, main_module):
        orchestrator.mode = "question_only"
        action = Action(action_type="test", content="CBC", reasoning="Check")
        case_state = main_module.CaseState(initial_vignette="v")
        corrected = orchestrator._validate_and_correct_action(action, case_state, 100)
        assert corrected.action_type == "ask"
        assert "Test ordering disabled" in corrected.content

    def test_budget_exhaustion_forces_diagnosis(self, orchestrator, main_module):
        orchestrator.mode = "budgeted"
        case_state = main_module.CaseState(initial_vignette="v", differential_diagnosis={"flu": 0.9})
        action = Action(action_type="test", content="X-ray", reasoning="why not")
        corrected = orchestrator._validate_and_correct_action(action, case_state, 0)
        assert corrected.action_type == "diagnose"
        assert corrected.content == "flu"
        assert corrected.reasoning == "Forced diagnosis due to budget exhaustion."


class TestPerformTurn:
    def test_retry_after_hypothesis_failure(self, orchestrator, main_module, monkeypatch):
        for role in [AgentRole.HYPOTHESIS, AgentRole.CONSENSUS, AgentRole.TEST_CHOOSER, AgentRole.CHALLENGER, AgentRole.STEWARDSHIP, AgentRole.CHECKLIST]:
            orchestrator.agents[role] = main_module.Agent(agent_name=role.value)

        calls = {"hypo": 0}

        def fake_run(self, agent, prompt):
            if agent.agent_name == AgentRole.HYPOTHESIS.value:
                if calls["hypo"] == 0:
                    calls["hypo"] += 1
                    return main_module.AgentResult(success=False, error="boom")
                return main_module.AgentResult(success=True, data="{}")
            if agent.agent_name == AgentRole.CONSENSUS.value:
                return main_module.AgentResult(
                    success=True,
                    data={"action_type": "diagnose", "content": "flu", "reasoning": "done"},
                )
            return main_module.AgentResult(success=True, data="ok")

        monkeypatch.setattr(orchestrator, "_safe_agent_run", types.MethodType(fake_run, orchestrator))

        case_state = main_module.CaseState(initial_vignette="v")
        action1, _ = orchestrator._perform_turn(case_state)
        assert action1.action_type == "ask"  # fallback due to failure

        action2, _ = orchestrator._perform_turn(case_state)
        assert action2.action_type == "diagnose" and action2.content == "flu"


class TestRun:
    def _configure(self, orchestrator, main_module, monkeypatch):
        for role in AgentRole:
            orchestrator.agents[role] = main_module.Agent(agent_name=role.value)

        responses = {
            AgentRole.HYPOTHESIS.value: '{"differential": [{"diagnosis": "flu", "probability": 1.0}]}',
            AgentRole.TEST_CHOOSER.value: "test",
            AgentRole.CHALLENGER.value: "none",
            AgentRole.STEWARDSHIP.value: "cost",
            AgentRole.CHECKLIST.value: "ok",
            AgentRole.CONSENSUS.value: {"action_type": "diagnose", "content": "flu", "reasoning": "all"},
            AgentRole.GATEKEEPER.value: "ack",
            AgentRole.JUDGE.value: {"score": 5, "reasoning": "correct"},
        }

        def fake_run(self, agent, prompt):
            return main_module.AgentResult(success=True, data=responses[agent.agent_name])

        monkeypatch.setattr(orchestrator, "_safe_agent_run", types.MethodType(fake_run, orchestrator))

    def test_run_happy_path(self, orchestrator, main_module, monkeypatch):
        self._configure(orchestrator, main_module, monkeypatch)
        result = orchestrator.run("fever", "details", "flu")
        assert result.final_diagnosis == "flu"
        assert result.iterations == 1

    def test_run_is_idempotent(self, orchestrator, main_module, monkeypatch):
        self._configure(orchestrator, main_module, monkeypatch)
        r1 = orchestrator.run("fever", "details", "flu")
        r2 = orchestrator.run("fever", "details", "flu")
        assert r1 == r2
