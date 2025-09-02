"""MAI Diagnostic Orchestrator (MAI-DxO) core logic.

This module implements the core of the "Sequential Diagnosis with Language
Models" framework using the `swarms` library. It simulates a panel of
physician-agents to perform iterative medical diagnosis with
cost-effectiveness optimization. The orchestrator supports both autonomous and
interactive (turn-based) modes.

Based on the paper "Sequential Diagnosis with Language Models"
(arXiv:2306.022405v1) by Nori et al.
"""

import os
import json
import sys
import time
import re
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from mai_dx.structures import AgentRole, Action
from mai_dx.prompts import get_prompt_for_role
from mai_dx.costing import estimate_cost
from mai_dx.utils import resilient_parser

# --- Dependency Management ---
# Dependencies are listed in requirements.txt and can be installed using
# `pip install -r requirements.txt` or the `scripts/install_dependencies.py` script.
try:
    import swarms
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError(
        "Required dependencies for MAI-DxO are missing."
        " Please install them with 'pip install -r requirements.txt' or run"
        " 'python scripts/install_dependencies.py'."
    ) from e

missing_attrs = [attr for attr in ("Agent", "Conversation") if not hasattr(swarms, attr)]
if missing_attrs:
    raise ImportError(
        "The 'swarms' package is missing required attributes: "
        f"{', '.join(missing_attrs)}. This may occur if a local package named 'swarms'"
        " is shadowing the intended external dependency. Please rename or remove any"
        " local directories named 'swarms'."
    )

Agent = swarms.Agent
Conversation = swarms.Conversation

load_dotenv()

# --- Logging Configuration ---
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

if os.getenv("MAIDX_DEBUG", "0").lower() in ("1", "true", "yes"):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger.add(
        "logs/maidx_debug_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="3 days",
    )
    logger.info(
        "🛠 Debug logging enabled - logs will be written to logs/ directory"
    )


# --- Data Structures and Enums ---
class DifferentialEntry(BaseModel):
    """Structured representation of a single diagnosis probability."""

    diagnosis: str
    probability: float = Field(..., ge=0.0, le=1.0)


class DifferentialSchema(BaseModel):
    """Schema for a differential diagnosis list."""

    differential: List[DifferentialEntry]


@dataclass
class AgentResult:
    """Standardized result from agent executions."""

    success: bool
    data: Any = None
    error: Optional[str] = None


@dataclass
class CaseState:
    """Structured state management for the diagnostic process.

    This dataclass represents the single source of truth for the *current*
    state of an ongoing diagnostic session. It is mutable and updated by the
    orchestrator during each turn. It should be used by all agents to inform
    their decisions.
    """

    initial_vignette: str
    evidence_log: List[str] = field(default_factory=list)
    differential_diagnosis: Dict[str, float] = field(default_factory=dict)
    tests_performed: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    cumulative_cost: int = 0
    iteration: int = 0
    last_actions: List[Action] = field(default_factory=list)

    def add_evidence(self, evidence: str):
        """Add a new piece of evidence to the log."""
        self.evidence_log.append(f"[Turn {self.iteration}] {evidence}")

    def update_differential(self, diagnosis_dict: Dict[str, float]):
        """Update the differential diagnosis with new probabilities."""
        self.differential_diagnosis.update(diagnosis_dict)

    def add_test(self, test_name: str, cost: int):
        """Record a performed test and its associated cost."""
        self.tests_performed.append(test_name)
        self.cumulative_cost += cost

    def add_question(self, question: str):
        """Record a question posed during the diagnostic process."""
        self.questions_asked.append(question)

    def add_action(self, action: Action):
        """Track the latest action, keeping only the most recent three."""
        self.last_actions.append(action)
        if len(self.last_actions) > 3:
            self.last_actions.pop(0)

    def is_stagnating(self) -> bool:
        """Return True if the two most recent actions are identical."""
        if len(self.last_actions) < 2:
            return False
        return self.last_actions[-1] == self.last_actions[-2]

    def get_max_confidence(self) -> float:
        """Return the highest diagnosis probability."""
        if not self.differential_diagnosis:
            return 0.0
        return max(self.differential_diagnosis.values())

    def get_leading_diagnosis(self) -> str:
        """Return the diagnosis with the highest confidence."""
        if not self.differential_diagnosis:
            return "No diagnosis formulated"
        return max(self.differential_diagnosis.items(), key=lambda x: x[1])[0]

    def summarize_evidence(self) -> str:
        """Summarize the accumulated evidence for use in prompts."""
        if len(self.evidence_log) <= 5:
            return "\n".join(self.evidence_log)
        summary_parts = (
            self.evidence_log[:2]
            + [f"[... {len(self.evidence_log) - 5} findings ...]"]
            + self.evidence_log[-3:]
        )
        return "\n".join(summary_parts)


@dataclass
class DeliberationState:
    """Structured state for panel deliberation."""

    hypothesis_analysis: str = ""
    test_chooser_analysis: str = ""
    challenger_analysis: str = ""
    stewardship_analysis: str = ""
    checklist_analysis: str = ""
    situational_context: str = ""
    stagnation_detected: bool = False

    def to_consensus_prompt(self) -> str:
        """Build a prompt summarizing panel deliberations for consensus."""
        prompt = f"""
You are the Consensus Coordinator. Here is the panel's analysis:

**Differential Diagnosis (Dr. Hypothesis):**
{self.hypothesis_analysis or 'Not yet formulated'}

**Test Recommendations (Dr. Test-Chooser):**
{self.test_chooser_analysis or 'None provided'}

**Critical Challenges (Dr. Challenger):**
{self.challenger_analysis or 'None identified'}

**Cost Assessment (Dr. Stewardship):**
{self.stewardship_analysis or 'Not evaluated'}

**Quality Control (Dr. Checklist):**
{self.checklist_analysis or 'No issues noted'}
"""
        if self.stagnation_detected:
            prompt += "\n**STAGNATION DETECTED** - The panel is repeating actions. You MUST make a decisive choice or provide a final diagnosis."
        if self.situational_context:
            prompt += f"\n**Situational Context:** {self.situational_context}"
        prompt += "\n\nBased on this comprehensive panel input, use the make_consensus_decision function to provide your structured action."
        return prompt


@dataclass
class DiagnosisResult:
    """Stores the final result of a diagnostic session."""

    final_diagnosis: str
    ground_truth: str
    accuracy_score: float
    accuracy_reasoning: str
    total_cost: int
    iterations: int
    conversation_history: str


def _get_fallback_action(reason: str) -> Action:
    """Creates a standardized fallback action for error states."""
    return Action(
        action_type="ask",
        content="Could you clarify the next step? (System experienced a processing issue)",
        reasoning=f"Fallback Action: {reason}",
    )


class MaiDxOrchestrator:
    """Implement the MAI Diagnostic Orchestrator (MAI-DxO) framework.

    This class orchestrates a virtual panel of AI agents to perform sequential
    medical diagnosis.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",
        physician_visit_cost: int = 300,
        request_delay: float = 1.0,
        test_costs: Optional[Dict[str, int]] = None,
        prompt_overrides: Optional[Dict[str, str]] = None,
    ):
        """Initialize the orchestrator with model and runtime settings."""
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.request_delay = request_delay
        self.top_p = 1.0  # Default value, will be used by compatible models

        # Initialize the test cost database, allowing overrides for customization
        if test_costs is not None:
            self.test_cost_db = dict(test_costs)
        else:
            from config import DEFAULT_TEST_COSTS
            self.test_cost_db = DEFAULT_TEST_COSTS.copy()

        # Store optional system prompt overrides for agents
        self.prompt_overrides: Dict[AgentRole, str] = {}
        if prompt_overrides:
            for role_key, prompt in prompt_overrides.items():
                try:
                    role = AgentRole[role_key]
                except KeyError:
                    continue
                self.prompt_overrides[role] = prompt

        self._init_agents()
        logger.info(
            f"🥼 MAI-DxO initialized in '{mode}' mode with model '{self.model_name}'."
        )

    def _init_agents(self):
        """Initialize all required agents."""
        consensus_tool = {
            "type": "function",
            "function": {
                "name": "make_consensus_decision",
                "description": "Make a structured consensus decision for the next diagnostic action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["ask", "test", "diagnose"],
                        },
                        "content": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["action_type", "content", "reasoning"],
                },
            },
        }
        
        judge_tool = {
            "type": "function",
            "function": {
                "name": "evaluate_diagnosis",
                "description": "Evaluate the final diagnosis against the ground truth.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Accuracy score from 1 (Incorrect) to 5 (Perfect)",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Concise justification for the score",
                        },
                    },
                    "required": ["score", "reasoning"],
                },
            },
        }

        self.agents = {}
        failed_roles: List[str] = []
        for role in AgentRole:
            max_tokens = 600 if role != AgentRole.HYPOTHESIS else 1000
            agent_args = {
                "agent_name": role.value,
                "system_prompt": get_prompt_for_role(role, self.prompt_overrides),
                "model_name": self.model_name,
                "max_loops": 1,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }
            # Add top_p only for compatible models (legacy chat completions)
            m = self.model_name.lower()
            if m.startswith("gpt-4o") or m.startswith("gpt-4-") or m.startswith("gpt-3"):
                 agent_args["top_p"] = self.top_p

            if role == AgentRole.CONSENSUS:
                agent_args["tools_list_dictionary"] = [consensus_tool]
                agent_args["tool_choice"] = "auto"
            elif role == AgentRole.JUDGE:
                agent_args["tools_list_dictionary"] = [judge_tool]
                agent_args["tool_choice"] = "auto"

            try:
                self.agents[role] = Agent(**agent_args)
            except Exception as e:
                logger.error(f"Failed to initialize {role.value}: {e}")
                failed_roles.append(role.value)

        if failed_roles:
            raise RuntimeError(
                "Agent initialization failed for: "
                + ", ".join(failed_roles)
                + ". Check API keys and configuration."
            )

        logger.info(
            f"👥 {len(self.agents)} virtual physician agents initialized."
        )

    def update_runtime_params(
        self,
        model_name: Optional[str] = None,
        prompt_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update model or agent prompts at runtime.

        Args:
            model_name: New model name to use for all agents.
            prompt_overrides: Mapping of AgentRole names to new system prompts.

        Returns:
            None
        """
        need_reinit = False
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            need_reinit = True

        if prompt_overrides:
            for role_key, prompt in prompt_overrides.items():
                try:
                    role = AgentRole[role_key]
                except KeyError:
                    continue
                self.prompt_overrides[role] = prompt
            need_reinit = True

        if need_reinit:
            self._init_agents()

    def _safe_agent_run(self, agent: Agent, prompt: str) -> AgentResult:
        """Run an agent safely, returning a standardized result."""
        time.sleep(self.request_delay)
        try:
            data = agent.run(prompt)
            return AgentResult(success=True, data=data)
        except Exception as e:
            logger.error(f"Agent {agent.agent_name} run failed: {e}")
            return AgentResult(success=False, error=str(e))

    def _extract_function_call_output(
        self, agent_response: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Robustly extract tool-call arguments from various possible agent
        response structures.
        """
        if not agent_response:
            return None

        # Helper to validate if a dictionary has the required keys for an Action
        def is_action_dict(data: Any) -> bool:
            return isinstance(data, dict) and all(
                k in data for k in ("action_type", "content", "reasoning")
            )

        # Path 1: The response is already a dictionary that looks like an action
        if is_action_dict(agent_response):
            return agent_response

        # Path 2: The entire response is a string that might contain a JSON/dict
        if isinstance(agent_response, str):
            match = re.search(r'\{.*\}', agent_response, re.DOTALL)
            if match:
                parsed_string = resilient_parser(match.group(0))
                # Check for either Action or Judge tool format
                if parsed_string and (
                    is_action_dict(parsed_string) or
                    all(k in parsed_string for k in ("score", "reasoning"))
                ):
                    return parsed_string

        # Path 3: A general recursive search for a valid action dictionary as a fallback
        def find_action_recursive(data: Any) -> Optional[Dict[str, Any]]:
            # Check for either Action or Judge tool format
            if isinstance(data, dict) and (
                is_action_dict(data) or
                all(k in data for k in ("score", "reasoning"))
            ):
                return data
            
            # If the dict contains 'arguments', try parsing them
            if isinstance(data, dict):
                args = data.get("arguments")
                if isinstance(args, str):
                    parsed_args = resilient_parser(args)
                    if parsed_args and (
                        is_action_dict(parsed_args) or
                        all(k in parsed_args for k in ("score", "reasoning"))
                    ):
                        return parsed_args
                elif isinstance(args, dict) and (
                    is_action_dict(args) or
                    all(k in args for k in ("score", "reasoning"))
                ):
                    return args

            # Recurse through nested structures
            if isinstance(data, dict):
                for key, value in data.items():
                    found = find_action_recursive(value)
                    if found:
                        return found
            elif isinstance(data, list):
                for item in data:
                    found = find_action_recursive(item)
                    if found:
                        return found
            return None

        action_dict = find_action_recursive(agent_response)
        if action_dict:
            return action_dict

        logger.error(f"Could not find valid arguments in response: {agent_response}")
        return None


    def _build_base_context(
        self, case_state: CaseState, remaining_budget: int
    ) -> str:
        """Build the base case context shared with panel agents."""
        return f"""
        === CASE STATUS - ROUND {case_state.iteration} ===
        Initial Presentation: {case_state.initial_vignette}
        Evidence Gathered:
        {case_state.summarize_evidence()}
        ---
        Current State:
        - Cost: ${case_state.cumulative_cost}
        - Remaining Budget: ${remaining_budget}
        - Mode: {self.mode}
        - Current Differential: {case_state.differential_diagnosis}
        """

    def _collect_panel_deliberation(
        self, base_context: str, case_state: CaseState
    ) -> DeliberationState:
        """Gather analyses from individual panel agents, running them in parallel."""
        deliberation_state = DeliberationState()
        unavailable_msg = "Analysis not available due to a technical error."

        # Dr. Hypothesis must run first as its output informs other agents.
        hypo_result = self._safe_agent_run(
            self.agents[AgentRole.HYPOTHESIS], base_context
        )
        if not hypo_result.success:
            raise RuntimeError(f"Critical agent Dr. Hypothesis failed: {hypo_result.error}")

        deliberation_state.hypothesis_analysis = hypo_result.data
        self._update_differential_from_text(
            case_state, deliberation_state.hypothesis_analysis
        )

        # These agents can run in parallel.
        parallel_agents = {
            AgentRole.TEST_CHOOSER: "test_chooser_analysis",
            AgentRole.CHALLENGER: "challenger_analysis",
            AgentRole.STEWARDSHIP: "stewardship_analysis",
            AgentRole.CHECKLIST: "checklist_analysis",
        }

        def run_agent(role: AgentRole) -> Tuple[AgentRole, AgentResult]:
            return role, self._safe_agent_run(self.agents[role], base_context)

        with ThreadPoolExecutor(max_workers=len(parallel_agents)) as executor:
            futures = [executor.submit(run_agent, role) for role in parallel_agents]
            for future in futures:
                role, result = future.result()
                attr_name = parallel_agents[role]
                setattr(
                    deliberation_state,
                    attr_name,
                    result.data if result.success else unavailable_msg
                )
        return deliberation_state

    def _determine_next_action(
        self,
        deliberation_state: DeliberationState,
        case_state: CaseState,
        remaining_budget: int,
    ) -> Action:
        """Derive the next action from deliberation and budget."""
        consensus_prompt = deliberation_state.to_consensus_prompt()
        consensus_result = self._safe_agent_run(
            self.agents[AgentRole.CONSENSUS], consensus_prompt
        )

        if not consensus_result.success:
            logger.error(
                f"Consensus agent run failed. Error: {consensus_result.error}"
            )
            return _get_fallback_action("Consensus agent failed to run.")

        action_dict = self._extract_function_call_output(consensus_result.data)
        if not action_dict:
            logger.error(
                "Failed to extract structured action from Consensus agent. Full response: "
                f"{consensus_result.data}"
            )
            return _get_fallback_action("Could not parse consensus agent output.")

        try:
            action = Action(**action_dict)
            return self._validate_and_correct_action(
                action, case_state, remaining_budget
            )
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for extracted data: {action_dict}. "
                f"Error: {e}. Full response: {consensus_result.data}"
            )
            return _get_fallback_action(f"Action validation failed: {e}")


    def _perform_turn(
        self, case_state: CaseState
    ) -> Tuple[Action, DeliberationState]:
        """
        Perform one deliberation turn and return the action and state.

        This is the core loop of the diagnostic process. It involves:
        1. Building a shared context for all agents.
        2. Collecting analyses from the agent panel in parallel.
        3. Checking for diagnostic stagnation.
        4. Deriving a final consensus action.
        5. Handling any critical errors that occur during the process.
        """
        logger.info(
            f"--- Starting Diagnostic Loop {case_state.iteration}/{self.max_iterations} ---"
        )

        try:
            # 1. Build the context with all current case information.
            remaining_budget = self.initial_budget - case_state.cumulative_cost
            base_context = self._build_base_context(case_state, remaining_budget)

            # 2. Run the panel deliberation.
            deliberation_state = self._collect_panel_deliberation(
                base_context, case_state
            )

            # 3. Check if the last few actions have been repetitive.
            deliberation_state.stagnation_detected = case_state.is_stagnating()

            # 4. Synthesize deliberations into a single, structured action.
            action = self._determine_next_action(
                deliberation_state, case_state, remaining_budget
            )
        except RuntimeError as e:
            # 5. Catch critical failures (e.g., Hypothesis agent) and return a safe action.
            logger.error(f"A critical error occurred during the turn: {e}")
            action = Action(
                action_type="ask",
                content="A critical agent failed. Please review the error and try again.",
                reasoning=str(e),
            )
            deliberation_state = DeliberationState(hypothesis_analysis=f"CRITICAL ERROR: {e}")

        logger.info(
            f"⚕️ Panel decision: {action.action_type.upper()} -> {action.content}"
        )
        return action, deliberation_state

    def _validate_and_correct_action(
        self, action: Action, case_state: CaseState, remaining_budget: int
    ) -> Action:
        """Validate and correct actions based on mode constraints and context."""
        if self.mode == "question_only" and action.action_type == "test":
            action.action_type = "ask"
            action.content = "Can you provide more details? (Test ordering disabled in question_only mode)"
        if (
            self.mode == "budgeted"
            and action.action_type == "test"
            and remaining_budget <= 0
        ):
            action.action_type = "diagnose"
            action.content = case_state.get_leading_diagnosis()
            action.reasoning = "Forced diagnosis due to budget exhaustion."
        return action

    def _update_differential_from_text(self, case_state: CaseState, text: str):
        """Extract and update differential diagnosis from JSON-formatted text."""
        try:
            if not isinstance(text, str):
                text = str(text)

            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in hypothesis analysis")

            json_str = text[start:end]
            data = json.loads(json_str)
            schema = DifferentialSchema(**data)
            new_differential = {
                entry.diagnosis.strip(): entry.probability
                for entry in schema.differential
            }
            if new_differential:
                case_state.update_differential(new_differential)
                logger.debug(f"Updated differential: {new_differential}")
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.warning(
                f"Could not parse differential diagnosis from text: {e}"
            )

    def _interact_with_gatekeeper(
        self, action: Action, full_case_details: str
    ) -> str:
        """Send the panel's action to the Gatekeeper and return its response."""
        if action.action_type == "diagnose":
            return "No interaction needed for 'diagnose' action."
        request = f"Request from Diagnostic Panel: {action.action_type} - {action.content}"
        prompt = f"Full Case Details (for your reference only):\n---\n{full_case_details}\n---\n\n{request}"
        result = self._safe_agent_run(self.agents[AgentRole.GATEKEEPER], prompt)
        return result.data if result.success else f"Error: {result.error}"

    def _judge_diagnosis(
        self, candidate_diagnosis: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Use the Judge agent to evaluate the final diagnosis."""
        prompt = f"""
        Please evaluate the following diagnosis against the ground truth.
        Ground Truth: '{ground_truth}'
        Candidate Diagnosis: '{candidate_diagnosis}'
        Use the `evaluate_diagnosis` function to provide your score and justification.
        """
        result = self._safe_agent_run(self.agents[AgentRole.JUDGE], prompt)

        if not result.success:
            logger.error(f"Judge agent run failed: {result.error}")
            return {"score": 0.0, "reasoning": "Judge agent failed to run."}

        judgement = self._extract_function_call_output(result.data)

        if not judgement or "score" not in judgement or "reasoning" not in judgement:
            logger.error(f"Failed to extract structured evaluation from Judge agent. Full response: {result.data}")
            return {"score": 0.0, "reasoning": "Could not parse judge's response."}

        return {
            "score": float(judgement.get("score", 0.0)),
            "reasoning": str(judgement.get("reasoning", "No reasoning provided.")),
        }

    def run(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
    ) -> DiagnosisResult:
        """Execute the full autonomous sequential diagnostic process."""
        case_state = CaseState(
            initial_vignette=initial_case_info,
            cumulative_cost=self.physician_visit_cost,
        )
        conversation = Conversation(time_enabled=True, autosave=False)
        conversation.add("System", f"Initial Case: {initial_case_info}")
        case_state.add_evidence(f"Initial presentation: {initial_case_info}")

        final_diagnosis = None
        for i in range(self.max_iterations):
            case_state.iteration = i + 1
            action, _ = self._perform_turn(case_state)
            case_state.add_action(action)

            if action.action_type == "diagnose":
                final_diagnosis = str(action.content)
                break

            response = self._interact_with_gatekeeper(
                action, full_case_details
            )
            conversation.add("Gatekeeper", response)
            case_state.add_evidence(response)

            if action.action_type == "test":
                cost = estimate_cost(action.content, self.test_cost_db)
                case_state.add_test(str(action.content), cost)
            elif action.action_type == "ask":
                case_state.add_question(str(action.content))

            if (
                self.mode == "budgeted"
                and case_state.cumulative_cost >= self.initial_budget
            ):
                final_diagnosis = case_state.get_leading_diagnosis()
                logger.warning(
                    "Budget limit reached. Forcing final diagnosis."
                )
                break
        else:
            final_diagnosis = (
                case_state.get_leading_diagnosis()
                or "Diagnosis not reached within iterations."
            )

        judgement = self._judge_diagnosis(
            final_diagnosis, ground_truth_diagnosis
        )
        return DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth_diagnosis,
            accuracy_score=judgement["score"],
            accuracy_reasoning=judgement["reasoning"],
            total_cost=case_state.cumulative_cost,
            iterations=case_state.iteration,
            conversation_history=conversation.get_str(),
        )

    @classmethod
    def create_variant(cls, variant: str, **kwargs) -> "MaiDxOrchestrator":
        """Create a preconfigured orchestrator variant."""
        configs = {
            "instant": {"mode": "instant", "max_iterations": 1},
            "question_only": {"mode": "question_only"},
            "budgeted": {
                "mode": "budgeted",
                "initial_budget": kwargs.get("budget", 5000),
            },
            "no_budget": {"mode": "no_budget"},
            "ensemble": {"mode": "no_budget"},
        }
        config = configs.get(variant, {})
        config.update(kwargs)
        config.pop("budget", None)
        return cls(**config)


# Aliases for compatibility
DiagnosticOrchestrator = MaiDxOrchestrator
AutonomousMode = MaiDxOrchestrator