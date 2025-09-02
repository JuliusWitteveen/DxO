import uuid
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field, asdict

from mai_dx.main import (
    MaiDxOrchestrator,
    CaseState,
    DeliberationState,
)
from mai_dx.structures import Action, AgentRole
from mai_dx.persistence import save_session
from mai_dx.costing import estimate_cost


@dataclass
class Turn:
    """Represents a single turn in the diagnostic session.

    This dataclass is an immutable snapshot of the session's state at a
    specific point in time. It is used for historical logging, auditing,
    and for generating analytics visualizations. It should not be used to
    inform current agent decisions; for that, use the CaseState object.
    """

    turn_number: int
    action_request: Action
    physician_input: Optional[str] = None
    deliberation: Optional[Dict[str, str]] = None
    differential_at_turn: Dict[str, float] = field(default_factory=dict)
    cost_at_turn: int = 0


@dataclass
class InteractiveDxSession:
    """Stateful wrapper for an interactive diagnostic session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    orchestrator_config: Dict[str, Any] = field(default_factory=dict)
    case_state: Optional[CaseState] = None
    turns: List[Turn] = field(default_factory=list)
    is_complete: bool = False

    _orchestrator: MaiDxOrchestrator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the underlying orchestrator after dataclass initialization."""
        self._orchestrator = MaiDxOrchestrator(**self.orchestrator_config)

    def start(self, initial_info: str) -> None:
        """Start a new diagnostic session.

        Args:
            initial_info: Initial vignette describing the case.

        Returns:
            None
        """
        self.case_state = CaseState(
            initial_vignette=initial_info,
            cumulative_cost=self._orchestrator.physician_visit_cost,
        )
        self.case_state.add_evidence(f"Initial presentation: {initial_info}")
        self._next_turn()

    def step(self, physician_input: str, ui_turn_number: int) -> None:
        """Process physician input and advance the session.

        Args:
            physician_input: New evidence supplied by the physician.
            ui_turn_number: The turn number the UI believes it is on.

        Returns:
            None

        Raises:
            ValueError: If the session is complete or if there is a
                        mismatch between the UI and session turn number.
        """
        if self.is_complete:
            raise ValueError("Session is already complete.")

        expected_turn_number = len(self.turns)
        if ui_turn_number != expected_turn_number:
            raise ValueError(
                f"State mismatch detected. The UI is on turn {ui_turn_number}, "
                f"but the session expects action for turn {expected_turn_number}. "
                "Please refresh the page."
            )

        # Record physician input in the last turn
        if self.turns:
            self.turns[-1].physician_input = physician_input

        # Update case state with new evidence
        if self.case_state:
            self.case_state.add_evidence(physician_input)

            last_action = self.turns[-1].action_request
            if last_action.action_type == "test":
                cost = estimate_cost(last_action.content, self._orchestrator.test_cost_db)
                self.case_state.add_test(str(last_action.content), cost)
            elif last_action.action_type == "ask":
                self.case_state.add_question(str(last_action.content))

        self._next_turn()

    def _next_turn(self) -> None:
        """Advance the session by performing one deliberation turn.

        Updates the case state, records the deliberation, and marks the
        session as complete if a final diagnosis is reached.

        Returns:
            None
        """
        if not self.case_state:
            return 

        self.case_state.iteration = len(self.turns) + 1

        action, deliberation = self._orchestrator._perform_turn(
            self.case_state
        )
        self.case_state.add_action(action)

        if action.action_type == "diagnose":
            self.is_complete = True

        new_turn = Turn(
            turn_number=self.case_state.iteration,
            action_request=action,
            deliberation={
                AgentRole.HYPOTHESIS.value: deliberation.hypothesis_analysis,
                AgentRole.TEST_CHOOSER.value: deliberation.test_chooser_analysis,
                AgentRole.CHALLENGER.value: deliberation.challenger_analysis,
                AgentRole.STEWARDSHIP.value: deliberation.stewardship_analysis,
                AgentRole.CHECKLIST.value: deliberation.checklist_analysis,
            },
            differential_at_turn=self.case_state.differential_diagnosis.copy(),
            cost_at_turn=self.case_state.cumulative_cost,
        )
        self.turns.append(new_turn)
        self.save()

    def update_runtime_params(
        self,
        model_name: Optional[str] = None,
        prompt_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update orchestrator runtime configuration mid-session.

        Args:
            model_name: New language model name.
            prompt_overrides: Mapping of agent role names to new prompts.

        Returns:
            None
        """
        self._orchestrator.update_runtime_params(
            model_name=model_name, prompt_overrides=prompt_overrides
        )
        if model_name:
            self.orchestrator_config["model_name"] = model_name
        if prompt_overrides:
            current = self.orchestrator_config.get("prompt_overrides", {})
            current.update(prompt_overrides)
            self.orchestrator_config["prompt_overrides"] = current
        self.save()

    def last_transparency(self) -> Dict[str, str]:
        """Return deliberation details from the latest turn.

        Returns:
            Dictionary mapping agent names to their reasoning for the most
            recent turn. Empty if no turns exist.
        """
        return self.turns[-1].deliberation if self.turns else {}

    def differential_timeline(self) -> List[Dict[str, Any]]:
        """Return the evolution of differential diagnoses over turns.

        Returns:
            List of dictionaries containing turn number, diagnosis, and
            probability.
        """
        timeline: List[Dict[str, Any]] = []
        for turn in self.turns:
            for diagnosis, prob in turn.differential_at_turn.items():
                timeline.append(
                    {
                        "turn": turn.turn_number,
                        "diagnosis": diagnosis,
                        "probability": prob,
                    }
                )
        return timeline

    # ---------- JSON-safe serialization ----------
    @staticmethod
    def _action_to_dict(a: Action) -> Dict[str, Any]:
        """Convert a Pydantic model to a plain dictionary.

        Args:
            a: Pydantic ``Action`` instance.

        Returns:
            A JSON-serializable dictionary.
        """
        return a.model_dump() if hasattr(a, "model_dump") else a.dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the session.
        """
        # Case state (asdict won't convert Pydantic models inside)
        cs = asdict(self.case_state) if self.case_state else None
        if cs and self.case_state and self.case_state.last_actions:
            cs["last_actions"] = [
                self._action_to_dict(a) for a in self.case_state.last_actions
            ]

        # Turns (ensure action_request is a plain dict)
        turns_list: List[Dict[str, Any]] = []
        for t in self.turns:
            td = asdict(t)
            td["action_request"] = self._action_to_dict(t.action_request)
            turns_list.append(td)

        return {
            "session_id": self.session_id,
            "orchestrator_config": self.orchestrator_config,
            "case_state": cs,
            "turns": turns_list,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractiveDxSession":
        """Deserialize a session from a dictionary.

        Args:
            data: Serialized session structure.

        Returns:
            A reconstructed :class:`InteractiveDxSession` instance.
        """
        session_id = data.get("session_id")
        if not session_id:
            raise ValueError(
                "Session data is missing the required 'session_id'."
            )
        session = cls(
            session_id=session_id,
            orchestrator_config=data.get("orchestrator_config", {}),
            is_complete=data.get("is_complete", False),
        )

        # Restore case state
        if data.get("case_state"):
            cs_data = data["case_state"]
            # Rehydrate last_actions to Pydantic Action objects first
            last_actions_data = cs_data.get("last_actions", [])
            cs_data["last_actions"] = [
                Action(**a) if isinstance(a, dict) else a
                for a in last_actions_data
            ]
            session.case_state = CaseState(**cs_data)


        # Restore turns
        restored_turns: List[Turn] = []
        for t in data.get("turns", []):
            turn_number = t.get("turn_number")
            if turn_number is None:
                raise ValueError("A turn entry is missing the 'turn_number'.")
            ar = t.get("action_request")
            if ar is None:
                raise ValueError(
                    f"Turn {turn_number} is missing the 'action_request'."
                )
            action_obj = ar if isinstance(ar, Action) else Action(**ar)
            restored_turns.append(
                Turn(
                    turn_number=turn_number,
                    action_request=action_obj,
                    physician_input=t.get("physician_input"),
                    deliberation=t.get("deliberation"),
                    differential_at_turn=t.get("differential_at_turn", {}),
                    cost_at_turn=t.get("cost_at_turn", 0),
                )
            )
        session.turns = restored_turns

        return session

    def save(self) -> None:
        """Persist the current session state to disk."""
        save_session(self.to_dict(), self.session_id)