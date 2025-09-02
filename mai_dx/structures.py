"""
Shared data structures for the MAI-DxO project.

This file centralizes common data classes and enums to prevent circular imports
and improve code organization.
"""
from enum import Enum
from typing import List, Union, Literal
from pydantic import BaseModel, Field


class AgentRole(Enum):
    """Enumeration of roles for the virtual physician panel."""
    HYPOTHESIS = "Dr. Hypothesis"
    TEST_CHOOSER = "Dr. Test-Chooser"
    CHALLENGER = "Dr. Challenger"
    STEWARDSHIP = "Dr. Stewardship"
    CHECKLIST = "Dr. Checklist"
    CONSENSUS = "Consensus Coordinator"
    GATEKEEPER = "Gatekeeper"
    JUDGE = "Judge"


class Action(BaseModel):
    """Pydantic model for a structured action decided by the consensus agent."""
    action_type: Literal["ask", "test", "diagnose"] = Field(
        ..., description="The type of action to perform."
    )
    content: Union[str, List[str]] = Field(
        ...,
        description="The content of the action (question, test name, or diagnosis).",
    )
    reasoning: str = Field(
        ..., description="The reasoning behind choosing this action."
    )