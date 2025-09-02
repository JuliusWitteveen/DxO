"""
MAI-DxO Core Package.

This package contains the core logic for the multi-agent diagnostic orchestrator.
"""
from .main import MaiDxOrchestrator, DiagnosticOrchestrator, AutonomousMode
from .structures import AgentRole, Action

# ``InteractiveDxSession`` depends on optional packages (e.g., ``dotenv`` with
# ``set_key``) that are not always available during lightweight testing.  Import
# it lazily so that basic functionality can be used without those extras.
try:  # pragma: no cover - gracefully handle missing optional deps
    from .interactive import InteractiveDxSession
except Exception:  # pragma: no cover
    InteractiveDxSession = None  # type: ignore

__all__ = [
    "MaiDxOrchestrator",
    "DiagnosticOrchestrator",
    "AutonomousMode",
    "AgentRole",
    "Action",
    "InteractiveDxSession",
]