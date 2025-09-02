# swarms/__init__.py
# Compatible local shim for `swarms` to satisfy runtime usage in DxO.
# Provides Agent and Conversation with attributes commonly accessed by the app.
#
# Place this file at: C:\DxO\swarms\__init__.py

from __future__ import annotations

class Agent:
    """
    Minimal stand-in with the attributes the app expects.
    Accepts both `agent_name` and `name` and exposes both.
    Carries through any extra kwargs so callers can stash config.
    """
    def __init__(self, agent_name: str | None = None, name: str | None = None, **kwargs):
        if agent_name is None and name is None:
            agent_name = "Agent"
        # unify
        self.agent_name = agent_name if agent_name is not None else name
        self.name = self.agent_name
        # optional/commonly referenced fields
        self.role = kwargs.get("role", "assistant")
        self.system_prompt = kwargs.get("system_prompt") or kwargs.get("instructions")
        self.tools = kwargs.get("tools", [])
        self.config = kwargs
        # simple state
        self._history: list[tuple[str, str]] = []

    def run(self, message: str | None = None, **kwargs):
        """
        Placeholder execute method. In the real library this would produce
        a response; here we just record and return a no-op structure.
        """
        if message is not None:
            self._history.append((self.agent_name, str(message)))
        return {"content": "", "tool_calls": []}

    def __repr__(self) -> str:
        return f"<Agent {self.agent_name!r} role={self.role!r}>"


class Conversation:
    """Very small conversation log with Stream()-like helpers."""
    def __init__(self):
        self.history: list[tuple[str, str]] = []

    def add(self, speaker: str, message: str):
        self.history.append((speaker, message))

    def get_str(self) -> str:
        return "\n".join(f"{s}: {m}" for s, m in self.history)

__all__ = ["Agent", "Conversation"]
