"""Audit logging utilities for MAI-DxO UI components."""

import json
import os
import hashlib
import time
from dataclasses import asdict
from typing import Any, Dict, Optional


class AuditLogger:
    """Simple append-only audit logging with hash chaining for integrity."""

    def __init__(self, log_path: str = "audit_logs/audit.log"):
        """Initialize the logger and load the last entry hash."""
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Load the previous entry hash from disk."""
        if not os.path.exists(self.log_path):
            return "0" * 64
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = None
                for last_line in f:
                    pass
                if last_line:
                    data = json.loads(last_line)
                    return data.get("hash", "0" * 64)
        except (json.JSONDecodeError, OSError):
            pass
        return "0" * 64

    def _write_entry(self, entry: Dict[str, Any]):
        """Write an entry to the log, updating the hash chain."""
        entry["prev_hash"] = self.last_hash
        entry_bytes = json.dumps(entry, sort_keys=True).encode("utf-8")
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry["hash"] = entry_hash
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.last_hash = entry_hash

    def log_state_change(self, state: Any, description: str = ""):
        """Record a state change in the audit log."""
        try:
            state_dict = asdict(state)
        except Exception:
            state_dict = state
        entry = {
            "timestamp": time.time(),
            "type": "state_change",
            "description": description,
            "state": state_dict,
        }
        self._write_entry(entry)

    def log_decision(self, decision: str, details: Optional[Dict[str, Any]] = None):
        """Record a decision made by the system."""
        entry = {
            "timestamp": time.time(),
            "type": "decision",
            "decision": decision,
            "details": details or {},
        }
        self._write_entry(entry)
