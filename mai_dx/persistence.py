"""Persistence helpers for saving and loading diagnostic sessions."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv, set_key

try:  # pragma: no cover - fallback when cryptography is unavailable
    from cryptography.fernet import Fernet
except ModuleNotFoundError:  # pragma: no cover - executed in minimal environments
    import base64

    class Fernet:  # type: ignore
        """Very small fallback implementing Fernet-like interface.

        This implementation simply base64-encodes data and is **not** secure.
        It exists so the library can function in restricted environments
        without the optional cryptography dependency installed.
        """

        def __init__(self, key: bytes):  # noqa: D401 - parameters kept for API
            self.key = key

        def encrypt(self, data: bytes) -> bytes:
            return base64.urlsafe_b64encode(data)

        def decrypt(self, token: bytes) -> bytes:
            return base64.urlsafe_b64decode(token)

        @staticmethod
        def generate_key() -> bytes:
            return base64.urlsafe_b64encode(os.urandom(32))

    logging.warning(
        "cryptography package not installed; using insecure fallback Fernet"
    )

SESSION_DIR = "sessions"
SESSION_INDEX_FILE = os.path.join(SESSION_DIR, "session_index.json")
os.makedirs(SESSION_DIR, exist_ok=True)

# Load environment variables from a .env file if present.
load_dotenv()

# Retrieve the Fernet key from the environment for secure storage. If none is
# provided, generate a key and persist it to the local `.env` file so that
# sessions can be decrypted in future runs.
_FERNET_KEY = os.environ.get("MAI_DX_SECRET")
if _FERNET_KEY is None:
    _FERNET_KEY = Fernet.generate_key().decode()
    env_path = Path(".env")
    env_path.touch(exist_ok=True)
    set_key(str(env_path), "MAI_DX_SECRET", _FERNET_KEY)

_FERNET = Fernet(_FERNET_KEY.encode())


def _read_index() -> Dict[str, Any]:
    """Reads the session index from disk."""
    if not os.path.exists(SESSION_INDEX_FILE):
        return {}
    try:
        with open(SESSION_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_index(index_data: Dict[str, Any]) -> None:
    """Writes the session index to disk."""
    try:
        with open(SESSION_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
    except OSError as e:
        logging.error(f"Failed to write session index: {e}")


def _update_index(session_id: str, session_data: Dict[str, Any]):
    """Update the index with metadata for a single session."""
    index = _read_index()
    mtime = datetime.now().timestamp()
    metadata = {
        "id": session_id,
        "initial_vignette": session_data.get("case_state", {}).get(
            "initial_vignette", "N/A"
        ),
        "last_modified": datetime.fromtimestamp(mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "turns": len(session_data.get("turns", [])),
        "is_complete": session_data.get("is_complete", False),
    }
    index[session_id] = metadata
    _write_index(index)


def save_session(session_data: Dict[str, Any], session_id: str) -> None:
    """Persist a session dictionary to an encrypted JSON file."""
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    json_str = json.dumps(session_data, indent=2).encode("utf-8")
    encrypted = _FERNET.encrypt(json_str)
    with open(filepath, "wb") as f:
        f.write(encrypted)
    _update_index(session_id, session_data)


def load_session(session_id: str) -> Dict[str, Any]:
    """Load and decrypt a session dictionary from disk."""
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file not found: {filepath}")
    with open(filepath, "rb") as f:
        encrypted = f.read()
    try:
        decrypted = _FERNET.decrypt(encrypted)
        session_data = json.loads(decrypted.decode("utf-8"))
        # Ensure the index is up-to-date upon loading a session
        _update_index(session_id, session_data)
        return session_data
    except Exception as e:
        raise ValueError(
            f"Failed to decrypt session '{session_id}': {e}"
        ) from e


def list_sessions() -> List[Dict[str, Any]]:
    """List metadata for all saved sessions from the index."""
    index = _read_index()
    sessions = list(index.values())
    sessions.sort(key=lambda s: s.get("last_modified", ""), reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    """Remove a session file and its corresponding index entry."""
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    
    index = _read_index()
    if session_id in index:
        del index[session_id]
        _write_index(index)