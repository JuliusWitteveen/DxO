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


def save_session(session_data: Dict[str, Any], session_id: str) -> None:
    """Persist a session dictionary to an encrypted JSON file.

    Args:
        session_data: Data representing the session.
        session_id: Identifier used for the output filename.

    Returns:
        None
    """
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    json_str = json.dumps(session_data, indent=2).encode("utf-8")
    encrypted = _FERNET.encrypt(json_str)
    with open(filepath, "wb") as f:
        f.write(encrypted)


def load_session(session_id: str) -> Dict[str, Any]:
    """Load and decrypt a session dictionary from disk.

    Args:
        session_id: Identifier of the session to load.

    Returns:
        The decrypted session data.

    Raises:
        FileNotFoundError: If the session file does not exist.
        ValueError: If the file cannot be decrypted.
    """
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Session file not found: {filepath}")
    with open(filepath, "rb") as f:
        encrypted = f.read()
    try:
        decrypted = _FERNET.decrypt(encrypted)
        return json.loads(decrypted.decode("utf-8"))
    except Exception as e:
        raise ValueError(
            f"Failed to decrypt session '{session_id}': {e}"
        ) from e


def list_sessions() -> List[Dict[str, Any]]:
    """List metadata for all saved sessions.

    Returns:
        A list of dictionaries containing session metadata such as ID,
        initial vignette, last modified timestamp, and completion status.
    """
    sessions: List[Dict[str, Any]] = []
    files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(SESSION_DIR, f)),
        reverse=True,
    )
    for filename in files:
        try:
            session_id = filename.replace(".json", "")
            data = load_session(session_id)
            mtime = os.path.getmtime(os.path.join(SESSION_DIR, filename))
            sessions.append(
                {
                    "id": session_id,
                    "initial_vignette": data.get("case_state", {}).get(
                        "initial_vignette", "N/A"
                    ),
                    "last_modified": datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "turns": len(data.get("turns", [])),
                    "is_complete": data.get("is_complete", False),
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(f"Could not load session file {filename}: {e}")
    return sessions


def delete_session(session_id: str) -> None:
    """Remove a session file from disk.

    Args:
        session_id: Identifier of the session to delete.

    Returns:
        None
    """
    filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
