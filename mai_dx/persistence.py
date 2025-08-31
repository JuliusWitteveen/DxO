"""Persistence helpers for saving and loading diagnostic sessions."""

import base64
import itertools
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Simple XOR-based encryption/decryption. The key can be provided via the
# MAI_DX_SECRET environment variable; otherwise a default is used.
_SECRET_KEY = os.environ.get("MAI_DX_SECRET", "mai_dx_default_key").encode()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR the data with the given key (repeating the key as needed).

    Args:
        data: Bytes to encrypt or decrypt.
        key: Secret key used for XOR operation.

    Returns:
        The transformed bytes after XOR.
    """
    return bytes(a ^ b for a, b in zip(data, itertools.cycle(key)))


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
    encrypted = base64.b64encode(_xor_bytes(json_str, _SECRET_KEY))
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
        decrypted = _xor_bytes(base64.b64decode(encrypted), _SECRET_KEY)
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
