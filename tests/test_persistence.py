import base64
import importlib
import json
import os
import sys

import pytest


def test_encrypted_session_round_trip(tmp_path, monkeypatch):
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("MAI_DX_SECRET", key)
    sys.modules.pop("mai_dx.persistence", None)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    persistence = importlib.import_module("mai_dx.persistence")
    importlib.reload(persistence)

    # Use a temporary directory for session storage
    monkeypatch.setattr(persistence, "SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(
        persistence, "SESSION_INDEX_FILE", os.path.join(str(tmp_path), "session_index.json")
    )
    os.makedirs(persistence.SESSION_DIR, exist_ok=True)

    session_id = "abc123"
    data = {
        "case_state": {"initial_vignette": "Intro"},
        "turns": ["t1"],
        "is_complete": False,
    }

    persistence.save_session(data, session_id)
    path = tmp_path / f"{session_id}.json"
    assert path.exists()

    content = path.read_bytes()
    assert content != json.dumps(data, indent=2).encode("utf-8")

    loaded = persistence.load_session(session_id)
    assert loaded == data

    sessions = persistence.list_sessions()
    assert len(sessions) == 1
    meta = sessions[0]
    assert meta["id"] == session_id
    assert meta["initial_vignette"] == "Intro"
    assert meta["turns"] == 1
    assert meta["is_complete"] is False

    persistence.delete_session(session_id)
    assert not path.exists()


def test_save_session_atomic_on_failure(tmp_path, monkeypatch):
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("MAI_DX_SECRET", key)
    sys.modules.pop("mai_dx.persistence", None)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    persistence = importlib.import_module("mai_dx.persistence")
    importlib.reload(persistence)

    monkeypatch.setattr(persistence, "SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(
        persistence, "SESSION_INDEX_FILE", os.path.join(str(tmp_path), "session_index.json")
    )
    os.makedirs(persistence.SESSION_DIR, exist_ok=True)

    session_id = "atomic"
    data = {"case_state": {}, "turns": [], "is_complete": False}
    persistence.save_session(data, session_id)
    path = tmp_path / f"{session_id}.json"
    original = path.read_bytes()

    def boom(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", boom)

    data2 = {"case_state": {}, "turns": ["x"], "is_complete": True}
    with pytest.raises(OSError):
        persistence.save_session(data2, session_id)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("tmp_session_*"))
