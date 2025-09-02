import importlib
import sys
import types

import pytest


def test_shadowed_swarms_package_warns(monkeypatch):
    """Importing mai_dx.main with a shadowed swarms module should raise hint."""
    fake_swarms = types.ModuleType("swarms")
    monkeypatch.setitem(sys.modules, "swarms", fake_swarms)

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    sys.modules.pop("mai_dx.main", None)
    with pytest.raises(ImportError) as exc:
        importlib.import_module("mai_dx.main")
    assert "rename or remove any local directories named 'swarms'" in str(exc.value)
