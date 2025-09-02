#!/usr/bin/env python
"""
DxO Preflight Check (v2) — fixes root import path issue

Run from repo root:
    python scripts\preflight_check.py
"""

from __future__ import annotations
import os, sys, importlib

# Ensure repo root is importable even though this file lives in scripts/
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

OK = 0
FAIL = 1

def _print_hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return FAIL

def _ok(msg: str) -> int:
    print(f"[ OK ] {msg}")
    return OK

def check_env() -> int:
    _print_hdr("1) ENVIRONMENT")
    print(f"Python: {sys.version.split()[0]}  (exe: {sys.executable})")
    print(f"CWD   : {os.getcwd()}")

    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        print(f"[info] .env found at {dotenv_path}")
    else:
        print("[info] .env not found (that's OK if you use real env vars)")

    missing = []
    required = ["MAI_DX_SECRET"]
    for k in required:
        if not os.getenv(k):
            missing.append(k)

    if missing:
        return _fail(f"Missing required env var(s): {', '.join(missing)}")
    return _ok("Env vars look OK (OPENAI_API_KEY optional for offline tests)")

def check_paths() -> int:
    _print_hdr("2) PATH SANITY")
    repo = os.getcwd()
    nested = os.path.join(repo, "DxO")
    if os.path.isdir(nested) and os.path.exists(os.path.join(nested, "tests")):
        print(f"[warn] Detected nested project directory: {nested}")
        print("       This often causes pytest import mismatches on Windows.")
        return _fail("Nested DxO directory detected")
    print("[ OK ] No nested DxO/ path detected")

    # Stale __pycache__ folders
    stale = []
    for root, dirs, files in os.walk(repo):
        for d in list(dirs):
            if d == "__pycache__":
                stale.append(os.path.join(root, d))
    if stale:
        print("[warn] __pycache__ dirs present. If pytest reports 'import file mismatch', clean them:")
        print("       for /r %i in (*__pycache__*) do @if exist \"%i\" rmdir /S /Q \"%i\"")
    else:
        print("[ OK ] No __pycache__ dirs found")
    return OK

def check_api_selection() -> int:
    _print_hdr("3) MODEL → API SELECTION")
    try:
        sel = importlib.import_module("model_api_selector")
        use_responses_api = getattr(sel, "use_responses_api")
    except Exception as e:
        return _fail(f"Could not import model_api_selector.use_responses_api: {e}")

    probes = [
        "gpt-4o-mini", "gpt-4o", "gpt-4-0613",
        "gpt-5.1", "o3-mini", "o4-mini", "gpt-4.1-mini"
    ]
    for m in probes:
        try:
            api = "Responses" if use_responses_api(m) else "ChatCompletions"
            print(f"  - {m:<12} → {api}")
        except Exception as e:
            print(f"[FAIL] exception on probe {m}: {e}")
            return FAIL
    return _ok("API selection probe OK")

def check_parser() -> int:
    _print_hdr("4) PARSER SANITY (MaiDxOrchestrator._extract_function_call_output)")
    try:
        main = importlib.import_module("mai_dx.main")
        Mai = getattr(main, "MaiDxOrchestrator")
    except Exception as e:
        return _fail(f"Could not import mai_dx.main: {e}")

    orch = Mai.__new__(Mai)  # construct without __init__ side-effects
    try:
        fn = getattr(orch, "_extract_function_call_output")
    except AttributeError:
        return _fail("MaiDxOrchestrator._extract_function_call_output not found")

    tests = [
        ({"action_type": "ask", "content": "Age?", "reasoning": "Need age"},
         {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}),
        ('Some text before... {"action_type": "ask", "content": "Age?", "reasoning": "Need age",} and after',
         {"action_type": "ask", "content": "Age?", "reasoning": "Need age"}),
        ("This is just a regular string without any dict or json.", None),
    ]

    failures = 0
    for sample, expected in tests:
        try:
            got = fn(sample)
        except Exception as e:
            print(f"[FAIL] exception on input {sample!r}: {e}")
            failures += 1
            continue
        if got != expected:
            print(f"[FAIL] expected {expected!r} but got {got!r} for input {sample!r}")
            failures += 1
        else:
            print(f"[ OK ] parsed: {sample!r} → {got!r}")

    return OK if failures == 0 else FAIL

def main() -> int:
    statuses = [
        check_env(),
        check_paths(),
        check_api_selection(),
        check_parser(),
    ]
    rc = OK if all(s == OK for s in statuses) else FAIL
    print("\nResult:", "OK" if rc == OK else "FAIL")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
