"""Repo-level pytest hygiene.

House rules (user: "always delete generated runs" + "delete pycache files
anytime they appear"): after every test session we remove the artifacts the
run produced, so neither accumulates in the working tree.

  * ``__pycache__/`` + ``*.pyc`` / ``*.pyo`` — compiled bytecode, always safe
    to delete (Python regenerates it on import).
  * the repo-level ``runs/`` dir — some tests use the default
    ``runs_dir="runs"``, so test-generated run folders leak into the repo.
    We delete only the subdirs *created during this session*, preserving any
    run you intentionally kept outside a test. ``runs/`` is a gitignored
    scratch area (see ``.gitignore``); to keep a run, run it elsewhere with
    ``--runs-dir <dir>``.

Both cleanups run inside a ``try/except`` so a hiccup can never fail a green
suite. Source files, ``examples/`` data, and pre-existing runs are never
touched.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
REGISTRY = "registry.json"


def _existing_run_dirs() -> set[str]:
    """Run subdirs already present at session start (to be preserved)."""
    if RUNS.is_dir():
        return {d.name for d in RUNS.iterdir() if d.is_dir()}
    return set()


def _clean_bytecode() -> None:
    for d in REPO.rglob("__pycache__"):
        if d.is_dir() and ".git" not in d.parts:
            shutil.rmtree(d, ignore_errors=True)
    for pattern in ("*.pyc", "*.pyo"):
        for f in REPO.rglob(pattern):
            if ".git" not in f.parts:
                f.unlink(missing_ok=True)


def _clean_generated_runs(preserve: set[str]) -> None:
    """Remove run subdirs created this session (not in `preserve`) and drop
    their `registry.json` entries; keep pre-existing runs and their rows."""
    if not RUNS.is_dir():
        return
    for d in RUNS.iterdir():
        if d.is_dir() and d.name not in preserve:
            shutil.rmtree(d, ignore_errors=True)
    reg = RUNS / REGISTRY
    if not reg.exists():
        return
    try:
        data = json.loads(reg.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    if not isinstance(data, list):
        return
    kept = [e for e in data
            if isinstance(e, dict) and e.get("run_id") in preserve]
    try:
        if kept:
            reg.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        else:
            reg.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def _clean_generated_artifacts_after_session():
    """Yield, then remove this session's bytecode + test-generated runs."""
    preserve = _existing_run_dirs()
    yield
    try:
        _clean_bytecode()
        _clean_generated_runs(preserve)
    except OSError:
        pass
