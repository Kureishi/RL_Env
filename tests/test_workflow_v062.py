"""Workflow strip — finished is all-green (v0.62, SPEC.md 76, A66).

SPEC.md 76: the terminal `Results` step of the four-step workflow state
machine (69.4.1) is `done` (green) when a result exists and no run is in
flight, so a finished run renders all four steps `done`:
  * 76.2.1 — the `Results` branch of `workflow_state` returns `STEP_DONE`
    for the finished state (`has_result and not running`), not the v0.61
    `STEP_CURRENT`; the finished condition is unchanged, so the in-flight
    and idle states stay byte-identical (their single `current` is `Run`
    / `Data` respectively);
  * 76.2.2 — the `at most one current` invariant holds: the finished
    state has zero `current`, the idle/in-flight states keep their one;
    `dashboard_app` needs no edit (it renders whatever the state returns,
    and `svg_workflow_strip` colors `done` = the accepted-green token);
  * 76.2.3 — the A59 re-derivation in `tests/test_workflow_v055.py` (the
    `_expect` helper) is green; the SVG/ARIA tests there are unaffected
    (they assert structure, not the `Results` status).

House rules: pure / deterministic (G2), stdlib only, no cross-test
imports (every fixture synthesized here), the core stays streamlit-free
(`svg_workflow_strip` runs without the optional dependency).
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import resolve_tokens, svg_is_well_formed
from autorefine.workflow import (
    STEP_CURRENT,
    STEP_DONE,
    STEP_TODO,
    WORKFLOW_STEPS,
    svg_workflow_strip,
    workflow_state,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
INIT = REPO / "src" / "autorefine" / "__init__.py"


def _status(name: str, state: list[dict]) -> str:
    return next(r["status"] for r in state if r["step"] == name)


def _node_fill(svg: str, step: str) -> str:
    """The `fill` of the step's circle (each node is rendered as
    `<circle … fill="HEX"><title>STEP — STATUS</title></circle>`)."""
    m = re.search(
        r'<circle [^>]*fill="([^"]+)"><title>' + re.escape(step) + r" — ",
        svg)
    assert m, f"no circle node for {step!r}"
    return m.group(1)


# --- 76.2.1 the finished state is all four `done` (A66) -----------------------

def test_finished_state_is_all_done():
    """A66 (76.2.1): the finished state (data, not running, result) is
    all four `done` — the `Results` terminal is green, not `current`."""
    state = workflow_state(True, False, True)
    assert [r["status"] for r in state] == [STEP_DONE] * 4
    assert _status("Results", state) == STEP_DONE
    assert _status("Results", state) != STEP_CURRENT  # the v0.61 behavior


def test_in_flight_and_idle_unchanged():
    """A66 (76.2.1/76.2.2): the in-flight and idle states keep their
    single `current` (Run / Data) — only the finished state changed."""
    inflight = workflow_state(True, True, False)
    assert _status("Run", inflight) == STEP_CURRENT
    assert _status("Results", inflight) == STEP_TODO
    idle = workflow_state(False, False, False)
    assert _status("Data", idle) == STEP_CURRENT
    assert _status("Results", idle) == STEP_TODO


def test_reachable_states_single_current():
    """A66 (76.2.2): over the app-reachable states the `current` step is
    unique — the finished state has zero `current` (all four `done`), the
    idle state has exactly one (`Data`), and the in-flight-with-data state
    has exactly one (`Run`). (The raw degenerate state `no data + a run in
    flight` is never produced by the app — a run cannot start without data
    — so it is not asserted here.)"""
    # finished: all four `done`, zero `current` (the v0.62 change)
    finished = workflow_state(True, False, True)
    assert sum(1 for r in finished if r["status"] == STEP_CURRENT) == 0
    # idle: exactly one `current`, and it is `Data`
    idle = workflow_state(False, False, False)
    assert sum(1 for r in idle if r["status"] == STEP_CURRENT) == 1
    assert _status("Data", idle) == STEP_CURRENT
    # in-flight (data set): exactly one `current`, and it is `Run`
    inflight = workflow_state(True, True, False)
    assert sum(1 for r in inflight if r["status"] == STEP_CURRENT) == 1
    assert _status("Run", inflight) == STEP_CURRENT


def test_non_bool_inputs_rejected():
    """A66 (76.5): a non-bool input is a `ValueError` (the state machine
    cannot mis-report a state)."""
    for bad in (("yes", False, False), (True, "no", False), (True, False, 1)):
        with pytest.raises(ValueError):
            workflow_state(*bad)


def test_determinism_equal_inputs_equal_rows():
    """A66 (76.2.1/76.3.1): equal inputs give byte-equal rows (G2)."""
    assert workflow_state(True, False, True) == \
        workflow_state(True, False, True)


# --- 76.2.2 / 76.3.2 the finished strip is green + well-formed (A66) ----------

def test_finished_strip_results_node_is_accepted_green():
    """A66 (76.3.2): the finished strip colors the `Results` node with the
    accepted-green token (not the v0.61 baseline-amber `current`)."""
    tok = resolve_tokens()
    svg = svg_workflow_strip(workflow_state(True, False, True))
    assert _node_fill(svg, "Results") == tok["accepted"]
    assert _node_fill(svg, "Results") != tok["baseline"]  # not the old amber
    # the finished state is all `done` → every node is accepted-green
    for step in WORKFLOW_STEPS:
        assert _node_fill(svg, step) == tok["accepted"]
    assert "<title>Results — done</title>" in svg


def test_finished_strip_well_formed_deterministic_aria():
    """A66 (76.3.2): the finished strip is well-formed (its own guard
    passes), byte-deterministic, ARIA-labelled, and names all four
    step labels."""
    state = workflow_state(True, False, True)
    svg = svg_workflow_strip(state)
    assert svg_is_well_formed(svg)
    assert svg == svg_workflow_strip(state)  # deterministic (G2)
    assert 'role="img"' in svg
    assert 'aria-label="Workflow' in svg
    for label in WORKFLOW_STEPS:
        assert label in svg
    assert "<title>" in svg


def test_inflight_strip_results_stays_todo_grey():
    """A66 (76.2.2): while a run is in flight the `Results` node is
    `todo` (rejected-grey), not green — the live state wins."""
    tok = resolve_tokens()
    svg = svg_workflow_strip(workflow_state(True, True, False))
    assert _node_fill(svg, "Results") == tok["rejected"]
    assert _node_fill(svg, "Run") == tok["baseline"]  # the current step


# --- 33.1 / A25 the round bookkeeping (A66) ----------------------------------

def test_version_and_spec_round():
    """A66 (76.5 / 33.1 / A25): the version steps to `0.62.0` in both
    sources; SPEC §76 cites A66; the acceptance index row M65 points at
    this test file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.62.0"
    assert '"0.62.0"' in INIT.read_text(encoding="utf-8")
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 76. Workflow strip" in spec
    assert "### 76.5 Acceptance (A66)" in spec
    assert "### 76.6 Milestone (M65)" in spec
    assert ("| M65 | v0.62   | 76     | A66 "
            "| tests/test_workflow_v062.py |" in spec)
