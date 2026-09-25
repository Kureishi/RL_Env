"""Workflow-strip settle — the strip shows the finished state (v0.63,
SPEC.md 77, A67).

SPEC.md 77: the reported "Run and Results stay grey when finished"
state is a script-ordering defect — the strip renders at the top of the
app script, before the Run tab's drain sets ``st.session_state["result"]``
— so the frame where a run finishes shows the pre-run strip beside the
verdict banner (77.1). The fix:
  * 77.2.1 — ``workflow.settle_needed(has_result, running, state)`` is
    the single settle decision: ``True`` iff a result exists, no run is
    in flight, and the strip state this frame rendered is not yet
    all-``done``; pure and deterministic (G2);
  * 77.2.2 — the app calls it once, at the end of ``main()``; when it is
    ``True`` the app performs exactly one ``st.rerun()``. The settled
    frame is all-``done`` → ``settle_needed`` is ``False`` → bounded, no
    loop;
  * 77.2.4 — in-flight, idle, and steady frames never settle (their
    facts make the decision ``False``); Streamlit resets momentary
    button values after a re-run, so the settle can never re-trigger the
    run.

House rules: pure / deterministic (G2), stdlib only (the AppTest block
skips without the optional dependency), no cross-test imports (every
fixture synthesized here).
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine.workflow import (
    STEP_DONE,
    settle_needed,
    workflow_state,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
INIT = REPO / "src" / "autorefine" / "__init__.py"
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# the app-reachable strip states (76.2.1/76.2.2):
STALE_FINISHED = workflow_state(True, False, False)  # strip-time, result not set yet
FINISHED = workflow_state(True, False, True)         # settled frame
INFLIGHT = workflow_state(True, True, False)         # a run in flight
IDLE = workflow_state(False, False, False)           # no data, no run, no result


# --- 77.2.1 the pure settle decision (A67) -----------------------------------

def test_only_the_stale_finished_frame_settles():
    """A67 (77.2.1): exactly one frame settles — a result exists, no run
    in flight, and the strip this frame rendered is not yet all-`done`
    (the button-press / reattach frame, 77.1)."""
    assert settle_needed(True, False, STALE_FINISHED) is True


def test_settled_inflight_idle_frames_never_settle():
    """A67 (77.2.4): the settled frame (all `done`), the in-flight frame
    (a run is live), and the idle frame (no result) never re-trigger the
    settle — the re-run is bounded and can never loop (77.2.2)."""
    assert settle_needed(True, False, FINISHED) is False
    assert settle_needed(True, True, STALE_FINISHED) is False
    assert settle_needed(False, False, STALE_FINISHED) is False
    assert settle_needed(False, True, INFLIGHT) is False


def test_invalid_inputs_rejected():
    """A67 (77.2.1): non-bool inputs and an empty/absent `state` are a
    `ValueError` — the settle decision cannot mis-report a frame."""
    with pytest.raises(ValueError):
        settle_needed("yes", False, FINISHED)
    with pytest.raises(ValueError):
        settle_needed(True, "no", FINISHED)
    with pytest.raises(ValueError):
        settle_needed(True, False, [])
    with pytest.raises(ValueError):
        settle_needed(True, False, None)  # type: ignore[arg-type]


def test_settle_deterministic():
    """A67 (77.2.1 / G2): equal inputs give an equal bool."""
    assert (settle_needed(True, False, STALE_FINISHED)
            == settle_needed(True, False, STALE_FINISHED))


def test_settle_consumes_state_machine_output():
    """A67 (77.4): `settle_needed` does not restate the state machine —
    it consumes `workflow_state` rows: with a result and no run, the
    settle on/off tracks exactly the ``not yet all-`done`` condition."""
    for state, expected in ((STALE_FINISHED, True), (FINISHED, False)):
        not_done = any(r["status"] != STEP_DONE for r in state)
        assert settle_needed(True, False, state) == not_done == expected


# --- 77.2.2 the app wiring (A67) ----------------------------------------------

def test_app_wires_the_settle():
    """A67 (77.2.2): the app imports `settle_needed` and, at the end of
    the tab section, re-runs exactly once when the strip is stale while a
    result exists — no other settle site, no second `st.rerun()` in the
    block."""
    src = APP.read_text(encoding="utf-8")
    assert "settle_needed" in src
    assert "from autorefine.workflow import (" in src
    assert re.search(
        r"if settle_needed\(result is not None, _wrunning, _wstate\):\s*"
        r"\n\s*st\.rerun\(\)",
        src), "the settle block: one guarded st.rerun() at the end of main()"
    # the settle sits after the last tab (t_exps) — end-of-main, not mid-script
    i_exps = src.index("with t_exps:")
    i_settle = src.index("if settle_needed(")
    i_main_call = src.rindex("main()")
    assert i_exps < i_settle < i_main_call


# --- 77.3 the AppTest settle flow (A67) ---------------------------------------

def _write_xor_csv(tmp: Path) -> Path:
    """The shared tiny classification CSV (quadrant XOR, 2 classes) —
    the same fixture the v0.41 AppTest uses."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp / "data.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _workflow_svg(at) -> str:
    md = " ".join(m.value for m in at.markdown)
    m = re.search(r'<svg[^>]*aria-label="Workflow.*?</svg>', md, re.DOTALL)
    assert m, "no workflow strip SVG in the frame"
    return m.group(0)


def _node_fill(svg: str, step: str) -> str:
    m = re.search(
        r'<circle [^>]*fill="([^"]+)"><title>' + re.escape(step) + r" — ",
        svg)
    assert m, f"no circle node for {step!r}"
    return m.group(1)


def test_app_test_settled_frame_is_all_green_with_verdict():
    """A67 (77.3): the button-press flow ends on the settled frame —
    all four strip nodes the accepted-green token with the PASS banner
    visible — and one further run leaves the strip byte-identical (the
    settle is bounded, 77.2.2)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest
    import tempfile

    from autorefine.plotting import resolve_tokens
    tok = resolve_tokens()
    tmp = Path(tempfile.mkdtemp())
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_xor_csv(tmp)))
    at.session_state["runs_dir"] = str(tmp / "runs")
    at.session_state["experiments"] = 2
    at.session_state["max_train"] = 5.0
    at.session_state["target"] = 0.0  # the tiny run PASSes → the banner is green
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True)
    at.run()  # Streamlit 1.51's AppTest resolves the settle st.rerun() here
    assert not at.exception, at.exception
    assert at.session_state["result"] is not None
    # the verdict is visible in this same (settled) frame
    assert any("PASS: final" in s.value for s in at.success), \
        "the PASS banner renders in the settled frame"
    # and the strip shows the finished state: all four accepted-green
    svg = _workflow_svg(at)
    for step in ("Data", "Preview", "Run", "Results"):
        assert _node_fill(svg, step) == tok["accepted"], \
            f"{step} is not the accepted-green in the settled frame"
    # bounded: one further run leaves the strip byte-identical (no loop)
    at.run()
    assert not at.exception
    assert _workflow_svg(at) == svg


# --- 33.1 / A25 the round bookkeeping (A67) ----------------------------------

def test_version_and_spec_round():
    """A67 (77.5 / 33.1 / A25): the version steps to `0.63.0` in both
    sources; SPEC §77 cites A67; the acceptance index row M66 points at
    this test file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.63.0"
    assert '"0.63.0"' in INIT.read_text(encoding="utf-8")
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 77. Workflow-strip settle" in spec
    assert "### 77.5 Acceptance (A67)" in spec
    assert "### 77.6 Milestone (M66)" in spec
    assert ("| M66 | v0.63   | 77     | A67 "
            "| tests/test_workflow_v063.py |" in spec)
