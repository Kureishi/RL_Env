"""Workflow efficiency + result-view tabs (v0.55, SPEC.md 69, A59).

SPEC.md 69: the app's procedural workflow becomes legible and the result
views reachable by name:
  * 69.1 — the Results page groups into six nested tabs (Export &
    artifacts, Plots & decisions, Research & conclusions, Parameters &
    what-if, Learning views, Advanced analysis);
  * 69.2 — the live drain splits into Live / Decision views /
    Notes & drill-down sub-tabs;
  * 69.3 — `svg_is_well_formed` (pure, in `plotting`) + the app's
    `_svg_html` / `_svg_block` guards degrade a broken renderer to a
    caption instead of a broken image;
  * 69.4 — the streamlit-free `workflow` core: the four-step state
    machine, the stepper SVG, and the post-run next-steps advice;
  * 69.5 — A1–A58 stay green (every addition is additive or
    app-structural; the bandit/search default path is untouched).

House rules: pure / deterministic (G2), stdlib + numpy core leaves, no
cross-test imports (every fixture synthesized here), the core stays
streamlit-free (the app tests touch `dashboard_app` only under AppTest,
streamlit optional-skipped). Envs stay tiny (quadrant-XOR CSV, 2
experiments) for speed.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import svg_is_well_formed
from autorefine.workflow import (
    STEP_CURRENT,
    STEP_DONE,
    STEP_TODO,
    WORKFLOW_STEPS,
    next_steps,
    svg_workflow_strip,
    workflow_state,
)

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
SPEC = REPO / "SPEC.md"
PLOT = REPO / "src" / "autorefine" / "plotting.py"

TOP_TABS = ["Setup", "Run", "Results", "Compare", "Experiments"]
INNER_TABS = ["Export & artifacts", "Plots & decisions",
              "Research & conclusions", "Parameters & what-if",
              "Learning views", "Advanced analysis"]
DRAIN_TABS = ["Live", "Decision views", "Notes & drill-down"]


# --- fixtures (synthesized here — no cross-test imports) ----------------------

def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes)."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --- 69.4.2 the state machine (A59) -------------------------------------------

def _expect(has_data, running, has_result) -> list[tuple[str, str]]:
    """The 69.4.2 rules, stated independently of the implementation."""
    out = []
    for name in WORKFLOW_STEPS:
        if name == "Data":
            st = STEP_DONE if has_data else STEP_CURRENT
        elif name == "Preview":
            st = STEP_DONE if has_data else STEP_TODO
        elif name == "Run":
            st = (STEP_CURRENT if running
                  else STEP_DONE if has_result else STEP_TODO)
        else:  # Results
            st = STEP_CURRENT if (has_result and not running) else STEP_TODO
        out.append((name, st))
    return out


def test_workflow_state_all_combinations():
    """A59 (69.4.2): all 8 (data, running, result) combinations give the
    expected per-step rows — the `Run` step is `current` while in
    flight, and the terminal `Results` step is `current` only when
    finished and no run is in flight."""
    for has_data in (False, True):
        for running in (False, True):
            for has_result in (False, True):
                rows = workflow_state(has_data, running, has_result)
                assert [tuple(r.values()) for r in rows] == _expect(
                    has_data, running, has_result)
    # while a run is in flight, `Run` is the `current` step (the live
    # state the user most needs to see)
    assert workflow_state(True, True, False)[2]["status"] == STEP_CURRENT
    assert workflow_state(False, True, False)[2]["status"] == STEP_CURRENT
    # equal inputs give byte-equal rows (G2 determinism)
    assert (workflow_state(True, False, True)
            == workflow_state(True, False, True))


def test_workflow_state_rejects_non_bool():
    """A59 (69.4.2): the app's three facts are bools — a non-bool is a
    `ValueError` (the state machine cannot mis-report a state)."""
    for bad in (("yes", False, False), (True, "no", False),
                (True, False, 1)):
        with pytest.raises(ValueError):
            workflow_state(*bad)


# --- 69.4.3 the workflow strip SVG (A59) --------------------------------------

def test_workflow_strip_well_formed_deterministic_aria():
    """A59 (69.4.3): the strip is well-formed (its own guard passes),
    byte-deterministic, ARIA-labelled (the 51.4.3 convention), and names
    all four step labels."""
    state = workflow_state(True, False, True)  # finished
    svg = svg_workflow_strip(state)
    assert svg_is_well_formed(svg)
    assert svg == svg_workflow_strip(state)  # deterministic (G2)
    assert 'role="img"' in svg
    assert 'aria-label="Workflow' in svg
    for label in WORKFLOW_STEPS:
        assert label in svg
    assert "<title>" in svg


def test_workflow_strip_rejects_bad_state():
    """A59 (69.4.3): a non-list, an empty list, or a malformed row is a
    `ValueError` — the strip never renders a nonsense state."""
    with pytest.raises(ValueError):
        svg_workflow_strip("not a list")
    with pytest.raises(ValueError):
        svg_workflow_strip([])
    with pytest.raises(ValueError):
        svg_workflow_strip([{"step": "Data"}])  # missing `status`
    with pytest.raises(ValueError):
        svg_workflow_strip([{"step": "Data", "status": "maybe"}])


# --- 69.4.4 the next-steps advice (A59) ---------------------------------------

def test_next_steps_pass():
    """A59 (69.4.4): `PASS` leads with the two always-on lines (export,
    compare) and names the seed-confirmation lever; deterministic."""
    lines = next_steps("PASS")
    assert lines[0].startswith("Export & artifacts")
    assert "Compare" in lines[1]
    assert any("seeds" in line for line in lines)
    assert lines == next_steps("PASS")  # equal inputs, equal list


def test_next_steps_miss_and_stopped():
    """A59 (69.4.4): `MISS` names the budget/steering levers; a stopped
    run appends the partial-run line; unknown verdicts are a
    `ValueError` (never guess a verdict)."""
    miss = next_steps("MISS")
    assert any("budget" in line for line in miss)
    assert any("steer" in line.lower() for line in miss)
    stopped = next_steps("MISS", finished_reason="stopped")
    assert stopped[-1].startswith("The run was stopped")
    assert len(stopped) == len(miss) + 1
    with pytest.raises(ValueError):
        next_steps("MAYBE")


# --- 69.3.1 the SVG well-formedness guard (A59) -------------------------------

def test_svg_guard_accepts_and_rejects():
    """A59 (69.3.1): well-formed SVG passes, truncated/missing markup
    does not — the app's degrade branch is keyed on this exact test."""
    good = svg_workflow_strip(workflow_state(True, False, True))
    assert svg_is_well_formed(good)
    assert not svg_is_well_formed("<svg></svg")  # truncated close
    assert not svg_is_well_formed("")
    assert not svg_is_well_formed("not svg at all")


# --- 69.1 / 69.2 / 69.4 the app (A59) ------------------------------------------

def _tab_labels(at) -> list[str]:
    """The raw-tree walk for `tab` node labels — `at.get("tabs")` is
    empty in this Streamlit build, but the nodes exist with a `.label`
    (the A41 pattern; nested tabs flatten into the same walk)."""
    out: list[str] = []

    def walk(node) -> None:
        if getattr(node, "type", None) == "tab":
            lab = getattr(node, "label", None)
            if lab is not None:
                out.append(lab)
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            for el in ch.values():
                walk(el)

    walk(at.main)
    return out


def _setup_app(at, tmp_path: Path, experiments: int = 2) -> None:
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["experiments"] = experiments
    at.session_state["max_train"] = 5.0
    at.run()
    assert not at.exception


def test_app_idle_renders_workflow_strip(tmp_path):
    """A59 (69.4.3): the idle screen shows the workflow strip (Data is
    the current step) before the Quickstart stops the script — the user
    sees where they are in the procedure before doing anything."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert 'role="img"' in md and "Workflow" in md
    assert "Data" in md and "Results" in md  # all four steps named


def test_app_result_subtabs_and_next_steps(tmp_path):
    """A59 (69.1/69.2/69.4): after a tiny run the tree carries the top
    five tabs, the six result sub-tabs, and the three drain sub-tabs;
    the pinned subheaders survive the regrouping; the Next steps block
    and the Run-tab finished hand-off render."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    # the drain sub-tabs exist in the run's own script tree (the drain
    # renders while its worker record is live, 51.2.3)
    labels_during = _tab_labels(at)
    for tab in DRAIN_TABS:
        assert tab in labels_during, (tab, labels_during)
    # a fresh script run (button un-pressed) — the Run tab's finished
    # hand-off lives in the `elif result is not None` branch, so it needs
    # this second run to appear (the drain is re-runnable, 51.2.3)
    at.run()
    assert not at.exception, at.exception

    labels = _tab_labels(at)
    assert labels[0] == "Setup"
    for tab in TOP_TABS + INNER_TABS:
        assert tab in labels, (tab, labels)

    subs = {s.value for s in at.subheader}
    for h in ("Plots", "Decision views", "Research views", "Conclusions",
              "Parameter inspector", "What-if & comparison",
              "Advanced analysis", "Learning views"):
        assert h in subs, h

    caps = [c.value for c in at.caption]
    assert "Next steps" in caps
    assert any(c.startswith("- ") for c in caps[caps.index("Next steps"):])
    infos = " ".join(i.value for i in at.info)
    assert "Run finished" in infos and "Results tab" in infos


def test_app_malformed_svg_degrades(tmp_path, monkeypatch):
    """A59 (69.3.2): a renderer that emits truncated markup degrades to
    a 'plot unavailable' caption — the app completes, never paints a
    broken fragment, never raises. The patch targets `autorefine.plotting`
    — the script's `from autorefine.plotting import …` re-binds the name
    on every `at.run()` (full script re-execution), so the binding's
    source is the durable seam."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest
    import autorefine.plotting as plotting

    monkeypatch.setattr(
        plotting, "svg_spec_lineage", lambda *a, **k: "<svg></svg")
    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "plot unavailable" in caps


# --- 69.2–69.4 app/core wiring (source scan) (A59) -----------------------------

def test_app_source_wires_69():
    """A59 (69.2–69.4): the app wires the new machinery — the workflow
    import + state/strip/next-steps calls, the SVG guards, and the two
    tab regroupings — and `plotting` defines the guard."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "from autorefine.workflow import (",
        "workflow_state(",
        "svg_workflow_strip(",
        "next_steps(",
        "def _svg_html(",
        "def _svg_block(",
        '"Export & artifacts"',
        '"Parameters & what-if"',
        '"Learning views"',
        '"Advanced analysis"',
        '"Live", "Decision views", "Notes & drill-down"',
    ):
        assert token in src, token
    plot_src = PLOT.read_text(encoding="utf-8")
    assert "def svg_is_well_formed(" in plot_src


# --- 33.1 exports + version, 25.x SPEC index (A59) -----------------------------

def test_exports_and_version():
    """A59 (33.1): the new names are in `autorefine.__all__` and
    resolvable (the package re-exports the very objects); the version
    steps to `0.55.0` in both sources; SPEC §69 cites A59."""
    for name in ("workflow_state", "svg_workflow_strip", "next_steps",
                 "WORKFLOW_STEPS", "STEP_DONE", "STEP_CURRENT",
                 "STEP_TODO", "svg_is_well_formed"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None, name
    import autorefine.workflow as wf
    assert autorefine.workflow_state is wf.workflow_state
    assert autorefine.svg_workflow_strip is wf.svg_workflow_strip
    assert autorefine.next_steps is wf.next_steps
    assert autorefine.WORKFLOW_STEPS is wf.WORKFLOW_STEPS
    assert autorefine.svg_is_well_formed is svg_is_well_formed
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.55.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.55.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 69.6 Acceptance (A59)" in spec
