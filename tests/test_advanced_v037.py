"""v0.37 — app interactivity: tabs, stop/cancel, rejection reasons,
accessibility (SPEC.md 51, A41, M40).

Covers:

- 51.2.1 `AutoRefineEnv(stop_check=...)` — the driver stop hook: the
  first `step()` completes normally; once the flag flips, the next
  `step()` returns `done=True` with `info["reason"] == "stopped"` and
  the full artifact set (`summary.json` with
  `finished_reason == "stopped"`, `best_spec.json`, `best_model.npz`,
  a registry entry, 38.1); a further `step()` raises (the done guard);
  a non-callable non-None `stop_check` is a construction-time
  `ValueError`; the default (`None`) env steps unchanged;
- 51.2.2 `DashboardRunner.request_stop()` — the flag is honored at the
  next `next()`: `u["done"]` + `u["reason"] == "stopped"`, and
  `finish()["finished_reason"] == "stopped"` with the honest verdict;
- 51.3.1 `accounting.candidate_reason` — the five branches
  (accepted / unscored → `dup` / score ≤ best → `score` / above-best +
  gap > 0.05·score → `overfit` / above-best + small gap → `ci`), the
  18.5 boundary (strict `>`), `best_before=None` skipping the score
  branch, and row-for-row agreement with the `_rejections` buckets
  (the 39.2.2 priority);
- 51.4.1/51.4.2/51.4.3 `plotting` — for each of the eight multi-series
  SVG functions: the default call is byte-identical with
  `palette="default", dark=False` explicit (G2), `palette="okabe"`
  uses the `OKABE_ITO` hexes (the default output uses none), an
  unknown palette is a `ValueError`; `dark=True` carries the
  `#0e1117` background + `#c9d1d9` axis (the default neither); every
  default-call SVG carries `role="img"` + an `aria-label`;
- 51.1.1/51.2.3/51.3.2/51.4.4 the app — the five tab labels render; a
  run completes end-to-end (verdict, `download_button >= 4`,
  `metric == 4`) with the live table carrying the reason column; a Stop
  press honored by the worker yields `finished_reason == "stopped"` +
  the neutral warning; the `cb_palette` checkbox exists and, when on,
  the result markdown carries the Okabe-Ito hexes; the app source
  wires `st.tabs`, `candidate_reason`, `request_stop`, and the
  `stop_check` hand-off;
- the A41 round regression — the version stepped to `0.39.0` in both
  sources (33.1) and SPEC carries the A41 block + M40 row.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy core; streamlit is optional (the app tests skip without
it).
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.accounting import _rejections, candidate_reason
from autorefine.dashboard import DashboardRunner
from autorefine.plotting import (
    OKABE_ITO,
    svg_frontier_overlay,
    svg_mutation_timeline,
    svg_run_curves,
    svg_score_curve,
    svg_score_gap_scatter,
    svg_score_strip,
    svg_seed_curves,
    svg_seed_variance,
)
from autorefine.registry import load_registry

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


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


# 51.4 inputs — one valid (non-empty) input per wrapped function, built
# from the shapes each renderer parses (21.2 log rows, 23.1 update
# stream, 49.3.3 frontier pairs, 50.1.3 curves, 29.1/30.1 seed sweeps)
ROWS = [
    {"kind": "baseline", "holdout_score": 60.0, "train_seconds": 0.5},
    {"kind": "experiment", "holdout_score": 80.0, "train_seconds": 1.0,
     "accepted": True},
]
UPDATES = [
    {"index": 1, "accepted": True, "candidate_score": 70.0, "gen_gap": 0.4,
     "mutation": ["hidden_dim"], "reason": None},
    {"index": 2, "accepted": False, "candidate_score": 55.0, "gen_gap": 0.8,
     "mutation": ["lr"], "reason": None},
    {"index": 3, "accepted": False, "candidate_score": None, "gen_gap": None,
     "mutation": [], "reason": "duplicate"},
]
NAMED = [
    ("a", [{"score": 60.0, "train_seconds": 1.0},
           {"score": 70.0, "train_seconds": 2.0}]),
    ("b", [{"score": 55.0, "train_seconds": 1.5},
           {"score": 65.0, "train_seconds": 2.5}]),
]
RUNNAMED = [("a", [60.0, 75.0]), ("b", [50.0, 80.0, 90.0])]
SEEDS = [
    {"seed": 1, "baseline": 60.0, "final": 70.0, "curve": [60.0, 70.0],
     "target": 95.0, "pass": False},
    {"seed": 2, "baseline": 60.0, "final": 65.0, "curve": [60.0, 65.0],
     "target": 95.0, "pass": False},
]
# (function, args) — the eight 51.4.1 multi-series SVGs
PALETTE_CASES = (
    (svg_score_curve, (ROWS,)),
    (svg_score_strip, (UPDATES,)),
    (svg_score_gap_scatter, (UPDATES,)),
    (svg_mutation_timeline, (UPDATES,)),
    (svg_frontier_overlay, (NAMED,)),
    (svg_run_curves, (RUNNAMED,)),
    (svg_seed_variance, (SEEDS,)),
    (svg_seed_curves, (SEEDS,)),
)
# 51.3.1: the 39.2.2 agreement log — baseline 60, an accepted 70 (the
# running best), then three scored rejections: below-best, above-best
# over the 18.5 gap, above-best under the gap
AGREE_LOG = [
    {"kind": "baseline", "holdout_score": 60.0, "accepted": True,
     "gen_gap": 0.0, "train_seconds": 0.5},
    {"kind": "experiment", "holdout_score": 70.0, "accepted": True,
     "gen_gap": 0.2, "train_seconds": 1.0},
    {"kind": "experiment", "holdout_score": 50.0, "accepted": False,
     "gen_gap": 0.5, "train_seconds": 1.0},
    {"kind": "experiment", "holdout_score": 80.0, "accepted": False,
     "gen_gap": 6.0, "train_seconds": 1.0},
    {"kind": "experiment", "holdout_score": 85.0, "accepted": False,
     "gen_gap": 1.0, "train_seconds": 1.0},
]


# --- 51.2.1 stop core — the env hook (A41) -------------------------------------

def test_env_stop_check_flip_between_steps(tmp_path):
    """A41 (SPEC.md 51.2.1): a flip-on `stop_check` completes the first
    `step()` normally; once the flag flips, the next `step()` returns
    `done=True` with `info["reason"] == "stopped"` and the full
    artifact set (summary / best_spec / best_model / registry entry),
    and a further `step()` raises (the done guard)."""
    flag = {"stop": False}
    env = AutoRefineEnv(seed=7, budget=Budget(5, 300, 30), runs_dir=tmp_path,
                        stop_check=lambda: flag["stop"])
    policy = SearchPolicy(seed=7)
    state = env.reset()
    state, reward, done, info = env.step(policy.propose(state))
    assert done is False, "the first step completes normally"
    assert info.get("reason") != "stopped"
    flag["stop"] = True
    state, reward, done, info = env.step(policy.propose(state))
    assert done is True
    assert info["accepted"] is False
    assert info["reason"] == "stopped"
    assert info["candidate_score"] is None
    assert reward == 0.0
    # the full artifact set (51.2.1) — the normal _finish path
    run_dir = Path(env.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["finished_reason"] == "stopped"
    assert (run_dir / "best_spec.json").is_file()
    assert (run_dir / "best_model.npz").is_file()
    # the registry entry (38.1) carries the honest reason
    reg = load_registry(str(tmp_path))
    assert reg, "the stopped run still registers (38.1)"
    assert any(e.get("finished_reason") == "stopped" for e in reg)
    # the done guard: a further step() raises
    with pytest.raises(RuntimeError):
        env.step(policy.propose(state))


def test_env_stop_check_non_callable_is_valueerror(tmp_path):
    """A41 (SPEC.md 51.2.1): a non-callable non-None `stop_check` is a
    construction-time `ValueError` (the fail-loud style)."""
    for bad in (True, "stop", [1, 2]):
        with pytest.raises(ValueError):
            AutoRefineEnv(seed=7, budget=Budget(2, 300, 30),
                          runs_dir=tmp_path, stop_check=bad)


def test_env_stop_check_default_unchanged(tmp_path):
    """A41 (SPEC.md 51.2.1): the default (`None`) env steps without
    consulting — the A1–A40 loop pins' default path is untouched."""
    env = AutoRefineEnv(seed=7, budget=Budget(1, 300, 30), runs_dir=tmp_path)
    assert env.stop_check is None
    policy = SearchPolicy(seed=7)
    state = env.reset()
    state, _r, done, info = env.step(policy.propose(state))
    assert info.get("reason") != "stopped"  # never stopped on the default path


# --- 51.2.2 stop core — the runner hook (A41) ----------------------------------

def test_runner_request_stop(tmp_path):
    """A41 (SPEC.md 51.2.2): `DashboardRunner.request_stop()` before a
    `next()` ends the loop with `finish()["finished_reason"] ==
    "stopped"` and the honest verdict (PASS/MISS by the gate)."""
    csv = _write_csv(tmp_path)
    runner = DashboardRunner(
        csv_path=csv, seed=7, experiments=3, max_train_seconds=5.0,
        runs_dir=str(tmp_path / "runs"))
    info = runner.start()
    runner.request_stop()
    u = runner.next()
    assert u["done"] is True
    assert u["reason"] == "stopped"
    res = runner.finish()
    assert res["finished_reason"] == "stopped"
    assert res["verdict"] in ("PASS", "MISS")  # the honest gate (51.2.2)
    run_dir = Path(info["run_dir"])
    assert (run_dir / "best_spec.json").is_file()
    assert (run_dir / "best_model.npz").is_file()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["finished_reason"] == "stopped"


def test_runner_request_stop_after_done_is_noop(tmp_path):
    """A41 (SPEC.md 51.2.4): a Stop press after the loop has already
    ended is a no-op — `finish()` stays valid and the reason is the
    loop's own, not `stopped`."""
    csv = _write_csv(tmp_path)
    runner = DashboardRunner(
        csv_path=csv, seed=7, experiments=1, max_train_seconds=5.0,
        runs_dir=str(tmp_path / "runs"))
    runner.start()
    while not runner.done:
        runner.next()
    runner.request_stop()  # no-op after done (51.2.4)
    res = runner.finish()
    assert res["finished_reason"] != "stopped"
    assert res["verdict"] in ("PASS", "MISS")


# --- 51.3.1 rejection reasons — the core verdict (A41) --------------------------

def test_candidate_reason_five_branches():
    """A41 (SPEC.md 51.3.1): the five branches — accepted; unscored →
    `dup`; score ≤ best → `score` (inclusive); above-best + gap >
    0.05·score → `overfit` (strict `>`; the boundary is `ci`);
    above-best + small gap → `ci`; `best_before=None` skips the score
    branch."""
    assert candidate_reason(True, 80.0, 1.0, 60.0) == "accepted"
    assert candidate_reason(False, None, None, 60.0) == "dup"
    assert candidate_reason(False, "n/a", 1.0, 60.0) == "dup"  # non-numeric
    assert candidate_reason(False, 50.0, 0.5, 60.0) == "score"
    assert candidate_reason(False, 60.0, 0.5, 60.0) == "score"  # == best
    assert candidate_reason(False, 80.0, 6.0, 60.0) == "overfit"  # 6 > 4
    assert candidate_reason(False, 80.0, 4.0, 60.0) == "ci"  # boundary: strict >
    assert candidate_reason(False, 80.0, 1.0, 60.0) == "ci"
    assert candidate_reason(False, 80.0, None, None) == "ci"
    assert candidate_reason(False, 80.0, 6.0, None) == "overfit"


def test_candidate_reason_agrees_with_rejections():
    """A41 (SPEC.md 51.3.1): the scored-rejection branching agrees with
    the `_rejections` buckets (39.2.2) row for row over a synthetic log
    (scored rejections only)."""
    # replay the running best exactly as `_rejections` does (39.2.2):
    # the baseline seeds it, an accepted candidate raises it
    best = 60.0
    verdicts = []
    for e in AGREE_LOG:
        if e["kind"] == "baseline":
            best = e["holdout_score"]
            continue
        if e["accepted"]:
            best = e["holdout_score"]
            continue
        verdicts.append(candidate_reason(
            False, e["holdout_score"], e["gen_gap"], best))
    assert verdicts == ["score", "overfit", "ci"]
    # the same log through the report's bucketing (39.2.2) — one hit
    # per bucket, row for row
    recs = _rejections(AGREE_LOG)
    assert recs["accepted"] == 2
    assert (recs["score"], recs["overfit"], recs["ci"]) == (1, 1, 1)
    assert recs["duplicate"] == 0  # free rejections are unlogged (39.2.1)


def test_candidate_reason_exported():
    """A41 (SPEC.md 51.3.1): `candidate_reason` is in
    `accounting.__all__` (35.1-style leaf module)."""
    import autorefine.accounting as accounting
    assert "candidate_reason" in accounting.__all__


# --- 51.4.1 Okabe-Ito palette over the eight SVGs (A41) -------------------------

def test_palette_default_byte_identical():
    """A41 (SPEC.md 51.4.1): for each of the eight functions the default
    call is byte-identical with `palette="default", dark=False` explicit
    (G2 — the A16/A18/A19/A20/A39 pinned hexes stay pinned)."""
    for fn, args in PALETTE_CASES:
        assert fn(*args) == fn(*args, palette="default", dark=False), \
            fn.__name__


def test_palette_okabe_hexes():
    """A41 (SPEC.md 51.4.1): `palette="okabe"` output uses the
    `OKABE_ITO` hexes; the default output uses none of them."""
    for fn, args in PALETTE_CASES:
        default = fn(*args)
        okabe = fn(*args, palette="okabe")
        assert any(h in okabe for h in OKABE_ITO), fn.__name__
        assert not any(h in default for h in OKABE_ITO), fn.__name__


def test_palette_unknown_value_error():
    """A41 (SPEC.md 51.4.1): an unknown palette is a `ValueError` for
    each of the eight functions."""
    for fn, args in PALETTE_CASES:
        with pytest.raises(ValueError):
            fn(*args, palette="nope")


# --- 51.4.2 dark mode over the eight SVGs (A41) ---------------------------------

def test_dark_mode_colors():
    """A41 (SPEC.md 51.4.2): `dark=True` output carries the `#0e1117`
    background and the `#c9d1d9` axis/text color; the default output
    carries neither."""
    for fn, args in PALETTE_CASES:
        default = fn(*args)
        dark = fn(*args, dark=True)
        assert "#0e1117" in dark and "#c9d1d9" in dark, fn.__name__
        assert "#0e1117" not in default and "#c9d1d9" not in default, \
            fn.__name__


# --- 51.4.3 ARIA on every SVG (A41) ---------------------------------------------

def test_aria_on_every_default_svg():
    """A41 (SPEC.md 51.4.3): every default-call SVG from the eight
    functions carries `role="img"` and an `aria-label` (always — the
    screen reader gets the chart's title)."""
    for fn, args in PALETTE_CASES:
        out = fn(*args)
        assert 'role="img"' in out, fn.__name__
        assert "aria-label=" in out, fn.__name__


# --- 51.1.1 / 51.2.3 / 51.3.2 / 51.4.4 the app (A41) ----------------------------

def _tab_labels(at) -> list[str]:
    """The raw-tree walk for `tab` node labels — `at.get("tabs")` is
    empty in this Streamlit build, but the nodes exist with a `.label`
    (51.1.1)."""
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
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(experiments)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception


def test_app_five_tabs_render(tmp_path):
    """A41 (SPEC.md 51.1.1): the app renders its content as the five
    `st.tabs` — Setup / Run / Results / Compare / Experiments (the
    sidebar + the idle screen unchanged)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path)
    assert _tab_labels(at) == ["Setup", "Run", "Results", "Compare",
                               "Experiments"]


def test_app_run_end_to_end_reason_column(tmp_path):
    """A41 (SPEC.md 51.1.2/51.3.2): a run completes end-to-end inside
    the tabs (verdict, `download_button >= 4`, `metric == 4`) and the
    live table carries the 51.3.2 reason column (the running-best
    verdict per candidate)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    # the honest verdict (22.1 gate) in the Results tab
    verdict = " ".join(e.value for e in list(at.success) + list(at.error))
    assert "PASS:" in verdict or "MISS:" in verdict, verdict
    # the artifacts (23.2) + the one-click exports (50.2)
    assert len(at.get("download_button")) >= 4
    assert len(at.metric) == 4
    # the 51.3.2 reason column — the live table (or its Experiments-tab
    # twin) carries it; values are the gate verdicts + the baseline row
    allowed = {"—", "accepted", "score", "overfit", "ci", "dup", "stopped"}
    found = False
    for e in at.dataframe:
        df = e.value
        if "reason" in getattr(df, "columns", []):
            found = True
            vals = {str(v) for v in df["reason"]}
            assert vals <= allowed, vals
            assert "—" in vals and len(vals) >= 2, vals
    assert found, "the live table carries the reason column (51.3.2)"


def test_app_stop_button_honored(tmp_path):
    """A41 (SPEC.md 51.2.3): a Stop press (with Run) is honored by the
    worker — the run finishes `stopped` with the neutral warning next
    to the honest verdict (51.2.4 semantics: between experiments)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path, experiments=3)
    at.button(key="stop_button").set_value(True)
    at.button(key="run_button").set_value(True)
    at.run()
    assert not at.exception, at.exception
    warn = " ".join(e.value for e in at.warning)
    assert "Stopped by user" in warn, warn
    # the partial run's artifacts are honest (51.2.1)
    dirs = sorted((tmp_path / "runs").glob("csv-seed7-*"))
    assert dirs, "the stopped run wrote its run dir"
    summary = json.loads(
        (dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["finished_reason"] == "stopped"
    assert (dirs[-1] / "best_spec.json").is_file()
    assert (dirs[-1] / "best_model.npz").is_file()


def test_app_cb_palette_checkbox_renders_okabe(tmp_path):
    """A41 (SPEC.md 51.4.4): the Setup-tab `cb_palette` checkbox
    (default off) exists and, when on, the result view's multi-series
    SVGs re-render from the stored data with the Okabe-Ito hexes
    (a view preference, not a run knob — 48.1 unchanged)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path)
    assert any(e.key == "cb_palette" for e in at.checkbox), \
        "the Setup-tab cb_palette checkbox exists"
    at.checkbox(key="cb_palette").set_value(True)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert any(h in md for h in OKABE_ITO), \
        "the result SVGs re-render Okabe-Ito when cb_palette is on"


def test_app_source_wiring():
    """A41 (SPEC.md 51.1.1/51.2.3/51.3.2/51.4.4): the app source wires
    `st.tabs` with the five labels, the reason column via
    `candidate_reason`, the Stop button → `request_stop` → the seeded stop
    flag (`initial_stop`, the app-side of the core `stop_check` hand-off
    in `dashboard.py`), and the `cb_palette` checkbox."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "st.tabs(",
        '"Setup", "Run", "Results", "Compare", "Experiments"',
        "candidate_reason(",
        "request_stop",
        "initial_stop",
        'key="stop_button"',
        'key="cb_palette"',
        'key="run_button"',
    ):
        assert token in src, token


# --- A41 round regression ---------------------------------------------------------

def test_version_round_v037():
    """A41 (SPEC.md 51.5, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.41.0"


def test_spec_cites_a41_and_round():
    """A41 (SPEC.md 51.5/51.6): SPEC.md defines the A41 acceptance block
    and the M40 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 42)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 51.5 Acceptance (A41)" in spec
    assert re.search(r"^\s*\| M40 \| v0\.37\s*\|\s*51\s*\|\s*A41\s*\|",
                     spec, re.MULTILINE)
    assert "**M40**" in spec
