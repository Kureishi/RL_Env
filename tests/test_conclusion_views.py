"""The "conclusion surfaces" round — four *pure derivations* that turn the
history already logged (``experiments.jsonl`` + ``summary.json``) into direct
answers about a finished run:

- **C1 — the run verdict card** — ``research.run_verdict(summary, entries)``
  (target vs final best, the champion's CI = the last *accepted* scored row
  whose ``holdout_score`` equals the final best, ``z`` defaulting to the gate's
  ``z_accept`` = 1.0, ``margin`` / ``pass`` None without a target or a best) +
  ``svg_run_verdict`` (the 0–100 axis, the PASS/MISS verdict line, the CI band,
  the empty state);
- **C2 — the decisive-knob ranking** — ``research.knob_signal(entries)``
  (per-field score-impact spread = best value mean − worst value mean over the
  57.2 value→mean surface, tiered decisive / weak / noise / untried by the fixed
  ``KNOB_SPREAD_*`` thresholds, plus the field's mutation wins/trials; ordered
  tier → spread desc → field name) + ``svg_knob_signal`` (the [worst, best]
  interval bars, the star at the best value, the tier chips, the empty state);
- **C3 — the efficiency knee** — ``research.frontier_knee(entries, ...)``
  (built on ``frontier3``: the **best** = max-score point, the **knee** = the
  frontier point farthest from the endpoint segment in normalized space,
  ``delta_score`` / ``time_saving_pct``; knee None with < 2 frontier points) +
  ``svg_efficiency_knee`` (the frontier polyline, the best + efficient markers,
  the trade caption, the empty state);
- **C4 — the rejection anatomy** — ``research.rejection_anatomy(summary,
  entries)`` (the accepted / rejected-by-first-gate mix read through the public
  ``accounting.account_run``, the dominant gate, the time-to-first-improvement,
  and the deterministic one-line stall story) + ``svg_rejection_anatomy`` (the
  stacked bar + legend, the empty state);
- **57.5-style app surface + exports** — the "Conclusions" block in
  ``dashboard_app._render_result`` (the four renderers over the four shapers,
  the friendly-error path), the eight names in ``autorefine.__all__``, and one
  end-to-end AppTest (streamlit optional).

House rules (A47): no cross-test imports (all fixtures synthesized here); the
pure core (``research.py`` + the four ``plotting`` renderers) is tested by
hand-computation; the app is a thin renderer tested by source-token assertions
plus one optional end-to-end AppTest. Every renderer output is validated as XML
(``xml.dom.minidom``); the default-theme output is deterministic (G2), and the
okabe / dark re-render + the bad-palette ``ValueError`` are covered (A46).
"""
from __future__ import annotations

import re
import tomllib
import xml.dom.minidom as minidom
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import (
    svg_efficiency_knee,
    svg_knob_signal,
    svg_rejection_anatomy,
    svg_run_verdict,
)
from autorefine.research import (
    KNOB_SPREAD_DECISIVE,
    KNOB_SPREAD_WEAK,
    frontier_knee,
    knob_signal,
    rejection_anatomy,
    run_verdict,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


def _parse(svg: str) -> None:
    minidom.parseString(svg)  # valid XML (G2)


# --- C1 the run verdict card (A47) ---------------------------------------------

VERDICT_SUMMARY = {
    "final_best_score": 98.0,
    "baseline_score": 85.0,
    "run_config": {"target": 95.0, "z_accept": 2.0},
    "finished_reason": "target_reached",
    "improvement_factor": 1.15,
}
VERDICT_ENTRIES = [
    {"kind": "baseline", "accepted": True, "holdout_score": 85.0, "std": 1.5},
    {"kind": "experiment", "accepted": True, "holdout_score": 98.0, "std": 0.5},
]


def test_run_verdict_hand_computed():
    """A47 (C1): the champion's CI is the ``std`` of the last *accepted*
    scored row whose ``holdout_score`` equals the final best (here the
    accepted candidate at 98, not the baseline at 85); ``margin = best −
    target`` and ``pass = margin >= 0``; ``z`` reads ``run_config.z_accept``."""
    v = run_verdict(VERDICT_SUMMARY, VERDICT_ENTRIES)
    assert v == {
        "target": 95.0,
        "baseline": 85.0,
        "best": 98.0,
        "std": 0.5,
        "z": 2.0,
        "margin": 3.0,
        "pass": True,
        "finished_reason": "target_reached",
        "improvement_factor": 1.15,
    }


def test_run_verdict_miss_no_target_and_empty():
    """A47 (C1): a miss flips ``pass``; without a target ``margin``/``pass``
    are None; ``z`` falls back to the gate default 1.0 when ``z_accept`` is
    absent; the champion ``std`` still comes from the matching accepted row;
    an empty run is fully None (pure, deterministic — G2)."""
    # miss: best below target
    v = run_verdict({"final_best_score": 90.0,
                    "run_config": {"target": 95.0}},
                    [{"kind": "baseline", "accepted": True,
                      "holdout_score": 90.0, "std": 2.0}])
    assert v["target"] == 95.0 and v["best"] == 90.0
    assert v["std"] == 2.0 and v["z"] == 1.0
    assert v["margin"] == -5.0 and v["pass"] is False

    # no target: margin / pass are None (best is still reported)
    v = run_verdict({"final_best_score": 90.0},
                    [{"kind": "baseline", "accepted": True,
                      "holdout_score": 90.0, "std": 2.0}])
    assert v["target"] is None and v["best"] == 90.0
    assert v["margin"] is None and v["pass"] is None and v["z"] == 1.0

    # empty run: fully None
    assert run_verdict({}, []) == {
        "target": None, "baseline": None, "best": None, "std": None,
        "z": 1.0, "margin": None, "pass": None,
        "finished_reason": None, "improvement_factor": None,
    }


def test_svg_run_verdict_render():
    """A47 (C1): the verdict SVG — ARIA, the PASS verdict line with its CI
    tooltip, the finished-reason caption, the MISS line, the no-target line,
    and the empty state; all valid XML."""
    s = svg_run_verdict(run_verdict(VERDICT_SUMMARY, VERDICT_ENTRIES))
    _parse(s)
    assert 'aria-label="run verdict card"' in s
    assert "PASS — +3.0 vs target 95.0" in s
    assert "final best 98.00 · target 95.0 · met" in s  # the CI tooltip
    assert "target_reached" in s  # the finished-reason caption

    miss = svg_run_verdict(run_verdict(
        {"final_best_score": 90.0, "run_config": {"target": 95.0}},
        [{"kind": "baseline", "accepted": True, "holdout_score": 90.0}]))
    _parse(miss)
    assert "MISS — -5.0 vs target 95.0" in miss

    no_target = svg_run_verdict(run_verdict({"final_best_score": 90.0},
                                            [{"kind": "baseline",
                                              "accepted": True,
                                              "holdout_score": 90.0}]))
    _parse(no_target)
    assert "no target set — final best 90.0" in no_target

    empty = svg_run_verdict(run_verdict({}, []))
    _parse(empty)
    assert "no scored run yet" in empty


# --- C2 the decisive-knob ranking (A47) ----------------------------------------

# Disjoint field groups (each entry's spec carries only its field) so the
# per-field value→mean surface is hand-computable without cross-field credit.
KNOB = [
    # w: decisive (spread 10) — values 1→10, 2→20, 3→15
    {"kind": "baseline", "spec": {"w": 1}, "holdout_score": 10.0,
     "accepted": True},
    {"kind": "experiment", "spec": {"w": 2}, "holdout_score": 20.0,
     "accepted": True, "mutation": ["w"]},
    {"kind": "experiment", "spec": {"w": 3}, "holdout_score": 15.0,
     "accepted": False, "mutation": ["w"]},
    # x: weak (spread 3) — values 1→50, 2→53
    {"kind": "baseline", "spec": {"x": 1}, "holdout_score": 50.0,
     "accepted": True},
    {"kind": "experiment", "spec": {"x": 2}, "holdout_score": 53.0,
     "accepted": True, "mutation": ["x"]},
    # z: noise (spread 1) — values 1→40, 2→41
    {"kind": "baseline", "spec": {"z": 1}, "holdout_score": 40.0,
     "accepted": True},
    {"kind": "experiment", "spec": {"z": 2}, "holdout_score": 41.0,
     "accepted": False, "mutation": ["z"]},
    # u: untried (a single value) — value 1→30
    {"kind": "baseline", "spec": {"u": 1}, "holdout_score": 30.0,
     "accepted": True},
]


def test_knob_signal_hand_computed():
    """A47 (C2): per-field ``spread = max(value means) − min(value means)``
    over the 57.2 surface; tiers by the fixed ``KNOB_SPREAD_*`` thresholds
    (decisive ≥ 5, weak ≥ 2, noise above 0, untried = < 2 values); wins /
    trials from the mutation lists; ordered tier → spread desc → field name."""
    k = knob_signal(KNOB)
    assert [f["field"] for f in k["fields"]] == ["w", "x", "z", "u"]
    by = {f["field"]: f for f in k["fields"]}
    # w: decisive — spread 20−10 = 10; best value 2 (mean 20); worst 1 (mean 10)
    assert by["w"]["tier"] == "decisive" and by["w"]["spread"] == 10.0
    assert by["w"]["n_values"] == 3
    assert (by["w"]["best_value"], by["w"]["best_mean"]) == ("2", 20.0)
    assert (by["w"]["worst_value"], by["w"]["worst_mean"]) == ("1", 10.0)
    assert (by["w"]["trials"], by["w"]["wins"]) == (2, 1)
    # x: weak — spread 53−50 = 3
    assert by["x"]["tier"] == "weak" and by["x"]["spread"] == 3.0
    assert (by["x"]["best_value"], by["x"]["worst_value"]) == ("2", "1")
    assert (by["x"]["trials"], by["x"]["wins"]) == (1, 1)
    # z: noise — spread 41−40 = 1
    assert by["z"]["tier"] == "noise" and by["z"]["spread"] == 1.0
    assert (by["z"]["trials"], by["z"]["wins"]) == (1, 0)
    # u: untried — one value, spread 0.0, never mutated
    assert by["u"]["tier"] == "untried" and by["u"]["spread"] == 0.0
    assert by["u"]["n_values"] == 1
    assert (by["u"]["trials"], by["u"]["wins"]) == (0, 0)
    # the thresholds pass through the fixed constants (A47)
    assert k["thresholds"] == {
        "decisive": KNOB_SPREAD_DECISIVE, "weak": KNOB_SPREAD_WEAK}


def test_knob_signal_empty():
    """A47 (C2): no scored specs → ``fields: []`` (pure, deterministic)."""
    assert knob_signal([]) == {"fields": [],
                               "thresholds": {"decisive": KNOB_SPREAD_DECISIVE,
                                              "weak": KNOB_SPREAD_WEAK}}
    assert knob_signal(None) == {"fields": [],
                                 "thresholds": {"decisive": KNOB_SPREAD_DECISIVE,
                                                "weak": KNOB_SPREAD_WEAK}}


def test_svg_knob_signal_render():
    """A47 (C2): the knob-ranking SVG — ARIA, the tier chips, the
    best / worst readouts, the impact caption, and the empty state; valid
    XML."""
    s = svg_knob_signal(knob_signal(KNOB))
    _parse(s)
    assert 'aria-label="decisive-knob ranking"' in s
    for token in ("decisive", "weak", "noise", "impact = best"):
        assert token in s
    # the per-field readouts: best / worst value means + wins / trials
    assert "2" in s and "10.0" in s  # w best value + worst mean
    assert "wins" in s  # the wins/trials readout
    empty = svg_knob_signal({"fields": []})
    _parse(empty)
    assert "no scored specs to rank" in empty


# --- C3 the efficiency knee (A47) ----------------------------------------------

# A monotone score×time frontier (tree family → ``size`` is None, so
# dominance is purely score/time). All three points are non-dominated.
KNEE = [
    {"kind": "baseline", "spec": {"model_family": "tree", "max_depth": 3},
     "spec_hash": "F0", "holdout_score": 60.0, "train_seconds": 1.0,
     "accepted": True},
    {"kind": "experiment", "spec": {"model_family": "tree", "max_depth": 4},
     "spec_hash": "F1", "holdout_score": 90.0, "train_seconds": 3.0,
     "accepted": True, "mutation": ["max_depth"]},
    {"kind": "experiment", "spec": {"model_family": "tree", "max_depth": 5},
     "spec_hash": "F2", "holdout_score": 98.0, "train_seconds": 5.0,
     "accepted": False, "mutation": ["max_depth"]},
]


def test_frontier_knee_hand_computed():
    """A47 (C3): ``best`` is the max-score point (F2, 98 @ 5s); the ``knee``
    is the frontier point farthest from the endpoint segment — the middle
    point F1 (90 @ 3s) sits off the (F0 → F2) line, so it is the efficient
    pick; ``delta_score = 98 − 90 = 8``; ``time_saving_pct = (5−3)/5·100``.
    A single-point frontier has no trade-off → ``knee`` None; empty → empty."""
    fk = frontier_knee(KNEE, state_dim=2, n_out=3)
    assert fk["best"] == 2 and fk["knee"] == 1
    assert fk["delta_score"] == pytest.approx(8.0)
    assert fk["time_saving_pct"] == pytest.approx(40.0)
    # all three points are on the (size-less) frontier
    assert [p["frontier"] for p in fk["points"]] == [True, True, True]
    assert fk["unsized"] == 3  # tree family → no size axis

    # one point → no segment to read a knee off
    fk1 = frontier_knee(KNEE[:1], state_dim=2, n_out=3)
    assert fk1["best"] == 0 and fk1["knee"] is None
    assert fk1["delta_score"] is None and fk1["time_saving_pct"] is None

    # empty
    fke = frontier_knee([])
    assert fke == {"points": [], "best": None, "knee": None, "unsized": 0,
                   "delta_score": None, "time_saving_pct": None}


def test_svg_efficiency_knee_render():
    """A47 (C3): the efficiency-knee SVG — ARIA, the best + efficient markers
    (the star), the trade caption, and the empty state; valid XML."""
    s = svg_efficiency_knee(frontier_knee(KNEE, state_dim=2, n_out=3))
    _parse(s)
    assert 'aria-label="efficiency knee (score vs train time)"' in s
    assert "best" in s and "efficient" in s
    assert "★" in s  # the efficient-knee marker
    empty = svg_efficiency_knee({"points": []})
    _parse(empty)
    assert "efficiency knee (empty)" in empty


# --- C4 the rejection anatomy (A47) --------------------------------------------

# baseline seeds the running best at 50; exp1 (accepted, 60) raises it; the
# three rejections then bucket score / overfit / ci in the 39.2.2 priority.
ANATOMY = [
    {"kind": "baseline", "holdout_score": 50.0, "accepted": True, "ts": 10.0},
    {"kind": "experiment", "holdout_score": 60.0, "gen_gap": 0.5,
     "accepted": True, "ts": 12.0, "mutation": ["a"]},
    {"kind": "experiment", "holdout_score": 55.0, "accepted": False,
     "mutation": ["a"]},
    {"kind": "experiment", "holdout_score": 70.0, "gen_gap": 5.0,
     "accepted": False, "mutation": ["a"]},
    {"kind": "experiment", "holdout_score": 72.0, "gen_gap": 1.0,
     "accepted": False, "mutation": ["a"]},
]
ANATOMY_ALL_ACCEPTED = [
    {"kind": "baseline", "holdout_score": 50.0, "accepted": True},
    {"kind": "experiment", "holdout_score": 60.0, "accepted": True,
     "mutation": ["a"]},
]


def test_rejection_anatomy_hand_computed():
    """A47 (C4): the buckets are read through the public
    ``accounting.account_run`` — the 39.2.2 score → overfit → ci priority
    (55 ≤ best 60 → score; 70 > 60 but gen_gap 5.0 > 0.05·70 → overfit;
    72 > 60, gen_gap 1.0 < 0.05·72 → ci). The dominant gate is the most
    populated bucket (a tie resolves to the priority order → score); the
    time-to-first-improvement is candidate 1 at 12 − 10 = 2s; the stall
    story is the deterministic one-liner."""
    ra = rejection_anatomy({}, ANATOMY)
    assert ra["accepted"] == 2 and ra["rejected"] == 3
    assert ra["by_reason"] == {"score": 1, "overfit": 1, "ci": 1}
    assert ra["dominant"] == "score"
    assert ra["first_improvement"] == {
        "improved": True, "experiment_index": 1, "seconds": 2.0}
    assert ra["story"] == (
        "3 of 5 candidates rejected — mostly the score gate (1/3); "
        "first improvement at candidate 1 (2.0s in) → consider more "
        "budget or a wider search space")


def test_rejection_anatomy_all_accepted_and_empty():
    """A47 (C4): every candidate improved → no dominant gate + the
    "signal on every try" story; an empty run → zeroed buckets + the empty
    story (pure, deterministic — G2)."""
    ra = rejection_anatomy({}, ANATOMY_ALL_ACCEPTED)
    assert ra["accepted"] == 2 and ra["rejected"] == 0
    assert ra["by_reason"] == {"score": 0, "overfit": 0, "ci": 0}
    assert ra["dominant"] is None
    assert ra["story"] == "every candidate improved — signal on every try"

    rae = rejection_anatomy({}, [])
    assert rae["accepted"] == 0 and rae["rejected"] == 0
    assert rae["dominant"] is None
    assert rae["first_improvement"] == {
        "improved": False, "experiment_index": None, "seconds": None}
    assert rae["story"] == "no scored candidates yet"


def test_svg_rejection_anatomy_render():
    """A47 (C4): the rejection-anatomy SVG — ARIA, the accepted / gate
    segment labels + legend, the stall story, and the empty state; valid
    XML."""
    s = svg_rejection_anatomy(rejection_anatomy({}, ANATOMY))
    _parse(s)
    assert 'aria-label="rejection anatomy"' in s
    for token in ("accepted", "score", "overfit", "ci", "rejected"):
        assert token in s
    # the deterministic stall story renders in the SVG
    assert "mostly the score gate (1/3)" in s
    empty = svg_rejection_anatomy(
        {"accepted": 0, "by_reason": {"score": 0, "overfit": 0, "ci": 0}})
    _parse(empty)
    assert "no scored candidates yet" in empty


# --- the renderers: palette / dark / validation (A46) --------------------------

def test_renderers_palette_dark_and_validation():
    """A46 (C1–C4): the four renderers honor the palette / dark contract —
    the okabe + dark re-render is valid XML, and a bad palette is a
    ``ValueError``."""
    verdict = run_verdict(VERDICT_SUMMARY, VERDICT_ENTRIES)
    knobs = knob_signal(KNOB)
    knee = frontier_knee(KNEE, state_dim=2, n_out=3)
    anatomy = rejection_anatomy({}, ANATOMY)
    for fn, data in ((svg_run_verdict, verdict), (svg_knob_signal, knobs),
                     (svg_efficiency_knee, knee),
                     (svg_rejection_anatomy, anatomy)):
        _parse(fn(data, palette="okabe", dark=True))
        with pytest.raises(ValueError):
            fn(data, palette="nope")


# --- the app wiring (A47) -------------------------------------------------------

def test_app_source_wires_conclusions():
    """A47 (C1–C4): the app is a thin renderer — the "Conclusions" subheader
    in ``_render_result``, the four svg renderers over the four shapers, the
    four research + four plotting names imported, and the friendly-error path
    (49.4.3). The palette/dark passthrough feeds every render."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        # the block
        'st.subheader("Conclusions")',
        "svg_run_verdict(run_verdict(_summ_c, _ren), **pal)",
        "svg_knob_signal(knob_signal(_ren), **pal)",
        "svg_efficiency_knee(",
        'frontier_knee(_ren, state_dim=res.get("state_dim"),',
        "svg_rejection_anatomy(rejection_anatomy(_summ_c, _ren),",
        # the core imports
        "    run_verdict,",
        "    knob_signal,",
        "    frontier_knee,",
        "    rejection_anatomy,",
        "    svg_run_verdict,",
        "    svg_knob_signal,",
        "    svg_efficiency_knee,",
        "    svg_rejection_anatomy,",
        # the friendly-error path (49.4.3)
        "conclusions unavailable",
    ):
        assert token in src, token


def test_app_default_path_end_to_end(tmp_path):
    """A47 (C1–C4): the *default* path runs end-to-end — a finished run
    renders the "Conclusions" block (the verdict + knob SVGs among the
    markdown) with no new widget (AppTest-safe)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    # a small classification CSV (quadrant XOR, 2 classes)
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["experiments"] = 2
    at.session_state["max_train"] = 5.0
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    assert at.session_state["result"] is not None
    sub_headers = [str(s.value) for s in at.subheader]
    assert "Conclusions" in sub_headers, sub_headers
    assert any("run verdict" in str(m.value) for m in at.markdown), \
        "the run-verdict SVG renders in the Results view"


# --- the round regression (A47) -------------------------------------------------

def test_all_exports_conclusion_round():
    """A47 (C1–C4, 33.1): the four shapers + the four renderers are exported
    from the package top level and ``__all__``."""
    for name in ("run_verdict", "knob_signal", "frontier_knee",
                 "rejection_anatomy",
                 "svg_run_verdict", "svg_knob_signal", "svg_efficiency_knee",
                 "svg_rejection_anatomy"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_version_round_conclusion():
    """A47 (33.1): the version is pinned and byte-identical in both sources
    (the conclusion surfaces are an additive, pin-safe round)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.57.0"
    assert re.search(r"\bA47\b", Path(__file__).read_text(encoding="utf-8"))
