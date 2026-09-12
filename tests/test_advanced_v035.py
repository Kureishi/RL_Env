"""v0.35 — advanced analysis: the app's three advanced panels over new
pure core (SPEC.md 49, A39, M38).

Covers:

- 49.1 `simulate.what_if_block` — the honest union of the two existing
  CLI gate surfaces: score/train objectives re-gate the logged pool
  (40.2 `what_if`), a model objective evaluates the final best model's
  artifact (37.2 `gate.evaluate`, 49.1.2), the combined verdict is the
  AND of the halves, an empty objective set is a MISS; pure, JSON-safe,
  deterministic (G2);
- 49.2 `advanced.retrain_spec` — one controlled training pass against a
  finished run's reconstructed task with the run's own seed
  (deterministic re-score, 49.2.2); the run dir is read-only (49.2.1);
  a bad dict is a `SpecError`; an unknown task / missing summary is a
  `ValueError`;
- 49.3 `advanced.opposite_policy` — bandit ↔ search, anything else is a
  `ValueError` (49.3.1); and `plotting.svg_frontier_overlay` — the
  two-frontier A/B overlay on a shared axis: legends, one polyline per
  series, the target line only when in range, the all-invalid empty
  case, valid XML, deterministic (G2) (49.3.3);
- 49.4 the app renders the "Advanced analysis" section with the three
  panels wired in and every action behind a unique-keyed button
  (inert until pressed — checked by scanning the app source);
- the A39 round regression — the version stepped to `0.39.0` in both
  sources (33.1).

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy only.
"""
from __future__ import annotations

import json
import math
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    DEFAULT_SPEC,
    SearchPolicy,
    SpecError,
)
from autorefine.advanced import opposite_policy, retrain_spec
from autorefine.gate import Objective
from autorefine.plotting import svg_frontier_overlay
from autorefine.simulate import what_if, what_if_block

REPO = Path(__file__).resolve().parent.parent
APP_SRC = REPO / "src" / "autorefine" / "dashboard_app.py"

# 40.2-shaped log rows: the pool is baseline + experiment (scored); the
# screen row is excluded from the pool (40.2.2).
ENTRIES = [
    {"kind": "baseline", "holdout_score": 60.0,
     "train_seconds": 1.0, "spec_hash": "aaaaaaaabbbbbbbb"},
    {"kind": "experiment", "holdout_score": 80.0,
     "train_seconds": 2.0, "spec_hash": "ccccccccdddddddd"},
    {"kind": "experiment", "holdout_score": 90.0,
     "train_seconds": 8.0, "spec_hash": "eeeeeeeeffffffff"},
    {"kind": "screen", "holdout_score": 50.0,
     "train_seconds": 0.5, "spec_hash": "gggggggghhhhhhhh"},
]

FRONT_A = [{"score": 60.0, "train_seconds": 1.0},
           {"score": 75.0, "train_seconds": 3.0}]
FRONT_B = [{"score": 55.0, "train_seconds": 0.5},
           {"score": 70.0, "train_seconds": 2.0},
           {"score": 82.0, "train_seconds": 5.0}]


# --- 49.1 what_if_block (A39) -------------------------------------------------

def test_what_if_block_score_train_halves():
    """A39 (SPEC.md 49.1.1): score/train objectives reproduce the
    `what_if` (40.2) half exactly — pool, verdict, counterfactual final;
    the model half is absent (None) when no model objective is
    selected; the screen row stays out of the pool (40.2.2)."""
    objs = (Objective("score", ">=", 85.0), Objective("train", "<=", 10.0))
    block = what_if_block(ENTRIES, objs)
    assert block["model"] is None
    wf = block["what_if"]
    assert wf == what_if(ENTRIES, objs)  # the same half, byte-identical
    assert wf["pool"] == 3  # baseline + 2 experiments, not the screen row
    assert wf["passing"] == 1  # only the 90.0 candidate clears 85
    assert block["pass"] is True
    assert wf["final"]["cand"] == 3  # log-order position of the 90.0 row
    # a bar no candidate clears is a MISS on the half (and the block)
    miss = what_if_block(ENTRIES, (Objective("score", ">=", 95.0),))
    assert miss["pass"] is False and miss["what_if"]["passing"] == 0


def test_what_if_block_model_objective():
    """A39 (SPEC.md 49.1.2): a model objective evaluates the final best
    model's artifact size (`model_actual`) — the 37.2 `evaluate` rule;
    a missing actual fails the objective honestly, never guessed; the
    what_if half is None when only model objectives are selected."""
    objs = (Objective("model", "<=", 5000),)
    ok = what_if_block(ENTRIES, objs, model_actual=4000)
    assert ok["pass"] is True and ok["what_if"] is None
    assert ok["model"]["objectives"][0]["pass"] is True
    bad = what_if_block(ENTRIES, objs, model_actual=6000)
    assert bad["pass"] is False
    assert bad["model"]["objectives"][0]["pass"] is False
    missing = what_if_block(ENTRIES, objs, model_actual=None)
    assert missing["pass"] is False  # honest MISS, not a guess


def test_what_if_block_combined_verdict_and_empty():
    """A39 (SPEC.md 49.1.1/49.1.3): the combined verdict is the AND of
    the halves — one half passing, the other failing, is a MISS in both
    directions; an empty objective set is a MISS with both halves
    None."""
    objs = (Objective("score", ">=", 80.0), Objective("model", "<=", 100))
    half_ok = what_if_block(ENTRIES, objs, model_actual=1000)
    assert half_ok["what_if"]["pass"] is True
    assert half_ok["model"]["pass"] is False
    assert half_ok["pass"] is False
    objs2 = (Objective("score", ">=", 99.0), Objective("model", "<=", 10**6))
    other = what_if_block(ENTRIES, objs2, model_actual=1000)
    assert other["what_if"]["pass"] is False
    assert other["model"]["pass"] is True
    assert other["pass"] is False
    assert what_if_block(ENTRIES, ()) == \
        {"pass": False, "what_if": None, "model": None}


def test_what_if_block_pure_and_json_safe():
    """A39 (SPEC.md 49.1.3): pure and deterministic (G2) — same inputs,
    same dict — and the result is JSON-safe (no numpy types)."""
    objs = (Objective("score", ">=", 85.0), Objective("model", "<=", 5000))
    a = what_if_block(ENTRIES, objs, model_actual=4000)
    b = what_if_block(ENTRIES, objs, model_actual=4000)
    assert a == b
    json.dumps(a)  # serializable (49.1.1)


# --- 49.2 retrain_spec (A39) ---------------------------------------------------

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:
    """A tiny finished run (1-experiment budget, the coherency-test
    shape) — the retrain_spec fixture (49.2.1)."""
    runs = tmp_path_factory.mktemp("runs")
    env = AutoRefineEnv(seed=7, budget=Budget(1, 300, 30), runs_dir=runs)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return env.run_dir


def test_retrain_spec_valid_dict_and_read_only(tiny_run):
    """A39 (SPEC.md 49.2.1/49.2.2): a valid dict spec returns a finite,
    non-negative score (task-specific — this fixture's default task is
    cartpole-v1, whose mean_steps score is in [0, max_steps]) with
    gen_score/gen_gap/train_seconds/final_loss; the re-score with the
    run's own seed is deterministic (same score on the second call); the
    run dir is unmodified — no new files."""
    before = sorted(p.name for p in tiny_run.iterdir())
    out = retrain_spec(tiny_run, DEFAULT_SPEC.to_dict())
    assert out["spec"] == DEFAULT_SPEC.to_dict()
    for key in ("score", "gen_score", "gen_gap", "train_seconds",
                "final_loss"):
        v = out[key]
        assert isinstance(v, float) and math.isfinite(v), key
    assert 0.0 <= out["score"]  # finite, non-negative (task-specific range)
    assert isinstance(out["time_capped"], bool)
    out2 = retrain_spec(tiny_run, DEFAULT_SPEC.to_dict())
    assert out2["score"] == out["score"]  # 49.2.2: deterministic
    assert sorted(p.name for p in tiny_run.iterdir()) == before  # 49.2.1


def test_retrain_spec_bad_spec_raises_specerror(tiny_run):
    """A39 (SPEC.md 49.2.1): a bad dict is a `SpecError` (a
    `ValueError` subclass — a loud typed error, not a crash); a
    non-spec input is rejected the same way."""
    with pytest.raises(SpecError):
        retrain_spec(tiny_run, {"architecture": "nope"})
    with pytest.raises(SpecError):
        retrain_spec(tiny_run, {"architecture": [8]})  # missing keys
    with pytest.raises(ValueError):
        retrain_spec(tiny_run, 42)


def test_retrain_spec_unknown_task_and_missing_summary(tmp_path):
    """A39 (SPEC.md 49.2.1/49.2.2): an unknown task name (the
    reconstruction returns None) is a ValueError naming the run dir; a
    dir without summary.json is one too."""
    d = tmp_path / "run-x"
    d.mkdir()
    (d / "summary.json").write_text(
        json.dumps({"task": "no-such-task", "seed": 7}), encoding="utf-8")
    with pytest.raises(ValueError):
        retrain_spec(d, DEFAULT_SPEC.to_dict())
    with pytest.raises(ValueError):
        retrain_spec(tmp_path, DEFAULT_SPEC.to_dict())


# --- 49.3 opposite_policy + svg_frontier_overlay (A39) ------------------------

def test_opposite_policy_mapping():
    """A39 (SPEC.md 49.3.1): bandit ↔ search; anything else (rl, an
    unknown name, a non-string) raises ValueError — rl stays CLI-only."""
    assert opposite_policy("bandit") == "search"
    assert opposite_policy("search") == "bandit"
    for bad in ("rl", "x", 3, None):
        with pytest.raises(ValueError):
            opposite_policy(bad)


def test_svg_frontier_overlay_two_series():
    """A39 (SPEC.md 49.3.3): two named frontiers render both legend
    names, one polyline per series, the point circles, a valid-XML
    <svg> block, and the target line when inside the y-range."""
    svg = svg_frontier_overlay(
        [("bandit", FRONT_A), ("search", FRONT_B)], target=80.0)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    ET.fromstring(svg)  # valid XML (G2)
    assert svg.count("<polyline") == 2
    assert "bandit" in svg and "search" in svg  # the legend
    assert svg.count("<circle") >= len(FRONT_A) + len(FRONT_B)
    assert "target 80.00" in svg  # 80 is inside 55..82


def test_svg_frontier_overlay_target_out_of_range():
    """A39 (SPEC.md 49.3.3): the target line appears only when inside
    the y-range — 99 (above the 55..82 scores) renders no target
    element."""
    svg = svg_frontier_overlay(
        [("a", FRONT_A), ("b", FRONT_B)], target=99.0)
    ET.fromstring(svg)
    assert "target 99.00" not in svg


def test_svg_frontier_overlay_skips_invalid_and_empty_case():
    """A39 (SPEC.md 49.3.3): empty/invalid series are skipped; an
    all-invalid input renders the standard empty header + message (the
    svg_pareto convention, 21.2) with no polylines."""
    svg = svg_frontier_overlay(
        [("a", []), ("b", [{"score": None, "train_seconds": 1.0}])])
    ET.fromstring(svg)
    assert "no frontier points to overlay" in svg
    assert "<polyline" not in svg


def test_svg_frontier_overlay_deterministic():
    """A39 (SPEC.md 49.3.3): pure and deterministic (G2) — same inputs,
    byte-identical SVG."""
    named = [("bandit", FRONT_A), ("search", FRONT_B)]
    assert svg_frontier_overlay(named, target=80.0) == \
        svg_frontier_overlay(named, target=80.0)


# --- 49.4 the app section (A39) -------------------------------------------------

def test_app_renders_advanced_section():
    """A39 (SPEC.md 49.4): the result view contains the
    'Advanced analysis' section — after the learning views (28), before
    the artifacts (23.2) — with the three panels' core calls wired in
    (49.1.4/49.2.3/49.3.4) and every action behind a unique-keyed
    st.button (inert until pressed, 49.4.2)."""
    src = APP_SRC.read_text(encoding="utf-8")
    for token in (
        'st.subheader("Advanced analysis")',
        "from autorefine.advanced import opposite_policy, retrain_spec",
        "from autorefine.gate import Objective, model_size",
        "from autorefine.simulate import what_if_block",
        "svg_frontier_overlay,",  # the plotting import
        "block = what_if_block(",
        "out = retrain_spec(run_dir, spec)",
        "res2 = _ab_rerun(res)",
    ):
        assert token in src, token
    # placement (49.4.1): in the result view the advanced section renders
    # after the Learning views and before the Artifacts. The section's
    # subheader lives in the `_render_advanced` helper (defined after the
    # result view), so the runtime order is fixed by its call site in
    # `_render_result` — assert on that, not on the source-text order.
    result_view = src[src.index("def _render_result("):
                    src.index("def _render_advanced(")]
    pos_adv = result_view.index("_render_advanced(res)")
    assert pos_adv > result_view.index('st.subheader("Learning views")')
    assert pos_adv < result_view.index('st.subheader("Artifacts")')
    # inert-until-pressed (49.4.2): each action sits behind its button
    for button, action in (
        ('key="wi_button"', "block = what_if_block("),
        ('key="spec_button"', "out = retrain_spec(run_dir, spec)"),
        ('key="ab_button"', "res2 = _ab_rerun(res)"),
    ):
        assert button in src, button
        assert src.index(button) < src.index(action), action


# --- A39 round regression -------------------------------------------------------

def test_version_round_v035():
    """A39 (SPEC.md 49.5, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"
