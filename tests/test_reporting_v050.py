"""v0.50 — "B. Reporting for a wider range of audiences — the benchmark
report + uncertainty on headline numbers" (SPEC.md 64, A54, M53).

Covers the A54 items as implemented — the program-of-runs view and the
±-read on the headline, all additive / opt-in (A1–A53 stay green; the
default and ``technical`` report paths are byte-identical, 64.4):

- 64.2.1 **B6 statistics** (``uncertainty.py``, one home) —
  ``seed_spread_stats`` hand-computed (mean / min / max / spread;
  ``None`` on an empty population; a single value → spread 0; bools and
  non-finite values skipped); ``format_pm`` → ``"97 ± 0.8"``;
  ``headline_uncertainty`` → ``None`` with no headline, no population, or
  1 run; with ≥ 2 runs the ``{value, plus_minus, n_runs, text}`` shape
  (half-spread ``plus_minus``, ``text == "97 ± 0.8 (2 runs)"``).
- 64.1.2 **B5 fingerprint** — 12 hex chars; key-order invariant; a
  different spec → a different fp; ``None`` / ``{}`` → ``None``; and the
  top-level ``autorefine.spec_fingerprint`` stays the v0.46 whatif "DNA"
  bars (one name, one binding — 64.1.2).
- 64.1.3 **B5 view** — ``benchmark_view`` on a synthetic registry: the
  per-task leaderboard (best + ``best_run`` with the tie-goes-to-first
  rule, the gate label), the generalization matrix (only ≥ 2 tasks, the
  ``"?"`` bucket for unknown specs, sorted by fp), and the trend
  ``best_so_far`` chains (a non-finite score carrying the previous best
  forward; the ``global`` chain). ``render_benchmark`` +
  ``render_benchmark_md`` carry the section headers, are byte-stable
  (G2), and an empty view does not crash.
- 64.2.2 **B6 views** — the four audience views with ``seed_spread`` →
  the correct hand-computed ``uncertainty`` block; without → ``None``
  (the A52 pin shape: the pre-v0.50 keys unchanged); ``decision_view``
  gains a correct ``headline_uncertainty`` (``None`` with no spread —
  the A53 pin intact); ``user_guide_view(headline=...)`` appends the
  spread line to ``how_to_read`` (``headline=None`` → unchanged).
- **CLI** — a real tiny run → ``report --benchmark --runs-dir DIR`` rc 0
  + the section headers; ``--benchmark --json`` parses with the view
  keys; an empty registry dir → rc 1 + the hint; symmetric exclusions
  rc 1 (``--benchmark`` vs ``--run`` / ``--history`` / ``--user`` /
  ``--format`` / ``--decision`` / ``--what-if`` / a non-technical
  ``--audience``, and the reverse); the default report stays
  byte-identical to ``--audience technical`` (64.4 pin anchor).
- **exports (64.5.5)** — the six new top-level names are in ``__all__``
  (33.1); the B5 spec-identity hash is
  ``autorefine.reporting.spec_fingerprint`` (64.1.2).

House rules (A54): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; ``dashboard_app`` is never
imported.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    SearchPolicy,
    benchmark_view,
    build_view,
    decision_view,
    domain_view,
    exec_view,
    format_pm,
    headline_uncertainty,
    regulator_view,
    render_benchmark,
    render_benchmark_md,
    seed_spread_stats,
    technical_view,
    user_guide_view,
)
from autorefine.cli import main as cli_main
from autorefine.reporting import spec_fingerprint

REPO = Path(__file__).resolve().parents[1]

# A54: the six new top-level exports (64.5.5). ``spec_fingerprint`` is
# deliberately NOT in this list — the top-level name stays the v0.46
# whatif "DNA" bars (one name, one binding, 64.1.2).
NEW_EXPORTS = (
    "benchmark_view", "render_benchmark", "render_benchmark_md",
    "seed_spread_stats", "format_pm", "headline_uncertainty",
)

# Two distinct model specs for the B5 fixtures (64.1.3).
SPEC_A = {"model_family": "mlp", "architecture": [64, 32],
          "activation": "tanh"}
SPEC_B = {"model_family": "knn", "knn_k": 5}

SPREAD = [95.6, 97.2]  # the B6 fixture population: min 95.6, max 97.2
HL_TEXT = "97 \u00b1 0.8 (2 runs)"  # the hand-computed headline read


# --- 64.2.1 B6: the uncertainty helpers (A54) ---------------------------------

def test_seed_spread_stats_hand_computed():
    """A54 (SPEC.md 64.2.1): the descriptive statistics over a population
    of final scores — hand-computed (mean, min, max, spread)."""
    st = seed_spread_stats([95.6, 97.2, 96.4])
    assert st["n_runs"] == 3
    assert st["min"] == pytest.approx(95.6)
    assert st["max"] == pytest.approx(97.2)
    assert st["mean"] == pytest.approx((95.6 + 97.2 + 96.4) / 3)
    assert st["spread"] == pytest.approx(97.2 - 95.6)


def test_seed_spread_stats_degenerates():
    """A54 (SPEC.md 64.2.1): ``None`` on an empty population (``None`` /
    ``[]`` / no finite values); a single value is a valid one-point
    population (spread 0); bools and non-finite values are skipped,
    never averaged."""
    assert seed_spread_stats(None) is None
    assert seed_spread_stats([]) is None
    assert seed_spread_stats(["x", True, float("nan")]) is None
    st = seed_spread_stats([50.0])
    assert st == {"n_runs": 1, "min": 50.0, "max": 50.0, "mean": 50.0,
                  "spread": 0.0}
    st = seed_spread_stats([True, 95.0, 97.0, "x", float("nan")])
    assert st["n_runs"] == 2 and st["min"] == 95.0 and st["max"] == 97.0
    assert st["spread"] == pytest.approx(2.0)


def test_format_pm():
    """A54 (SPEC.md 64.2.1): the compact ``value ± plus_minus`` read
    (``%g`` — no trailing zeros; U+00B1)."""
    assert format_pm(97.0, 0.8) == "97 \u00b1 0.8"
    assert format_pm(100.0, 0.0) == "100 \u00b1 0"


def test_headline_uncertainty_requires_two_runs():
    """A54 (SPEC.md 64.2.1): ``None`` when there is no finite headline
    value, no spread population, or fewer than 2 runs — a single run
    has no variance to display (the A52/A53 pin rule)."""
    assert headline_uncertainty(None, SPREAD) is None
    assert headline_uncertainty(97.0, None) is None
    assert headline_uncertainty(97.0, [95.6]) is None
    assert headline_uncertainty(float("nan"), SPREAD) is None


def test_headline_uncertainty_shape():
    """A54 (SPEC.md 64.2.1): with ≥ 2 runs the ``{value, plus_minus,
    n_runs, text}`` shape — ``plus_minus`` is the *half-spread* and
    ``text`` is the ready-to-print line."""
    hl = headline_uncertainty(97.0, SPREAD)
    assert hl is not None
    assert set(hl) == {"value", "plus_minus", "n_runs", "text"}
    assert hl["value"] == pytest.approx(97.0)
    assert hl["plus_minus"] == pytest.approx((97.2 - 95.6) / 2)
    assert hl["n_runs"] == 2
    assert hl["text"] == HL_TEXT


# --- 64.1.2 B5: the spec identity (A54) ----------------------------------------

def test_spec_fingerprint_rules():
    """A54 (SPEC.md 64.1.2): the spec's identity — first 12 hex chars of
    the SHA-256 of the canonical JSON; key-order invariant; a different
    spec → a different fp; ``None`` / ``{}`` → ``None``."""
    fp = spec_fingerprint(SPEC_A)
    assert re.fullmatch(r"[0-9a-f]{12}", fp)
    reordered = {"activation": SPEC_A["activation"],
                 "model_family": SPEC_A["model_family"],
                 "architecture": [64, 32]}
    assert spec_fingerprint(reordered) == fp
    assert spec_fingerprint(SPEC_B) != fp
    assert spec_fingerprint({"hidden_dim": 64}) != fp
    assert spec_fingerprint(None) is None
    assert spec_fingerprint({}) is None


def test_top_level_spec_fingerprint_stays_the_whatif_bars():
    """A54 (SPEC.md 64.1.2, the naming rule): the top-level
    ``autorefine.spec_fingerprint`` remains the v0.46 whatif "DNA" bars
    (one name, one binding); the B5 identity hash is the module-qualified
    ``autorefine.reporting.spec_fingerprint`` (64.1.2)."""
    bars = autorefine.spec_fingerprint(SPEC_A)
    assert isinstance(bars, list) and all(
        isinstance(r, dict) and "name" in r and "t" in r for r in bars)
    assert isinstance(autorefine.reporting.spec_fingerprint(SPEC_A), str)


# --- 64.1.3 B5: the benchmark view (A54) ---------------------------------------

def _syn_registry() -> list:
    """A synthetic program of runs (7 entries, 4 tasks). Hand-computable:

    - parity-v1: two runs tied at 80.0 (r1, r2) — the tie goes to the
      first in append order (r1); r2's spec is unknown (→ ``"?"``).
    - sine-v1: r3 at 90.0 (SPEC_A) + r6 with a ``None`` score (carries
      the previous best forward in the trend chain).
    - cartpole-v1: r4 at 97.0 (SPEC_B, met) + r5 at 85.0 (unknown spec —
      the ``"?"`` bucket on a second task).
    - gridnav-v1: r7 with a ``None`` score (no finite best → the
      all-``None`` leaderboard row).
    """
    return [
        {"run_id": "r1", "task": "parity-v1", "seed": 7,
         "final_score": 80.0, "target": 95.0, "met_target": False},
        {"run_id": "r2", "task": "parity-v1", "seed": 8,
         "final_score": 80.0, "target": 95.0, "met_target": False},
        {"run_id": "r3", "task": "sine-v1", "seed": 7,
         "final_score": 90.0, "target": 95.0, "met_target": False},
        {"run_id": "r4", "task": "cartpole-v1", "seed": 1,
         "final_score": 97.0, "target": 95.0, "met_target": True},
        {"run_id": "r5", "task": "cartpole-v1", "seed": 2,
         "final_score": 85.0, "target": 95.0, "met_target": False},
        {"run_id": "r6", "task": "sine-v1", "seed": 9,
         "final_score": None, "target": 95.0, "met_target": None},
        {"run_id": "r7", "task": "gridnav-v1", "seed": 3,
         "final_score": None, "target": 95.0, "met_target": None},
    ]


def _syn_spec_map() -> dict:
    return {"r1": SPEC_A, "r3": SPEC_A, "r4": SPEC_B}


def test_benchmark_view_leaderboard():
    """A54 (SPEC.md 64.1.3): the per-task leaderboard — best score + the
    run achieving it (ties keep the *first* in append order), the best
    run's fingerprint, and the gate label (PASS/MISS/—); a task with no
    finite scores degrades to all-``None`` + the ``—`` gate."""
    v = benchmark_view(_syn_registry(), _syn_spec_map())
    lb = {r["task"]: r for r in v["leaderboard"]}
    assert [r["task"] for r in v["leaderboard"]] == sorted(lb)  # sorted
    assert v["n_runs"] == 7
    assert v["tasks"] == ["cartpole-v1", "gridnav-v1", "parity-v1",
                          "sine-v1"]

    c = lb["cartpole-v1"]
    assert c["n_runs"] == 2 and c["best_score"] == pytest.approx(97.0)
    assert c["best_run"] == "r4"
    assert c["spec_fp"] == spec_fingerprint(SPEC_B)
    assert c["met_target"] == "PASS"

    p = lb["parity-v1"]
    assert p["n_runs"] == 2 and p["best_score"] == pytest.approx(80.0)
    assert p["best_run"] == "r1"  # the tie at 80.0 goes to the first run
    assert p["spec_fp"] == spec_fingerprint(SPEC_A)
    assert p["met_target"] == "MISS"

    s = lb["sine-v1"]
    assert s["n_runs"] == 2 and s["best_score"] == pytest.approx(90.0)
    assert s["best_run"] == "r3"  # r6's None score does not count
    assert s["met_target"] == "MISS"

    g = lb["gridnav-v1"]  # no finite score at all
    assert g == {"task": "gridnav-v1", "n_runs": 1, "best_score": None,
                 "best_run": None, "spec_fp": None, "met_target": "\u2014"}


def test_benchmark_view_generalization():
    """A54 (SPEC.md 64.1.3): the same-spec-across-tasks matrix — only
    fingerprints seen on ≥ 2 distinct tasks; runs with an unknown spec
    fall into one shared ``"?"`` bucket; sorted by ``spec_fp``."""
    v = benchmark_view(_syn_registry(), _syn_spec_map())
    gen = v["generalization"]
    # sorted by fp string: "?" (0x3F) sorts before hex digits
    assert [g["spec_fp"] for g in gen] == ["?", spec_fingerprint(SPEC_A)]
    q = gen[0]  # the "?" bucket: r2 (parity) + r5 (cartpole)
    a = gen[1]
    assert a["tasks"] == {"parity-v1": 80.0, "sine-v1": 90.0}
    assert a["runs"] == ["r1", "r3"] and a["n_tasks"] == 2
    assert q["tasks"] == {"cartpole-v1": 85.0, "parity-v1": 80.0}
    assert q["runs"] == ["r2", "r5"] and q["n_tasks"] == 2
    # SPEC_B is on one task only — not a generalization signal
    assert spec_fingerprint(SPEC_B) not in [g["spec_fp"] for g in gen]


def test_benchmark_view_trend():
    """A54 (SPEC.md 64.1.3): the best-score trend — per-task + the
    ``global`` ``best_so_far`` chains; a non-finite score carries the
    previous best forward (``None`` until the first finite one)."""
    v = benchmark_view(_syn_registry(), _syn_spec_map())
    t = v["trend"]
    assert set(t) == {"cartpole-v1", "gridnav-v1", "parity-v1", "sine-v1",
                      "global"}

    assert t["parity-v1"] == [
        {"run_id": "r1", "seed": 7, "score": 80.0, "best_so_far": 80.0},
        {"run_id": "r2", "seed": 8, "score": 80.0, "best_so_far": 80.0}]
    assert t["sine-v1"] == [
        {"run_id": "r3", "seed": 7, "score": 90.0, "best_so_far": 90.0},
        {"run_id": "r6", "seed": 9, "score": None, "best_so_far": 90.0}]
    assert t["cartpole-v1"] == [
        {"run_id": "r4", "seed": 1, "score": 97.0, "best_so_far": 97.0},
        {"run_id": "r5", "seed": 2, "score": 85.0, "best_so_far": 97.0}]
    assert t["gridnav-v1"] == [
        {"run_id": "r7", "seed": 3, "score": None, "best_so_far": None}]
    assert t["global"] == [
        {"run_id": "r1", "seed": 7, "score": 80.0, "best_so_far": 80.0},
        {"run_id": "r2", "seed": 8, "score": 80.0, "best_so_far": 80.0},
        {"run_id": "r3", "seed": 7, "score": 90.0, "best_so_far": 90.0},
        {"run_id": "r4", "seed": 1, "score": 97.0, "best_so_far": 97.0},
        {"run_id": "r5", "seed": 2, "score": 85.0, "best_so_far": 97.0},
        {"run_id": "r6", "seed": 9, "score": None, "best_so_far": 97.0},
        {"run_id": "r7", "seed": 3, "score": None, "best_so_far": 97.0}]


def test_benchmark_view_is_pure_over_its_inputs():
    """A54 (SPEC.md 64.1): absent ``spec_map`` → every fingerprint reads
    ``None`` (rendered as an em-dash, never invented); non-dict entries
    are ignored; a second call is byte-identical (G2)."""
    v1 = benchmark_view(_syn_registry())
    assert all(r["spec_fp"] is None for r in v1["leaderboard"])
    assert [g["spec_fp"] for g in v1["generalization"]] == ["?"]
    v2 = benchmark_view(_syn_registry())
    assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)
    v3 = benchmark_view([{"run_id": "r1"}, "junk", None, 42])
    # junk entries are dropped (n_runs counts the surviving dicts); the
    # lone entry has no task/score, so no leaderboard rows and no tasks
    assert v3["n_runs"] == 1 and v3["leaderboard"] == [] and v3["tasks"] == []


def test_render_benchmark_is_stable_and_complete():
    """A54 (SPEC.md 64.1.4): the text rendering carries the section
    headers, the leaderboard row, the ``"?"`` bucket, and the trend
    chains; a re-render is byte-identical (G2); an empty view does not
    crash."""
    v = benchmark_view(_syn_registry(), _syn_spec_map())
    r1 = render_benchmark(v)
    assert r1 == render_benchmark(v)  # G2 byte-stability
    assert r1.startswith("=== AutoRefine benchmark (longitudinal) ===")
    assert "leaderboard (per task)" in r1
    assert "generalization (same spec across tasks)" in r1
    assert "trend (best score across runs)" in r1
    assert "runs : 7" in r1
    assert "PASS" in r1 and "MISS" in r1
    assert spec_fingerprint(SPEC_A) in r1  # the best run's fingerprint
    assert "?  (2 tasks," in r1            # the unknown-spec bucket
    assert "global: r1: 80" in r1          # the global chain
    assert "|" not in r1                   # no table pipes (terminal)
    e = render_benchmark({})
    assert "(no runs)" in e                # an empty view does not crash


def test_render_benchmark_md():
    """A54 (SPEC.md 64.1.4): the Markdown rendering — ``#`` headings +
    ``|`` tables, byte-stable (G2)."""
    v = benchmark_view(_syn_registry(), _syn_spec_map())
    m1 = render_benchmark_md(v)
    assert m1 == render_benchmark_md(v)
    assert m1.startswith("# AutoRefine benchmark (longitudinal)")
    assert "|" in m1                        # the leaderboard table
    assert "## Leaderboard (per task)" in m1
    assert "## Generalization (same spec across tasks)" in m1
    assert "## Trend (best score across runs)" in m1


# --- 64.2.2 B6: the views' uncertainty block (A54) ------------------------------

def _syn_summary(**over) -> dict:
    """A minimal, hand-computed summary the views can consume (best 97
    vs target 95 — GO)."""
    s = {
        "task": "parity-v1",
        "seed": 7,
        "finished_reason": "target",
        "baseline_score": 80.0,
        "final_best_score": 97.0,
        "improvement_factor": 1.2125,
        "experiments_run": 3,
        "wall_seconds": 12.5,
        "target": 95.0,
        "best_spec": SPEC_A,
    }
    s.update(over)
    return s


def _expected_block() -> dict:
    """The hand-computed B6 block for best=97.0 over SPREAD (64.2.1)."""
    return {"value": 97.0, "plus_minus": (97.2 - 95.6) / 2,
            "n_runs": 2, "text": HL_TEXT}


def test_audience_views_uncertainty_block():
    """A54 (SPEC.md 64.2.2): all four audience views carry the correct
    ``uncertainty`` block with a ``seed_spread`` (one home:
    ``uncertainty.headline_uncertainty``); without → ``None`` (rendered
    as ``—``; the A52 pin shape — the pre-v0.50 keys unchanged)."""
    s = _syn_summary()
    want = _expected_block()
    for view in (exec_view(s, [], seed_spread=SPREAD),
                 domain_view(s, [], seed_spread=SPREAD),
                 technical_view(s, [], seed_spread=SPREAD),
                 regulator_view(s, [], seed_spread=SPREAD)):
        block = view["uncertainty"]
        assert block is not None
        assert set(block) == {"value", "plus_minus", "n_runs", "text"}
        assert block["value"] == pytest.approx(want["value"])
        assert block["plus_minus"] == pytest.approx(want["plus_minus"])
        assert block["n_runs"] == want["n_runs"]
        assert block["text"] == want["text"]
    # absent spread → None on all four (the A52 pin shape)
    assert exec_view(s, [])["uncertainty"] is None
    assert domain_view(s, [])["uncertainty"] is None
    assert technical_view(s, [])["uncertainty"] is None
    assert regulator_view(s, [])["uncertainty"] is None


def test_audience_views_keep_their_pre_v050_keys():
    """A54 (SPEC.md 64.2.3, the A52 pin anchor): the B6 key is additive —
    the pre-v0.50 key sets of the four views are unchanged. (The v0.52
    B7 key ``robustness`` is likewise additive on the exec view only —
    A56, SPEC.md 66.2: the three other views do not render a go/no-go
    verdict, so they keep their key sets intact.)"""
    s = _syn_summary()
    assert set(exec_view(s, [])) == {
        "what_we_built", "target_met", "uncertainty", "robustness",
        "cost", "risk", "go_no_go"}
    assert set(domain_view(s, [])) == {
        "task", "final_score", "uncertainty", "per_class",
        "weakest_class", "calibration", "worked_examples"}
    t = technical_view(s, [])
    assert set(t) == {"task", "seed", "finished_reason", "baseline_score",
                      "final_best_score", "improvement_factor",
                      "experiments_run", "wall_seconds", "best_spec",
                      "uncertainty", "experiments"}
    assert set(regulator_view(s, [])) == {
        "tool", "version", "task", "seed", "target", "config_hash",
        "environment", "data_fingerprint", "chain_of_custody",
        "uncertainty", "decision_trace"}


def test_build_view_forwards_the_spread():
    """A54 (SPEC.md 64.2.2): ``build_view`` forwards ``seed_spread`` to
    every view's ``uncertainty`` block."""
    s = _syn_summary()
    for audience in ("exec", "domain", "technical", "regulator"):
        v = build_view(audience, s, [], seed_spread=SPREAD)
        assert v["uncertainty"] is not None
        assert v["uncertainty"]["text"] == HL_TEXT
        assert build_view(audience, s, [])["uncertainty"] is None


def test_decision_view_headline_uncertainty():
    """A54 (SPEC.md 64.2.2): ``decision_view``'s result gains the
    ``headline_uncertainty`` read (correct with a spread, ``None``
    without — the A53 pin: the pre-v0.50 keys are all still there)."""
    s = _syn_summary()
    d = decision_view(s, [], seed_spread=SPREAD)
    assert d["headline_uncertainty"] is not None
    assert d["headline_uncertainty"]["text"] == HL_TEXT
    assert d["headline_uncertainty"]["n_runs"] == 2
    assert d["headline_uncertainty"]["plus_minus"] == pytest.approx(
        (97.2 - 95.6) / 2)
    d0 = decision_view(s, [])
    assert d0["headline_uncertainty"] is None
    # the A53 key set, intact and additive (the v0.52 B7 ``robustness``
    # key is additive too — A56, SPEC.md 66.2)
    assert set(d0) == {"verdict", "target_met", "confidence",
                       "failure_modes", "next_step",
                       "headline_uncertainty", "robustness"}
    assert d0["verdict"]["verdict"] == "GO"  # 97 >= 95 (62.2.2 rule)


def test_user_guide_view_headline_line():
    """A54 (SPEC.md 64.2.2): ``user_guide_view(headline=...)`` appends
    the spread line to ``how_to_read``; ``headline=None`` leaves the
    pre-v0.50 text byte-identical (the A53 pin)."""
    s = _syn_summary()
    base = user_guide_view(s, [])
    assert "same-task runs" not in base["how_to_read"]
    with_hl = user_guide_view(s, [], headline=_expected_block())
    assert with_hl["how_to_read"] == (
        base["how_to_read"] + f" Across same-task runs the final score "
        f"reads {HL_TEXT}")
    assert with_hl["how_to_read"].endswith(HL_TEXT)
    # a headline with no text degrades to the base text
    assert user_guide_view(s, [], headline={"value": 1.0})["how_to_read"] \
        == base["how_to_read"]


# --- CLI (a real tiny run; A54) -------------------------------------------------

@pytest.fixture
def real_run(tmp_path):
    """A tiny, self-consistent live run (parity-v1, 2 experiments, G2) —
    the registry + ``summary.json`` (with ``best_spec``) land in
    ``tmp_path``."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    assert env.memory.load_summary()
    return Path(env.run_dir)


def test_cli_report_benchmark(real_run, capsys):
    """A54 (SPEC.md 64.1): ``report --benchmark --runs-dir DIR`` rc 0 —
    the section headers, the parity row, and the loaded spec fingerprint
    (from the run's ``summary.json`` ``best_spec``)."""
    rc = cli_main(["report", "--benchmark",
                   "--runs-dir", str(real_run.parent)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("=== AutoRefine benchmark (longitudinal) ===")
    assert "leaderboard (per task)" in out
    assert "generalization (same spec across tasks)" in out
    assert "trend (best score across runs)" in out
    assert "runs : 1" in out and "parity-v1" in out
    assert re.search(r"[0-9a-f]{12}", out)  # the best run's spec fp
    assert "global:" in out                  # the global trend chain


def test_cli_report_benchmark_json(real_run, capsys):
    """A54 (SPEC.md 64.1): ``--benchmark --json`` prints the view model
    (the machine form) — parseable, with the view keys."""
    rc = cli_main(["report", "--benchmark", "--runs-dir",
                   str(real_run.parent), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    view = json.loads(out)
    assert set(view) == {"n_runs", "tasks", "leaderboard",
                         "generalization", "trend"}
    assert view["n_runs"] == 1
    assert view["tasks"] == ["parity-v1"]
    assert view["leaderboard"][0]["task"] == "parity-v1"
    assert "global" in view["trend"]


def test_cli_report_benchmark_empty_registry(tmp_path, capsys):
    """A54 (SPEC.md 64.1.5): an empty registry is rc 1 + the
    "finish at least one run" hint (the 38.3.3 rule)."""
    rc = cli_main(["report", "--benchmark", "--runs-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no runs registered" in err
    assert "finish at least one run" in err


def test_cli_report_benchmark_exclusions(real_run, capsys):
    """A54 (SPEC.md 64.1.5 / 64.3.1): ``--benchmark`` is mutually
    exclusive with ``--run``, ``--history``, and every single-run view —
    and the table is symmetric (``X --benchmark`` is rc 1 too)."""
    cases = (
        (["--benchmark", "--run", str(real_run)],),
        (["--benchmark", "--history"],),
        (["--benchmark", "--user", "--run", str(real_run)],),
        (["--benchmark", "--format", "md", "--run", str(real_run)],),
        (["--benchmark", "--decision", "--run", str(real_run)],),
        (["--benchmark", "--what-if", "score>=90", "--run",
          str(real_run)],),
        (["--benchmark", "--audience", "exec"],),
        (["--user", "--benchmark", "--run", str(real_run)],),
        (["--audience", "exec", "--benchmark", "--run", str(real_run)],),
        (["--what-if", "score>=90", "--benchmark", "--run", str(real_run)],),
    )
    for (flags,) in cases:
        rc = cli_main(["report", *flags])
        err = capsys.readouterr().err
        assert rc == 1, flags
        assert "mutually exclusive" in err, (flags, err)


def test_cli_default_report_pin_anchor(real_run, capsys):
    """A54 (SPEC.md 64.4, the pin anchor): the default report is
    untouched — still the technical summary with ``final_best_score`` —
    and byte-identical to ``--audience technical`` (extending the
    63.5 anchor)."""
    rc1 = cli_main(["report", "--run", str(real_run)])
    out1 = capsys.readouterr().out
    assert rc1 == 0 and "final_best_score" in out1
    rc2 = cli_main(["report", "--run", str(real_run),
                    "--audience", "technical"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert out1 == out2


def test_cli_audience_exec_uncertainty_line(real_run, capsys):
    """A54 (SPEC.md 64.2.2): the exec view rendered by the CLI carries
    the ``uncertainty`` block — here ``—`` (a single run in the registry
    has no spread to display, the 64.2.3 pin rule)."""
    rc = cli_main(["report", "--run", str(real_run), "--audience", "exec"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== AutoRefine report (exec view) ===" in out
    assert "uncertainty: \u2014" in out


# --- exports + version (A54) -----------------------------------------------------

def test_new_names_are_exported():
    """A54 (SPEC.md 64.5.5): the six new top-level names are in
    ``__all__`` and resolvable (33.1); the B5 spec-identity hash is
    ``autorefine.reporting.spec_fingerprint`` (64.1.2)."""
    for name in NEW_EXPORTS:
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None
    assert callable(autorefine.reporting.spec_fingerprint)
    # one name, one binding (64.1.2): the two same-named functions are
    # distinct objects in distinct modules
    assert autorefine.reporting.spec_fingerprint is not \
        autorefine.spec_fingerprint
    assert autorefine.reporting.spec_fingerprint.__module__ == \
        "autorefine.reporting"
    assert autorefine.spec_fingerprint.__module__ == "autorefine.whatif"


def test_version_round_v050():
    """A54 (33.1): the version stepped with the round — ``0.50.0`` in
    both sources (pyproject and the package)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.61.0"
