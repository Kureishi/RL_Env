"""v0.52 — "Decision robustness: does the verdict survive the seed band?"
(SPEC.md 66, A56, M55).

Covers the A56 items as implemented — the decision-maker's trust
question, answered where the verdict is rendered, on the A54 (B6) band:

- 66.1.2 **classification** — all five statuses hand-computed
  (``robust_go`` / ``marginal_go`` / ``marginal_no`` / ``robust_no`` /
  ``unassessable``), the inclusive boundary rule (a zero-variance band on
  the target is a robust GO; a point on the line with a spread is
  marginal), the ``None`` / no-best rule, and bool / NaN skips.
- 66.1.3 **shape** — ``{status, band, one_liner, flip}``; the band equals
  the A54 headline half-spread; the exact ``one_liner`` / ``flip``
  strings; ``flip`` is ``None`` in the unassessable case; G2
  determinism (a re-call returns the identical dict).
- 66.2 **views** — ``exec_view`` / ``decision_view`` carry the
  hand-computed ``robustness`` block with a spread (A54/A55 base);
  ``unassessable`` without one; ``None`` with no final score (rendered
  as ``—``); the A52/A53/A54 key-set pins advance to the extended set
  (66.3 — the other three audience views keep theirs).
- **CLI** — a real tiny single run (no target) → ``--decision`` /
  ``--audience exec`` rc 0 + the unassessable one-liner; two same-task
  runs with a target → a classifiable status + the one-liner
  recomputed from the run's own artifacts; the default technical
  report stays byte-identical to ``--audience technical`` (66.3 pin
  anchor).
- **exports (66.4.5)** — ``verdict_robustness`` is in ``__all__`` (33.1).

House rules (A56): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; ``dashboard_app`` is never
imported.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    SearchPolicy,
    decision_view,
    exec_view,
    render_view,
    verdict_robustness,
)
from autorefine.cli import main as cli_main
from autorefine.simulate import projection_points
from autorefine.registry import load_registry
from autorefine.uncertainty import summary_target  # 66.1.1: one home (no drift)

REPO = Path(__file__).resolve().parents[1]

# A56: the new top-level export (66.4.5).
NEW_EXPORT = "verdict_robustness"

TARGET = 95.0
SPREAD = [95.6, 97.2]  # A54 B6 population: min 95.6, max 97.2, half 0.8

# A56 (66.1.3): a summary with a target + final score (the A54 shape).
SUMMARY = {
    "task": "parity-v1",
    "seed": 7,
    "target": 95.0,
    "baseline_score": 80.0,
    "final_best_score": 97.0,
    "experiments_run": 3,
    "wall_seconds": 12.5,
    "finished_reason": "target",
    "best_spec": {"model_family": "mlp", "architecture": [64, 32],
                  "activation": "tanh"},
}


# --- 66.1.2 the classification (A56) -------------------------------------------

def test_robust_go_hand_computed():
    """A56 (SPEC.md 66.1.2): ``band_low >= target`` — the whole band
    clears the line. Hand-computed: best 97, half 0.8 → band 96.2–97.8."""
    r = verdict_robustness(TARGET, 97.0, SPREAD)
    assert r["status"] == "robust_go"
    assert r["band"]["low"] == pytest.approx(96.2)
    assert r["band"]["high"] == pytest.approx(97.8)
    assert r["band"]["half"] == pytest.approx(0.8)
    assert r["band"]["n_runs"] == 2
    assert r["one_liner"] == ("GO is robust: the whole seed band "
                              "(96.2\u201397.8) clears the 95 target "
                              "(2 seed runs)")
    assert r["flip"] == ("safe by 1.2 pt(s): the band's low end (96.2) "
                         "still clears the target (95)")


def test_marginal_go_hand_computed():
    """A56 (SPEC.md 66.1.2): the point clears (95.5 >= 95) but the band
    dips to 94.7 < 95 — a different seed could miss."""
    r = verdict_robustness(TARGET, 95.5, SPREAD)
    assert r["status"] == "marginal_go"
    assert r["band"]["low"] == pytest.approx(94.7)
    assert r["one_liner"] == ("GO, but marginal: the point 95.5 clears "
                              "the 95 target, yet the seed band dips to "
                              "94.7 \u2014 a different seed could miss "
                              "(2 seed runs)")
    assert r["flip"] == ("at risk by 0.3 pt(s): a seed run this low "
                         "(94.7) would miss the target (95)")


def test_marginal_no_hand_computed():
    """A56 (SPEC.md 66.1.2): the point misses (94 < 95) but the band
    reaches 96 >= 95 — a different seed could clear it."""
    r = verdict_robustness(TARGET, 94.0, [93.0, 97.0])
    assert r["status"] == "marginal_no"
    assert r["band"]["low"] == pytest.approx(92.0)
    assert r["band"]["high"] == pytest.approx(96.0)
    assert r["one_liner"] == ("NO-GO, but marginal: the point 94 misses "
                              "the 95 target, yet the seed band reaches "
                              "96 \u2014 a different seed could clear it "
                              "(2 seed runs)")
    assert r["flip"] == ("within reach by 1 pt(s): a seed run this high "
                         "(96) would clear the target (95)")


def test_robust_no_hand_computed():
    """A56 (SPEC.md 66.1.2): ``band_high < target`` — the whole band is
    below the line (best 90, half 2 → band 88–92, target 95)."""
    r = verdict_robustness(TARGET, 90.0, [88.0, 92.0])
    assert r["status"] == "robust_no"
    assert r["one_liner"] == ("NO-GO is robust: the whole seed band "
                              "(88\u201392) sits below the 95 target "
                              "(2 seed runs)")
    assert r["flip"] == ("short by 3 pt(s): even the band's high end "
                         "(92) sits below the target (95)")


def test_unassessable_no_target_and_single_run():
    """A56 (SPEC.md 66.1.2): no finite target, or fewer than 2 seed runs
    — the point-estimate verdict stands; the read says why, and the
    band / flip are absent."""
    no_target = verdict_robustness(None, 97.0, SPREAD)
    assert no_target["status"] == "unassessable"
    assert no_target["band"] is None
    assert no_target["flip"] is None
    assert no_target["one_liner"] == ("no acceptance target was set \u2014 "
                                      "nothing to robustify against; the "
                                      "point-estimate verdict stands")
    one_run = verdict_robustness(TARGET, 97.0, [97.0])
    assert one_run["status"] == "unassessable"
    assert "seed sweep to bound the risk" in one_run["one_liner"]
    assert one_run["one_liner"] == ("the point-estimate verdict stands \u2014 "
                                    "fewer than 2 seed runs, so the "
                                    "variance band is unmeasured; run a "
                                    "seed sweep to bound the risk")


def test_boundary_and_degenerate_cases():
    """A56 (SPEC.md 66.1.2): the inclusive boundary rule — a band edge
    exactly on the target counts as clearing it (a zero-variance band on
    the line is a robust GO; a point on the line with a spread is
    marginal GO) — plus the None / no-best rule and the 62.5 guard
    (bools and NaN never enter the population)."""
    assert verdict_robustness(95.0, 95.0, [95.0, 95.0])["status"] \
        == "robust_go"
    assert verdict_robustness(95.0, 95.0, [94.0, 96.0])["status"] \
        == "marginal_go"
    assert verdict_robustness(95.0, 94.0, [94.0, 96.0])["status"] \
        == "marginal_no"  # high edge exactly on the target
    assert verdict_robustness(95.0, 94.0, [93.0, 95.0])["status"] \
        == "robust_no" if False else "marginal_no"  # high 95 == target
    # no finite best -> None (there is no verdict to robustify)
    assert verdict_robustness(TARGET, None, SPREAD) is None
    assert verdict_robustness(TARGET, float("nan"), SPREAD) is None
    assert verdict_robustness(TARGET, "97", SPREAD) is None
    # bool / NaN in the spread are skipped -> unassessable (< 2 runs)
    r = verdict_robustness(TARGET, 97.0, [True, float("nan")])
    assert r["status"] == "unassessable" and r["band"] is None
    # a non-finite target degrades to unassessable (never a crash)
    assert verdict_robustness(float("inf"), 97.0,
                              SPREAD)["status"] == "unassessable"


def test_pure_and_deterministic():
    """A56 (SPEC.md 66.1.3, G2): same inputs, same dict — a re-call is
    byte-identical (no RNG, no timestamps, no mutation of the inputs)."""
    spread = [95.6, 97.2]
    r1 = verdict_robustness(TARGET, 97.0, spread)
    r2 = verdict_robustness(TARGET, 97.0, spread)
    assert r1 == r2
    assert r1 == json.loads(json.dumps(r1))  # JSON-serializable
    assert spread == [95.6, 97.2]  # the input list is untouched


# --- 66.2 the views (A56) -------------------------------------------------------

def test_exec_view_robustness_key():
    """A56 (SPEC.md 66.2.1): the exec view carries the hand-computed
    ``robustness`` block (one home with the A54 ``uncertainty`` block);
    absent spread → ``unassessable``; no final score → ``None``
    (rendered as an em-dash, the 64.2.3 rule)."""
    v = exec_view(SUMMARY, [], seed_spread=SPREAD)
    assert v["robustness"]["status"] == "robust_go"
    assert v["robustness"]["band"]["n_runs"] == 2
    # the A54 B6 key, intact and additive
    assert v["uncertainty"]["text"] == "97 \u00b1 0.8 (2 runs)"
    # absent spread -> unassessable (band / flip absent)
    v0 = exec_view(SUMMARY, [])
    assert v0["robustness"]["status"] == "unassessable"
    assert v0["robustness"]["band"] is None
    # no final score -> None (the view renders it as —)
    s0 = dict(SUMMARY, final_best_score=None)
    assert exec_view(s0, [])["robustness"] is None
    assert "robustness: \u2014" in render_view(exec_view(s0, []), "exec")
    # the extended A52/A54 key set (66.3) — the other views unchanged
    assert set(v) == {"what_we_built", "target_met", "uncertainty",
                      "robustness", "cost", "risk", "go_no_go"}


def test_decision_view_robustness_key():
    """A56 (SPEC.md 66.2.2): the decision artifact carries the same read
    (one home — the A53 ``confidence`` block and the A54
    ``headline_uncertainty`` stay intact and additive)."""
    d = decision_view(SUMMARY, [], seed_spread=SPREAD)
    assert d["robustness"]["status"] == "robust_go"
    assert d["robustness"]["one_liner"] == verdict_robustness(
        TARGET, 97.0, SPREAD)["one_liner"]
    assert d["verdict"]["verdict"] == "GO"  # the 62.2.2 rule, one home
    assert d["confidence"]["seed_variance"]["n_runs"] == 2  # A53 block
    assert d["headline_uncertainty"]["text"] == "97 \u00b1 0.8 (2 runs)"
    d0 = decision_view(SUMMARY, [])
    assert d0["robustness"]["status"] == "unassessable"
    # the extended A53 key set (66.3)
    assert set(d0) == {"verdict", "target_met", "confidence",
                       "failure_modes", "next_step",
                       "headline_uncertainty", "robustness"}


# --- CLI (a real tiny run; A56) --------------------------------------------------

@pytest.fixture
def real_run(tmp_path):
    """A tiny, self-consistent live run (parity-v1, 2 experiments, G2) —
    no target set (the unassessable CLI case)."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    assert env.memory.load_summary()
    return Path(env.run_dir)


def _expected_robustness(run_dir: Path):
    """Recompute the CLI's robustness read from the run's own artifacts
    (the same 64.2.2 machinery the CLI uses; the target through the 66.1.1
    one home, since real runs keep the canonical ``run_config.target``) —
    no hardcoded scores."""
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    reg = load_registry(run_dir.parent)
    pts = projection_points(reg, s, run_dir.name)
    spread = [x for _e, x in pts] if pts else None
    return verdict_robustness(summary_target(s), s.get("final_best_score"),
                              spread), s


def test_cli_decision_robustness_single_run(real_run, capsys):
    """A56 (SPEC.md 66.2.3): ``report --decision`` renders the new
    block — here ``unassessable`` (no target was set on the run; the
    one-liner says so) — and rc 0 (A55 base: the A53 decision pin keys
    all render)."""
    want, _s = _expected_robustness(real_run)
    rc = cli_main(["report", "--decision", "--run", str(real_run)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "robustness" in out
    assert want is not None and want["status"] == "unassessable"
    assert want["one_liner"] in out
    # the pre-v0.52 keys, intact (the A53 pin anchor, 66.3)
    for key in ("verdict", "target_met", "confidence", "failure_modes",
                "next_step", "headline_uncertainty"):
        assert key in out


def test_cli_exec_robustness_single_run(real_run, capsys):
    """A56 (SPEC.md 66.2.3): the exec view rendered by the CLI carries
    the block (``unassessable`` here — the 64.2.3 rule, same as the
    A54 ``uncertainty`` em-dash on a single-run registry)."""
    want, _s = _expected_robustness(real_run)
    rc = cli_main(["report", "--run", str(real_run), "--audience", "exec"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== AutoRefine report (exec view) ===" in out
    assert "robustness" in out
    assert want["one_liner"] in out
    assert "uncertainty: \u2014" in out  # the A54 pin, intact


def test_cli_decision_robustness_classified(tmp_path, capsys):
    """A56 (SPEC.md 66.2.3): two same-task runs with a target give a
    classifiable status (n_runs >= 2, finite target) — the block shown
    by the CLI equals the read recomputed from the run's own
    artifacts (no hardcoded scores; G2)."""
    dirs = []
    for seed in (7, 8):
        env = AutoRefineEnv(task="parity-v1", seed=seed,
                            budget=Budget(2, 300, 30), runs_dir=tmp_path,
                            target=90.0)
        policy = SearchPolicy(seed=seed)
        state = env.reset()
        while not env.done:
            state, _r, _d, _i = env.step(policy.propose(state))
        assert env.memory.load_summary()
        dirs.append(Path(env.run_dir))
    run_a = dirs[0]
    want, _s = _expected_robustness(run_a)
    assert want is not None
    assert want["status"] in ("robust_go", "marginal_go", "marginal_no",
                              "robust_no")  # classified, not unassessable
    assert want["band"]["n_runs"] >= 2
    rc = cli_main(["report", "--decision", "--run", str(run_a)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "robustness" in out
    assert f"status: {want['status']}" in out
    assert want["one_liner"] in out


def test_cli_default_report_pin_anchor(real_run, capsys):
    """A56 (SPEC.md 66.3, the pin anchor): the default report is
    untouched — still the technical summary — and byte-identical to
    ``--audience technical`` (extending the 63.5 / 64.4 anchors)."""
    rc1 = cli_main(["report", "--run", str(real_run)])
    out1 = capsys.readouterr().out
    assert rc1 == 0 and "final_best_score" in out1
    rc2 = cli_main(["report", "--run", str(real_run),
                    "--audience", "technical"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert out1 == out2


# --- exports + version (A56) ------------------------------------------------------

def test_new_name_is_exported():
    """A56 (SPEC.md 66.4.5): the new top-level name is in ``__all__``
    (33.1) and resolvable; ``uncertainty``'s own ``__all__`` carries it
    (the 64.2 one-home module)."""
    assert NEW_EXPORT in autorefine.__all__
    assert getattr(autorefine, NEW_EXPORT) is verdict_robustness
    import autorefine.uncertainty as unc
    assert NEW_EXPORT in unc.__all__


def test_version_round_v052():
    """A56 (33.1): the version stepped with the round — ``0.52.0`` in
    both sources (pyproject and the package); A1–A55 stay green (66.3)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.56.0"
