"""Simulation III tests (SPEC.md 41, A31, v0.27).

S3 (41.1) `report --project`: project the budget to a target — a
Michaelis–Menten saturation curve fit (Lineweaver–Burk OLS) over the
same-task history (the 38.1 registry + the current run, deduped by run
id); CEILING / ~N-more-experiments / insufficient-history verdicts, all
rc 0 (an informational view).

S4 (41.2) `report --trace`: terminal replay — one decision line per
`experiments.jsonl` entry, with the 39.2.2 rejection priority (score
gate -> overfit -> CI gate) and the running best reconstructed exactly
as in 39.2.2 (baseline seeds it, an acceptance raises it, a curriculum
step-up re-pins it).

S5 (41.3) `run --demo`: a tiny fixed parity-v1 loop (3 experiments),
finished in seconds, printed narrated with the 41.2 renderer (one
renderer, two entry points); a demo run is a run (run dir, artifacts,
registry entry).

House rules: stdlib + numpy only, no cross-test imports (the `_make_run`
fixture is duplicated here, as in test_simulation_v026.py).
"""
import json
import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.cli import main as cli_main
from autorefine.memory import RunMemory
from autorefine.simulate import (
    project_budget,
    projection_points,
    trace_lines,
)

REPO = Path(__file__).resolve().parent.parent


# --- SPEC.md 41.1.3/41.1.4 (A31): project_budget (pure) ----------------------

def test_project_budget_insufficient_fewer_than_two_points():
    """A31 (41.1.4): < 2 usable points degrades gracefully to
    'insufficient' (ok False)."""
    r = project_budget([], 95.0, 0.0)
    assert r["ok"] is False and r["verdict"] == "insufficient"
    assert r["points"] == 0 and r["vmax"] is None
    r = project_budget([(10, 50.0)], 95.0, 10.0)
    assert r["ok"] is False and r["verdict"] == "insufficient"
    assert r["points"] == 1


def test_project_budget_exact_fit_more_verdict():
    """A31 (41.1.3/41.1.4): points exactly on V=100, K=10 recover
    Vmax=100, Km=10; target 90 -> e_T = 90 (more = 60 past e=30)."""
    pts = [(10, 50.0), (20, 200.0 / 3.0), (30, 75.0)]  # score(e)=100e/(10+e)
    r = project_budget(pts, 90.0, 30.0)
    assert r["ok"] is True and r["verdict"] == "more"
    assert r["points"] == 3
    assert r["vmax"] == pytest.approx(100.0)
    assert r["km"] == pytest.approx(10.0)
    assert r["e_target"] == pytest.approx(90.0)
    assert r["more"] == 60


def test_project_budget_ceiling_when_vmax_at_or_below_target():
    """A31 (41.1.4): Vmax <= target is the CEILING verdict (no e_T/more)."""
    pts = [(10, 50.0), (20, 200.0 / 3.0), (30, 75.0)]
    r = project_budget(pts, 100.0, 30.0)
    assert r["ok"] is True and r["verdict"] == "ceiling"
    assert r["vmax"] == pytest.approx(100.0)
    assert r["e_target"] is None and r["more"] is None


def test_project_budget_more_floors_at_zero():
    """A31 (41.1.4): N = max(0, ceil(e_T - e_current)) — already past the
    target means 0 more experiments."""
    pts = [(10, 50.0), (20, 200.0 / 3.0), (30, 75.0)]
    r = project_budget(pts, 90.0, 95.0)  # e_T = 90 < 95
    assert r["ok"] is True and r["verdict"] == "more"
    assert r["more"] == 0


def test_project_budget_decreasing_history_insufficient():
    """A31 (41.1.4): a degenerate fit (decreasing history -> negative Km)
    degrades gracefully instead of failing."""
    r = project_budget([(10, 80.0), (20, 40.0)], 95.0, 20.0)
    assert r["ok"] is False and r["verdict"] == "insufficient"


def test_project_budget_filters_nonpositive_points():
    """A31 (41.1.2/41.1.3): non-positive / non-finite points are dropped
    before the fit (41.1.2 usability)."""
    pts = [(0, 50.0), (10, 50.0), (20, 200.0 / 3.0), (30, 75.0),
           (-1, 80.0), (40, 0.0)]
    r = project_budget(pts, 90.0, 30.0)
    assert r["points"] == 3
    assert r["vmax"] == pytest.approx(100.0)


def test_project_budget_pure_and_deterministic():
    """A31 (41.1.5): `project_budget(p, t, e) == project_budget(p, t, e)`
    (pure, no training, no writes)."""
    pts = [(10, 50.0), (20, 200.0 / 3.0)]
    a = project_budget(pts, 90.0, 10.0)
    assert a == project_budget(pts, 90.0, 10.0)


# --- SPEC.md 41.1.2 (A31): projection_points (pure) --------------------------

def test_projection_points_filters_task_and_positive_rows():
    """A31 (41.1.2): only same-task rows with finite positive
    experiments_run and final_score are points."""
    reg = [
        {"task": "sine-v1", "run_id": "r1",
         "experiments_run": 10, "final_score": 40.0},
        {"task": "parity-v1", "run_id": "r2",
         "experiments_run": 10, "final_score": 99.0},  # other task
        {"task": "sine-v1", "run_id": "r3",
         "experiments_run": 0, "final_score": 40.0},   # er <= 0
        {"task": "sine-v1", "run_id": "r4",
         "experiments_run": 10, "final_score": 0.0},   # score <= 0
    ]
    summary = {"task": "sine-v1", "experiments_run": 2, "final_best_score": 1.0}
    pts = projection_points(reg, summary, "current")
    assert pts == [(10.0, 40.0), (2.0, 1.0)]


def test_projection_points_includes_current_run():
    """A31 (41.1.2): the current run's own point is added (no registry)."""
    summary = {"task": "sine-v1", "experiments_run": 4, "final_best_score": 30.0}
    assert projection_points([], summary, "cur") == [(4.0, 30.0)]


def test_projection_points_dedups_by_run_id():
    """A31 (41.1.2): when the current run already has a registry entry, the
    registry row is the point — the summary's is not added a second time."""
    reg = [{"task": "sine-v1", "run_id": "cur",
            "experiments_run": 4, "final_score": 29.0}]
    summary = {"task": "sine-v1", "experiments_run": 4, "final_best_score": 30.0}
    assert projection_points(reg, summary, "cur") == [(4.0, 29.0)]


def test_projection_points_none_task():
    """A31 (41.1.2): no task in the summary -> no points."""
    reg = [{"task": "sine-v1", "run_id": "r1",
            "experiments_run": 4, "final_score": 29.0}]
    assert projection_points(reg, {}, "cur") == []


# --- SPEC.md 41.2.2 (A31): trace_lines (pure) --------------------------------

def test_trace_lines_baseline_accepted_rejected():
    """A31 (41.2.2): one line per entry — baseline seed champion, accepted
    (new best), rejected with the 39.2.2 reason."""
    e = [
        {"kind": "baseline", "holdout_score": 50.0},
        {"kind": "experiment", "accepted": True, "holdout_score": 60.0,
         "gen_gap": 0.01, "mutation": ["hidden_dim"]},
        {"kind": "experiment", "accepted": False, "holdout_score": 55.0,
         "gen_gap": 0.0, "mutation": []},
    ]
    lines = trace_lines(e)
    assert len(lines) == 3
    assert lines[0].startswith("#1 [baseline]") and "BASELINE (seed champion)" in lines[0]
    assert lines[1].startswith("#2 [experiment]")
    assert "mutation=hidden_dim" in lines[1] and "ACCEPTED (new best)" in lines[1]
    assert "REJECTED" in lines[2] and "score gate" in lines[2]
    assert "running best 60.00" in lines[2]


def test_trace_lines_three_reject_reasons():
    """A31 (41.2.2): the documented 39.2.2 priority — score gate, then
    overfit (gen-gap above GEN_GAP_TOL*score), then the CI gate."""
    e = [
        {"kind": "baseline", "holdout_score": 60.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 55.0,
         "gen_gap": 0.0},                                    # score gate
        {"kind": "experiment", "accepted": False, "holdout_score": 65.0,
         "gen_gap": 5.0},                                    # overfit
        {"kind": "experiment", "accepted": False, "holdout_score": 65.0,
         "gen_gap": 1.0},                                    # CI gate
    ]
    lines = trace_lines(e)
    assert "score gate" in lines[1]
    assert "overfit" in lines[2]
    assert "CI gate" in lines[3]


def test_trace_lines_curriculum_repins_running_best():
    """A31 (41.2.2): a curriculum step-up re-pins the running best — a
    candidate above the old best but below the new baseline is a score-gate
    rejection, not a CI-gate one."""
    e = [
        {"kind": "baseline", "holdout_score": 50.0},
        {"kind": "curriculum", "new_baseline_score": 70.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 65.0,
         "gen_gap": 0.0},
    ]
    lines = trace_lines(e)
    assert "new baseline 70.00" in lines[1] and "re-pins" in lines[1]
    assert "score gate" in lines[2] and "running best 70.00" in lines[2]
    assert "CI gate" not in lines[2]


def test_trace_lines_screen_and_invalid_spec():
    """A31 (41.2.2): screen rejections and invalid specs get their lines."""
    e = [
        {"kind": "baseline", "holdout_score": 50.0},
        {"kind": "screen", "holdout_score": 40.0},
        {"kind": "invalid_spec", "error": "hidden_dim out of space"},
    ]
    lines = trace_lines(e)
    assert "SCREEN-REJECTED" in lines[1] and "40.00" in lines[1]
    assert "invalid spec" in lines[2] and "hidden_dim out of space" in lines[2]


def test_trace_lines_pure_and_deterministic():
    """A31 (41.2.3): `trace_lines(e) == trace_lines(e)` (pure, no I/O)."""
    e = [
        {"kind": "baseline", "holdout_score": 50.0},
        {"kind": "experiment", "accepted": True, "holdout_score": 60.0,
         "gen_gap": 0.0},
    ]
    assert trace_lines(e) == trace_lines(e)


# --- SPEC.md 41.1 (A31): report --project (CLI) ------------------------------

def _make_run(tmp_path: Path) -> Path:
    """A finished sine-v1 run (best score ~27.9, positive) — duplicated from
    test_simulation_v026.py (house rule: no cross-test imports)."""
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs")
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def _append_registry_entry(runs_dir: Path, **kw) -> None:
    """Append a same-task finished-run entry to the 38.1 registry."""
    reg_file = runs_dir / "registry.json"
    entries = json.loads(reg_file.read_text(encoding="utf-8")) \
        if reg_file.exists() else []
    entries.append(kw)
    reg_file.write_text(json.dumps(entries), encoding="utf-8")


def test_report_project_ceiling_rc0(tmp_path, capsys):
    """A31 (41.1.5): a sub-target asymptote renders the CEILING verdict,
    rc 0 (an informational view)."""
    run = _make_run(tmp_path)
    _append_registry_entry(run.parent, task="sine-v1", run_id="other",
                           experiments_run=10, final_score=40)
    rc = cli_main(["report", "--run", str(run), "--project"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "projection" in out and "CEILING" in out
    assert "Vmax" in out and "Km" in out


def test_report_project_more_rc0(tmp_path, capsys):
    """A31 (41.1.5): a super-target asymptote renders the ~N-more verdict,
    rc 0."""
    run = _make_run(tmp_path)
    _append_registry_entry(run.parent, task="sine-v1", run_id="other",
                           experiments_run=50, final_score=92)
    rc = cli_main(["report", "--run", str(run), "--project"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "projection" in out and "MORE" in out and "more experiment" in out


def test_report_project_target_flag(tmp_path, capsys):
    """A31 (41.1.1): --target changes the bar (target 40 is below the
    curve's asymptote -> MORE, where 95 was CEILING)."""
    run = _make_run(tmp_path)
    _append_registry_entry(run.parent, task="sine-v1", run_id="other",
                           experiments_run=10, final_score=40)
    rc = cli_main(["report", "--run", str(run), "--project", "--target", "40"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "target 40" in out and "MORE" in out


def test_report_project_insufficient_history_rc0(tmp_path, capsys):
    """A31 (41.1.4): only the current run in the history -> < 2 points ->
    INSUFFICIENT HISTORY, still rc 0 (graceful)."""
    run = _make_run(tmp_path)  # the registry holds this one run only
    rc = cli_main(["report", "--run", str(run), "--project"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "projection" in out and "INSUFFICIENT HISTORY" in out


def test_report_project_json_rc1(tmp_path, capsys):
    """A31 (41.1.1): --project + --json is an error (A11 machine path
    stays pure)."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--project", "--json"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err


def test_report_project_history_rc1(capsys):
    """A31 (41.1.1): --project + --history is an error."""
    rc = cli_main(["report", "--project", "--history"])
    assert rc == 1


def test_report_project_whatif_rc1(tmp_path, capsys):
    """A31 (41.1.1): --project + --what-if is an error (the other view
    path is untouched)."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--project",
                   "--what-if", "score>=0"])
    assert rc == 1


# --- SPEC.md 41.2 (A31): report --trace (CLI) --------------------------------

def test_report_trace_one_line_per_entry(tmp_path, capsys):
    """A31 (41.2.3): one decision line per experiments.jsonl entry, rc 0."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--trace"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== trace (one line per logged experiment) ===" in out
    entries = RunMemory(run).load_experiments()
    drawn = [ln for ln in out.splitlines() if re.match(r"\s*#\d+ \[", ln)]
    assert len(drawn) == len(entries)
    assert any("BASELINE (seed champion)" in ln for ln in drawn)
    assert any(("ACCEPTED" in ln) or ("REJECTED" in ln)
               for ln in drawn if "[experiment]" in ln)


def test_report_trace_json_rc1(tmp_path, capsys):
    """A31 (41.2.1): --trace + --json is an error (A11 stays pure)."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--trace", "--json"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err


def test_report_json_invariant_no_project_or_trace_block(tmp_path, capsys):
    """A31 (41.4): the A11 `--json` invariant stays green — stdout is pure
    JSON equal to summary.json, with no projection/trace block."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out) == json.loads(
        (run / "summary.json").read_text(encoding="utf-8"))
    assert "projection:" not in out
    assert "=== trace" not in out


# --- SPEC.md 41.3 (A31): run --demo (CLI) ------------------------------------

def test_run_demo_narrates_tiny_loop_and_creates_artifacts(tmp_path, capsys):
    """A31 (41.3.2/41.3.3): `run --demo` finishes a tiny parity-v1 loop in
    seconds (rc 0), prints the narration (baseline -> candidates -> gate ->
    best spec), and creates a normal run dir with artifacts + a registry
    entry. Other run flags (--task) are ignored (41.3.1)."""
    runs = str(tmp_path / "runs")
    rc = cli_main(["run", "--demo", "--runs-dir", runs,
                   "--task", "cartpole-v1"])  # ignored by the demo
    out = capsys.readouterr().out
    assert rc == 0
    assert "AutoRefine demo" in out
    assert "task parity-v1" in out  # the demo is always parity-v1
    assert "BASELINE (seed champion)" in out
    assert "best spec" in out
    assert "experiments_run" in out and "finished_reason" in out
    run_dirs = [p for p in Path(runs).iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for f in ("summary.json", "experiments.jsonl", "best_spec.json"):
        assert (run_dir / f).is_file(), f
    reg = json.loads((Path(runs) / "registry.json").read_text(encoding="utf-8"))
    assert len(reg) == 1
    assert reg[0]["run_id"] == run_dir.name
    assert reg[0]["task"] == "parity-v1"


# --- SPEC.md 33.1 / 41.4 (A31): version --------------------------------------

def test_version_027_both_sources():
    """A31 (41.4, 33.1): the version stepped to `0.37.0` in both sources
    (advanced again with v0.37 ⇒ `0.37.0`, M40, SPEC.md 51 — per 33.1)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.37.0"
