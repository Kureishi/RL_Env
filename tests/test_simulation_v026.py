"""Simulation tests (SPEC.md 40, A30, v0.26).

S1 (40.1) `fit --dry-run`: resolve data -> task -> head/metric -> split and
print the plan, creating **no** run dir / artifacts, rc 1 on a bad label
column (the failure mode the flag exists to kill).

S2 (40.2) `report --what-if`: re-gate the logged history against a NEW
objective set — pure derivation from `experiments.jsonl`, no retraining.
`what_if`/`estimate_wall` live in the `autorefine.simulate` leaf; the CLI
wires them. House rules: stdlib + numpy only, no cross-test imports (the
`_make_run` fixture is duplicated here, as in test_infra_dx.py).
"""
import json
import random
import tomllib
from pathlib import Path

import numpy as np  # noqa: F401  (house rule: core+tests may use numpy)
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.cli import main as cli_main
from autorefine.gate import Objective
from autorefine.simulate import estimate_wall, what_if

REPO = Path(__file__).resolve().parent.parent


# --- SPEC.md 40.2 (A30): what_if (pure) ---------------------------------------

def _entries():
    """A small logged history: baseline + 2 experiments + one unscored row."""
    return [
        {"kind": "baseline", "holdout_score": 50.0,
         "train_seconds": 0.5, "spec_hash": "aaaa1111"},
        {"kind": "experiment", "holdout_score": 55.0,
         "train_seconds": 1.0, "spec_hash": "bbbb2222"},
        {"kind": "experiment", "holdout_score": 55.0,
         "train_seconds": 0.8, "spec_hash": "cccc3333"},
        {"kind": "curriculum", "difficulty": "parity-4"},  # unscored
    ]


def test_what_if_picks_best_passing_score_only():
    """A30 (40.2.2): best passing candidate under a score-only bar; the
    unscored curriculum row is not in the pool."""
    r = what_if(_entries(), (Objective("score", ">=", 52.0),))
    assert r["pool"] == 3
    assert r["pass"] is True
    assert r["passing"] == 2  # the two 55.0 candidates
    assert r["final"]["score"] == 55.0


def test_what_if_score_plus_train_excludes_slow():
    """A30 (40.2.2): a higher-score candidate that is too slow is excluded
    when a `train` objective is present."""
    o = (Objective("score", ">=", 52.0), Objective("train", "<=", 0.9))
    r = what_if(_entries(), o)
    assert r["pass"] is True
    assert r["passing"] == 1  # only the 55.0 / 0.8s candidate
    assert r["final"]["train"] == 0.8
    assert r["final"]["cand"] == 3


def test_what_if_tie_break_lower_train_then_log_order():
    """A30 (40.2.2): ties on score break on lower train_seconds, then log order."""
    e = [
        {"kind": "experiment", "holdout_score": 60.0,
         "train_seconds": 2.0, "spec_hash": "x"},
        {"kind": "experiment", "holdout_score": 60.0,
         "train_seconds": 1.0, "spec_hash": "y"},
    ]
    r = what_if(e, (Objective("score", ">=", 55.0),))
    assert r["final"]["train"] == 1.0
    assert r["final"]["cand"] == 2  # the lower-train (later) row wins the tie


def test_what_if_baseline_in_pool():
    """A30 (40.2.2): the baseline row is a candidate (kind in the pool)."""
    e = [
        {"kind": "baseline", "holdout_score": 70.0,
         "train_seconds": 0.3, "spec_hash": "b"},
        {"kind": "experiment", "holdout_score": 40.0,
         "train_seconds": 0.3, "spec_hash": "c"},
    ]
    r = what_if(e, (Objective("score", ">=", 60.0),))
    assert r["pool"] == 2
    assert r["final"]["cand"] == 1  # the baseline is the passing one


def test_what_if_miss_when_none_pass():
    """A30 (40.2.2): MISS when no candidate meets the bar."""
    r = what_if(_entries(), (Objective("score", ">=", 999999.0),))
    assert r["pass"] is False
    assert r["passing"] == 0
    assert r["final"] is None


def test_what_if_missing_train_fails_train_objective():
    """A30 (40.2.2): a missing actual fails its objective (37.2 rule)."""
    e = [{"kind": "baseline", "holdout_score": 80.0, "spec_hash": "b"}]
    assert what_if(e, (Objective("train", "<=", 100.0),))["pass"] is False
    assert what_if(e, (Objective("score", ">=", 50.0),))["pass"] is True


def test_what_if_rejects_model():
    """A30 (40.2.3): `model` is rejected with a clear error (no second size
    calculator is invented)."""
    with pytest.raises(ValueError, match="model"):
        what_if(_entries(), (Objective("model", "<=", 5),))


def test_what_if_pure_and_deterministic():
    """A30 (40.3): `what_if(e, o) == what_if(e, o)` (pure, no side effects)."""
    o = (Objective("score", ">=", 52.0),)
    assert what_if(_entries(), o) == what_if(_entries(), o)


# --- SPEC.md 40.1.2 (A30): estimate_wall (pure) -------------------------------

def _registry():
    return [
        {"task": "csv", "experiments_run": 10, "wall_seconds": 100.0},  # 10.0/e
        {"task": "csv", "experiments_run": 20, "wall_seconds": 300.0},  # 15.0/e
        {"task": "csv", "experiments_run": 30, "wall_seconds": 150.0},  # 5.0/e
        {"task": "parity-v1", "experiments_run": 10, "wall_seconds": 5.0},
        {"task": "csv", "experiments_run": 0, "wall_seconds": 10.0},  # er<=0
        {"task": "csv", "experiments_run": 5, "wall_seconds": -1.0},  # ws<=0
    ]


def test_estimate_wall_median_times_experiments():
    """A30 (40.1.2): median of wall/exp over same-task history × experiments,
    excluding other-task and non-positive rows."""
    r = estimate_wall(_registry(), "csv", 30)
    assert r["runs"] == 3
    assert r["estimate"] == pytest.approx(10.0 * 30)  # median [5,10,15] = 10


def test_estimate_wall_even_count_median():
    """A30 (40.1.2): an even history averages the middle two."""
    r = estimate_wall(_registry()[:2], "csv", 10)
    assert r["estimate"] == pytest.approx(12.5 * 10)  # median [10,15] = 12.5


def test_estimate_wall_no_history_is_none():
    """A30 (40.1.2): no usable same-task history -> the fallback (estimate
    None, runs 0)."""
    assert estimate_wall(_registry(), "gridnav-v1", 10) == {
        "estimate": None, "runs": 0}


# --- SPEC.md 40.1 (A30): fit --dry-run (CLI) ----------------------------------

def _csv(tmp_path: Path) -> Path:
    random.seed(0)
    rows = ["x1,x2,label"]
    for _ in range(60):
        a, b = random.uniform(-1, 1), random.uniform(-1, 1)
        rows.append(f"{a:.3f},{b:.3f},{1 if (a + b) > 0 else 0}")
    p = tmp_path / "data.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_fit_dry_run_prints_plan_and_creates_nothing(tmp_path, capsys):
    """A30 (40.1): the plan prints (rc 0) and no run dir / artifacts exist."""
    p = _csv(tmp_path)
    runs = str(tmp_path / "runs")
    rc = cli_main(["fit", "--dry-run", "--data", str(p), "--runs-dir", runs])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run: plan only" in out
    assert "metric accuracy" in out  # task/head/metric line
    assert "dataset" in out and "budget" in out and "catalog" in out
    assert "estimate" in out
    assert not Path(runs).exists()  # no run dir / artifacts


def test_fit_dry_run_bad_label_rc1(tmp_path, capsys):
    """A30 (40.1.3): a wrong label column is reported rc 1 (the failure mode
    the flag exists to kill) before any training."""
    p = _csv(tmp_path)
    rc = cli_main(["fit", "--dry-run", "--data", str(p),
                   "--label", "NOPE", "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "NOPE" in err


def test_fit_dry_run_from_run_rc1(capsys):
    """A30 (40.1.3): `--dry-run --from-run` is mutually exclusive (rc 1)."""
    assert cli_main(["fit", "--dry-run", "--from-run", "foo"]) == 1


def test_fit_dry_run_estimate_from_registry(tmp_path, capsys):
    """A30 (40.1.2): the estimate line reflects the registry's same-task
    median (median 12.5 × 30 = 375)."""
    p = _csv(tmp_path)
    runs = Path(tmp_path / "runs")
    runs.mkdir(parents=True)
    (runs / "registry.json").write_text(json.dumps([
        {"task": "csv", "experiments_run": 10, "wall_seconds": 100.0},
        {"task": "csv", "experiments_run": 20, "wall_seconds": 300.0},
    ]), encoding="utf-8")
    rc = cli_main(["fit", "--dry-run", "--data", str(p), "--runs-dir",
                   str(runs), "--experiments", "30"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "375" in out and "past run" in out


# --- SPEC.md 40.2 (A30): report --what-if (CLI) -------------------------------

def _make_run(tmp_path: Path) -> Path:
    """A finished sine-v1 run (best score ≈ 27.9, positive) — duplicated from
    test_infra_dx.py (house rule: no cross-test imports)."""
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs")
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def test_report_what_if_pass_rc0(tmp_path, capsys):
    """A30 (40.2.5): rc 0 PASS; the block shows the bar/pool/passing/final."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--what-if", "score>=0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "what-if" in out and "PASS" in out
    assert "pool" in out and "counterfactual final" in out
    assert "actual final" in out


def test_report_what_if_miss_rc2(tmp_path, capsys):
    """A30 (40.2.5): rc 2 MISS when the bar is unmeetable by the history."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--what-if", "score>=999999"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "MISS" in out


def test_report_what_if_model_rc1(tmp_path, capsys):
    """A30 (40.2.3): the `model` objective is rejected with rc 1."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--what-if", "model<=5"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "model" in err


def test_report_what_if_json_rc1(tmp_path, capsys):
    """A30 (40.2.1): `--what-if` + `--json` is an error (machine path pure)."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run),
                   "--what-if", "score>=0", "--json"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err


def test_report_what_if_history_rc1(capsys):
    """A30 (40.2.1): `--what-if` + `--history` is an error."""
    rc = cli_main(["report", "--what-if", "score>=0", "--history"])
    assert rc == 1


def test_report_json_unchanged_no_what_if_block(tmp_path, capsys):
    """A30 (40.3): the A11 `--json` invariant stays green — stdout is pure
    JSON equal to summary.json and carries no what-if block."""
    run = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    summary = json.loads(out)
    on_disk = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary == on_disk
    assert "what-if" not in out


# --- SPEC.md 33.1 / 40.3 (A30): version ---------------------------------------

def test_version_026_both_sources():
    """A30 (40.3, 33.1): the version stepped to `0.39.0` in both sources
    (advanced again with v0.39 ⇒ `0.39.0`, M42, SPEC.md 53 — per 33.1)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"
