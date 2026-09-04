"""v0.17 optimization — SPEC.md 31, A21:

- 31.1 plateau early stop — `AutoRefineEnv(stall_patience=K)`: K consecutive
  non-improving (scored) experiments end the run with
  `finished_reason="stalled"`; acceptance / `reset()` / a curriculum step-up
  reset the streak; the flag is off by default (A1–A20 pins unchanged).
  CLI: `run` / `fit` / `variance` accept `--stall-patience`.
- 31.2 parallel seed sweep — `DashboardRunner.seed_sweep(workers=N)` runs the
  independent seeds on a `concurrent.futures.ProcessPoolExecutor`,
  bit-identical per-seed results (except the timestamped `run_dir`);
  `workers=1` (default) keeps the exact serial loop; `on_update` raises with
  `workers > 1`; empty seeds → `[]`. CLI: `variance --workers N`.

House rules: no cross-test imports (fixtures duplicated); tiny budgets
(constant-score fake task + a 60-row quadrant-XOR CSV, 2-3 experiments);
every SVG asserted as valid XML; determinism (G2) throughout.
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget, DashboardRunner
from autorefine.cli import main as cli_main
from autorefine.tasks import TASKS

# ---------------------------------------------------------------------------
# fakes (SPEC.md 31.4): a constant / scripted-score task so stall patience is
# fully deterministic — `train` is monkeypatched to hand back models whose
# `.score_val` the task's `score()` returns; the model satisfies the
# `save_best` contract (`model.save(path)`, SPEC.md 22 artifacts).
# ---------------------------------------------------------------------------

FAKE_TASK = "fake-stall-v1"


class _FakeModel:
    """A stand-in model: the task scores it from `.score_val`."""

    def __init__(self, score_val: float) -> None:
        self.score_val = score_val

    def save(self, path: str) -> None:
        np.savez(path, a=np.zeros(1))  # the model.save contract (memory.save_best)


class _FakeTask:
    """Minimal AutoRefineEnv task (name/head/n_outputs/make_dataset/score)."""

    name = FAKE_TASK
    head = "softmax"
    n_outputs = 2
    default_dataset_size = 16

    def __init__(self, seed: int = 7, **_kw) -> None:
        pass

    def make_dataset(self, size: int):
        return (np.zeros((size, 2)), np.zeros(size, dtype=np.int64))

    def score(self, model, _split: str, _n: int) -> float:
        return float(model.score_val)


class _FakeCurriculum:
    """A stub of the SPEC.md 20.1 curriculum contract for the white-box
    step-up test (task() + level metadata read by `_advance_curriculum`)."""

    level = 1
    n_bits = 2
    p_flip = 0.1
    ceiling = 100.0

    def task(self):
        return _FakeTask(seed=7)

    def level_description(self):
        return "fake level 1"

    def levels_left(self):
        return 0

    def step_up_if_saturated(self, _best_score: float) -> bool:
        return False


def _install_fake_train(monkeypatch, scores: list[float]) -> None:
    """`autorefine.improver.meta_env.train` → models scored from `scores`
    (call 0 = the reset baseline, call i = experiment i; the last entry is
    repeated once the list is exhausted — a constant tail)."""
    calls = {"i": 0}

    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        i = calls["i"]
        calls["i"] += 1
        sv = scores[min(i, len(scores) - 1)]
        return SimpleNamespace(
            model=_FakeModel(sv), train_seconds=0.001,
            loss_history=[], time_capped=False)

    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _fake_env(tmp_path, budget_experiments: int, stall_patience, **kw) -> AutoRefineEnv:
    TASKS[FAKE_TASK] = _FakeTask  # the worked-example registration pattern
    try:
        return AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(budget_experiments, 300, 30), runs_dir=tmp_path,
            stall_patience=stall_patience, **kw)
    except Exception:
        TASKS.pop(FAKE_TASK, None)
        raise
    return None  # unreachable; keeps the try/finally shape explicit


def _drive_to_done(env: AutoRefineEnv, seed: int = 7) -> None:
    policy = BanditPolicy(seed=seed)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


def _summary(tmp_path: Path) -> dict:
    dirs = sorted(Path(tmp_path).glob("fake-stall-v1-*"))
    assert dirs, "no run dir was written"
    return json.loads((dirs[-1] / "summary.json").read_text(encoding="utf-8"))


# --- 31.1 stall patience: deterministic fake-task runs (A21) -----------------

def test_stall_patience_stops_constant_score_run(tmp_path, monkeypatch):
    """A21: constant score, `stall_patience=2`, 5-experiment budget → stops
    after exactly 2 scored experiments with `finished_reason="stalled"`."""
    _install_fake_train(monkeypatch, [50.0])
    env = _fake_env(tmp_path, 5, stall_patience=2)
    try:
        _drive_to_done(env)
        assert env.done and env.done_reason == "stalled"
        summary = _summary(tmp_path)
        assert summary["finished_reason"] == "stalled"
        assert summary["experiments_run"] == 2  # budget of 5, stopped at 2
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_stall_patience_acceptance_resets_streak(tmp_path, monkeypatch):
    """A21: 50 → 60 (accepted, streak reset) → 55 → 52 with
    `stall_patience=2` → the run stops at experiment 3, best stays 60."""
    _install_fake_train(monkeypatch, [50.0, 60.0, 55.0, 52.0])
    env = _fake_env(tmp_path, 5, stall_patience=2)
    try:
        _drive_to_done(env)
        assert env.done and env.done_reason == "stalled"
        assert env.best_score == 60.0  # the accepted experiment is the best
        summary = _summary(tmp_path)
        assert summary["finished_reason"] == "stalled"
        assert summary["experiments_run"] == 3
        assert summary["final_best_score"] == 60.0
        assert summary["baseline_score"] == 50.0
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_stall_patience_off_burns_full_budget(tmp_path, monkeypatch):
    """A21 pin guard: patience unset → the pre-v0.17 behavior exactly — the
    constant-score run burns the full 3-experiment budget."""
    _install_fake_train(monkeypatch, [50.0])
    env = _fake_env(tmp_path, 3, stall_patience=None)
    try:
        _drive_to_done(env)
        assert env.done and env.done_reason == "budget_exhausted"
        summary = _summary(tmp_path)
        assert summary["finished_reason"] == "budget_exhausted"
        assert summary["experiments_run"] == 3
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_curriculum_step_up_resets_stall_streak(tmp_path, monkeypatch):
    """A21 (SPEC.md 31.1): a curriculum step-up re-arms patience — the new
    difficulty level is a new ceiling, so the streak resets to 0."""
    _install_fake_train(monkeypatch, [50.0])
    env = _fake_env(tmp_path, 5, stall_patience=2, curriculum=_FakeCurriculum())
    try:
        env.reset()
        env._stall_streak = 1  # a rejection so far at this level
        env._advance_curriculum()  # white-box: the step-up itself
        assert env._stall_streak == 0
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_reset_resets_stall_streak(tmp_path, monkeypatch):
    """SPEC.md 31.1: a fresh episode (`reset()`) re-arms patience."""
    _install_fake_train(monkeypatch, [50.0])
    env = _fake_env(tmp_path, 3, stall_patience=2)
    try:
        state = env.reset()
        assert state["best_score"] == 50.0
        env._stall_streak = 2  # would be "stalled" on the next check
        env.reset()  # a fresh episode
        assert env._stall_streak == 0
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_stall_patience_invalid_values_rejected(tmp_path, monkeypatch):
    """SPEC.md 31.1: K must be an int >= 1 or None (off)."""
    _install_fake_train(monkeypatch, [50.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        for bad in (0, 1.5, "2", -1):
            with pytest.raises(ValueError, match="stall_patience"):
                AutoRefineEnv(
                    task=FAKE_TASK, seed=7,
                    budget=Budget(3, 300, 30), runs_dir=tmp_path,
                    stall_patience=bad)
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_run_cli_stall_patience_smoke(tmp_path, capsys):
    """A21 (SPEC.md 31.1): `run --stall-patience 1` finishes with a reason
    in {stalled, budget_exhausted} (the gate/reason contract is unchanged)."""
    rc = cli_main([
        "run", "--task", "sine-v1", "--experiments", "2",
        "--stall-patience", "1", "--search-quality", "legacy",
        "--max-seconds", "120", "--max-train-seconds", "5",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    out = capsys.readouterr().out
    assert rc == 0  # `run` reports; the gate is a `fit` concern (SPEC.md 22.1)
    assert "finished_reason" in out
    assert out.split('"finished_reason"')[1].split('"')[1] in (
        "stalled", "budget_exhausted")


def test_fit_cli_stall_patience_smoke(tmp_path, capsys):
    """A21 (SPEC.md 31.1): `fit --stall-patience` is accepted and the
    summary carries the finished reason (exit code: the §22.1 gate only)."""
    csv = _write_csv(tmp_path)
    rc = cli_main([
        "fit", "--data", str(csv), "--label", "churn",
        "--experiments", "2", "--max-seconds", "120",
        "--max-train-seconds", "5", "--stall-patience", "1",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    out = capsys.readouterr().out
    assert rc in (0, 2)  # PASS/MISS — the gate on final score (SPEC.md 22.1)
    assert "finished_reason" in out


# --- 31.2 parallel seed sweep (A21) ------------------------------------------

def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    (Duplicated fixture — tests/test_comprehension_visuals2.py and
    tests/test_multi_run_policy_views.py.)"""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _runner(tmp_path, **kw) -> DashboardRunner:
    kw.setdefault("csv_path", str(_write_csv(tmp_path)))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 2)
    kw.setdefault("policy", "bandit")
    kw.setdefault("max_train_seconds", 5.0)
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    return DashboardRunner(**kw)


def _without_run_dir(sweep: list[dict]) -> list[dict]:
    """SPEC.md 31.2: the only per-seed value that legitimately differs
    between the serial and parallel paths is the timestamped `run_dir`."""
    return [{k: v for k, v in s.items() if k != "run_dir"} for s in sweep]


def test_parallel_sweep_matches_serial(tmp_path):
    """A21 (SPEC.md 31.2): `workers=2` returns the same per-seed dicts
    (excluding `run_dir`) as `workers=1`, in the same input seed order."""
    serial = _runner(tmp_path).seed_sweep([7, 8], workers=1)
    parallel = _runner(tmp_path).seed_sweep([7, 8], workers=2)
    assert [s["seed"] for s in parallel] == [7, 8]  # pool.map order (31.2)
    assert _without_run_dir(parallel) == _without_run_dir(serial)
    for s in parallel:  # the A19 key set, unchanged (SPEC.md 29.1/30.1)
        assert set(s) == {"seed", "baseline", "final", "curve", "target",
                          "pass", "verdict", "experiments_run", "run_dir"}
        assert np.isfinite(s["baseline"]) and np.isfinite(s["final"])
        assert len(s["curve"]) >= 2 and s["curve"][0] == s["baseline"]
        assert Path(s["run_dir"]).name  # each seed got its own run dir


def test_seed_sweep_default_is_serial(tmp_path):
    """A21: `workers=1` is the default and matches the explicit serial path
    (the pre-v0.17 loop, byte-identical per-seed results)."""
    default = _runner(tmp_path).seed_sweep([7])
    explicit = _runner(tmp_path).seed_sweep([7], workers=1)
    assert _without_run_dir(default) == _without_run_dir(explicit)


def test_seed_sweep_empty_and_validation(tmp_path):
    """A21 (SPEC.md 31.2): empty seeds → [] without spawning; `on_update`
    raises with `workers > 1`; bad `workers` values raise."""
    r = _runner(tmp_path)
    assert r.seed_sweep([], workers=2) == []
    assert r.seed_sweep(None, workers=2) == []  # forgiving None, same []
    with pytest.raises(ValueError, match="on_update"):
        r.seed_sweep([7], on_update=lambda u: None, workers=2)
    for bad in (0, -1, True, "2"):
        with pytest.raises(ValueError, match="workers"):
            r.seed_sweep([7], workers=bad)


def test_variance_cli_workers_smoke(tmp_path, capsys):
    """A21 (SPEC.md 31.2): `variance --workers 2` exits 0 and writes the
    same artifacts (a 2-entry seed_sweep.json, valid SVGs)."""
    csv = _write_csv(tmp_path)
    rc = cli_main([
        "variance", "--data", str(csv), "--label", "churn",
        "--seeds", "2", "--workers", "2", "--experiments", "2",
        "--max-seconds", "120", "--max-train-seconds", "5",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    out = capsys.readouterr().out
    assert rc == 0  # a variance report is a measurement (SPEC.md 29.1)
    runs = tmp_path / "runs"
    sweep = json.loads((runs / "seed_sweep.json").read_text(encoding="utf-8"))
    assert [s["seed"] for s in sweep] == [7, 8]
    ET.fromstring((runs / "seed_variance.svg").read_text(encoding="utf-8"))
    ET.fromstring((runs / "seed_curves.svg").read_text(encoding="utf-8"))
    assert str(runs / "seed_variance.svg") in out
    assert str(runs / "seed_curves.svg") in out


# --- house invariants ---------------------------------------------------------

def test_import_autorefine_stays_clean():
    """SPEC.md 31.2/31.3: no new dependencies — a fresh interpreter importing
    `autorefine` must never pull in streamlit/PIL/soundfile (the pool is
    stdlib, created only inside `seed_sweep`)."""
    code = (
        "import sys, autorefine; "
        "bad = [m for m in ('PIL', 'streamlit', 'soundfile') if m in sys.modules]; "
        "assert not bad, bad"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
