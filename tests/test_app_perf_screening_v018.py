"""v0.18 optimization — SPEC.md 32, A22:

- 32.1 app perceived perf (app-only) — the "Run the seed sweep" button
  streams the per-step `on_update` (SPEC.md 31.2 serial callback path) into
  an `st.progress` bar that finishes at 1.0 with "complete" text; the data
  preview computation is cached on (path, mtime_ns, size) — a second call
  with the same key does not re-probe, a changed stamp re-probes.
- 32.2 two-stage candidate screening (opt-in) —
  `AutoRefineEnv(screen_frac=F, 0 < F < 1)`: screen each candidate on the
  first floor(F·n) rows; only a strict beat of the fixed champion (the
  baseline spec screened once at `reset()`) spends the full training.
  Screen rejections are logged `kind="screen"`, spend one budget
  experiment, count for the v0.17 stall patience, and never touch Pareto /
  ensemble / curriculum. Default (1.0 = off) is pre-v0.18 exactly.
  CLI: `run` / `fit` accept `--screen-frac` (A22).

House rules: no cross-test imports (fixtures duplicated); tiny budgets
(scripted-score fake task + a 60-row quadrant-XOR CSV, 2-3 experiments);
determinism (G2) throughout.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget
from autorefine.cli import main as cli_main
from autorefine.improver.meta_env import candidate_screening
from autorefine.tasks import TASKS, CsvTask

# ---------------------------------------------------------------------------
# fakes (SPEC.md 32.4): a scripted-score task so two-stage screening is
# fully deterministic — `train` is monkeypatched to hand back models whose
# `.score_val` the task's `score()` returns; screen vs full calls are
# distinguished by the dataset size (default_dataset_size=16, frac 0.25
# -> exactly 4 screen rows). The model satisfies the `save_best` contract
# (`model.save(path)`, SPEC.md 22 artifacts).
# ---------------------------------------------------------------------------

FAKE_TASK = "fake-screen-v1"


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


def _install_screen_train(monkeypatch, screen_scores: list[float],
                          full_scores: list[float]) -> None:
    """`autorefine.improver.meta_env.train` → models scored from
    `screen_scores` (calls on the ≤4-row prefix subsample) or `full_scores`
    (calls on the full 16-row dataset); the last entry of each list repeats
    once exhausted (a constant tail)."""
    calls = {"s": 0, "f": 0}

    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        n = int(dataset[0].shape[0])
        if n <= 4:  # the prefix subsample (floor(16 * 0.25) = 4 rows)
            sv = screen_scores[min(calls["s"], len(screen_scores) - 1)]
            calls["s"] += 1
        else:
            sv = full_scores[min(calls["f"], len(full_scores) - 1)]
            calls["f"] += 1
        return SimpleNamespace(
            model=_FakeModel(sv), train_seconds=0.001,
            loss_history=[], time_capped=False)

    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _install_const_train(monkeypatch, score: float) -> None:
    """A constant-score `train` (the screening-off pin guard)."""
    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        return SimpleNamespace(
            model=_FakeModel(score), train_seconds=0.001,
            loss_history=[], time_capped=False)

    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _drive_to_done(env: AutoRefineEnv, seed: int = 7) -> None:
    policy = BanditPolicy(seed=seed)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    (Duplicated fixture — tests/test_optimization_v017.py.)"""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --- 32.2 two-stage screening: the A22 scripted pin --------------------------

def test_screening_pin_scripted_run(tmp_path, monkeypatch, capsys):
    """A22 (SPEC.md 32.4): screen 50(baseline) → 40 → 60 → 55, full
    50 → 70 → 65, `screen_frac=0.25`, budget 3 → log kinds exactly
    `baseline, screen, experiment, experiment` (exp 1 screen-rejected,
    exp 2 promoted+accepted, exp 3 promoted+gate-rejected);
    `experiments_run=3`, `final_best_score=70`, the summary carries the
    `screening` block, and `report --run <dir>` exits 0 printing the
    `screen` row."""
    _install_screen_train(monkeypatch,
                          [50.0, 40.0, 60.0, 55.0], [50.0, 70.0, 65.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(3, 300, 30), runs_dir=tmp_path,
            screen_frac=0.25)
        policy = BanditPolicy(seed=7)
        state = env.reset()
        assert env._screen_champion == 50.0  # the baseline screened (champion)
        while not env.done:
            state, _r, _d, _i = env.step(policy.propose(state))
        assert env.done and env.done_reason == "budget_exhausted"

        entries = env.memory.load_experiments()
        assert [e["kind"] for e in entries] == \
            ["baseline", "screen", "experiment", "experiment"]
        screen = entries[1]
        assert screen["accepted"] is False and screen["screen_rejected"] is True
        assert screen["holdout_score"] == 40.0  # the screen holdout score
        exps = entries[2:]
        assert [e["holdout_score"] for e in exps] == [70.0, 65.0]
        assert [e["accepted"] for e in exps] == [True, False]

        summary = env.memory.load_summary()
        assert summary["finished_reason"] == "budget_exhausted"
        assert summary["experiments_run"] == 3  # the screen rejection counts
        assert summary["final_best_score"] == 70.0
        assert summary["baseline_score"] == 50.0
        assert summary["screening"] == \
            {"frac": 0.25, "baseline_screen_score": 50.0}
        run_dir = env.run_dir
    finally:
        TASKS.pop(FAKE_TASK, None)  # before the report: no model loader

    # A22: `report` on the pinned run exits 0 and prints the `screen` row
    # (float holdout_score/gen_gap satisfy the table formatter, 32.2)
    rc = cli_main(["report", "--run", str(run_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(l.startswith("screen") and "False" in l for l in out.splitlines())
    assert '"screening"' in json.dumps(summary)  # additive summary block


def test_screening_off_default_unchanged(tmp_path, monkeypatch):
    """A22 (SPEC.md 32.2): default `screen_frac=1.0` is pre-v0.18 exactly —
    no `screen` rows, no `screening` summary key."""
    _install_const_train(monkeypatch, 50.0)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(2, 300, 30), runs_dir=tmp_path)
        assert env.screen_frac == 1.0 and env.screen_active is False
        _drive_to_done(env)
        assert env.done and env.done_reason == "budget_exhausted"
        entries = env.memory.load_experiments()
        assert all(e["kind"] in ("baseline", "experiment") for e in entries)
        summary = env.memory.load_summary()
        assert "screening" not in summary
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_screen_frac_invalid_values_rejected(tmp_path):
    """SPEC.md 32.2: `screen_frac` must be a non-bool finite number in
    (0, 1]; anything else raises `ValueError`."""
    TASKS[FAKE_TASK] = _FakeTask
    try:
        # NB: "0.5" is a *valid* string number (float("0.5") = 0.5) — 32.2
        # only requires a finite number in (0, 1], so it is NOT in this list.
        for bad in (0.0, -0.5, 1.5, "abc", True, False, None,
                    float("nan"), float("inf")):
            with pytest.raises(ValueError, match="screen_frac"):
                AutoRefineEnv(
                    task=FAKE_TASK, seed=7,
                    budget=Budget(2, 300, 30), runs_dir=tmp_path,
                    screen_frac=bad)
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_screening_preset_and_boundary(tmp_path, monkeypatch):
    """SPEC.md 32.2: `candidate_screening` is the `search_quality_v04()`
    preset pattern; `frac=1.0` is valid but inactive (off)."""
    assert candidate_screening() == {"screen_frac": 0.25}
    assert candidate_screening(0.5) == {"screen_frac": 0.5}
    _install_const_train(monkeypatch, 50.0)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7, budget=Budget(1, 300, 30),
            runs_dir=tmp_path, **candidate_screening(1.0))
        assert env.screen_frac == 1.0 and env.screen_active is False
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_screen_rejections_count_for_stall_patience(tmp_path, monkeypatch):
    """SPEC.md 32.2: a screen rejection is a scored non-improving
    experiment — it extends the v0.17 stall streak (SPEC.md 31.1)."""
    _install_screen_train(monkeypatch, [50.0, 40.0, 40.0], [50.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(5, 300, 30), runs_dir=tmp_path,
            screen_frac=0.25, stall_patience=2)
        _drive_to_done(env)
        assert env.done and env.done_reason == "stalled"
        assert env.bm.used_experiments == 2  # both screen-rejected
    finally:
        TASKS.pop(FAKE_TASK, None)


# --- 32.2 CLI smoke (A22) -----------------------------------------------------

def test_run_cli_screen_frac_smoke(tmp_path, capsys):
    """A22 (SPEC.md 32.2): `run --screen-frac 0.5` is accepted and the
    small screened run finishes with a reason."""
    rc = cli_main([
        "run", "--task", "sine-v1", "--experiments", "2",
        "--screen-frac", "0.5", "--search-quality", "legacy",
        "--max-seconds", "120", "--max-train-seconds", "5",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    out = capsys.readouterr().out
    assert rc == 0  # `run` reports; the gate is a `fit` concern (SPEC.md 22.1)
    assert "finished_reason" in out


def test_fit_cli_screen_frac_smoke(tmp_path, capsys):
    """A22 (SPEC.md 32.2): `fit --screen-frac 0.5` is accepted; the exit
    code is the §22.1 gate on the final score (PASS/MISS)."""
    csv = _write_csv(tmp_path)
    rc = cli_main([
        "fit", "--data", str(csv), "--label", "churn",
        "--experiments", "2", "--max-seconds", "120",
        "--max-train-seconds", "5", "--screen-frac", "0.5",
        "--runs-dir", str(tmp_path / "runs"),
    ])
    out = capsys.readouterr().out
    assert rc in (0, 2)  # PASS/MISS — the gate (SPEC.md 22.1)
    assert "finished_reason" in out


# --- 32.1 app perceived perf (A22) --------------------------------------------

def test_app_seed_sweep_progress_streams(tmp_path, monkeypatch):
    """A22 (SPEC.md 32.1): pressing "Run the seed sweep" streams the
    per-step `on_update` into the progress bar — it finishes at `1.0`
    with "complete" text, and the after-sweep SVGs still render
    (SPEC.md 29.1)."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "src" / "autorefine" / \
        "dashboard_app.py"
    csv_path = _write_csv(tmp_path)
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()  # first render materializes the sidebar widgets
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.run()  # re-render so the path is picked up and the variance panel shows
    assert not at.exception
    at.number_input(key="var_seeds").set_value(2)
    at.number_input(key="experiments").set_value(2)  # keep the sweep small
    # Streamlit 1.51's AppTest does not expose `st.progress` elements, so we
    # spy on `st.progress` itself and record every update it receives.
    import autorefine.dashboard_app as app_mod
    updates: list[tuple[float, str]] = []
    calls = {"n": 0}

    class _SpyBar:
        def __init__(self, value: float, text: str) -> None:
            self.value = value
            self.text = text

        def progress(self, value: float, text: str = "", **_kw) -> "_SpyBar":
            updates.append((float(value), text))
            self.value, self.text = float(value), text
            return self

    def spy(value: float = 0.0, text: str = "", **_kw) -> _SpyBar:
        calls["n"] += 1
        return _SpyBar(float(value), text)

    monkeypatch.setattr(app_mod.st, "progress", spy)
    try:
        at.button(key="var_button").set_value(True).run()
        assert not at.exception
    finally:
        monkeypatch.undo()
    assert calls["n"] >= 1  # the sweep opened a progress bar (SPEC.md 32.1)
    assert updates, "no `on_update` steps streamed into the bar (SPEC.md 32.1)"
    assert updates[-1][0] == 1.0  # the bar finished
    assert "complete" in updates[-1][1]
    md = " ".join(m.value for m in at.markdown)
    assert "final score across 2 seed(s)" in md  # the sweep SVGs (SPEC.md 29.1)


def test_app_preview_cached(tmp_path, monkeypatch):
    """A22 (SPEC.md 32.1): `_preview_data` is cached on (path, stamp,
    size) — a second call with the same key does not re-probe (the task
    `__init__` counting wrapper fires once), while a changed stamp
    re-probes."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    import autorefine.dashboard_app as app_mod
    app_mod.st.cache_data.clear()  # isolate from any earlier app render
    csv_path = _write_csv(tmp_path, name="cache.csv")

    orig = CsvTask.__init__
    calls = {"n": 0}

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(CsvTask, "__init__", counting)
    p = Path(csv_path)
    stamp, size = p.stat().st_mtime_ns, p.stat().st_size
    d1 = app_mod._preview_data(str(csv_path), stamp, size)
    assert d1["kind"] == "csv" and d1["header"] and len(d1["rows"]) == 5
    d2 = app_mod._preview_data(str(csv_path), stamp, size)
    assert d1 == d2
    assert calls["n"] == 1  # the second call is served from the cache
    d3 = app_mod._preview_data(str(csv_path), stamp + 1, size)
    assert calls["n"] == 2  # a changed stamp re-probes (32.1)
    assert d3 == d1  # same data, same shape


# --- house invariants ----------------------------------------------------------

def test_import_autorefine_stays_clean():
    """SPEC.md 32.3: no new dependencies — a fresh interpreter importing
    `autorefine` must never pull in streamlit/PIL/soundfile (the app
    changes are app-only; screening is core stdlib/NumPy)."""
    code = (
        "import sys, autorefine; "
        "bad = [m for m in ('PIL', 'streamlit', 'soundfile') if m in sys.modules]; "
        "assert not bad, bad"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
