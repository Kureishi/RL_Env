"""v0.24 tracking — SPEC.md 38, A28:

- 38.1 (T1) the run registry — `<runs-dir>/registry.json` gets exactly
  one 13-key entry per finished run; the entry agrees with
  `summary.json`; `config_fp` is the 38.1.3 fingerprint (12 hex chars,
  deterministic, config-sensitive); two runs append in order; a
  corrupt registry is moved aside (`.corrupt-*` preserved) and the
  new entry still lands (38.1.4).
- 38.2 (T2) lineage — the `parent_run` kwarg lands in the *conditional*
  `summary.json` key (absent for a fresh run, so the A24 key sets stay
  untouched) and the registry entry; `fit --from-run` sets it to the
  source run's dir name.
- 38.3 `report --history` — the table (rc 0, run id + task + gate),
  `--json` equals the file, no registry → rc 1 + hint, neither flag →
  rc 1, both flags → rc 1.
- 38.4 the app — `diff_two_summaries` (score delta + per-field spec
  diff, streamlit-free) and the Past-runs section (empty state and
  populated).
- Regression — version stepped to `0.36.0` in both sources (33.1,
  advanced again with v0.32, SPEC.md 45).

House rules: no cross-test imports (fakes duplicated from
test_coherency_v020.py, the CSV fixture from test_generalization_v023.py);
stdlib + numpy only.
"""
import csv
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, BanditPolicy, Budget
from autorefine.cli import main as cli_main
from autorefine.registry import (
    ENTRY_KEYS,
    append_entry,
    config_fingerprint,
    format_table,
    gate_label,
    load_registry,
)
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# --- fakes (duplicated from test_coherency_v020.py; no cross-test imports) --

FAKE_TASK = "fake-track-v1"
FLAT_SCORE = 55.0


class _FakeModel:
    """A stand-in model: the task scores it from `.score_val`."""

    def __init__(self, score_val: float) -> None:
        self.score_val = score_val

    def save(self, path: str) -> None:
        np.savez(path, a=np.zeros(1))  # the model.save contract


class _FakeTask:
    """Minimal AutoRefineEnv task (name/head/n_outputs/make_dataset/score)."""

    name = FAKE_TASK
    head = "softmax"
    n_outputs = 2
    default_dataset_size = 16

    def __init__(self, seed: int = 7, **_kw) -> None:
        pass  # accepts task_config kwargs (22.1) without storing them

    def make_dataset(self, size: int):
        return (np.zeros((size, 2)), np.zeros(size, dtype=np.int64))

    def score(self, model, _split: str, _n: int) -> float:
        return float(getattr(model, "score_val", 0.0))


def _install_flat_train(monkeypatch, score: float) -> None:
    """`autorefine.improver.meta_env.train` → one constant score."""
    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        return SimpleNamespace(model=_FakeModel(score), train_seconds=0.001,
                               loss_history=[], time_capped=False)
    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _drive_to_done(env: AutoRefineEnv) -> None:
    policy = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


def _fake_env(tmp_path, **kw) -> AutoRefineEnv:
    defaults = dict(
        task=FAKE_TASK, seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs")
    defaults.update(kw)
    return AutoRefineEnv(**defaults)


# --- 38.1 the run registry (T1, A28) -----------------------------------------

def test_registry_entry_contract(tmp_path, monkeypatch):
    """A28 (SPEC.md 38.1.2): a finished run appends exactly one entry
    with the 13-key contract, and its values agree with summary.json."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = _fake_env(tmp_path, policy="bandit", target=80.0)
        _drive_to_done(env)
        summary = env.memory.load_summary()
        reg = load_registry(tmp_path / "runs")
        assert len(reg) == 1  # exactly one entry per finished run
        entry = reg[0]
        assert set(entry) == set(ENTRY_KEYS)
        assert len(ENTRY_KEYS) == 13
        # agrees with the run dir + summary.json (38.1.2)
        assert entry["run_id"] == env.run_dir.name
        assert entry["task"] == FAKE_TASK
        assert entry["seed"] == summary["seed"] == 7
        assert entry["policy"] == "bandit"
        assert entry["final_score"] == float(summary["final_best_score"])
        assert entry["finished_reason"] == summary["finished_reason"]
        assert entry["experiments_run"] == summary["experiments_run"]
        assert entry["wall_seconds"] == summary["wall_seconds"]
        # the §22.1 gate: 55 < 80 → met_target False
        assert entry["target"] == 80.0
        assert entry["met_target"] is False
        # the 38.1.3 fingerprint: 12 hex chars, equals the config's
        assert re.fullmatch(r"[0-9a-f]{12}", entry["config_fp"])
        assert entry["config_fp"] == config_fingerprint(
            env.run_config.to_dict())
        assert entry["parent_run"] is None  # a fresh run (38.2)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                            entry["timestamp"])
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_met_target_true_and_env_level_unset(tmp_path, monkeypatch):
    """A28 (SPEC.md 38.1.2): `met_target` is the §22.1 gate when a
    target exists; null (both fields) at the env level."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        # 55 >= 50 → PASS
        env = _fake_env(tmp_path, target=50.0)
        _drive_to_done(env)
        e = load_registry(tmp_path / "runs")[0]
        assert e["met_target"] is True and e["target"] == 50.0
        # env-level (no driver target): both fields null
        env2 = _fake_env(tmp_path / "e2")
        _drive_to_done(env2)
        e2 = load_registry(tmp_path / "e2" / "runs")[0]
        assert e2["met_target"] is None and e2["target"] is None
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_config_fingerprint_deterministic_and_sensitive():
    """A28 (SPEC.md 38.1.3): same config → same fp; any changed field →
    different fp; 12 hex chars."""
    d = {"a": 1, "b": [1, 2], "c": {"x": "y"}}
    fp = config_fingerprint(d)
    assert re.fullmatch(r"[0-9a-f]{12}", fp)
    assert config_fingerprint(dict(reversed(list(d.items())))) == fp
    assert config_fingerprint({"a": 1, "b": [1, 2], "c": {"x": "z"}}) != fp
    assert config_fingerprint({"a": 1, "b": [2, 1], "c": {"x": "y"}}) != fp
    assert config_fingerprint({"a": 1, "b": [1, 2]}) != fp


def test_two_runs_append_in_order(tmp_path, monkeypatch):
    """A28 (SPEC.md 38.1.1): two finished runs append two entries, in
    order, with distinct run ids."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        _drive_to_done(_fake_env(tmp_path))
        first = load_registry(tmp_path / "runs")[0]["run_id"]
        _drive_to_done(_fake_env(tmp_path))
        reg = load_registry(tmp_path / "runs")
        assert [e["run_id"] for e in reg] == [first, reg[1]["run_id"]]
        assert reg[0]["run_id"] != reg[1]["run_id"]
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_corrupt_registry_is_recovered_and_preserved(tmp_path, monkeypatch):
    """A28 (SPEC.md 38.1.4): a corrupt registry is moved aside (bytes
    preserved) and the new entry still lands — finish never crashes on
    registry state."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "registry.json").write_text("not json {{{", encoding="utf-8")
    try:
        env = _fake_env(tmp_path)
        _drive_to_done(env)
        reg = load_registry(runs)
        assert len(reg) == 1 and reg[0]["run_id"] == env.run_dir.name
        backups = sorted(runs.glob("registry.json.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "not json {{{"
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_load_registry_missing_is_empty_and_append_creates(tmp_path):
    """A28 (SPEC.md 38.1.1): a missing registry is `[]`; the first
    finished run creates the file."""
    assert load_registry(tmp_path / "nowhere") == []
    append_entry(tmp_path / "new" / "runs", {"run_id": "r1"})
    assert load_registry(tmp_path / "new" / "runs") == [{"run_id": "r1"}]


# --- 38.2 lineage (T2, A28) ----------------------------------------------------

def test_parent_run_kwarg_summary_key_and_entry(tmp_path, monkeypatch):
    """A28 (SPEC.md 38.2.1): `parent_run` lands in the conditional
    `summary.json` key + the registry entry; unset → the key is *absent*
    (the A24 key sets stay untouched) and the entry field is null."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    parent = "csv-seed7-20260101-000000"
    try:
        env = _fake_env(tmp_path, parent_run=parent)
        _drive_to_done(env)
        summary = env.memory.load_summary()
        assert summary["parent_run"] == parent
        entry = load_registry(tmp_path / "runs")[0]
        assert entry["parent_run"] == parent
        # a fresh run: absent from the summary, null in the entry
        env2 = _fake_env(tmp_path / "fresh")
        _drive_to_done(env2)
        assert "parent_run" not in env2.memory.load_summary()
        assert load_registry(tmp_path / "fresh" / "runs")[0]["parent_run"] is None
    finally:
        TASKS.pop(FAKE_TASK, None)


# --- 38.3 report --history (A28) ----------------------------------------------

def test_report_history_table_and_json(tmp_path, monkeypatch, capsys):
    """A28 (SPEC.md 38.3.1/38.3.2): the table shows run id + task + gate
    (rc 0); `--json` is parseable and equals the registry file."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    runs = tmp_path / "runs"
    try:
        _drive_to_done(_fake_env(tmp_path, target=50.0))  # PASS
        _drive_to_done(_fake_env(tmp_path, target=80.0))  # MISS
        rc = cli_main(["report", "--history", "--runs-dir", str(runs)])
        out = capsys.readouterr().out
        assert rc == 0
        assert FAKE_TASK in out and "PASS" in out and "MISS" in out
        reg = json.loads((runs / "registry.json").read_text(encoding="utf-8"))
        for e in reg:
            assert e["run_id"] in out

        capsys.readouterr()
        rc = cli_main(["report", "--history", "--json",
                       "--runs-dir", str(runs)])
        out = capsys.readouterr().out
        assert rc == 0
        assert json.loads(out) == reg  # pure JSON, equal to the file
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_report_history_errors(tmp_path, capsys):
    """A28 (SPEC.md 38.3.1/38.3.3): neither flag → rc 1; both flags →
    rc 1 (mutually exclusive); no/empty registry → rc 1 + hint."""
    capsys.readouterr()
    rc = cli_main(["report"])
    err = capsys.readouterr().err
    assert rc == 1 and "one of --run or --history" in err

    capsys.readouterr()
    rc = cli_main(["report", "--run", "x", "--history"])
    err = capsys.readouterr().err
    assert rc == 1 and "mutually exclusive" in err

    capsys.readouterr()
    rc = cli_main(["report", "--history", "--runs-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1 and "finish at least one run" in err

    # an explicitly *empty* registry is the same empty state (38.3.3)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "registry.json").write_text("[]", encoding="utf-8")
    capsys.readouterr()
    rc = cli_main(["report", "--history", "--runs-dir", str(empty)])
    err = capsys.readouterr().err
    assert rc == 1 and "finish at least one run" in err


def test_gate_label_and_format_table():
    """A28 (SPEC.md 38.3.1): the gate column is PASS/MISS/—; the table
    carries one line per entry."""
    assert gate_label(None) == "—"
    assert gate_label(True) == "PASS"
    assert gate_label(False) == "MISS"
    rows = format_table([
        {"run_id": "r-1", "task": "t", "seed": 7, "policy": None,
         "final_score": 90.0, "target": 95.0, "met_target": False,
         "wall_seconds": 1.5, "experiments_run": 4, "parent_run": "p"},
        {"run_id": "r-2", "task": "t", "seed": 8, "policy": "bandit",
         "final_score": 96.0, "target": None, "met_target": None,
         "wall_seconds": 2.0, "experiments_run": 5, "parent_run": None},
    ])
    assert "r-1" in rows and "r-2" in rows
    assert "MISS" in rows and "—" in rows
    assert "bandit" in rows


# --- fit --from-run lineage (38.2, real data) ---------------------------------

def _cls_csv(path: Path) -> Path:
    """Non-linear 2-D classification (quadrant XOR) → softmax head
    (duplicated from test_generalization_v023.py)."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = ((x1 > 0) == (x2 > 0)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "x1", "x2", "churn"])
        for i in range(n):
            w.writerow(["A" if i % 2 else "B", f"{x1[i]:.5f}",
                        f"{x2[i]:.5f}", int(y[i])])
    return path


def _one_run_dir(base: Path, prefix: str) -> Path:
    dirs = sorted(base.glob(f"{prefix}-*"))
    assert len(dirs) == 1
    return dirs[0]


def test_fit_from_run_records_lineage(tmp_path, capsys):
    """A28 (SPEC.md 38.2.1): `fit --from-run` records the source run's
    dir name as `parent_run` in both `summary.json` and the registry
    entry, so the iteration chain is recoverable."""
    p = _cls_csv(tmp_path / "cls.csv")
    capsys.readouterr()
    # --target 40: below the ~47.5 baseline, so the first fit PASSes
    # (rc 0) — the lineage assertions don't depend on the verdict
    rc = cli_main(["fit", "--data", str(p), "--label", "churn", "--target", "40",
                   "--experiments", "1", "--runs-dir", str(tmp_path / "r1")])
    assert rc == 0
    dir1 = _one_run_dir(tmp_path / "r1", "csv-seed7")

    capsys.readouterr()
    rc = cli_main(["fit", "--from-run", str(dir1),
                   "--runs-dir", str(tmp_path / "r2")])
    assert rc == 0
    dir2 = _one_run_dir(tmp_path / "r2", "csv-seed7")
    s2 = json.loads((dir2 / "summary.json").read_text(encoding="utf-8"))
    assert s2["parent_run"] == dir1.name  # the lineage key (38.2.1)
    reg = json.loads(
        (tmp_path / "r2" / "registry.json").read_text(encoding="utf-8"))
    assert len(reg) == 1
    assert reg[0]["parent_run"] == dir1.name  # the entry field (38.1.2)
    assert reg[0]["run_id"] == dir2.name
    # the source run itself is a fresh run (no parent)
    s1 = json.loads((dir1 / "summary.json").read_text(encoding="utf-8"))
    assert "parent_run" not in s1


# --- 38.4 the app's Past-runs view (A28) ---------------------------------------

def test_diff_two_summaries_helper():
    """A28 (SPEC.md 38.4.1): `diff_two_summaries` returns the score
    delta and the per-field best_spec diff (only differing fields; a
    one-sided field diffs against None); identical specs diff empty."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from autorefine.dashboard_app import diff_two_summaries
    sa = {"final_best_score": 80.0,
          "best_spec": {"family": "mlp", "hidden_dim": 64, "lr": 0.001}}
    sb = {"final_best_score": 84.5,
          "best_spec": {"family": "mlp", "hidden_dim": 128, "lr": 0.001,
                        "knn_k": 4}}
    d = diff_two_summaries(sa, sb)
    assert d["score_a"] == 80.0 and d["score_b"] == 84.5
    assert d["score_delta"] == pytest.approx(4.5)
    assert d["n_diff_fields"] == 2
    got = {row["field"]: (row["a"], row["b"]) for row in d["spec_diff"]}
    assert got == {"hidden_dim": (64, 128), "knn_k": (None, 4)}
    # identical specs → no diff rows; missing scores → null delta
    assert diff_two_summaries(sa, sa)["spec_diff"] == []
    assert diff_two_summaries({}, {})["score_delta"] is None


def _fake_registry(runs: Path, n: int = 2) -> list[str]:
    """Two registry entries + two run dirs with minimal summaries, so
    the app's compare widget has something to diff."""
    ids = []
    for i in range(n):
        rid = f"csv-seed7-20260101-00000{i}"
        ids.append(rid)
        (runs / rid).mkdir(parents=True)
        (runs / rid / "summary.json").write_text(
            json.dumps({"final_best_score": 80.0 + i,
                        "best_spec": {"family": "mlp",
                                      "hidden_dim": 64 * (i + 1)}}),
            encoding="utf-8")
        append_entry(runs, {
            "run_id": rid, "task": "csv", "seed": 7, "policy": "bandit",
            "final_score": 80.0 + i, "target": 95.0, "met_target": False,
            "finished_reason": "budget_exhausted", "experiments_run": 1,
            "wall_seconds": 1.0, "config_fp": "abcdef123456",
            "parent_run": None, "timestamp": "2026-01-01T00:00:00Z",
        })
    return ids


def test_app_past_runs_section_renders(tmp_path):
    """A28 (SPEC.md 38.4.1/38.4.2): the Past-runs section renders the
    empty state (no registry) and, once runs are registered, the table
    + the two-run compare pickers — no exceptions either way."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    csv = _cls_csv(tmp_path / "cls.csv")
    runs = tmp_path / "runs"
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.text_input(key="runs_dir").set_value(str(runs))
    at.run()
    assert not at.exception
    assert any(e.value == "Past runs" for e in at.subheader)
    # empty state (38.4.2): the caption, no compare pickers
    captions = " ".join(e.value for e in at.caption)
    assert "No finished runs yet" in captions
    # AppTest's at.get(key) does not traverse into st.expander — the typed
    # element list is the reliable surface
    assert not any(e.key == "past_run_a" for e in at.selectbox)

    # populated: the table + the compare pickers render
    _fake_registry(runs, n=2)
    at.run()
    assert not at.exception
    assert any(e.value == "Past runs" for e in at.subheader)
    keys = {e.key for e in at.selectbox}
    assert "past_run_a" in keys and "past_run_b" in keys


# --- regression (A28) -----------------------------------------------------------

def test_version_round_v024():
    """A28 (SPEC.md 38.5, 33.1): the version stepped to `0.36.0` in both
    sources (advanced again with v0.36 ⇒ `0.36.0`, M39, SPEC.md 49)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.36.0"
