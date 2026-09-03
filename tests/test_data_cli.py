"""Data-in CLI & one-page report tests (SPEC.md 22.1/22.2, A12)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget, CsvTask
from autorefine.cli import _report_extras, main as cli_main
from autorefine.memory import RunMemory
from autorefine.plotting import html_report
from autorefine.tasks import TASKS

# --- deterministic fixture CSVs ------------------------------------------------


def _cls_csv(path: Path) -> Path:
    """Non-linear 2-D classification (quadrant XOR) + a non-numeric column."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = ((x1 > 0) == (x2 > 0)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "x1", "x2", "churn"])
        for i in range(n):
            w.writerow(["A" if i % 2 else "B", f"{x1[i]:.5f}", f"{x2[i]:.5f}", int(y[i])])
    return path


def _reg_csv(path: Path) -> Path:
    """Near-linear regression (fractional labels → mse head)."""
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = 2.5 * x - 1.0 + 0.05 * rng.standard_normal(n)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y"])
        for i in range(n):
            w.writerow([f"{x[i]:.5f}", f"{y[i]:.5f}"])
    return path


# --- SPEC.md 22.1: the csv task ------------------------------------------------


def test_csv_task_classification_protocol(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    t = CsvTask(seed=7, path=str(p), label="churn")
    assert t.head == "softmax" and t.n_outputs == 2 and t.state_dim == 2
    assert t.default_dataset_size == 320  # 400 * 0.8
    x, y = t.make_dataset()
    assert x.shape == (320, 2)
    assert y.dtype == np.int64 and set(np.unique(y)) <= {0, 1}
    # G2: a same-seed instance reproduces the dataset exactly
    t2 = CsvTask(seed=7, path=str(p), label="churn")
    x2, y2 = t2.make_dataset()
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert np.array_equal(t._x_ho, t2._x_ho) and np.array_equal(t._y_ho, t2._y_ho)
    assert np.array_equal(t._x_ge, t2._x_ge) and np.array_equal(t._y_ge, t2._y_ge)


def test_csv_task_split_sizes_and_prefix_matching(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    t = CsvTask(seed=7, path=str(p), label="churn")
    tr, ho, ge = t._rows_for("train"), t._rows_for("holdout"), t._rows_for("gen")
    assert len(tr[0]) == 320 and len(ho[0]) == 40 and len(ge[0]) == 40
    # §18.3 block-bootstrap names resolve by prefix (SPEC.md 22.1)
    assert len(t._rows_for("holdout-b3")[0]) == 40
    assert len(t._rows_for("gen-b2")[0]) == 40
    # n clamps to the fixed split length
    assert t.initial_conditions("holdout", 1000).shape[0] == 40
    # the three splits partition all 400 rows (no row twice)
    allx = np.vstack([tr[0], ho[0], ge[0]])
    assert allx.shape == (400, 2)


def test_csv_task_regression_head(tmp_path):
    p = _reg_csv(tmp_path / "reg.csv")
    t = CsvTask(seed=7, path=str(p))  # auto label: "y" hint
    assert t.head == "mse" and t.n_outputs == 1 and t.class_values is None
    x, y = t.make_dataset()
    assert x.shape == (240, 1) and y.dtype == np.float64
    assert t.label_name == "y"


def test_csv_task_label_rules_and_ignored_columns(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    # non-numeric "region" column is ignored (still 2 features)
    assert CsvTask(seed=7, path=str(p), label="churn").state_dim == 2
    # explicit label wins over hints; a missing name is an error
    assert CsvTask(seed=7, path=str(p), label="CHURN").label_name == "churn"
    with pytest.raises(ValueError, match="no label column"):
        CsvTask(seed=7, path=str(p), label="does-not-exist")


def test_csv_task_error_cases(tmp_path):
    p = tmp_path / "bad.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "c"])
        w.writerow(["s1", "s2", "0"])
        w.writerow(["s3", "s4", "1"])
    with pytest.raises(ValueError, match="no numeric feature columns"):
        CsvTask(seed=7, path=str(p), label="c")
    q = tmp_path / "tiny.csv"
    with q.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y"])
        w.writerow(["1", "0"])
        w.writerow(["2", "1"])
    with pytest.raises(ValueError, match="at least 3 rows"):
        CsvTask(seed=7, path=str(q), label="y")
    with pytest.raises(ValueError, match="data path"):
        CsvTask(seed=7)  # no path


# --- SPEC.md 22.1: task_config through the env, summary, eval -------------------


def _drive_to_done(env, seed=7):
    state = env.reset()
    policy = BanditPolicy(seed=seed)
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return env.memory.load_summary()


def test_env_task_config_roundtrip_and_eval(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
    )
    summary = _drive_to_done(env)
    assert summary["task"] == "csv"
    assert summary["task_config"] == {"path": str(p), "label": "churn"}
    # eval reconstructs the task from the recorded config and re-scores
    assert cli_main(["eval", "--run", str(env.run_dir)]) == 0


def test_builtin_env_task_config_none_unchanged(tmp_path):
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(1, 300, 30),
        runs_dir=tmp_path / "runs", task_config=None,
    )
    assert type(env.task).__name__ == "SineRegressionV1"
    summary = _drive_to_done(env)
    assert "task_config" not in summary  # absent for built-in tasks


def test_run_task_csv_without_config_errors(tmp_path):
    with pytest.raises(ValueError, match="data path"):
        cli_main(["run", "--task", "csv", "--experiments", "1",
                  "--runs-dir", str(tmp_path / "runs")])
    assert "csv" in TASKS  # registered, just needs a file


# --- SPEC.md 22.1: the fit CLI --------------------------------------------------


def _run_dirs(base: Path) -> list[Path]:
    return sorted(base.glob("csv-seed7-*"))


def test_fit_cli_pass_and_summary_config(tmp_path, capsys):
    p = _cls_csv(tmp_path / "cls.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "8", "--policy", "bandit", "--seed", "7",
                   "--target", "80.0", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out
    dirs = _run_dirs(runs)
    assert len(dirs) == 1
    summary = json.loads((dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_config"] == {"path": str(p), "label": "churn"}
    assert summary["final_best_score"] >= 80.0


def test_fit_cli_regression_target(tmp_path, capsys):
    p = _reg_csv(tmp_path / "reg.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--experiments", "4",
                   "--policy", "bandit", "--seed", "7",
                   "--target", "50.0", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out


def test_fit_cli_missing_file_errors(capsys):
    rc = cli_main(["fit", "--data", "/definitely/not/here.csv"])
    err = capsys.readouterr().err
    assert rc == 1 and "no such file" in err


# --- SPEC.md 22.2: report --html -------------------------------------------------


def test_report_html_written_and_deterministic(tmp_path, capsys):
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
    )
    summary = _drive_to_done(env)
    assert cli_main(["report", "--run", str(env.run_dir), "--html"]) == 0
    html_path = Path(env.run_dir) / "report.html"
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    # the two core plots are embedded (SPEC.md 21.2); the v0.14 learning
    # views (C1-C4) add more SVGs, so assert the core ones by title, not a
    # fixed count (SPEC.md 28.1-28.4)
    assert text.count("<svg") >= 2
    assert "score vs experiment" in text
    assert "pareto frontier" in text
    assert f"{summary['final_best_score']:.3f}" in text
    assert "churn" in text  # task config section
    # G2: pure function of (summary, entries) + the same None-safe extras
    mem = RunMemory(Path(env.run_dir))
    extras = _report_extras(summary, Path(env.run_dir))
    assert html_report(summary, mem.load_experiments(),
                       diagnostics=extras["diagnostics"],
                       gallery=extras["gallery"],
                       arch_svg=extras["arch_svg"]) == text


def test_report_html_json_keeps_stdout_pure(tmp_path, capsys):
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
    )
    expected = _drive_to_done(env)
    assert cli_main(["report", "--run", str(env.run_dir), "--html", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == expected
    assert (Path(env.run_dir) / "report.html").exists()
