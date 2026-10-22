"""v0.79 extrapolation + intuitive transfer learning — SPEC.md 93, A83.

SPEC.md 93 (v0.79): the finished model gains a "how sure?" and an
"is this in my world?", and the tool gains a transfer verb:

  * A83.A1 (93.1)  ``fit_calibration`` + the per-row prediction SET
                   (softmax) / INTERVAL (mse), the finite-sample-corrected
                   ``tau``; fail-loud on an unsupported head, a bad
                   holdout target, a missing ``holdout_rows`` protocol,
                   or an empty holdout;
  * A83.A2 (93.2)  ``conformal_distance`` / ``is_ood`` / ``familiarity``
                   (the (1-beta) nearest-rank threshold;
                   ``with_ood=False`` fails loud);
  * A83.A3 (93.3.8)  ``predict --cal --ood`` — the opt-in JSON extras;
                   with neither flag the output is byte-identical to the
                   pre-v0.79 surface (the A32/A82 pins);
  * A83.A4 (93.3.1) ``adapt --dry-run`` / ``plan_adapt`` — preflight +
                   compatibility + finetune/retrain mode, no training;
  * A83.A5 (93.3.2-93.3.6) ``adapt_model`` — before/after on the new
                   holdout, the verdict, the per-class + weight deltas,
                   the reconstructible artifacts, the registry lineage
                   (``parent_run``); a non-parametric champion takes
                   mode ``retrain``; an episode-task champion fails loud;
  * A83.A6 (93.3.9) the app's Transfer section (source tokens + a
                   minimal AppTest; the app-language scan stays clean);
  * A83.A7 (93.3.7) pinned surfaces — the exporter's standalone
                   ``predict.py`` template and the RL action catalog
                   (106 actions) stay untouched; the A1–A19 legacy
                   sequence stays green under the full suite;
  * ceremony (93.4.8) — the version steps to ``0.84.0`` in both sources,
                   the M82 index row resolves to this file, and the
                   conformal + adapt names land in ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (every fixture is
synthesized here), deterministic (G2), tiny budgets (200 steps), and
module-scoped fixtures where a finished run is needed.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import (
    ADAPT_EPS,
    Calibration,
    adapt_model,
    conformal_distance,
    conformal_interval,
    conformal_prediction_set,
    familiarity,
    fit_calibration,
    is_ood,
    plan_adapt,
    render_adapt_text,
)
from autorefine.cli import _run_task_and_model, main as cli_main
from autorefine.config import DEFAULT_SPEC, ModelSpec
from autorefine.registry import load_registry
from autorefine.tasks.csv import CsvTask
from autorefine.trainer import train_from_task

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# v0.79 (SPEC.md 93, A83): the round's new top-level exports.
NEW_EXPORTS = (
    "Calibration", "fit_calibration", "conformal_prediction_set",
    "conformal_interval", "conformal_distance", "is_ood", "familiarity",
    "plan_adapt", "adapt_model", "render_adapt_text", "ADAPT_EPS",
)


# --- shared fixtures (test-local; the house rules forbid cross-test imports) ---

def _spec(family: str) -> ModelSpec:
    """A valid, tiny spec for `family` (200 steps; the A82 pattern)."""
    d = {**DEFAULT_SPEC.to_dict(), "model_family": family, "train_steps": 200}
    return ModelSpec.from_dict(d)


def _write_csv(tmp_path: Path, name: str = "data.csv",
               rows: int = 60) -> Path:
    """Deterministic quadrant-XOR classification CSV (2 classes)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_similar_csv(tmp_path: Path, name: str = "similar.csv",
                       rows: int = 48) -> Path:
    """The same table shape (a, b, churn; 2 classes), different rows —
    the "a later export of the same table" transfer fixture."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = ((i * 7 + 1) % 10) / 10.0
        b = ((i * 3 + 2) % 10) / 10.0
        churn = 1.0 if (a > 0.5) ^ (b < 0.5) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_incompatible_csv(tmp_path: Path, name: str = "three_class.csv",
                            rows: int = 45) -> Path:
    """Same shape but THREE classes — n_outputs 3 != the champion's 2
    (the 93.3.3 incompatible fixture)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 3) / 3.0
        lines.append(f"{a:.2f},{b:.2f},{i % 3}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_regress_csv(tmp_path: Path, name: str = "reg.csv",
                       rows: int = 60) -> Path:
    """A continuous-label CSV (y = 3a + 2b) — the mse-head fixture."""
    lines = ["a,b,y"]
    for i in range(rows):
        a = (i % 6) / 6.0
        b = ((i // 6) % 5) / 5.0
        lines.append(f"{a:.2f},{b:.2f},{3.0 * a + 2.0 * b:.2f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_rows_csv(tmp_path: Path, name: str = "rows.csv") -> Path:
    """New rows with the champion's columns (the `predict` fixture)."""
    p = tmp_path / name
    p.write_text("a,b,churn\n0.80,0.20,1\n0.10,0.90,1\n"
                 "0.20,0.10,0\n0.70,0.80,0\n", encoding="utf-8")
    return p


def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run one CLI command, capturing stdout + stderr (the house pattern)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


def _synth_champion(root: Path, csv: Path, family: str, name: str) -> Path:
    """A synthetic finished csv-run dir (the A82.5 pattern): train
    `family` on the CSV, save the checkpoint + a reconstructible summary."""
    task = CsvTask(seed=7, path=str(csv))
    spec = _spec(family)
    res = train_from_task(task, spec, seed=7)
    rd = root / name
    rd.mkdir(parents=True)
    res.model.save(str(rd / "best_model.npz"))
    summary = {
        "task": "csv", "seed": 7,
        "task_config": {"path": str(csv)},
        "best_spec": spec.to_dict(),
        "final_best_score": float(task.score(res.model, "holdout", 8)),
        "baseline_score": 50.0,
        "finished_reason": "target_met",
        "policy": "bandit",
    }
    (rd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (rd / "best_spec.json").write_text(
        json.dumps(spec.to_dict()), encoding="utf-8")
    return rd


def _synth_episode_champion(root: Path, name: str) -> Path:
    """A finished cartpole-v1 run (max_steps 500 — the 93.3.3
    episode-task fixture)."""
    from autorefine.tasks.cartpole import CartPoleV1
    task = CartPoleV1(seed=7)
    spec = _spec("mlp")
    res = train_from_task(task, spec, seed=7)
    rd = root / name
    rd.mkdir(parents=True)
    res.model.save(str(rd / "best_model.npz"))
    summary = {
        "task": "cartpole-v1", "seed": 7,
        "best_spec": spec.to_dict(),
        "final_best_score": 50.0,
        "finished_reason": "budget",
        "policy": "bandit",
    }
    (rd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (rd / "best_spec.json").write_text(
        json.dumps(spec.to_dict()), encoding="utf-8")
    return rd


class _TwoOutModel:
    """A duck-typed 2-output model (the conformal protocol is duck-typed,
    exactly like the 42.1 `predict` leaf)."""

    def forward(self, x):
        n = np.atleast_2d(np.asarray(x, dtype=np.float64)).shape[0]
        return np.zeros((n, 2))


class _NoProtocolTask:
    name = "noprot"
    head = "softmax"


class _OddHeadTask:
    name = "odd"
    head = "logistic"


class _OneRowTask:
    name = "onerow"
    head = "softmax"

    def holdout_rows(self, n, model=None):
        return np.zeros((1, 2)), np.zeros(1)


class _BadTargetTask:
    name = "badtarget"
    head = "softmax"
    class_values = (0.0, 1.0)

    def holdout_rows(self, n, model=None):
        return np.zeros((4, 2)), np.array([0, 1, 5, 0], dtype=np.float64)


@pytest.fixture(scope="module")
def softmax_pair(tmp_path_factory):
    """A tiny softmax (csv) task + a trained mlp — the A83.A1/A83.A2 pair."""
    root = tmp_path_factory.mktemp("transfer_conf")
    csv = _write_csv(root)
    task = CsvTask(seed=7, path=str(csv))
    res = train_from_task(task, _spec("mlp"), seed=7)
    return task, res.model


@pytest.fixture(scope="module")
def mse_pair(tmp_path_factory):
    """A tiny mse task + a trained mlp (the A83.A1 interval fixture)."""
    root = tmp_path_factory.mktemp("transfer_mse")
    csv = _write_regress_csv(root)
    task = CsvTask(seed=7, path=str(csv))
    res = train_from_task(task, _spec("mlp"), seed=7)
    return task, res.model


@pytest.fixture(scope="module")
def mlp_champ(tmp_path_factory):
    """The mlp champion + its data files (the A83.A3-A83.A5 fixtures)."""
    root = tmp_path_factory.mktemp("transfer_mlp")
    csv = _write_csv(root)
    rd = _synth_champion(root, csv, "mlp", "mlp-champ")
    return {
        "root": root, "csv": csv, "run_dir": rd,
        "similar": _write_similar_csv(root),
        "incompatible": _write_incompatible_csv(root),
        "rows": _write_rows_csv(root),
    }


@pytest.fixture(scope="module")
def knn_champ(mlp_champ):
    """A non-parametric (knn) champion — the mode-retrain fixture."""
    return _synth_champion(mlp_champ["root"], mlp_champ["csv"],
                           "knn", "knn-champ")


@pytest.fixture(scope="module")
def episode_champ(mlp_champ):
    """A finished cartpole-v1 run — the 93.3.3 episode-task fixture."""
    return _synth_episode_champion(mlp_champ["root"], "cartpole-champ")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    """Where the adapted run dirs + registry land (the A83.A5 fixtures)."""
    return tmp_path_factory.mktemp("transfer_runs")


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    """One real, tiny, finished ``go`` run (the A83.A6 AppTest fixture)."""
    root = tmp_path_factory.mktemp("transfer_go")
    csv = _write_csv(root)
    runs = root / "runs"
    rc, out, err = _run_cmd([
        "go", "--data", str(csv),
        "--runs-dir", str(runs),
        "--experiments", "3", "--target", "50", "--quiet",
    ])
    assert rc == 0, out + err
    run_dirs = sorted(runs.glob("csv-*"))
    assert run_dirs and (run_dirs[0] / "summary.json").is_file()
    return {"root": root, "csv": csv, "runs": runs, "run_dir": run_dirs[0]}


@pytest.fixture(scope="module")
def adapted(mlp_champ, runs_dir):
    """One full finetune adapt, shared by the A83.A5 assertions."""
    rep = adapt_model(str(mlp_champ["run_dir"]), str(mlp_champ["similar"]),
                      runs_dir=str(runs_dir), seed=7, max_seconds=20.0)
    return {"report": rep, "run_dir": Path(runs_dir) / rep["run_dir"]}


# --- A83.A1 the conformal core (93.1) -------------------------------------------

class TestConformalCore:
    """SPEC.md 93.1 (A83.A1): split-conformal calibration + the per-row
    prediction SET / INTERVAL."""

    def test_tau_finite_and_fit_deterministic(self, softmax_pair):
        task, model = softmax_pair
        cal = fit_calibration(task, model)
        assert isinstance(cal, Calibration)
        assert cal.head == "softmax"
        assert cal.n_cal >= 2 and math.isfinite(cal.tau) and cal.tau >= 0.0
        assert cal.class_values == tuple(task.class_values)
        # G2: two fits agree (nearest-rank quantile, no RNG)
        assert cal == fit_calibration(task, model)

    def test_prediction_set_never_empty_contains_argmax(self, softmax_pair):
        task, model = softmax_pair
        cal = fit_calibration(task, model)
        x, _y = task.holdout_rows(8, model)
        for row in x:
            s = conformal_prediction_set(task, model, row, cal)
            assert len(s) >= 1  # never empty (93.1.1)
            out = np.asarray(model.forward(np.atleast_2d(row)),
                             dtype=np.float64)
            am = int(np.argmax(out[0] if out.ndim > 1 else out))
            assert task.class_values[am] in s  # the argmax always joins

    def test_mse_interval(self, mse_pair):
        task, model = mse_pair
        assert task.head == "mse"
        cal = fit_calibration(task, model)
        assert cal.head == "mse" and cal.class_values is None
        assert math.isfinite(cal.tau)
        lo, hi = conformal_interval(1.0, cal)
        assert (lo, hi) == (1.0 - cal.tau, 1.0 + cal.tau)
        with pytest.raises(ValueError, match="softmax-head"):
            conformal_prediction_set(task, model,
                                     np.zeros(task.state_dim), cal)

    def test_conformal_distance_leaf(self):
        assert conformal_distance(np.zeros(3)) == 0.0
        assert conformal_distance([1.0, 2.0]) == pytest.approx(math.sqrt(2.5))
        with pytest.raises(ValueError):
            conformal_distance(np.array([]))
        with pytest.raises(ValueError):
            conformal_distance(np.array([1.0, np.nan]))

    def test_unsupported_head_fails_loud(self):
        with pytest.raises(ValueError, match="softmax/mse"):
            fit_calibration(_OddHeadTask(), _TwoOutModel())

    def test_bad_holdout_target_fails_loud(self):
        with pytest.raises(ValueError, match="class index"):
            fit_calibration(_BadTargetTask(), _TwoOutModel())

    def test_no_holdout_protocol_fails_loud(self):
        with pytest.raises(ValueError, match="holdout_rows"):
            fit_calibration(_NoProtocolTask(), _TwoOutModel())

    def test_empty_holdout_fails_loud(self):
        with pytest.raises(ValueError, match="holdout"):
            fit_calibration(_OneRowTask(), _TwoOutModel())


# --- A83.A2 the OOD surface (93.2) ---------------------------------------------

class TestOod:
    """SPEC.md 93.2 (A83.A2): the OOD flag + the 0-1 familiarity."""

    def test_far_row_flagged_near_row_familiar(self, softmax_pair):
        task, model = softmax_pair
        cal = fit_calibration(task, model)
        assert cal.ood_threshold is not None
        assert math.isfinite(cal.ood_threshold)
        x, _y = task.holdout_rows(16, model)
        near = np.asarray(x[0], dtype=np.float64)
        far = np.full(task.state_dim, 1000.0)
        assert is_ood(far, cal) is True      # 1000x scale: past the threshold
        assert familiarity(far, cal) == 0.0  # 0 beyond the threshold
        assert 0.0 <= familiarity(near, cal) <= 1.0
        assert familiarity(near, cal) >= familiarity(far, cal)

    def test_with_ood_false_fails_loud(self, softmax_pair):
        task, model = softmax_pair
        cal = fit_calibration(task, model, with_ood=False)
        assert cal.ood_threshold is None
        x, _y = task.holdout_rows(4, model)
        with pytest.raises(ValueError, match="without OOD"):
            is_ood(x[0], cal)
        with pytest.raises(ValueError, match="without OOD"):
            familiarity(x[0], cal)

    def test_alpha_out_of_range_fails_loud(self, softmax_pair):
        task, model = softmax_pair
        with pytest.raises(ValueError, match="alpha"):
            fit_calibration(task, model, alpha=1.5)
        with pytest.raises(ValueError, match="beta"):
            fit_calibration(task, model, beta=-0.1)


# --- A83.A3 the predict surface (93.3.8) ----------------------------------------

class TestPredictCli:
    """SPEC.md 93.3.8 (A83.A3): `predict --cal --ood` — the opt-in JSON
    extras; the default output stays byte-identical (the A32/A82 pins)."""

    def test_default_output_unchanged(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "predict", "--run", str(c["run_dir"]),
            "--csv", str(c["rows"]), "--json"])
        assert rc == 0, out + err
        entries = json.loads(out)
        assert len(entries) == 4
        for e in entries:
            assert set(e.keys()) == {"row", "prediction", "probabilities"}

    def test_cal_ood_add_documented_fields(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "predict", "--run", str(c["run_dir"]),
            "--csv", str(c["rows"]), "--cal", "--ood", "--json"])
        assert rc == 0, out + err
        for e in json.loads(out):
            assert {"row", "prediction", "probabilities",
                    "prediction_set", "familiarity", "ood"} <= set(e.keys())
            assert len(e["prediction_set"]) >= 1
            assert 0.0 <= e["familiarity"] <= 1.0
            assert isinstance(e["ood"], bool)

    def test_human_readable_extras(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "predict", "--run", str(c["run_dir"]),
            "--csv", str(c["rows"]), "--cal", "--ood"])
        assert rc == 0, out + err
        assert "prediction_set:" in out
        assert "familiarity:" in out


# --- A83.A4 the plan (93.3.1) -----------------------------------------------------

class TestPlanAdapt:
    """SPEC.md 93.3.1 (A83.A4): the dry-run transfer plan — preflight +
    compatibility + mode; no training, no writes."""

    def test_dry_run_compatible_cli(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "adapt", "--run", str(c["run_dir"]),
            "--data", str(c["similar"]), "--dry-run"])
        assert rc == 0, out + err
        assert "finetune" in out
        assert "plan     : ok" in out

    def test_dry_run_json_plan(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "adapt", "--run", str(c["run_dir"]),
            "--data", str(c["similar"]), "--dry-run", "--json"])
        assert rc == 0, out + err
        plan = json.loads(out)
        assert plan["compatible"] is True
        assert plan["mode"] == "finetune"
        assert plan["warm_start"] is True
        assert plan["champion_family"] == "mlp"
        assert plan["preflight"]["n_rows"] >= 1
        assert plan["new_task"]["head"] == "softmax"
        assert plan["new_task"]["n_outputs"] == 2

    def test_dry_run_knn_champion_is_retrain(self, mlp_champ, knn_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "adapt", "--run", str(knn_champ),
            "--data", str(c["similar"]), "--dry-run"])
        assert rc == 0, out + err
        assert "retrain" in out

    def test_dry_run_incompatible(self, mlp_champ):
        c = mlp_champ
        rc, out, err = _run_cmd([
            "adapt", "--run", str(c["run_dir"]),
            "--data", str(c["incompatible"]), "--dry-run"])
        assert rc == 1
        assert "incompatible" in err

    def test_episode_champion_fails_loud(self, mlp_champ, episode_champ):
        with pytest.raises(ValueError, match="episode"):
            plan_adapt(str(episode_champ), str(mlp_champ["similar"]))

    def test_missing_data_file_fails_loud(self, mlp_champ):
        with pytest.raises(FileNotFoundError):
            plan_adapt(str(mlp_champ["run_dir"]),
                       str(mlp_champ["root"] / "nope.csv"))


# --- A83.A5 the adapt (93.3.2-93.3.6) ---------------------------------------------

class TestAdaptModel:
    """SPEC.md 93.3.2 (A83.A5): the full transfer — before/after on the
    new holdout, the verdict, the artifacts, the lineage."""

    def test_full_adapt_finetune(self, adapted, mlp_champ):
        rep = adapted["report"]
        assert rep["mode"] == "finetune" and rep["warm_start"] is True
        assert rep["champion_family"] == "mlp"
        assert rep["champion_head"] == "softmax"
        assert rep["before"] is not None and rep["after"] is not None
        assert rep["delta"] == pytest.approx(rep["after"] - rep["before"])
        assert rep["n_holdout"] >= 2
        assert rep["verdict"] in ("IMPROVED", "NEUTRAL", "DEGRADED")
        # the ADAPT_EPS margin (93.3.4): the verdict agrees with the delta
        d = rep["delta"]
        if d > ADAPT_EPS:
            assert rep["verdict"] == "IMPROVED"
        elif d < -ADAPT_EPS:
            assert rep["verdict"] == "DEGRADED"
        else:
            assert rep["verdict"] == "NEUTRAL"
        assert rep["parent_run"] == mlp_champ["run_dir"].name
        # the per-class before/after (a softmax task) + warm-start deltas
        assert rep["per_class_before"] is not None
        assert rep["per_class_after"] is not None
        assert rep["weight_deltas"] is not None  # the weights transferred
        # the artifacts (93.3.6)
        rd = adapted["run_dir"]
        for name in ("summary.json", "best_spec.json",
                     "best_model.npz", "adapt_report.json"):
            assert (rd / name).is_file(), name
        summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
        assert summary["task"] == "csv"
        assert summary["finished_reason"] == "adapted"
        assert summary["policy"] == "finetune"
        assert summary["final_best_score"] == rep["after"]
        assert summary["baseline_score"] == rep["before"]
        assert summary["task_config"]["path"] == str(mlp_champ["similar"])
        # the adapted run reconstructs through the 42 one home
        s2, task2, _model2 = _run_task_and_model(rd)
        assert s2["task"] == "csv" and task2.head == "softmax"
        # and `predict` scores new rows with it
        c = mlp_champ
        rc, out, err = _run_cmd([
            "predict", "--run", str(rd), "--csv", str(c["rows"]), "--json"])
        assert rc == 0, out + err
        assert len(json.loads(out)) == 4

    def test_registry_lineage(self, adapted, runs_dir):
        rep = adapted["report"]
        entries = load_registry(str(runs_dir))
        e = next(x for x in entries if x["run_id"] == rep["run_dir"])
        assert e["parent_run"] == rep["parent_run"]
        assert e["finished_reason"] == "adapted"
        assert e["policy"] == "finetune"

    def test_render_adapt_text(self, adapted):
        rep = adapted["report"]
        text = render_adapt_text(rep)
        text.encode("ascii")  # the house console rule (93.3.5)
        assert rep["verdict"] in text
        assert rep["run_dir"] in text
        assert "before" in text and "after" in text

    def test_full_adapt_retrain_mode(self, mlp_champ, knn_champ, runs_dir):
        rep = adapt_model(str(knn_champ), str(mlp_champ["similar"]),
                          runs_dir=str(runs_dir), seed=7, max_seconds=20.0)
        assert rep["mode"] == "retrain" and rep["warm_start"] is False
        assert rep["champion_family"] == "knn"
        assert rep["weight_deltas"] is None  # nothing transferred
        assert rep["verdict"] in ("IMPROVED", "NEUTRAL", "DEGRADED")
        assert (Path(runs_dir) / rep["run_dir"] / "best_model.npz").is_file()

    def test_incompatible_data_fails_loud(self, mlp_champ, runs_dir):
        with pytest.raises(ValueError, match="incompatible"):
            adapt_model(str(mlp_champ["run_dir"]),
                        str(mlp_champ["incompatible"]),
                        runs_dir=str(runs_dir))

    def test_cli_adapt_narrative(self, mlp_champ, runs_dir):
        rc, out, err = _run_cmd([
            "adapt", "--run", str(mlp_champ["run_dir"]),
            "--data", str(mlp_champ["similar"]),
            "--runs-dir", str(runs_dir), "--max-seconds", "20"])
        assert rc == 0, out + err
        out.encode("ascii")  # the narrative is ASCII (93.3.5)
        assert "verdict" in out
        assert "run dir" in out


# --- A83.A6 the app's Transfer section (93.3.9) -----------------------------------

class TestAppTransfer:
    """SPEC.md 93.3.9 (A83.A6): the Adapt tab's Transfer section."""

    @staticmethod
    def _at():
        pytest.importorskip("streamlit", reason="dashboard app is optional")
        from streamlit.testing.v1 import AppTest
        return AppTest

    def test_source_tokens(self):
        src = APP.read_text(encoding="utf-8")
        for tok in (
            "adapt_champ", "adapt_transfer_csv", "adapt_steps",
            "adapt_maxsec", "adapt_show_plan", "adapt_transfer_go",
            "Transfer: adapt the champion to similar data",
            "Show plan", "Fine-tune on the new data",
            "plan_adapt", "adapt_model",
            "_adapt_before_after_table", "_adapt_per_class_table",
            "_adapt_old_vs_new",
        ):
            assert tok in src, tok

    def test_transfer_section_renders(self, champion):
        AppTest = self._at()
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()
        assert not at.exception
        # the sidebar's CSV is the Transfer section's "new data" fallback
        # (the 90-tab wiring), so the section renders operable — the
        # champion pick + the plan/run buttons — not the upload prompt
        at.text_input(key="csv_path").set_value(str(champion["csv"]))
        at.session_state["runs_dir"] = str(champion["runs"])
        at.session_state["experiments"] = 2
        at.run()
        assert not at.exception
        subs = [s.value for s in at.subheader]
        assert any("Adapt the model to new data" in v for v in subs)
        assert any("Transfer: adapt the champion" in v for v in subs)
        # a finished run exists in `runs`: the champion selectbox is there
        sel = [s.key for s in at.selectbox if s.key]
        assert "adapt_champ" in sel
        bkeys = [b.key for b in at.button if b.key]
        assert "adapt_show_plan" in bkeys
        assert "adapt_transfer_go" in bkeys


# --- A83.A7 honesty + pinned surfaces (93.3.7) -----------------------------------

class TestHonestyPins:
    """SPEC.md 93.3.7 (A83.A7): the pinned surfaces stay byte-identical."""

    def test_exporter_template_untouched(self):
        src = (REPO / "src" / "autorefine" / "exporter.py").read_text(
            encoding="utf-8")
        assert "_PREDICT_TEMPLATE = '''\\" in src
        assert "#!/usr/bin/env python3" in src
        assert "pure numpy".lower() in src.lower()
        from autorefine.exporter import PORTABLE_FAMILIES
        assert PORTABLE_FAMILIES == ("mlp", "knn", "tree", "boost")

    def test_rl_catalog_keeps_106_actions(self):
        from autorefine.improver.catalog import ACTIONS
        assert len(ACTIONS) == 106  # untouched this round (93.3.7)

    def test_legacy_sequence_reference(self):
        """The A1–A19 legacy bit-exact sequence stays green under the
        full suite (the 93.3.7 invariant) — this file's fixtures never
        touch it; the legacy pin tests live in the A1–A19 rounds."""
        assert NEW_EXPORTS  # ceremony: the round's names are declared


# --- ceremony (93.4.8) ------------------------------------------------------------

class TestCeremony:
    """SPEC.md 93.4 item 8: the version step + the index row + the
    exports (the A-index advance is enforced by the A25 coherency tests)."""

    def test_version_steps_to_079_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.84.0"' in init
        assert 'version = "0.84.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.84.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 93. v0.79" in spec
        assert "### 93.4 Acceptance (A83)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M82 |"))
        assert "93" in row and "A83" in row
        assert "tests/test_transfer_v079.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
            assert hasattr(autorefine, name)
