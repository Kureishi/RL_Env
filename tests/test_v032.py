"""v0.32 — probabilistic outputs, curriculum beyond parity, portfolio mode.

SPEC.md 46, A36, M35.

46.1 probabilistic CSV outputs — `predict --prob` prints one
`<row>\t<class>: p=…` line per row, per class (softmax head only; a mse
head is rc 1 naming the head, SPEC.md 46.1); `--json` is unchanged
(display-only flag). `CsvTask(metric="logloss")` declares the metric as
a task property (G1): a perfect model → 100, a constant (uniform) model
→ 0 (clamped), an mse head is a construction ValueError citing 46.1;
`fit --metric logloss` runs end-to-end (the gate line names logloss),
the default `--metric accuracy` is byte-identical (the dry-run plan is
unchanged), and `fit --dry-run --metric logloss` names the metric while
omitting the accuracy headroom note (46.1.3).

46.2 curriculum beyond parity — `SineRegressionV1(seed, noise,
freq_scale, amplitude)` (defaults bit-identical to the historical
constants), the `sine_ceiling` monotonicity, the 3-axis SineCurriculum
(noise → freq → amplitude; 27 levels; the exact trigger threshold),
`CartPoleV1(seed, ic_scale)` (default ICs bit-identical; the scale-2 box
is exactly 2×), the single-axis CartPoleCurriculum (1.0 → 2.0 → 3.0;
ceiling proxy 500.0), the shared `level_params()` API (parity keeps its
historical n_bits/p_flip event-row keys), `run --curriculum` dispatch by
task (sine/cartpole rc 0; gridnav is rc 1 naming the three), and the
top-level package exports.

46.3 portfolio mode — `fit --tasks A.csv,B.csv` runs both loops under one
shared budget (two run dirs, two registry rows, rc 0 iff every task
PASSes), is mutually exclusive with `--data`/`--from-run` (rc 1), needs
≥ 2 paths (rc 1), and `fit --dry-run --tasks` prints the per-task plans
with zero artifacts.

Regression — A1–A35 stay green (the rest of the suite); the A25 index
advances (defined == set(range(1, 37))); the version stepped to `0.37.0`
in both sources (33.1).

House rules: no cross-test imports (fixtures duplicated per file);
stdlib + numpy only; every test cites A36 + its SPEC §.
"""
import csv as _csv
import json
import re
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget
from autorefine.cli import build_parser, main as cli_main
from autorefine.improver.curriculum import (
    CartPoleCurriculum,
    ParityCurriculum,
    SineCurriculum,
)
from autorefine.registry import load_registry
from autorefine.tasks.cartpole import CartPoleV1
from autorefine.tasks.csv import CsvTask
from autorefine.tasks.sine import (
    LABEL_NOISE,
    SineRegressionV1,
    sine_ceiling,
    target_function,
)

import autorefine

REPO = Path(__file__).resolve().parent.parent

# the `predict --prob` line: `<row>\t<class>: p=<4 decimals>` (46.1.2)
_P_LINE = re.compile(r"^(\d+)\t([^:]+): p=(\d\.\d{4})$")


# --- fixtures (duplicated per file, no cross-test imports) --------------------

def _cls_csv(base: Path, name: str = "cls.csv") -> Path:
    """60 rows (30 per class, exact 50/50), 2 features, linearly
    separable: label 1 iff x1 > 0 (SPEC.md 46.1/46.3 battery). An MLP at
    any budget fits it — the `fit` PASS gate and the logloss anchors."""
    rnd = np.random.default_rng(0)
    p = base / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x1", "x2", "label"])
        for i in range(60):
            y = 1 if i < 30 else 0
            x1 = float(rnd.uniform(0.01, 1.0)) if y \
                else float(rnd.uniform(-1.0, -0.01))
            x2 = float(rnd.uniform(-1.0, 1.0))
            w.writerow([f"{x1:.6f}", f"{x2:.6f}", y])
    return p


def _mse_csv(base: Path) -> Path:
    """40 rows, one feature, non-integer float labels — the §22.1 rule
    sends this to the mse head (the 46.1 logloss ValueError path)."""
    rnd = np.random.default_rng(1)
    p = base / "mse.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x", "y"])
        for _ in range(40):
            x = float(rnd.uniform(-1.0, 1.0))
            y = 2.0 * x + 1.5 + 0.1 * float(rnd.standard_normal())
            w.writerow([f"{x:.6f}", f"{y:.6f}"])
    return p


class _Oracle:
    """A per-row perfect predictor (test stub): class 1 iff x1 > 0 (the
    fixture's separability rule), the true class at +50 logits, the other
    at -50 — softmax p = 1 in float64 → NLL 0 → the logloss score 100
    (46.1.2). Predicting from x (not a fixed label array) keeps it
    row-aligned on score_fold's ho ∪ gen pool (44.1.2)."""

    def forward(self, x):
        x = np.asarray(x)
        pred = (x[:, 0] > 0).astype(int)
        logits = np.full((len(x), 2), -50.0)
        logits[np.arange(len(x)), pred] = 50.0
        return logits


class _Uniform:
    """Constant zero logits: every class p = 0.5 → NLL ln 2 → the
    logloss score 0 (clamped); argmax always class 0 → accuracy = the
    class-0 share (the unchanged 22.1 branch)."""

    def forward(self, x):
        return np.zeros((len(x), 2))


def _csv_run(tmp_path: Path) -> tuple:
    """A finished csv run (softmax, separable 0/1) for the predict tests;
    returns (the csv file, the run dir)."""
    p = _cls_csv(tmp_path)
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 10),
        runs_dir=tmp_path / "runs",
        task_config={"path": str(p)}, dataset_episodes=48)
    state = env.reset()
    policy = BanditPolicy(seed=7)
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return p, Path(env.run_dir)


def _sine_run(tmp_path: Path) -> Path:
    """A finished sine-v1 run (mse head — the `--prob` error path)."""
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(2, 300, 10),
                        runs_dir=tmp_path / "runs")
    state = env.reset()
    policy = BanditPolicy(seed=7)
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return Path(env.run_dir)


# --- 46.1 predict --prob -------------------------------------------------------

def test_predict_prob_prints_per_class_lines(tmp_path, capsys):
    """A36 (SPEC.md 46.1.2): `predict --prob` prints one
    `<row>\t<class>: p=…` line per row, per class — the user's class
    labels in class order, probabilities to 4 decimals, summing to 1
    within the 4-decimal rounding (rc 0)."""
    p, run = _csv_run(tmp_path)
    rc = cli_main(["predict", "--run", str(run), "--csv", str(p), "--prob"])
    out = capsys.readouterr().out
    assert rc == 0, out
    lines = out.splitlines()
    assert len(lines) == 60 * 2  # every row × the 2 classes
    for r in range(60):
        ma = _P_LINE.match(lines[2 * r])
        mb = _P_LINE.match(lines[2 * r + 1])
        assert ma is not None and mb is not None, lines[2 * r:2 * r + 2]
        assert int(ma.group(1)) == r and int(mb.group(1)) == r
        assert ma.group(2) == "0" and mb.group(2) == "1"  # class order
        assert float(ma.group(3)) + float(mb.group(3)) \
            == pytest.approx(1.0, abs=5e-4)  # the calibrated distribution


def test_predict_json_unchanged_by_prob(tmp_path, capsys):
    """A36 (SPEC.md 46.1.2): `--prob` is display-only — the `--json`
    output is byte-identical with and without it (the 42.1.4 shape)."""
    p, run = _csv_run(tmp_path)
    base = ["predict", "--run", str(run), "--csv", str(p)]
    assert cli_main(base + ["--json"]) == 0
    out_plain = capsys.readouterr().out
    assert cli_main(base + ["--json", "--prob"]) == 0
    out_prob = capsys.readouterr().out
    assert out_plain == out_prob
    arr = json.loads(out_plain)
    assert len(arr) == 60
    for row in arr:
        assert set(row) == {"row", "prediction", "probabilities"}


def test_predict_prob_mse_head_is_error(tmp_path, capsys):
    """A36 (SPEC.md 46.1.4): `--prob` on a non-softmax (mse) head is
    rc 1 with a stderr message naming the head (never a silent
    omission)."""
    run = _sine_run(tmp_path)
    rc = cli_main(["predict", "--run", str(run), "--row", "[0.5]", "--prob"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "softmax" in err and "46.1" in err


# --- 46.1 the logloss metric ----------------------------------------------------

def test_logloss_scoring_contract(tmp_path):
    """A36 (SPEC.md 46.1.2): a perfect model scores 100; a constant
    (uniform) model scores 0 (clamped); `metric` is the declared task
    property (G1); the default accuracy branch is unchanged (uniform →
    the class-0 share, exactly 50.0 on the 50/50 fixture)."""
    p = _cls_csv(tmp_path)
    t = CsvTask(seed=7, path=p, metric="logloss")
    assert t.head == "softmax" and t.metric == "logloss"
    assert t.score(_Oracle(), "holdout", 1000) == pytest.approx(100.0)
    assert t.score(_Uniform(), "holdout", 1000) == pytest.approx(0.0)
    # score_fold shares the body (44.1.2) — the same two anchors hold
    assert t.score_fold(_Oracle(), "holdout", 0, 1000) == pytest.approx(100.0)
    assert t.score_fold(_Uniform(), "holdout", 0, 1000) == pytest.approx(0.0)
    # the default is the §22.1 accuracy rule, untouched (46.1.5)
    t2 = CsvTask(seed=7, path=p)
    assert t2.metric == "accuracy"
    assert t2.score(_Oracle(), "holdout", 1000) == pytest.approx(100.0)
    assert t2.score(_Uniform(), "holdout", 1000) == pytest.approx(50.0)


def test_logloss_on_mse_csv_raises_citing_461(tmp_path):
    """A36 (SPEC.md 46.1.2/46.1.4): logloss is softmax-head only — a
    construction ValueError citing SPEC.md 46.1 (at the probe, before
    any training); an unknown metric is rejected the same way."""
    p_mse = _mse_csv(tmp_path)
    with pytest.raises(ValueError, match="46\\.1"):
        CsvTask(seed=7, path=p_mse, metric="logloss")
    p_cls = _cls_csv(tmp_path, "c2.csv")
    with pytest.raises(ValueError, match="46\\.1"):
        CsvTask(seed=7, path=p_cls, metric="f1")


def test_fit_metric_logloss_end_to_end(tmp_path, capsys):
    """A36 (SPEC.md 46.1.2/46.4): `fit --metric logloss` runs the loop
    end-to-end and the gate line names the declared metric (rc per the
    gate — PASS here, separable data)."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--metric", "logloss",
                   "--experiments", "4", "--policy", "bandit", "--seed", "7",
                   "--target", "70", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PASS" in out
    assert "on logloss" in out  # the gate line names the metric (36.1.5)
    assert len(load_registry(runs)) == 1


def test_fit_metric_default_is_byte_identical(tmp_path, capsys):
    """A36 (SPEC.md 46.1.5): the default `--metric accuracy` builds the
    identical config as no flag at all — the rendered dry-run plan is
    byte-identical (the config dict never gains a `metric` key)."""
    p = _cls_csv(tmp_path)
    base = ["fit", "--dry-run", "--data", str(p),
            "--runs-dir", str(tmp_path / "runs")]
    assert cli_main(base) == 0
    out_default = capsys.readouterr().out
    assert cli_main(base + ["--metric", "accuracy"]) == 0
    out_flag = capsys.readouterr().out
    assert out_default == out_flag
    assert "metric accuracy" in out_default  # the `task:` line names it


def test_fit_dry_run_logloss_names_metric_and_omits_headroom(tmp_path,
                                                             capsys):
    """A36 (SPEC.md 46.1.3/46.4): `fit --dry-run --metric logloss` names
    the metric in the plan (rc 0), keeps the class-balance block, and
    omits the accuracy headroom note (a constant model scores 0 on that
    scale, not `100·share`); accuracy tasks still print it."""
    p = _cls_csv(tmp_path)
    base = ["fit", "--dry-run", "--data", str(p),
            "--runs-dir", str(tmp_path / "runs")]
    assert cli_main(base) == 0
    out_acc = capsys.readouterr().out
    assert "balance" in out_acc and "headroom" in out_acc  # the control
    assert cli_main(base + ["--metric", "logloss"]) == 0
    out_ll = capsys.readouterr().out
    assert "metric logloss" in out_ll  # the `task:` line names it
    assert "balance" in out_ll  # the block still prints
    assert "headroom" not in out_ll  # suppressed (46.1.3)


def test_fit_dry_run_logloss_on_mse_csv_is_probe_error(tmp_path, capsys):
    """A36 (SPEC.md 46.1.4): `fit --dry-run --metric logloss` on an
    mse-head file is a probe-time failure (rc 1, before any training)."""
    p = _mse_csv(tmp_path)
    rc = cli_main(["fit", "--dry-run", "--data", str(p), "--metric",
                   "logloss", "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "46.1" in err and "softmax" in err


def test_metric_flag_on_fit_only():
    """A36 (SPEC.md 46.1.2): `--metric` is a data-task knob on `fit`
    only (the `--label`/`--temporal` precedent) — not a KNOBS row, not
    on `run`/`predict`."""
    parser = build_parser()
    sub = next(a for a in parser._actions
               if type(a).__name__ == "_SubParsersAction")
    subs = sub.choices
    where = [name for name, sp in subs.items()
             if any("--metric" in a.option_strings for a in sp._actions)]
    assert where == ["fit"]
    action = next(a for a in subs["fit"]._actions
                  if "--metric" in a.option_strings)
    assert action.default == "accuracy"


# --- 46.2 sine: the task knobs + the ceiling ------------------------------------

def test_sine_defaults_bit_identical_pre_462():
    """A36 (SPEC.md 46.2.3/46.2.6): `SineRegressionV1(seed)` is the
    historical task — the default (noise=LABEL_NOISE, freq_scale=1.0,
    amplitude=1.0) reproduces the pre-46.2 make_dataset exactly
    (reconstructed from the pinned module constants); bad constructor
    values raise (the 45.1-style contract)."""
    t = SineRegressionV1(7)
    assert (t.noise, t.freq_scale, t.amplitude) == (LABEL_NOISE, 1.0, 1.0)
    X, y = t.make_dataset(64)
    seq = np.random.SeedSequence([7, zlib.crc32(b"train")])
    Xh = np.random.default_rng(seq).uniform(0.0, 1.0, (64, 1))
    rng = np.random.default_rng(np.random.SeedSequence(
        [7, zlib.crc32(b"train-noise")]))
    yh = target_function(Xh[:, 0]) + rng.normal(0.0, LABEL_NOISE, 64)
    assert np.array_equal(X, Xh)  # the same unit inputs
    assert np.array_equal(y, yh)  # the historical target + noise
    # the explicit defaults are the same behavior (G2 determinism)
    t2 = SineRegressionV1(7, noise=0.05, freq_scale=1.0, amplitude=1.0)
    assert np.array_equal(t.make_dataset(64)[0], t2.make_dataset(64)[0])
    for bad in ({"noise": -0.01}, {"freq_scale": 0.0}, {"amplitude": -0.5}):
        with pytest.raises(ValueError):
            SineRegressionV1(7, **bad)


def test_sine_ceiling_monotone():
    """A36 (SPEC.md 46.2.3): the ceiling is monotone decreasing in
    noise and in a shrinking amplitude, on the 0–100 scale (clamped at
    0). The freq_scale dependence is not monotone in general — the sum
    of components' variance depends on phase alignment (SPEC 46.2.3
    only pins determinism there) — so it is checked bounded, not
    ordered."""
    assert sine_ceiling(0.05) > sine_ceiling(0.10) > sine_ceiling(0.15)
    assert sine_ceiling(amplitude=1.0) > sine_ceiling(amplitude=0.75) \
        > sine_ceiling(amplitude=0.5)
    assert 0.0 <= sine_ceiling(freq_scale=2.0) <= 100.0
    assert 0.0 <= sine_ceiling() <= 100.0
    assert sine_ceiling(noise=10.0) == 0.0  # clamped (noise > signal)


# --- 46.2 sine: the ladder --------------------------------------------------------

def test_sine_curriculum_ladder_order_and_exhaustion():
    """A36 (SPEC.md 46.2.3): a 3×3×3 grid (27 levels, 26 step-ups),
    noise swept first (fastest), then freq_scale, then amplitude;
    exhaustion at the last combination; `level_params` carries the three
    knobs; `task()` rebuilds the level task (same seed — G2)."""
    cur = SineCurriculum(seed=7)
    assert cur.levels_left() == 26
    assert (cur.noise, cur.freq_scale, cur.amplitude) == (0.05, 1.0, 1.0)
    order = []
    for _ in range(26):
        assert cur.step_up_if_saturated(1e18)
        order.append((cur.noise, cur.freq_scale, cur.amplitude))
    assert not cur.step_up_if_saturated(1e18)  # exhausted (46.2.3)
    assert cur.levels_left() == 0
    # noise is the fastest axis: the first step-ups raise it in place
    assert order[0] == (0.10, 1.0, 1.0)
    assert order[1] == (0.15, 1.0, 1.0)
    # then freq_scale sweeps, resetting noise to its start
    assert order[2] == (0.05, 1.5, 1.0)
    assert order[4] == (0.15, 1.5, 1.0)
    # then amplitude (the slowest axis), both inner axes reset
    assert order[7] == (0.15, 2.0, 1.0)  # last a=1.0 combination
    assert order[8] == (0.05, 1.0, 0.75)  # first a=0.75 combination
    assert order[25] == (0.15, 2.0, 0.5)  # the last combination
    assert set(cur.level_params()) == {"noise", "freq_scale", "amplitude"}
    t = cur.task()
    assert (t.noise, t.freq_scale, t.amplitude) == (0.15, 2.0, 0.5)
    assert t.seed == 7


def test_sine_curriculum_trigger_exact_and_validation():
    """A36 (SPEC.md 46.2.2/46.2.3): the trigger threshold is exact —
    `best >= 0.9 · ceiling` steps, a hair below does not; the
    constructor validates its grid (the 45.1-style contract)."""
    cur = SineCurriculum(seed=7)
    c0 = cur.ceiling
    assert not cur.step_up_if_saturated(0.9 * c0 - 1e-6)
    assert cur.step_up_if_saturated(0.9 * c0)
    assert cur.level == 1 and cur.noise == 0.10
    for bad in (dict(noise_step=0.0), dict(min_amp=1.5),
                dict(trigger_fraction=0.0), dict(freq_start=0.0)):
        with pytest.raises(ValueError):
            SineCurriculum(seed=7, **bad)


# --- 46.2 cartpole: the task knob + the ladder -----------------------------------

def test_cartpole_default_ics_bit_identical_and_scaled():
    """A36 (SPEC.md 46.2.4/46.2.6): the default IC box is the historical
    one (bit-identical — reconstructed from the pinned constants), and
    `ic_scale=2.0` widens every axis by exactly 2 (the same RNG stream;
    the affine uniform map is exact in float64); `ic_scale < 1.0`
    raises, citing 46.2.4."""
    t = CartPoleV1(7)
    assert t.ic_scale == 1.0
    ic = t.initial_conditions("holdout", 4)
    seq = np.random.SeedSequence([7, zlib.crc32(b"holdout")])
    rng = np.random.default_rng(seq)
    hist = np.stack([rng.uniform(-0.2, 0.2, 4),
                     rng.uniform(-0.1, 0.1, 4),
                     rng.uniform(-0.1, 0.1, 4),
                     rng.uniform(-0.5, 0.5, 4)], axis=1)
    assert np.array_equal(ic, hist)  # the default is the historical box
    t2 = CartPoleV1(7, ic_scale=2.0)
    assert np.array_equal(t2.initial_conditions("holdout", 4), 2.0 * hist)
    with pytest.raises(ValueError, match="46\\.2\\.4"):
        CartPoleV1(7, ic_scale=0.5)


def test_cartpole_curriculum_ladder_and_proxy_ceiling():
    """A36 (SPEC.md 46.2.4): the ladder is 1.0 → 2.0 → 3.0 (2 step-ups,
    the AUG_BOX cap); the ceiling proxy is `max_steps` (500.0) — the
    trigger is `best >= 0.9 · 500`; exhaustion after the last level;
    `level_params` carries `ic_scale`."""
    cur = CartPoleCurriculum(seed=7)
    assert cur.ceiling == pytest.approx(500.0)  # the max_steps proxy
    assert cur.levels_left() == 2
    assert cur.ic_scale == 1.0
    assert not cur.step_up_if_saturated(449.9)  # just below 0.9 · 500
    assert cur.step_up_if_saturated(450.0)
    assert cur.ic_scale == 2.0
    assert cur.step_up_if_saturated(450.0)
    assert cur.ic_scale == 3.0  # the cap
    assert not cur.step_up_if_saturated(450.0)  # exhausted
    assert cur.levels_left() == 0
    assert cur.level_params() == {"ic_scale": 3.0}
    t = cur.task()
    assert t.ic_scale == 3.0 and t.seed == 7
    assert cur.level_description().startswith("cartpole-ics")


# --- 46.2 the shared curriculum API ----------------------------------------------

def test_level_params_api_and_top_level_exports():
    """A36 (SPEC.md 46.2.2/46.2.5): `level_params()` exists on all three
    curricula — parity's is the historical `{"n_bits", "p_flip"}` shape
    (the A10 event-row keys), sine's the three knobs, cartpole's
    `ic_scale`; both new classes are exported from the top-level
    package (33.1-style additive `__all__`)."""
    par = ParityCurriculum(seed=7)
    assert par.level_params() == {"n_bits": 4, "p_flip": 0.08}
    assert set(ParityCurriculum(seed=7).level_params()) == \
        {"n_bits", "p_flip"}
    assert set(SineCurriculum(seed=7).level_params()) == \
        {"noise", "freq_scale", "amplitude"}
    assert set(CartPoleCurriculum(seed=7).level_params()) == {"ic_scale"}
    # the top-level re-exports are the same classes (additive entries)
    assert autorefine.SineCurriculum is SineCurriculum
    assert autorefine.CartPoleCurriculum is CartPoleCurriculum
    assert "SineCurriculum" in autorefine.__all__
    assert "CartPoleCurriculum" in autorefine.__all__


def test_env_curriculum_event_rows_keep_historical_keys(tmp_path):
    """A36 (SPEC.md 46.2.2): the env's curriculum event row splats the
    curriculum's `level_params()` — parity rows still carry exactly the
    historical keys (the A10 shape), sine rows carry
    noise/freq_scale/amplitude (white-box: a forced step-up, then
    `_advance_curriculum` — deterministic)."""
    cur = ParityCurriculum(seed=7)
    assert cur.step_up_if_saturated(1e18)  # level 1: p_flip 0.10
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(1, 30, 5),
                        runs_dir=tmp_path / "par", curriculum=cur)
    env.reset()
    env._advance_curriculum()  # the env-side effect of the step-up
    row = env.curriculum_events[-1]
    historical = {"level", "difficulty", "n_bits", "p_flip", "ceiling",
                  "best_before_step", "new_baseline_score"}
    assert historical <= set(row)  # the A10 keys, all present
    assert row["n_bits"] == 4 and row["p_flip"] == 0.10
    assert row["level"] == 1

    cur2 = SineCurriculum(seed=7)
    assert cur2.step_up_if_saturated(1e18)  # level 1: noise 0.10
    env2 = AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(1, 30, 5),
                         runs_dir=tmp_path / "sin", curriculum=cur2)
    env2.reset()
    env2._advance_curriculum()
    row2 = env2.curriculum_events[-1]
    assert {"level", "difficulty", "noise", "freq_scale", "amplitude",
            "ceiling", "best_before_step", "new_baseline_score"} \
        <= set(row2)
    assert row2["noise"] == 0.10 and row2["freq_scale"] == 1.0 \
        and row2["amplitude"] == 1.0


# --- 46.2 the run --curriculum CLI --------------------------------------------------

def test_run_curriculum_sine_and_cartpole_cli(tmp_path, capsys):
    """A36 (SPEC.md 46.2.5/46.4): `run --curriculum` dispatches by task
    — sine-v1 and cartpole-v1 complete rc 0 with the ladder in the
    summary; gridnav-v1 is rc 1 with a message naming the three
    supported tasks."""
    runs = tmp_path / "runs"
    rc = cli_main(["run", "--task", "sine-v1", "--curriculum",
                   "--experiments", "1", "--seed", "7",
                   "--max-seconds", "60", "--max-train-seconds", "5",
                   "--runs-dir", str(runs)])
    assert rc == 0, capsys.readouterr().out
    rd = next(d for d in sorted(runs.iterdir())
              if (d / "summary.json").is_file())
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    cur = summary["curriculum"]
    assert isinstance(cur["levels"], list)
    # a 1-experiment budget does not saturate: the ladder sits at 0
    assert cur["final_difficulty"] == "sine-n0.05-f1.00-a1.00"

    runs2 = tmp_path / "runs2"
    rc = cli_main(["run", "--task", "cartpole-v1", "--curriculum",
                   "--experiments", "1", "--seed", "7",
                   "--max-seconds", "60", "--max-train-seconds", "10",
                   "--runs-dir", str(runs2)])
    assert rc == 0, capsys.readouterr().out
    rd = next(d for d in sorted(runs2.iterdir())
              if (d / "summary.json").is_file())
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summary["curriculum"]["final_difficulty"] \
        .startswith("cartpole-ics")

    rc = cli_main(["run", "--task", "gridnav-v1", "--curriculum",
                   "--experiments", "1",
                   "--runs-dir", str(tmp_path / "r3")])
    err = capsys.readouterr().err
    assert rc == 1
    for name in ("parity-v1", "sine-v1", "cartpole-v1"):
        assert name in err  # the message names the three (46.2.5)


# --- 46.3 portfolio mode ----------------------------------------------------------

def test_portfolio_fit_both_pass(tmp_path, capsys):
    """A36 (SPEC.md 46.3.2/46.4): `fit --tasks a.csv,b.csv` runs both
    loops under one shared budget — in the given order, each with its
    own run dir + gate line; rc 0 because both PASS; two registry rows."""
    a = _cls_csv(tmp_path, "a.csv")
    b = _cls_csv(tmp_path, "b.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--tasks", f"{a},{b}",
                   "--experiments", "4", "--policy", "bandit", "--seed", "7",
                   "--target", "70", "--runs-dir", str(runs),
                   "--max-seconds", "180", "--max-train-seconds", "10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count("=== portfolio task") == 2  # both tasks, in order
    assert str(a) in out.split("=== portfolio task 1/")[1][:400]
    assert str(b) in out.split("=== portfolio task 2/")[1][:400]
    assert out.count("PASS") >= 2  # one gate line per task
    # two run dirs, two registry rows (each a complete fit)
    dirs = [d for d in sorted(runs.iterdir())
            if d.is_dir() and (d / "summary.json").is_file()]
    assert len(dirs) == 2
    assert len(load_registry(runs)) == 2


def test_portfolio_exclusivity_and_min_count(tmp_path, capsys):
    """A36 (SPEC.md 46.3.2): `--tasks` is mutually exclusive with
    `--data` and `--from-run` (rc 1, naming 46.3), and needs at least
    2 CSV paths (a single path is rc 1)."""
    a = _cls_csv(tmp_path, "a.csv")
    b = _cls_csv(tmp_path, "b.csv")
    rc = cli_main(["fit", "--tasks", f"{a},{b}", "--data", str(a),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err and "46.3" in err
    rc = cli_main(["fit", "--tasks", str(a),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1 and "at least 2" in err


def test_portfolio_dry_run_plans_zero_artifacts(tmp_path, capsys):
    """A36 (SPEC.md 46.3.2/40.1): `fit --dry-run --tasks a.csv,b.csv`
    prints the per-task plans with rc 0 — still zero training, zero
    artifacts (no run dirs, no registry)."""
    a = _cls_csv(tmp_path, "a.csv")
    b = _cls_csv(tmp_path, "b.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--dry-run", "--tasks", f"{a},{b}",
                   "--experiments", "4", "--seed", "7",
                   "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count("=== portfolio task") == 2
    assert "dry-run: plan only" in out
    # zero artifacts (40.1): no run dirs, no registry entry
    assert not any((d / "summary.json").is_file()
                   for d in runs.iterdir()) if runs.is_dir() else True
    assert not (runs / "registry.json").is_file()


# --- 46.4 regression ------------------------------------------------------------------

def test_version_round_v032():
    """A36 (SPEC.md 46.4, 33.1): the version stepped to `0.37.0` in
    both sources (v0.37 ⇒ `0.37.0`, M40)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.37.0"


def test_spec_cites_a36_and_round():
    """A36 (SPEC.md 46.4/46.5): SPEC.md defines the A36 acceptance block
    and the M35 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 37)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 46.4 Acceptance (A36)" in spec
    assert re.search(r"^\s*\| M35 \| v0\.32\s*\|\s*46\s*\|\s*A36\s*\|",
                     spec, re.MULTILINE)
    assert "**M35**" in spec
