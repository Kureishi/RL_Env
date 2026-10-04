"""Sim-in-the-loop — the autonomous sequential improvement loop
(SPEC.md 85, A75).

SPEC.md 85 (v0.71): the sequential decision-making pieces (84) are wired
into the **autonomous loop** — the environment improves a control policy
on its own and records the improvement curve:

  * 85.1  — `awr.py::awr_loop`: start from the **uniform baseline
    policy** (the "random" start, the sequential analogue of the
    AutoRefine baseline spec); iterate `rollout(current policy)` →
    AWR retrain (one `train(..., sample_weight=w)` call, 84.3.3);
    evaluate each improved policy on a **single shared evaluation
    seed** (`seed + 9000`) so the curve is apples-to-apples; record
    per-iteration mean / best total returns;
  * 85.2  — acceptance: the loop mechanics, determinism (bit-identical
    curves *and* policy weights, G2), the autonomous improvement
    (final mean total return beats the uniform baseline at the *same*
    shared evaluation seed), the version step, the A-index advance.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import ModelSpec, SpecError
from autorefine.awr import awr_loop
from autorefine.simulator import CartpoleSim

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


def _spec(**kw) -> ModelSpec:
    """A small, deterministic policy spec (test-local)."""
    base = dict(
        architecture=(32,),
        optimizer="momentum",
        learning_rate=0.05,
        batch_size=64,
        weight_decay=0.0,
        train_steps=800,
        input_noise=0.0,
        activation="tanh",
    )
    base.update(kw)
    return ModelSpec(**base)


def _run(seed: int = 100) -> dict:
    """The canonical A75 run: 3 iterations, 32 episodes × 32 steps."""
    return awr_loop(
        CartpoleSim(),
        iterations=3,
        episodes=32,
        horizon=32,
        seed=seed,
        spec=_spec(),
    )


# --- 85.1 loop mechanics (A75.1) ---------------------------------------------

class TestLoopMechanics:
    def test_run_end_to_end(self):
        """A75.1 (SPEC.md 85.1): `awr_loop` runs end-to-end and returns
        the improvement curve, a usable final policy, the uniform
        baseline's return at the *same shared evaluation seed*, and that
        evaluation seed."""
        out = _run()
        assert out["iterations"] == 3
        mean = out["mean_total_returns"]
        best = out["best_total_returns"]
        assert isinstance(mean, np.ndarray) and mean.shape == (3,)
        assert isinstance(best, np.ndarray) and best.shape == (3,)
        assert (mean >= 0).all() and (best >= mean).all()
        # the shared evaluation seed (85.1.2)
        assert out["eval_seed"] == 100 + 9000
        # the uniform baseline's mean total return — a finite float
        assert isinstance(out["baseline_mean_total_return"], float)
        assert np.isfinite(out["baseline_mean_total_return"])
        assert out["baseline_mean_total_return"] >= 0.0

    def test_final_policy_is_usable(self):
        """A75.1: the loop's final `policy` is a working model — a forward
        pass over a state yields `n_actions` finite logits."""
        sim = CartpoleSim()
        out = _run()
        state = sim.reset(42)
        logits = np.asarray(out["policy"].forward(state.reshape(1, -1)))
        assert logits.shape == (1, sim.n_actions)
        assert np.isfinite(logits).all()

    def test_zero_iterations_is_a_spec_error(self):
        """A75.1 (SPEC.md 85.1): `iterations < 1` is a clean SpecError."""
        with pytest.raises(SpecError):
            awr_loop(CartpoleSim(), iterations=0, episodes=4, horizon=4,
                     seed=1, spec=_spec())
        with pytest.raises(SpecError):
            awr_loop(CartpoleSim(), iterations=-2, episodes=4, horizon=4,
                     seed=1, spec=_spec())


# --- 85.1.4 determinism (A75.2, G2) -------------------------------------------

class TestDeterminism:
    def test_two_runs_bit_identical(self):
        """A75.2 (SPEC.md 85.2): two `awr_loop` calls with the same
        (sim, budget, seed, spec) yield bit-identical return curves and
        bit-identical final-policy weights — the loop is a pure function
        of its arguments (G2, 85.1.4)."""
        o1, o2 = _run(seed=202), _run(seed=202)
        assert np.array_equal(o1["mean_total_returns"],
                              o2["mean_total_returns"])
        assert np.array_equal(o1["best_total_returns"],
                              o2["best_total_returns"])
        assert o1["baseline_mean_total_return"] == \
            o2["baseline_mean_total_return"]
        assert o1["eval_seed"] == o2["eval_seed"]
        for l1, l2 in zip(o1["policy"].layers, o2["policy"].layers):
            assert np.array_equal(l1[0], l2[0])
            assert np.array_equal(l1[1], l2[1])

    def test_a_different_seed_changes_the_curve(self):
        """A75.2 (sanity): the seed *does* drive the RNGs — a different
        seed produces a different improvement curve (not a fixed
        constant)."""
        o1, o2 = _run(seed=202), _run(seed=203)
        assert not np.array_equal(o1["mean_total_returns"],
                                  o2["mean_total_returns"])


# --- 85.1.3 the autonomous improvement (A75.3) ---------------------------------

class TestImprovement:
    def test_final_policy_beats_the_uniform_baseline(self):
        """A75.3 (SPEC.md 85.3): the loop's final mean total return beats
        the uniform baseline's at the *same shared evaluation seed* —
        the autonomous improvement, measured apples-to-apples."""
        out = _run()
        assert out["mean_total_returns"][-1] > \
            out["baseline_mean_total_return"]

    def test_curve_has_no_nans(self):
        """A75.3: every recorded point of the improvement curve is
        finite — a NaN anywhere is a suite failure, not a silent curve
        hole."""
        out = _run()
        assert np.isfinite(out["mean_total_returns"]).all()
        assert np.isfinite(out["best_total_returns"]).all()
        assert np.isfinite(out["baseline_mean_total_return"])


# --- 85.2.4 ceremony (A75.4) ----------------------------------------------------

class TestCeremony:
    def test_version_round_v071(self):
        """A75.4 (SPEC.md 85.2): the version stepped to `0.72.0` in both
        sources (33.1)."""
        py = tomllib.loads(
            (REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert py["project"]["version"] == autorefine.__version__ == "0.72.0"

    def test_spec_section_85_present(self):
        """A75.4: SPEC.md carries the §85 heading and the A75
        acceptance heading."""
        text = SPEC.read_text(encoding="utf-8")
        assert "## 85." in text
        assert "85.2 Acceptance (A75)" in text

    def test_index_row_m74_resolves(self):
        """A75.4: the acceptance-index row M74 lists this test file."""
        text = SPEC.read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith("| M74 |"))
        assert "tests/test_simloop_v071.py" in row
        assert "A75" in row

    def test_awr_loop_exported(self):
        """A75.4: `awr_loop` is a top-level package export."""
        assert "awr_loop" in autorefine.__all__
        assert autorefine.awr_loop is awr_loop
