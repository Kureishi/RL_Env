"""Sequential decision-making — sample weights, cartpole sim, AWR, MPC
(SPEC.md 84, A74).

SPEC.md 84 (v0.70): the environment learns *policies* for sequential
control tasks, **offline** — from logged rollouts, with no environment
access at fit time:

  * 84.1  — `sample_weight` in the trainer contract: weighted-mean
    loss/gradient on the four parametric families; `None` = the exact
    legacy path (bit-identical, the T2 / §18.7 pins);
  * 84.2  — `simulator.Simulator` ABC + `CartpoleSim` (NumPy cartpole,
    discrete push left/right, +1/step survival reward);
  * 84.3  — AWR (`awr.py`): returns / advantages / `exp(adv/beta)` weights,
    `rollout` (the REINFORCE demonstration), `train_awr_policy` (one
    `train(..., sample_weight=w)` call);
  * 84.4  — learned-model MPC (`mpc.py`): `learn_dynamics` (mse head) +
    `mpc_act` (argmax over predicted survival, no RNG);
  * 84.5  — acceptance: the math, the sim, the offline improvement
    (AWR beats the uniform baseline), the MPC, the version step, the
    A-index advance.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import itertools
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import ModelSpec, SpecError
from autorefine.awr import (
    advantages,
    awr_weights,
    rollout,
    train_awr_policy,
    trajectory_returns,
    _zero_policy,
)
from autorefine.mpc import learn_dynamics, mpc_act
from autorefine.models.mlp import MLP
from autorefine.models.conv1d import Conv1D
from autorefine.models.convnet import ConvNet
from autorefine.models.rnn import RNN
from autorefine.simulator import CartpoleSim, Simulator
from autorefine.trainer import train

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


def _spec(**kw) -> ModelSpec:
    """A small, deterministic policy/dynamics spec (test-local)."""
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


def _weighted_ce_loss(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Hand-computed weighted softmax CE (the 84.1 reference)."""
    n = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    oh = np.zeros_like(logz)
    oh[np.arange(n), y] = 1.0
    W = float(w.sum())
    return float((w * (-(oh * logz).sum(axis=1))).sum() / W)


def _weighted_mse_loss(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Hand-computed weighted MSE (the 84.1 reference)."""
    target = np.asarray(y, dtype=np.float64).reshape(logits.shape[0], -1)
    per = ((logits - target) ** 2).sum(axis=1)
    return float((w * per).sum() / float(w.sum()))


# --- 84.1 sample_weight -------------------------------------------------------

class TestSampleWeight:
    """SPEC.md 84.1 (A74.1): the weighted-mean contract on all four
    parametric families."""

    def test_mlp_weighted_softmax_ce_matches_hand_computed(self):
        m = MLP(3, (4,), 2, "tanh", seed=3, head="softmax")
        x = np.random.default_rng(1).uniform(-1.0, 1.0, (8, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        w = np.array([0.5, 2.0, 1.0, 0.1, 3.0, 1.0, 0.7, 1.5])
        loss, _ = m.loss_and_grads(x, y, weights=w)
        assert loss == pytest.approx(_weighted_ce_loss(m.forward(x), y, w),
                                     rel=1e-12)

    def test_mlp_weighted_mse_matches_hand_computed(self):
        m = MLP(3, (4,), 1, "tanh", seed=4, head="mse")
        x = np.random.default_rng(2).uniform(-1.0, 1.0, (6, 3))
        y = np.random.default_rng(3).uniform(-2.0, 2.0, 6)
        w = np.array([1.0, 0.3, 2.0, 1.0, 0.5, 4.0])
        loss, _ = m.loss_and_grads(x, y, weights=w)
        assert loss == pytest.approx(_weighted_mse_loss(m.forward(x), y, w),
                                     rel=1e-12)

    def test_rnn_conv1d_convnet_weighted_ce_matches_hand_computed(self):
        rng = np.random.default_rng(7)
        w = np.array([0.5, 2.0, 1.0, 0.2, 3.0, 1.0])
        y = np.array([0, 1, 1, 0, 1, 0])

        r = RNN(3, 2, 4, 2, "tanh", seed=5, head="softmax")
        xs = rng.uniform(-1, 1, (6, 3, 2))
        loss, _ = r.loss_and_grads(xs, y, weights=w)
        assert loss == pytest.approx(_weighted_ce_loss(r.forward(xs), y, w),
                                     rel=1e-12)

        c1 = Conv1D(3, 2, 4, 2, 2, "tanh", seed=6, head="softmax")
        loss, _ = c1.loss_and_grads(xs, y, weights=w)
        assert loss == pytest.approx(_weighted_ce_loss(c1.forward(xs), y, w),
                                     rel=1e-12)

        cn = ConvNet((2, 8, 8), 4, 4, 2, "tanh", seed=8, head="softmax")
        xg = rng.uniform(-1, 1, (6, 2, 8, 8))
        loss, _ = cn.loss_and_grads(xg, y, weights=w)
        assert loss == pytest.approx(_weighted_ce_loss(cn.forward(xg), y, w),
                                     rel=1e-12)

    def test_mlp_weighted_gradient_finite_difference(self):
        """A74.1: gradients included — a finite-difference gradcheck on
        the mlp (the weighted softmax CE)."""
        m = MLP(3, (4,), 2, "tanh", seed=9, head="softmax")
        x = np.random.default_rng(10).uniform(-1.0, 1.0, (8, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        w = np.array([0.5, 2.0, 1.0, 0.1, 3.0, 1.0, 0.7, 1.5])
        loss0, grads = m.loss_and_grads(x, y, weights=w)
        eps = 1e-6
        for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 3)]:
            orig = m.layers[0][0][i, j]
            m.layers[0][0][i, j] = orig + eps
            lp, _ = m.loss_and_grads(x, y, weights=w)
            m.layers[0][0][i, j] = orig - eps
            lm, _ = m.loss_and_grads(x, y, weights=w)
            m.layers[0][0][i, j] = orig
            num = (lp - lm) / (2 * eps)
            assert num == pytest.approx(grads[0][0][i, j], abs=1e-5)
        # the loss itself is unchanged by the (restored) perturbation
        loss1, _ = m.loss_and_grads(x, y, weights=w)
        assert loss1 == loss0

    def test_none_weights_is_the_legacy_unweighted_path(self):
        """A74.1: `weights=None` is the exact legacy mean (bit-identical
        path — the §18.7 / T2 pins elsewhere cover the stream)."""
        m = MLP(3, (4,), 2, "tanh", seed=11, head="softmax")
        x = np.random.default_rng(12).uniform(-1.0, 1.0, (8, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss, _ = m.loss_and_grads(x, y)
        logits = m.forward(x)
        z = logits - logits.max(axis=1, keepdims=True)
        logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        oh = np.zeros_like(logz)
        oh[np.arange(8), y] = 1.0
        assert loss == pytest.approx(float(-(oh * logz).sum(axis=1).mean()),
                                     rel=1e-12)

    def test_equal_weights_match_the_unweighted_loss(self):
        """A74.1: w = 1 for all rows is mathematically the legacy mean
        (the weighted-mean contract, 84.1.1)."""
        m = MLP(3, (4,), 2, "tanh", seed=13, head="softmax")
        x = np.random.default_rng(14).uniform(-1.0, 1.0, (8, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        loss_w, _ = m.loss_and_grads(x, y, weights=np.ones(8))
        loss_u, _ = m.loss_and_grads(x, y)
        assert loss_w == pytest.approx(loss_u, rel=1e-9)

    def test_bad_weights_are_clean_specerrors(self):
        m = MLP(3, (4,), 2, "tanh", seed=15, head="softmax")
        x = np.random.default_rng(16).uniform(-1.0, 1.0, (8, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(SpecError):
            m.loss_and_grads(x, y, weights=np.ones(7))          # wrong length
        with pytest.raises(SpecError):
            m.loss_and_grads(x, y, weights=np.array([-1.0] + [1.0] * 7))
        with pytest.raises(SpecError):
            m.loss_and_grads(x, y, weights=np.zeros(8))         # zero sum

    def test_non_parametric_family_with_weights_is_a_specerror(self):
        """A74.1.4: the reweighting is defined for parametric policies."""
        S = np.random.default_rng(17).uniform(-1, 1, (40, 3))
        y = np.random.default_rng(18).integers(0, 2, 40)
        with pytest.raises(SpecError):
            train((S, y), _spec(model_family="tree"), 0,
                  sample_weight=np.ones(40))


# --- 84.2 CartpoleSim ---------------------------------------------------------

class TestCartpoleSim:
    """SPEC.md 84.2 (A74.2): the simulator contract + the cartpole."""

    def test_contract_shape(self):
        sim = CartpoleSim()
        assert isinstance(sim, Simulator)
        assert sim.n_actions == 2 and sim.state_dim == 4

        class _Plain(Simulator):
            n_actions = 2
            state_dim = 2

            def reset(self, seed):
                return np.zeros(2)

            def step(self, state, action):
                return np.zeros(2), 1.0, False

        # the default is_terminal is "never terminal" (the contract default)
        assert _Plain().is_terminal(np.zeros(2)) is False

    def test_reset_is_deterministic_and_in_box(self):
        sim = CartpoleSim()
        s1 = sim.reset(42)
        s2 = sim.reset(42)
        assert np.array_equal(s1, s2)
        assert np.isfinite(s1).all() and s1.shape == (4,)
        assert abs(s1[0]) <= 0.05 and abs(s1[2]) <= 0.02  # the reset box
        assert not np.array_equal(sim.reset(42), sim.reset(43))

    def test_step_invariants(self):
        sim = CartpoleSim()
        state = sim.reset(1)
        for a in (0, 1):
            nxt, reward, done = sim.step(state, a)
            assert reward == 1.0                      # the survival reward
            assert np.isfinite(nxt).all()
            assert done == sim.is_terminal(nxt)       # done == terminal box
        # a state already outside the box is terminal
        out = np.array([0.0, 0.0, 0.5, 0.0])          # |theta| > 0.2
        assert sim.is_terminal(out)
        out = np.array([1.7, 0.0, 0.0, 0.0])          # |x| > 1.6
        assert sim.is_terminal(out)
        assert not sim.is_terminal(np.array([0.0, 0.0, 0.0, 0.0]))

    def test_trajectory_reproduces_from_seed_and_actions(self):
        """A74.2: a full episode trajectory is a pure function of
        (seed, action sequence) (G2)."""
        def play(seed, actions):
            sim = CartpoleSim()
            state = sim.reset(seed)
            traj = [state]
            for a in actions:
                state, _r, done = sim.step(state, a)
                traj.append(state)
                if done:
                    break
            return traj

        actions = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]
        t1 = play(7, actions)
        t2 = play(7, actions)
        assert all(np.array_equal(a, b) for a, b in zip(t1, t2))
        # a different seed starts a different episode
        assert not np.array_equal(play(7, actions)[0], play(8, actions)[0])

    def test_a_falling_pole_ends_the_episode(self):
        """A74.2: the terminal box is respected — one action held long
        enough must end the episode (within the 500-step budget)."""
        sim = CartpoleSim()
        state = sim.reset(3)
        done = False
        for i in range(500):
            state, _r, done = sim.step(state, 0)
            if done:
                break
        assert done
        assert sim.is_terminal(state)


# --- 84.3 AWR ------------------------------------------------------------------

class TestAWR:
    """SPEC.md 84.3 (A74.3): returns / advantages / weights, the rollout,
    and the offline improvement."""

    def test_trajectory_returns_gamma1_and_gamma09(self):
        """A74.3.1: both gamma values match a hand-computed backward loop."""
        r = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        # two trajectories of horizon 4: [1,1,0,1] and [0,1,1,0]
        ret1 = trajectory_returns(r, 4, gamma=1.0)
        assert ret1.tolist() == [3.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0]
        # gamma = 0.9: hand loop
        expected = []
        for start in (0, 4):
            run = 0.0
            row = []
            for t in range(3, -1, -1):
                run = r[start + t] + 0.9 * run
                row.append(run)
            expected.extend(row[::-1])
        ret09 = trajectory_returns(r, 4, gamma=0.9)
        for a, b in zip(ret09, expected):
            assert a == pytest.approx(b, rel=1e-12)

    def test_advantages_and_weights(self):
        rets = np.array([10.0, 20.0, 30.0])
        adv = advantages(rets, "mean")
        assert np.allclose(adv, rets - 20.0)
        assert np.allclose(advantages(rets, "min"), rets - 10.0)
        assert np.allclose(advantages(rets, "zero"), rets)
        w = awr_weights(adv, beta=1.0)
        assert np.allclose(w, np.exp(adv))
        with pytest.raises(SpecError):
            awr_weights(adv, beta=0.0)
        with pytest.raises(SpecError):  # overflow → clean SpecError
            with np.errstate(over="ignore"):
                awr_weights(np.array([1000.0, -1000.0]), beta=0.001)
        with pytest.raises(SpecError):
            advantages(rets, "bogus")

    def test_rollout_is_deterministic_and_a_clean_grid(self):
        """A74.3.2: same policy + seed ⇒ same data; (E*H,) grid."""
        sim = CartpoleSim()
        base = _zero_policy(sim)
        E, H = 16, 24
        d1 = rollout(base, sim, E, H, seed=123)
        d2 = rollout(base, sim, E, H, seed=123)
        for key in ("states", "actions", "rewards", "returns", "total_returns"):
            assert np.array_equal(d1[key], d2[key])
        m = E * H
        assert d1["states"].shape == (m, 4)
        assert d1["actions"].shape == (m,) and d1["rewards"].shape == (m,)
        assert d1["total_returns"].shape == (E,)
        assert set(np.unique(d1["actions"])) <= {0, 1}
        # rewards are +1 (active) or 0 (padded past done)
        assert set(np.unique(d1["rewards"])) <= {0.0, 1.0}
        # the grid's returns are consistent with its rewards
        assert np.array_equal(d1["returns"], trajectory_returns(d1["rewards"], H))

    def test_awr_policy_beats_the_uniform_baseline(self):
        """A74.3.3: the offline improvement — a policy AWR-trained on
        collected rollouts beats the uniform baseline's mean total return
        on a *fresh* rollout (different seed)."""
        sim = CartpoleSim(max_steps=200)
        base = _zero_policy(sim)
        E, H = 48, 40
        d0 = rollout(base, sim, E, H, seed=123)
        baseline_mean = float(d0["total_returns"].mean())
        policy = train_awr_policy(
            d0["states"], d0["actions"], d0["rewards"], H,
            spec=_spec(), seed=7, beta=1.0, n_out=2,
        ).model
        d1 = rollout(policy, sim, E, H, seed=999)  # fresh, unseen seed
        assert float(d1["total_returns"].mean()) > baseline_mean

    def test_awr_fit_is_deterministic(self):
        """A74.3.3: (data, spec, seed) ⇒ the same weights, bit-exactly."""
        sim = CartpoleSim()
        base = _zero_policy(sim)
        E, H = 12, 16
        d = rollout(base, sim, E, H, seed=5)
        r1 = train_awr_policy(d["states"], d["actions"], d["rewards"], H,
                              spec=_spec(), seed=11, beta=1.0, n_out=2)
        r2 = train_awr_policy(d["states"], d["actions"], d["rewards"], H,
                              spec=_spec(), seed=11, beta=1.0, n_out=2)
        for l1, l2 in zip(r1.model.layers, r2.model.layers):
            assert np.array_equal(l1[0], l2[0]) and np.array_equal(l1[1], l2[1])


# --- 84.4 learned-model MPC -----------------------------------------------------

def _collect_transitions(sim: CartpoleSim, episodes: int, horizon: int,
                         seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(state, action, next_state) triples from random actions (test-local)."""
    rng = np.random.default_rng(seed)
    S: list[np.ndarray] = []
    A: list[int] = []
    N: list[np.ndarray] = []
    for e in range(episodes):
        state = np.asarray(sim.reset(5000 + e * 31 + seed),
                           dtype=np.float64).reshape(-1)
        done = False
        for _t in range(horizon):
            if done:
                break
            a = int(rng.integers(0, sim.n_actions))
            nxt, _r, done = sim.step(state, a)
            S.append(state)
            A.append(a)
            N.append(nxt)
            state = np.asarray(nxt, dtype=np.float64).reshape(-1)
    return np.array(S), np.array(A, dtype=np.int64), np.array(N)


def _survival(model, sim: CartpoleSim, state: np.ndarray,
              seq: tuple[int, ...]) -> int:
    """Predicted survival of an action sequence (mirrors mpc_act)."""
    s = state
    steps = 0
    for a in seq:
        x = np.concatenate([s, np.array([float(a)], dtype=np.float64)])
        s = np.asarray(model.forward(x.reshape(1, -1)),
                       dtype=np.float64).reshape(-1)
        if not np.isfinite(s).all() or sim.is_terminal(s):
            break
        steps += 1
    return steps


class TestMPC:
    """SPEC.md 84.4 (A74.4): the learned dynamics + the greedy act."""

    def test_learned_dynamics_beats_the_noop_baseline(self):
        """A74.4.1: the (state, action) -> next_state model beats the
        no-op (predict the current state)."""
        sim = CartpoleSim()
        S, A, N = _collect_transitions(sim, 64, 40, seed=0)
        dyn = learn_dynamics(S, A, N, spec=_spec(), seed=3).model
        X = np.hstack([S, A.reshape(-1, 1)])
        learned = float(((dyn.forward(X) - N) ** 2).mean())
        noop = float(((S - N) ** 2).mean())
        assert learned < noop

    def test_mpc_act_is_valid_deterministic_and_optimal(self):
        """A74.4.2: valid action; deterministic (no RNG); the chosen
        action's best predicted survival is >= the other action's."""
        sim = CartpoleSim()
        S, A, N = _collect_transitions(sim, 64, 40, seed=1)
        dyn = learn_dynamics(S, A, N, spec=_spec(), seed=5).model
        state = np.asarray(sim.reset(9), dtype=np.float64).reshape(-1)
        H = 4
        a1 = mpc_act(dyn, sim, state, horizon=H)
        a2 = mpc_act(dyn, sim, state, horizon=H)
        assert a1 in (0, 1) and a1 == a2
        best = {
            0: max(_survival(dyn, sim, state, seq)
                   for seq in itertools.product((0, 1), repeat=H)
                   if seq[0] == 0),
            1: max(_survival(dyn, sim, state, seq)
                   for seq in itertools.product((0, 1), repeat=H)
                   if seq[0] == 1),
        }
        assert best[a1] >= best[1 - a1]

    def test_mpc_rejects_bad_inputs(self):
        sim = CartpoleSim()
        S, A, N = _collect_transitions(sim, 8, 8, seed=2)
        dyn = learn_dynamics(S, A, N, spec=_spec(), seed=6).model
        with pytest.raises(SpecError):
            mpc_act(dyn, sim, np.zeros(4), horizon=0)
        with pytest.raises(SpecError):
            learn_dynamics(S, A[:10], N, spec=_spec())  # length mismatch


# --- 84.5 language / version / index ---------------------------------------------

class TestCeremony:
    """SPEC.md 84.5.5 (A74): the app-language scan, the version step, the
    A-index advance (M73 → this file)."""

    def test_version_steps_to_v070(self):
        py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert py["project"]["version"] == autorefine.__version__ == "0.73.0"

    def test_spec_section_and_index_row(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "## 84." in text
        assert "84.5 Acceptance (A74)" in text
        row = next(l for l in text.splitlines() if l.startswith("| M73 |"))
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert cells[1] == "v0.70" and "A74" in cells[3]
        assert "tests/test_sequential_v070.py" in cells[4]

    def test_exports_are_public(self):
        names = ("Simulator", "CartpoleSim", "trajectory_returns", "advantages",
                 "awr_weights", "rollout", "train_awr_policy",
                 "learn_dynamics", "mpc_act", "awr_loop")
        for n in names:
            assert hasattr(autorefine, n), f"autorefine.{n} missing"
            assert n in autorefine.__all__
