"""Advantage-Weighted Regression (AWR) over the existing trainer (SPEC.md 84.3).

Offline policy learning for sequential tasks: weight each (state, action)
example by `exp(advantage / beta)` and train the policy with the trainer's
`sample_weight` path (84.1) — **no new training loop, no environment access
at fit time**. The rollouts ARE the dataset.

`rollout` (84.3.2) is the REINFORCE-style demonstration: sample actions from
the current policy's softmax, collect (state, action, reward) triples.
`awr_loop` (85.1) iterates rollout → AWR retrain → repeat and records the
per-iteration returns — the autonomous sequential improvement loop.

Determinism (G2): every RNG is seeded; `rollout`'s action draws come from a
single `default_rng(seed)`, `reset` seeds are deterministic in (seed, episode);
AWR's weights are a pure function of the rollout data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import ModelSpec, SpecError
from .trainer import TrainResult, train

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the import cycle)
    from .simulator import Simulator


# --- returns / advantages / weights ------------------------------------------

def trajectory_returns(rewards: np.ndarray, horizon: int, gamma: float = 1.0) -> np.ndarray:
    """Per-step (discounted) returns for rollouts of a fixed `horizon`.

    `rewards` is `(episodes * horizon,)` in rollout order; the result has the
    same shape with, per trajectory, `returns[t] = sum_{k>=t} gamma**(k-t) r[k]`
    (84.3.1). `gamma == 1.0` (default) = undiscounted survival returns.
    Deterministic (no RNG).
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    horizon = int(horizon)
    if horizon < 1:
        raise SpecError(f"horizon must be >= 1, got {horizon}")
    if r.size % horizon != 0:
        raise SpecError(
            f"rewards length {r.size} is not a multiple of horizon {horizon}")
    R = r.reshape(r.size // horizon, horizon)
    if gamma == 1.0:
        # backward cumsum within each trajectory (exact, no float powers)
        return R[:, ::-1].cumsum(axis=1)[:, ::-1].reshape(-1)
    k = np.arange(horizon, dtype=np.float64)
    P = np.where(
        k[None, :] >= k[:, None],
        gamma ** (k[None, :] - k[:, None]),
        0.0,
    )
    return (R @ P.T).reshape(-1)


def advantages(returns: np.ndarray, baseline: str = "mean") -> np.ndarray:
    """Return advantages: `returns - baseline` (84.3.1).

    `baseline` in {"mean" (default), "min", "zero"}. Pure function (G2).
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if baseline == "mean":
        return r - float(r.mean())
    if baseline == "min":
        return r - float(r.min())
    if baseline == "zero":
        return r
    raise SpecError(f"unknown advantage baseline {baseline!r}")


def awr_weights(advantages_arr: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """AWR reweighting: `exp(advantage / beta)` (84.3.1).

    `beta` is the temperature — larger beta = flatter (more uniform) weights;
    the trainer divides by `w.sum()`, so a global rescale of the weights
    leaves the loss invariant (84.1). Overflow (huge advantage / tiny beta)
    is a clean SpecError, never `inf` weights.
    """
    a = np.asarray(advantages_arr, dtype=np.float64).reshape(-1)
    beta = float(beta)
    if beta <= 0.0:
        raise SpecError(f"beta must be positive, got {beta}")
    w = np.exp(a / beta)
    if not np.isfinite(w).all() or w.sum() <= 0.0:
        raise SpecError(
            "awr_weights overflowed (exp(advantage/beta)) — increase beta")
    return w


# --- rollout (the REINFORCE demonstration, 84.3.2) ---------------------------

def _episode_seed(seed: int, episode: int) -> int:
    """Deterministic, distinct per-episode seed (G2)."""
    return int(seed) * 1_000_003 + int(episode) * 7 + 11


def rollout(
    policy, sim: "Simulator", episodes: int, horizon: int, seed: int,
) -> dict:
    """Collect (states, actions, rewards) by sampling the current policy.

    `policy` is a trained model: `forward((1, state_dim)) -> (1, n_actions)`
    logits; the action is drawn from `softmax(logits)` with the seeded RNG.
    Episodes are truncated at `horizon` steps or earlier on `done`; after
    `done` the terminal state is repeated with reward 0 (keeps the dataset a
    clean `(episodes * horizon,)` grid — padded steps carry ~mean advantage,
    so their weight is ~1, 84.3.2).

    Returns `{"states": (m, d), "actions": (m,), "rewards": (m,),
    "returns": (m,), "total_returns": (episodes,)}` — deterministic in
    (policy, seed) (G2).
    """
    episodes = int(episodes)
    horizon = int(horizon)
    if episodes < 1 or horizon < 1:
        raise SpecError(f"episodes/horizon must be >= 1, got {episodes}/{horizon}")
    rng = np.random.default_rng(seed)
    S: list[np.ndarray] = []
    A: list[int] = []
    R: list[float] = []
    totals: list[float] = []
    for e in range(episodes):
        state = np.asarray(sim.reset(_episode_seed(seed, e)), dtype=np.float64).reshape(-1)
        done = False
        total = 0.0
        for _t in range(horizon):
            logits = np.asarray(
                policy.forward(state.reshape(1, -1)), dtype=np.float64
            ).reshape(-1)
            logits = logits - logits.max()
            p = np.exp(logits)
            p = p / p.sum()
            a = int(rng.choice(sim.n_actions, p=p))  # categorical sample
            S.append(state)
            A.append(a)
            if not done:
                state, r, done = sim.step(state, a)
                state = np.asarray(state, dtype=np.float64).reshape(-1)
                total += float(r)
            else:  # past done: frozen terminal state, zero reward (84.3.2)
                r = 0.0
            R.append(float(r))
        totals.append(total)
    rewards = np.asarray(R, dtype=np.float64)
    return {
        "states": np.asarray(S, dtype=np.float64),
        "actions": np.asarray(A, dtype=np.int64),
        "rewards": rewards,
        "returns": trajectory_returns(rewards, horizon),
        "total_returns": np.asarray(totals, dtype=np.float64),
    }


# --- AWR policy fit (84.3.3) --------------------------------------------------

def train_awr_policy(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    horizon: int,
    spec: ModelSpec | None = None,
    seed: int = 0,
    beta: float = 1.0,
    gamma: float = 1.0,
    baseline: str = "mean",
    n_out: int = 2,
    time_limit_seconds: float | None = None,
    time_check_every: int = 256,
) -> TrainResult:
    """Fit a policy by AWR: weight each example by `exp(advantage/beta)`.

    Builds `(X = states, y = actions)` and trains the parametric policy with
    the trainer's `sample_weight` path (84.1) — the whole AWR fit is one
    `train()` call (no new loop, 84.3.3). `n_out` is the action count
    (default 2 = CartpoleSim's). Deterministic in (data, spec, seed) (G2).
    """
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions).reshape(-1)
    R = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if not (S.shape[0] == A.shape[0] == R.shape[0]) or S.shape[0] == 0:
        raise SpecError("states/actions/rewards must be non-empty, equal length")
    rets = trajectory_returns(R, int(horizon), gamma)
    w = awr_weights(advantages(rets, baseline), beta)
    use_spec = spec if spec is not None else ModelSpec(model_family="mlp")
    return train(
        (S, A.astype(np.int64)), use_spec, seed, time_limit_seconds,
        time_check_every, n_out=int(n_out), head="softmax", sample_weight=w,
    )


# --- the autonomous loop (SPEC.md 85.1) ---------------------------------------

def awr_loop(
    sim: "Simulator",
    iterations: int = 4,
    episodes: int = 32,
    horizon: int = 32,
    seed: int = 0,
    beta: float = 1.0,
    gamma: float = 1.0,
    baseline: str = "mean",
    spec: ModelSpec | None = None,
    time_limit_seconds: float | None = None,
) -> dict:
    """Sim-in-the-loop: rollout → AWR retrain → repeat (SPEC.md 85.1).

    The sequential analogue of the AutoRefine loop, for control tasks:
    start from the **uniform baseline policy** (no learning), then iterate
    ``data = rollout(current policy)`` → ``AWR retrain on that data``. Each
    iteration's policy is then evaluated on a **single shared evaluation
    seed** (``seed + 9000`` — identical across iterations, so the policy is
    the only variable and the improvement curve is an apples-to-apples
    comparison). Records the per-iteration mean / best total returns.

    Determinism (G2): iteration ``i`` uses seed ``seed + 1000*(i+1)`` for
    its data rollout and ``seed + 2000*(i+1)`` for its AWR fit — the whole
    loop is a pure function of (sim, budget, seeds). No environment access
    at fit time (the rollouts are the dataset, 84.3.3).
    """
    iterations = int(iterations)
    if iterations < 1:
        raise SpecError(f"iterations must be >= 1, got {iterations}")
    n_out = int(sim.n_actions)
    eval_seed = seed + 9000  # shared across iterations (fair comparison)
    policy = _zero_policy(sim)  # the uniform baseline (the "random" start)
    mean_ret: list[float] = []
    best_ret: list[float] = []
    for i in range(iterations):
        data = rollout(policy, sim, episodes, horizon, seed + 1000 * (i + 1))
        policy = train_awr_policy(
            data["states"], data["actions"], data["rewards"], horizon,
            spec=spec, seed=seed + 2000 * (i + 1), beta=beta, gamma=gamma,
            baseline=baseline, n_out=n_out,
            time_limit_seconds=time_limit_seconds,
        ).model
        # evaluate the NEW policy on the shared evaluation seed
        ev = rollout(policy, sim, episodes, horizon, eval_seed)
        tr = ev["total_returns"]
        mean_ret.append(float(tr.mean()))
        best_ret.append(float(tr.max()))
    return {
        "mean_total_returns": np.asarray(mean_ret, dtype=np.float64),
        "best_total_returns": np.asarray(best_ret, dtype=np.float64),
        "policy": policy,
        "baseline_mean_total_return": float(
            rollout(_zero_policy(sim), sim, episodes, horizon, eval_seed)
            ["total_returns"].mean()),
        "eval_seed": eval_seed,
        "iterations": iterations,
    }


def _zero_policy(sim: "Simulator"):
    """A policy that emits uniform logits — the pre-learning baseline."""

    class _Uniform:
        def forward(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], sim.n_actions), dtype=np.float64)

    return _Uniform()
