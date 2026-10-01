"""Learned-model model-predictive control (SPEC.md 84.4).

The model-based counterpart to AWR (84.3, model-free):
  * `learn_dynamics` fits `(state, action) -> next_state` with the trainer's
    mse head — a parametric surrogate of the simulator's physics;
  * `mpc_act` greedily maximizes *predicted survival* over a short horizon:
    for every action sequence (n_actions ** horizon), roll the learned
    dynamics forward, count the steps until `sim.is_terminal` fires (or the
    state goes non-finite), and take the first action of the best sequence.

Determinism (G2): `learn_dynamics` is one `train()` call (seeded); `mpc_act`
has no RNG at all — the action-sequence enumeration is a fixed
`itertools.product` order and ties keep the *first* best (strict `>`).
"""
from __future__ import annotations

import itertools

import numpy as np

from .config import ModelSpec, SpecError
from .trainer import TrainResult, train


def learn_dynamics(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    spec: ModelSpec | None = None,
    seed: int = 0,
    time_limit_seconds: float | None = None,
    time_check_every: int = 256,
) -> TrainResult:
    """Fit the dynamics model `(state, action) -> next_state` (84.4.1).

    `actions` are encoded as a single 0/1 feature column (CartpoleSim's two
    actions); the model is the spec's parametric family (default: the
    mlp) with the mse head. One `train()` call — deterministic in
    (data, spec, seed) (G2).
    """
    S = np.asarray(states, dtype=np.float64)
    A = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    N = np.asarray(next_states, dtype=np.float64)
    if not (S.shape[0] == A.shape[0] == N.shape[0]) or S.shape[0] == 0:
        raise SpecError("states/actions/next_states must be non-empty, equal length")
    X = np.hstack([S, A])
    use_spec = spec if spec is not None else ModelSpec()
    return train(
        (X, N), use_spec, seed, time_limit_seconds, time_check_every,
        n_out=int(N.shape[1]), head="mse",
    )


def _predict_next(model, state: np.ndarray, action: int) -> np.ndarray:
    x = np.concatenate([state, np.array([float(action)], dtype=np.float64)])
    out = np.asarray(model.forward(x.reshape(1, -1)), dtype=np.float64)
    return out.reshape(-1)


def mpc_act(
    dynamics_model, sim, state: np.ndarray, horizon: int = 4,
) -> int:
    """Pick the action that maximizes predicted survival (84.4.2).

    Enumerates all `n_actions ** horizon` action sequences in a fixed order;
    for each, rolls the learned dynamics forward from `state` and counts the
    steps until `sim.is_terminal` (or a non-finite state). Returns the first
    action of the best sequence (ties keep the first — deterministic, G2).
    No RNG (pure enumeration + forward passes).
    """
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    horizon = int(horizon)
    if horizon < 1:
        raise SpecError(f"horizon must be >= 1, got {horizon}")
    best_action = 0
    best_steps = -1
    for seq in itertools.product(range(int(sim.n_actions)), repeat=horizon):
        s = state
        steps = 0
        for a in seq:
            s = _predict_next(dynamics_model, s, a)
            if not np.isfinite(s).all() or sim.is_terminal(s):
                break
            steps += 1
        if steps > best_steps:
            best_action, best_steps = int(seq[0]), steps
    return int(best_action)
