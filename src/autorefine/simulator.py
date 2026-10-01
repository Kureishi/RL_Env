"""Sequential decision-making simulators (SPEC.md 84).

The `Simulator` contract (reset / step / is_terminal) plus `CartpoleSim`,
a NumPy cartpole with the classic discrete push-left / push-right control
problem. The AWR loop (84.3) collects rollouts against it; MPC (84.4)
rollouts through a *learned* dynamics model consult `is_terminal`.

Pure NumPy, no new dependency (SPEC.md 3). Determinism (G2): the whole
trajectory of an episode is a pure function of (seed, action sequence) —
`reset` draws only from `np.random.default_rng(seed)` and `step` is a
deterministic Euler update.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Simulator(ABC):
    """Reset/step contract for offline sequential tasks (SPEC.md 84.2).

    A simulator is a pure function of its inputs: the same `seed` always
    yields the same initial state (G2). `step` advances one step and
    returns `(next_state, reward, done)`. `is_terminal` is the state-level
    out-of-bounds check that `step` uses for `done` — and that the learned
    dynamics' MPC rollouts (84.4) can consult without an `step` call.
    """

    n_actions: int
    state_dim: int

    @abstractmethod
    def reset(self, seed: int) -> np.ndarray:
        """Return the initial state of a fresh episode, deterministic in seed."""

    @abstractmethod
    def step(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        """Advance one step; return `(next_state, reward, done)`."""

    def is_terminal(self, state: np.ndarray) -> bool:
        """True when the state is outside the usable region (default: never)."""
        return False


class CartpoleSim(Simulator):
    """Classic cartpole, discrete control {0: push left, 1: push right}.

    State = `(x, v, theta, omega)` (cart position, cart velocity, pole
    angle, pole angular velocity). Fixed-dt Euler integration of the
    classic-control equations (CartPole-v0 constants): gravity 9.8 m/s²,
    cart 1.0 kg, pole 0.1 kg, pole length 0.5 m, force 10.0 N, dt 0.02 s.

    Reward: **+1 per step** (the classic survival reward) — maximizing
    total return == maximizing survival steps. Done when |theta| > 0.2 rad
    or |x| > 1.6 m; episodes are truncated at `max_steps` steps (a
    long-surviving policy saturates the horizon, which keeps the AWR
    dataset a clean fixed grid).
    """

    n_actions = 2
    state_dim = 4

    GRAVITY = 9.8
    CART_MASS = 1.0
    POLE_MASS = 0.1
    LENGTH = 0.5
    FORCE = 10.0
    DT = 0.02
    THETA_LIMIT = 0.2
    X_LIMIT = 1.6

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = int(max_steps)

    def reset(self, seed: int) -> np.ndarray:
        # small deterministic perturbation (like the classic formulation's
        # [-0.05, 0.05] box) — same seed, same start, always (G2).
        rng = np.random.default_rng(seed)
        return np.array(
            [
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.02, 0.02),
                rng.uniform(-0.02, 0.02),
            ],
            dtype=np.float64,
        )

    def is_terminal(self, state: np.ndarray) -> bool:
        return (
            abs(float(state[2])) > self.THETA_LIMIT
            or abs(float(state[0])) > self.X_LIMIT
        )

    def step(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        x, v, theta, omega = (float(state[i]) for i in range(4))
        force = self.FORCE if int(action) == 1 else -self.FORCE
        g = self.GRAVITY
        mc = self.CART_MASS
        mp = self.POLE_MASS
        l = self.LENGTH
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        # classic cartpole equations (OpenAI CartPole-v0 formulation)
        tmp = (force + mp * l * omega * omega * sin_t) / (mc + mp)
        theta_acc = (
            g * sin_t - tmp * cos_t
        ) / (l * (4.0 / 3.0 - mp * cos_t * cos_t / (mc + mp)))
        x_acc = tmp - mp * theta_acc * cos_t / (mc + mp)
        v = v + x_acc * self.DT
        x = x + v * self.DT
        omega = omega + theta_acc * self.DT
        theta = theta + omega * self.DT
        done = self.is_terminal(np.array([x, v, theta, omega]))
        return np.array([x, v, theta, omega], dtype=np.float64), 1.0, bool(done)
