"""CartPoleV1: deterministic cart-pole balancing task (SPEC.md 7).

Standard (Brockmann) cart-pole equations, semi-implicit (symplectic) Euler at
fixed dt, discrete action {-1, +1}. All splits are seed-derived blocks of
initial conditions (SPEC.md 5.3): the model is trained by imitation on the
train split and officially scored on the holdout / gen splits, which it never
sees as data.
"""
from __future__ import annotations

import zlib

import numpy as np

from .base import Task


class CartPoleV1:
    name = "cartpole-v1"
    state_dim = 4
    n_outputs = 2        # two action logits (left/right)
    head = "softmax"
    max_steps = 500
    default_dataset_size = 60  # SPEC.md 20.3 (episodes; v1 legacy value)

    # --- physics constants (fixed in v1; exposed for v2 difficulty knobs) ---
    g = 9.8
    m_cart = 1.0
    m_pole = 0.1
    pole_length = 0.5  # b in the standard equations
    force_max = 10.0
    dt = 0.02
    theta_limit = 0.25  # rad
    x_limit = 2.4  # m
    # covers the full training-data envelope (no contradictory clipped labels)
    state_clip = 3.0

    # Reference linear controller gains: u = a*th + b*w + c*x + d*v (tuned)
    GAINS = (20.0, 2.0, 0.5, -1.0)

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    # --- seeding -----------------------------------------------------------
    def _split_rng(self, split: str) -> np.random.Generator:
        """One child RNG per (seed, split); crc32 keeps this platform-stable."""
        seq = np.random.SeedSequence([self.seed, zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        rng = self._split_rng(split)
        x = rng.uniform(-0.2, 0.2, n)
        v = rng.uniform(-0.1, 0.1, n)
        th = rng.uniform(-0.1, 0.1, n)
        w = rng.uniform(-0.5, 0.5, n)
        return np.stack([x, v, th, w], axis=1)

    # --- dynamics ----------------------------------------------------------
    def step_vec(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, v, th, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        F = actions.astype(np.float64) * self.force_max
        s, c = np.sin(th), np.cos(th)
        denom = self.pole_length * (4.0 / 3.0 - self.m_pole * c * c / (self.m_cart + self.m_pole))
        tdd = (self.g * s - c * (F + self.m_pole * self.pole_length * w * w * s)) / denom
        ddot = (F - self.m_pole * self.pole_length * (w * w * c + tdd * s)) / (
            self.m_cart + self.m_pole
        )
        # semi-implicit Euler: velocities first, then positions
        v2 = v + ddot * self.dt
        w2 = w + tdd * self.dt
        x2 = x + v2 * self.dt
        th2 = th + w2 * self.dt
        nxt = np.stack([x2, v2, th2, w2], axis=1)
        terminated = (np.abs(th2) > self.theta_limit) | (np.abs(x2) > self.x_limit)
        return nxt, terminated

    def control_signal(self, states: np.ndarray) -> np.ndarray:
        """Raw (unclipped) reference control signal u(state), before sign."""
        a, b, c, d = self.GAINS
        return a * states[:, 2] + b * states[:, 3] + c * states[:, 0] + d * states[:, 1]

    def reference_action(self, states: np.ndarray) -> np.ndarray:
        u = np.clip(self.control_signal(states), -self.force_max, self.force_max)
        return np.where(u >= 0.0, 1, -1).astype(np.int64)

    # --- model interface (Task protocol, SPEC.md 15) -------------------------
    def prepare(self, states: np.ndarray) -> np.ndarray:
        """Bound the state inputs the model sees (envelope of training data)."""
        return clip_states(states, self.state_clip)

    def act(self, logits: np.ndarray) -> np.ndarray:
        """logits -> force: class index 0/1 -> -1/+1 (label convention)."""
        return np.where(logits.argmax(axis=1) == 1, 1, -1).astype(np.int64)

    def terminal_success(self, states: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Success = survived the full horizon (pole never fell)."""
        return steps == self.max_steps

    def score(self, model, split: str, n: int) -> float:
        """Official score: mean episode survival steps (SPEC.md 5.3)."""
        from ..evaluator import run_episodes

        return run_episodes(self, model, split, n)["mean_steps"]

    # --- imitation data (train split only) ----------------------------------
    # Off-manifold augmentation box: states the model may reach under its own
    # (imperfect) control but that on-trajectory rollouts rarely visit.
    AUG_BOX = (
        (-2.0, 2.0),   # x
        (-2.5, 2.5),   # v
        (-0.2, 0.2),   # theta
        (-1.5, 1.5),   # omega
    )
    # Only states where the reference controller is confident (|u| >= margin)
    # are labeled: boundary states (u ~ 0) carry ambiguous, contradictory
    # supervision that no finite net can fit and that compounds into failure.
    # The smooth policy learned on the confident regions still decides the
    # boundary correctly by interpolation.
    LABEL_MARGIN = 0.5

    def _labeled_states(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep states with a confident reference decision; return (X, labels {0,1})."""
        u = self.control_signal(states)
        keep = np.abs(u) >= self.LABEL_MARGIN
        return states[keep], (u[keep] >= 0.0).astype(np.int64)

    def make_dataset(
        self, n_episodes: int = 40, n_random_states: int = 40000
    ) -> tuple[np.ndarray, np.ndarray]:
        """(states, action_labels) for training: margin-labeled on-trajectory
        reference rollouts (train split) plus margin-labeled random states over
        the reachable state box — a dense, unambiguous policy target."""
        X: list[np.ndarray] = []
        A: list[np.ndarray] = []

        ics = self.initial_conditions("train", n_episodes)
        for ic in ics:
            st = ic.copy()
            for _ in range(self.max_steps):
                u = self.control_signal(st[None, :])[0]
                act = 1 if u >= 0 else -1
                if abs(u) >= self.LABEL_MARGIN:
                    X.append(st[None, :])
                    A.append(np.array([1 if act == 1 else 0]))
                nxt, term = self.step_vec(st[None, :], np.array([act]))
                st = nxt[0]
                if term[0]:
                    break

        rng = self._split_rng("train-augment")
        box = np.array(self.AUG_BOX)  # (state_dim, 2) rows: (low, high) per dim
        Xr = rng.uniform(box[:, 0], box[:, 1], (n_random_states, self.state_dim))
        Xr_k, Ar_k = self._labeled_states(Xr)
        X.append(Xr_k)
        A.append(Ar_k)

        X_all = np.vstack(X)
        A_all = np.concatenate(A).astype(np.int64)
        return X_all, A_all


def clip_states(states: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Bound the state inputs the model sees (envelope of the training data)."""
    return np.clip(states, -clip, clip)
