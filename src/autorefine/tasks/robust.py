"""RobustTask: worst-group robustness/OOD wrapper (SPEC.md 61.2.3, A2).

A reference *domain task pack* that wraps a base **classification** task and
scores it on a **shifted** holdout: for each of ``n_shift`` seeded Gaussian
noise levels (σ from 0 up to ``max_sigma``) it measures the base task's
holdout accuracy and takes the **worst group** —
``score = 100 · min_levels accuracy``. The improver searches the base task's
spec space (the curated mutation space) but the acceptance bar is the worst
group, so a model that cheats the clean holdout is not rewarded.

Deterministic given the seed (G2); stdlib + numpy only. Requires the base
task to expose ``holdout_rows`` (the §28.2 protocol) and a ``softmax`` head.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..config import DEFAULT_SPEC
from ..gate import Objective
from .base import Task

N_SHIFT = 3       # perturbation levels (0 = clean, up to max_sigma)
MAX_SIGMA = 0.5   # the worst perturbation's Gaussian σ


class RobustTask(Task):
    """SPEC.md 61.2.3 (A2): worst-group robustness wrapper,
    ``robust-v1``. ``score = 100 · min_levels accuracy`` over seeded
    Gaussian noise levels; delegates data to the wrapped base task."""

    name = "robust-v1"
    metric = "worst_group"   # score = 100·(worst-group accuracy)
    capabilities = frozenset()
    split_mode = "random"
    max_steps = 1

    def __init__(self, base: Task, seed: int, n_shift: int = N_SHIFT,
                 max_sigma: float = MAX_SIGMA) -> None:
        if getattr(base, "head", None) != "softmax":
            raise ValueError(
                "RobustTask wraps a classification (softmax-head) base "
                f"task; the base head is {getattr(base, 'head', None)!r}")
        if not callable(getattr(base, "holdout_rows", None)):
            raise ValueError(
                "RobustTask needs a base task exposing `holdout_rows` "
                "(the SPEC.md 28.2 protocol)")
        if int(n_shift) < 1:
            raise ValueError(f"n_shift must be >= 1, got {n_shift!r}")
        if not float(max_sigma) > 0.0:
            raise ValueError(f"max_sigma must be > 0, got {max_sigma!r}")
        self.base = base
        self.seed = int(seed)
        self.n_shift = int(n_shift)
        self.max_sigma = float(max_sigma)
        # mirror the wrapped task's identity
        self.head = base.head
        self.n_outputs = base.n_outputs
        self.state_dim = base.state_dim
        self.class_values = getattr(base, "class_values", None)
        self.default_dataset_size = base.default_dataset_size

    def make_dataset(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self.base.make_dataset(**kwargs)

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return self.base.prepare(states) if callable(getattr(self.base, "prepare", None)) else states

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 61.2.3 (A1): the *clean* holdout ``(x, y)`` (the
        zero-perturbation group) — the rows A1's actuals use."""
        return self.base.holdout_rows(n, model)

    def _level_accuracy(self, model, x: np.ndarray, y: np.ndarray,
                        level: int) -> float:
        """Holdout accuracy at perturbation ``level`` (0 = clean)."""
        sigma = self.max_sigma * level / self.n_shift
        if sigma > 0.0:
            seq = np.random.SeedSequence(
                [self.seed, zlib.crc32(f"robust-{level}".encode("utf-8"))])
            x = x + np.random.default_rng(seq).normal(0.0, sigma, x.shape)
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        y_true = np.asarray(y).ravel()
        return float((pred == y_true).mean())

    def score(self, model, split: str, n: int) -> float:
        """``100 · min_levels accuracy`` — the worst group across the
        seeded perturbation levels (higher = better; a clean-holdout
        cheater scores low)."""
        x, y = self.base.holdout_rows(min(int(n), 200), model)
        if len(x) == 0:
            return 0.0
        accs = [self._level_accuracy(model, x, y, lvl)
                for lvl in range(self.n_shift)]
        return float(100.0 * min(accs))

    # --- the domain's bar (61.2) ----------------------------------------------
    @property
    def starter_spec(self) -> dict:
        """The domain-appropriate starting ModelSpec (61.2.3)."""
        return DEFAULT_SPEC.to_dict()

    def default_objectives(self, target: float) -> tuple[Objective, ...]:
        """The domain's acceptance bar (61.2.3): the worst-group score
        gate plus a per-class constraint (when the base has classes)."""
        objs = (Objective(name="score", op=">=", threshold=float(target)),)
        if self.class_values:
            objs = objs + (
                Objective(name="per_class_f1", op=">=", threshold=0.6),)
        return objs
