"""SineRegressionV1: deterministic function-fitting task (SPEC.md 15).

The model approximates a fixed sum of sinusoids on a 1-D domain. This is a
*fitting* task (no episodes): it implements the minimal Task protocol
(make_dataset + score) and a regression (mse) head.

Score (holdout/gen): 100 * R^2 of the model on fresh seed-derived points
(clamped at 0), i.e. the percentage of output variance explained.
"""
from __future__ import annotations

import zlib

import numpy as np

from .base import Task

# fixed target function: four sinusoids, up to 3 cycles over the unit domain
# (the model sees the unit-scaled input u = x / SPAN, so activations stay in a
# non-saturated regime — SPEC.md 15 task design). Moderate difficulty: a small
# un-tuned net fits only part of it; a well-tuned one exceeds 90% R^2.
COMPONENTS = (
    (1.20, 0.5, 0.0),
    (0.80, 0.25, 0.5),
    (0.50, 0.15, 1.2),
    (0.35, 0.06, 2.1),
)
SPAN = 12.0 * np.pi
LABEL_NOISE = 0.05  # seed-derived label noise on the train split


def target_function(u: np.ndarray) -> np.ndarray:
    """Target values for unit inputs u in [0, 1]."""
    x = u * SPAN
    y = np.zeros_like(u, dtype=np.float64)
    for amp, freq, phase in COMPONENTS:
        y = y + amp * np.sin(freq * x + phase)
    return y


class SineRegressionV1(Task):
    name = "sine-v1"
    state_dim = 1
    n_outputs = 1
    head = "mse"
    max_steps = 1  # not an episode task; present for protocol completeness
    default_dataset_size = 2048  # SPEC.md 20.3 (points)
    # SPEC.md 36.1 (v0.22, G1): declared metric + capabilities
    metric = "r2"  # score() = 100 * R^2 (clamped at 0)
    capabilities = frozenset()

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def _split_rng(self, split: str) -> np.random.Generator:
        seq = np.random.SeedSequence([self.seed, zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    # --- data ----------------------------------------------------------------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived unit input points u in [0, 1] (n, 1)."""
        rng = self._split_rng(split)
        return rng.uniform(0.0, 1.0, (n, 1))

    def make_dataset(self, n_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
        """(u (n,1), y (n,)) on the train split, with seed-derived label noise."""
        X = self.initial_conditions("train", n_points)
        rng = self._split_rng("train-noise")
        y = target_function(X[:, 0]) + rng.normal(0.0, LABEL_NOISE, n_points)
        return X, y

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # unit-scaled inputs need no transform

    # --- scoring ---------------------------------------------------------------
    def score(self, model, split: str, n: int) -> float:
        """100 * R^2 on fresh points of `split` (clamped at 0); higher is better."""
        X = self.initial_conditions(split, n)
        y = target_function(X[:, 0])
        pred = np.asarray(model.forward(X), dtype=np.float64).reshape(-1)
        ss_res = float(((pred - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)
