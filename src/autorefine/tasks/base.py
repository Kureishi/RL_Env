"""Task protocol: the inner environment a model is trained/evaluated against.

Implementations must be deterministic given the task seed (SPEC.md G2, S3).

Core contract (what AutoRefineEnv needs — SPEC.md 15 "more tasks"):

    name, state_dim, n_outputs, head        - identity + model output shape
    default_dataset_size                    - default train-split size (SPEC.md 20.3)
    make_dataset(...) -> (X, y)             - training data for the train split
    score(model, split, n) -> float         - OFFICIAL scoring (holdout/gen)

Interactive tasks (policy episodes: CartPoleV1, GridNavV1) additionally
provide the episode primitives below; fitting tasks (SineRegressionV1) do not
need them and implement `score` directly.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Task(Protocol):
    name: str
    state_dim: int
    n_outputs: int      # number of model outputs (classes, or 1 for regression)
    head: str           # "softmax" (classification) or "mse" (regression)
    max_steps: int      # episode horizon (interactive tasks)
    # SPEC.md 20.3: default train-split size in the task's native units
    # (points for fitting tasks, episodes for episode tasks)
    default_dataset_size: int

    def make_dataset(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Training data for the train split: (features (n, d), targets)."""
        ...

    def score(self, model, split: str, n: int) -> float:
        """Official score on one split (holdout/gen); higher is better."""
        ...

    # --- interactive-task primitives (optional for fitting tasks) ----------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived initial-condition block for one of train/val/holdout/gen."""
        ...

    def prepare(self, states: np.ndarray) -> np.ndarray:
        """State transform the model sees (clipping/normalization; identity ok)."""
        ...

    def act(self, logits: np.ndarray) -> np.ndarray:
        """Map model output (n, n_outputs) to discrete actions (n,)."""
        ...

    def step_vec(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized dynamics: (states (n, d), actions (n,)) -> (next (n, d), terminated (n,))."""
        ...

    def terminal_success(self, states: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Per-episode success flag given the final state and steps taken."""
        ...

    def reference_action(self, states: np.ndarray) -> np.ndarray:
        """Deterministic reference policy used to generate imitation data (n,)."""
        ...
