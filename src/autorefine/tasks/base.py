"""Task ABC: the inner environment a model is trained/evaluated against.

Implementations must be deterministic given the task seed (SPEC.md G2, S3).

SPEC.md 36.1 (v0.22, G1): this is a nominal ABC, not the former structural
Protocol. Two methods are abstract — `make_dataset` and `score` — both
exist on every real task, so a task that forgets either is a
construction error, not a first-step crash. The identity attributes
(`name`, `state_dim`, `n_outputs`, `head`, `max_steps`,
`default_dataset_size`) are a documented class-attribute contract —
Python ABCs cannot enforce class attributes, so the A26 tests enforce
them instead — plus two new explicit attributes (36.1.3/36.1.4):

    metric        - the name of the score the task reports ("accuracy",
                    "r2", "mean_steps", "success"); declared instead of
                    inferred from n_outputs/label dtype (new metrics such
                    as F1 or log-loss are a task property, not a head
                    string)
    capabilities  - a frozenset of capability strings ("interactive" for
                    episode tasks, "grid" for grid-layout inputs 25.3,
                    "media" for file/directory items); empty by default

Core contract (what AutoRefineEnv needs — SPEC.md 15 "more tasks"):

    name, state_dim, n_outputs, head        - identity + model output shape
    default_dataset_size                    - default train-split size (SPEC.md 20.3)
    make_dataset(...) -> (X, y)             - training data for the train split
    score(model, split, n) -> float         - OFFICIAL scoring (holdout/gen)

Interactive tasks (policy episodes: CartPoleV1, GridNavV1) additionally
provide the episode primitives below; fitting tasks (SineRegressionV1) do
not need them and implement `score` directly.

Duck-typed fakes keep working: no core code performs `isinstance(...,
Task)`, so test fakes that structurally implement the contract remain
valid without listing Task as a base.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    name: str
    state_dim: int
    n_outputs: int      # number of model outputs (classes, or 1 for regression)
    head: str           # "softmax" (classification) or "mse" (regression)
    max_steps: int      # episode horizon (interactive tasks)
    # SPEC.md 20.3: default train-split size in the task's native units
    # (points for fitting tasks, episodes for episode tasks)
    default_dataset_size: int
    # SPEC.md 36.1.3: the metric the task reports (declared, not inferred)
    metric: str
    # SPEC.md 36.1.4: capability strings; empty by default
    capabilities: frozenset[str] = frozenset()
    # SPEC.md 45.1 (v0.31): the split rule for data-driven tasks —
    # "random" (the §22.1 seed permutation, the default; every existing
    # task keeps it) or "temporal" (walk-forward, file order — CsvTask
    # only; inert on the other tasks)
    split_mode: str = "random"

    # --- abstract core (the only two a task must implement) ----------------
    @abstractmethod
    def make_dataset(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Training data for the train split: (features (n, d), targets)."""
        ...

    @abstractmethod
    def score(self, model, split: str, n: int) -> float:
        """Official score on one split (holdout/gen); higher is better."""
        ...

    def score_fold(self, model, split: str, fold_index: int, n: int) -> float:
        """SPEC.md 44.1 (v0.30): one k-fold fold — the `fold_index`-th
        distinct held-out subset of `split`. ABC default: a fresh
        seed-derived point set, named like the §18.3 blocks (`holdout-
        kf3`) so every generative task works unchanged; CsvTask
        overrides with a deterministic subset of its non-train pool.
        Duck-typed fakes without this method are fine — only
        `kfold > 0` envs call it."""
        return float(self.score(model, f"{split}-kf{fold_index}", n))

    # --- interactive-task primitives (optional for fitting tasks) ----------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived initial-condition block for one of train/val/holdout/gen."""
        raise NotImplementedError

    def prepare(self, states: np.ndarray) -> np.ndarray:
        """State transform the model sees (clipping/normalization; identity ok)."""
        return states

    def act(self, logits: np.ndarray) -> np.ndarray:
        """Map model output (n, n_outputs) to discrete actions (n,)."""
        raise NotImplementedError

    def step_vec(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized dynamics: (states (n, d), actions (n,)) -> (next (n, d), terminated (n,))."""
        raise NotImplementedError

    def terminal_success(self, states: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Per-episode success flag given the final state and steps taken."""
        raise NotImplementedError

    def reference_action(self, states: np.ndarray) -> np.ndarray:
        """Deterministic reference policy used to generate imitation data (n,)."""
        raise NotImplementedError
