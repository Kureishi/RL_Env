"""BudgetManager: enforces the three hard caps (SPEC.md 5.2, S2)."""
from __future__ import annotations

import time

from .config import Budget


class BudgetManager:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.used_experiments = 0
        self._t0: float | None = None

    def start(self) -> None:
        """Begin (or restart) an episode: the budget is per-episode, so a fresh
        start restores the full caps (needed for repeated `env.reset()`, e.g.
        meta-RL training over episodes — SPEC.md 15)."""
        self.used_experiments = 0
        self._t0 = time.monotonic()

    @property
    def wall_seconds_left(self) -> float:
        if self._t0 is None:
            return self.budget.max_wall_seconds
        return max(0.0, self.budget.max_wall_seconds - (time.monotonic() - self._t0))

    @property
    def wall_exhausted(self) -> bool:
        return self.wall_seconds_left <= 0.0

    @property
    def experiments_left(self) -> int:
        return max(0, self.budget.max_experiments - self.used_experiments)

    def spend_experiment(self) -> None:
        if self.experiments_left <= 0:
            raise RuntimeError("experiment budget exhausted")
        self.used_experiments += 1

    def train_time_limit(self) -> float:
        """Cap a single training so it can't eat the remaining wall budget."""
        return min(self.budget.max_train_seconds, self.wall_seconds_left)
