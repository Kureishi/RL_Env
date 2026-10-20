"""BudgetManager: enforces the three hard caps (SPEC.md 5.2, S2)."""
from __future__ import annotations

import time

from .config import Budget


class BudgetManager:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.used_experiments = 0
        self._t0: float | None = None
        # SPEC.md 96.2.4 (v0.82): the pause hold — the wall clock is
        # suspended while a run is paused (a hold must not burn the wall
        # budget); `_paused_at` anchors the shift applied on resume()
        self._paused = False
        self._paused_at: float | None = None

    def start(self) -> None:
        """Begin (or restart) an episode: the budget is per-episode, so a fresh
        start restores the full caps (needed for repeated `env.reset()`, e.g.
        meta-RL training over episodes — SPEC.md 15). A fresh episode also
        starts unpaused (SPEC.md 96.2.4)."""
        self.used_experiments = 0
        self._t0 = time.monotonic()
        self._paused = False
        self._paused_at = None

    @property
    def paused(self) -> bool:
        """SPEC.md 96.2.4 (v0.82): True while the wall clock is suspended."""
        return self._paused

    def pause(self) -> None:
        """SPEC.md 96.2.4 (v0.82): suspend the wall clock — a paused run
        does not burn its wall budget while it holds. Idempotent; a no-op
        before `start()` (no clock yet). The experiment budget is
        untouched by a hold (96.2.4)."""
        if self._t0 is None or self._paused:
            return
        self._paused = True
        self._paused_at = time.monotonic()

    def resume(self) -> None:
        """SPEC.md 96.2.4 (v0.82): release a `pause()` hold — `_t0` shifts
        by the paused duration, so the clock reads as if the pause never
        happened (the hold is not charged). Idempotent."""
        if not self._paused or self._paused_at is None:
            return
        self._t0 += time.monotonic() - self._paused_at
        self._paused = False
        self._paused_at = None

    @property
    def wall_seconds_left(self) -> float:
        if self._t0 is None:
            return self.budget.max_wall_seconds
        now = time.monotonic()
        if self._paused and self._paused_at is not None:
            now = self._paused_at  # 96.2.4: the clock is frozen while paused
        return max(0.0, self.budget.max_wall_seconds - (now - self._t0))

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
