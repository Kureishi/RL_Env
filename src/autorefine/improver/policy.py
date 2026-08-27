"""V1 search policy: hill-climb + evolutionary restarts + local refinement
(SPEC.md 8). Deterministic given a seed; swappable for an RL/Bayesian policy
via the same `propose(env_state) -> spec_dict` contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .actions import SEARCH_FIELDS, mutate_spec_dict, uniform_random_spec

K_RESTART_AFTER = 5  # consecutive non-improvements before a uniform restart
REFINE_STEPS = 2  # extra mutations in the same field after a success


class SearchPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.fail_streak = 0
        self.refine_left = 0
        self.last_field: str | None = None
        self._started = False

    def propose(self, env_state: dict[str, Any]) -> dict:
        """Consume the previous step's outcome, then propose a candidate spec."""
        last = env_state.get("last")
        if last is not None:
            if last.get("accepted"):
                self.fail_streak = 0
                fields = last.get("fields") or []
                if len(fields) == 1:
                    self.last_field = fields[0]
                    self.refine_left = REFINE_STEPS
            else:
                self.fail_streak += 1
                self.refine_left = 0

        task = env_state.get("task")  # SPEC.md 25.5: task-aware family draw
        # evolutionary restart after K consecutive non-improvements
        if self.fail_streak >= K_RESTART_AFTER:
            self.fail_streak = 0
            self.last_field = None
            self.refine_left = 0
            return uniform_random_spec(self.rng, task)

        if self._started and self.refine_left > 0 and self.last_field is not None:
            # SPEC.md 18.1: refinement (re-mutating the successful field) is a
            # local neighborhood move; initial field mutations stay uniform
            field = self.last_field
            self.refine_left -= 1
            mode = "local"
        else:
            # SPEC.md 25.7: the v1 field space (14 legacy fields) keeps the
            # A1-A4 seeded streams bit-stable as the spec space grows
            field = str(self.rng.choice(SEARCH_FIELDS))
            self._started = True
            mode = "uniform"

        return mutate_spec_dict(env_state["best_spec"], field, self.rng,
                                mode=mode, task=task)
