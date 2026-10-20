"""V1 search policy: hill-climb + evolutionary restarts + local refinement
(SPEC.md 8). Deterministic given a seed; swappable for an RL/Bayesian policy
via the same `propose(env_state) -> spec_dict` contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .actions import (FIELD_NAMES, SEARCH_FIELDS, mutate_spec_dict,
                      uniform_random_spec)

K_RESTART_AFTER = 5  # consecutive non-improvements before a uniform restart
REFINE_STEPS = 2  # extra mutations in the same field after a success


class SearchPolicy:
    def __init__(self, seed: int, exclude_fields: tuple = ()) -> None:
        self.rng = np.random.default_rng(seed)
        # SPEC.md 59.2 (v0.45): steering pins — the excluded fields are never
        # chosen as a mutation target (empty default = the exact pre-v0.45
        # seeded proposal stream, A1-A4 pins green)
        self.exclude_fields = tuple(exclude_fields)
        self.fail_streak = 0
        self.refine_left = 0
        self.last_field: str | None = None
        self._started = False

    # --- live steering (SPEC.md 94.1.2, v0.80) --------------------------------
    def reset_field(self, field: str) -> None:
        """SPEC.md 94.1.2 (v0.80): drop this field's per-field tracking —
        when it is the current refinement target (``last_field``), the
        refine queue clears (``last_field = None``, ``refine_left = 0``),
        so a live steering change ends the old field's refinement run and
        the search re-picks fields from the pool (94.1.4). A field the
        policy never tracks (outside ``SEARCH_FIELDS``, e.g. ``knn_k`` —
        the 25.7 exclusion) is a legal no-op: there is nothing of its to
        reset (the pre-v0.45 pin path already tolerated such rules).

        ``fail_streak`` is loop-level, not per-field, and is untouched
        (resetting it would trigger an unwanted evolutionary restart).
        An unknown field (outside the spec registry, ``FIELD_NAMES``) is
        a `ValueError` (fail loud, the 59.2.1 style).
        """
        if field not in FIELD_NAMES:
            raise ValueError(
                f"search reset_field: {field!r} is not a spec field "
                f"(available: {', '.join(FIELD_NAMES)}) (SPEC.md 94.1.2)")
        if field in SEARCH_FIELDS and self.last_field == field:
            self.last_field = None
            self.refine_left = 0

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
            pool = [f for f in SEARCH_FIELDS if f not in self.exclude_fields] \
                or list(SEARCH_FIELDS)  # 59.2: pin-safe fallback
            field = str(self.rng.choice(pool))
            self._started = True
            mode = "uniform"

        return mutate_spec_dict(env_state["best_spec"], field, self.rng,
                                mode=mode, task=task)
