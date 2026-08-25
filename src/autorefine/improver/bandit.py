"""BanditPolicy: a UCB field-bandit improver (README "Extending" / SPEC.md 17).

A second reference policy showing the extension contract: implement
`propose(env_state) -> spec_dict`, consume `state["last"]` for feedback.

Each step mutates ONE field of the current best spec. The field is chosen by
UCB over per-field win rates observed through `state["last"]` (accepted /
not-accepted); every field is tried once before exploitation starts.
Deterministic given a seed (SPEC.md G2).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .actions import FIELD_NAMES, mutate_spec_dict
from .catalog import relevant_fields


class BanditPolicy:
    def __init__(self, seed: int, alpha: float = 1.0, mode: str = "local") -> None:
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)  # exploration weight in the UCB bonus
        # SPEC.md 18.1: "local" is the v0.4 default; "uniform" restores the
        # v0.3 proposal stream (both are deterministic given the seed, G2)
        self.mode = mode
        self.trials: dict[str, int] = {f: 0 for f in FIELD_NAMES}
        self.wins: dict[str, float] = {f: 0.0 for f in FIELD_NAMES}

    # --- feedback --------------------------------------------------------------
    def _update(self, env_state: dict[str, Any]) -> None:
        """Credit the fields mutated in the previous step with its outcome."""
        last = env_state.get("last")
        if not last:
            return
        accepted = bool(last.get("accepted"))
        for f in last.get("fields") or []:
            if f in self.trials:
                self.trials[f] += 1
                if accepted:
                    self.wins[f] += 1.0

    # --- action selection ------------------------------------------------------
    def _ucb(self, field: str) -> float:
        t = self.trials[field]
        if t == 0:
            return float("inf")  # cold start: try every field exactly once
        total = max(1, sum(self.trials.values()))
        return self.wins[field] / t + math.sqrt(self.alpha * math.log(total) / t)

    def propose(self, env_state: dict[str, Any]) -> dict:
        """Consume the previous step's outcome, then propose one-field mutation."""
        self._update(env_state)
        # SPEC.md 18.2: UCB over the fields the best spec's family actually
        # uses (trials/wins history is kept across family switches)
        best = env_state.get("best_spec") or {}
        fields = relevant_fields(best.get("model_family", "mlp"))
        # deterministic tie-break: highest UCB, then lexicographically largest field
        field = max(fields, key=lambda f: (self._ucb(f), f))
        return mutate_spec_dict(env_state["best_spec"], field, self.rng, mode=self.mode)
