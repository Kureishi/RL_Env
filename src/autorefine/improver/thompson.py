"""ThompsonPolicy: a Beta-Bernoulli Thompson-sampling action improver (A89.5).

A third reference policy beside ``BanditPolicy`` (UCB) and ``MetaRLPolicy``
(REINFORCE). Each catalog action is a Bernoulli arm; after every step the
arm proposed last is credited with a win (accepted) or a loss (rejected) and
its Beta posterior is updated. The next proposal samples one posterior
draw per allowed action and takes the argmax — the classic Thompson
exploration/exploitation trade, but fully deterministic given a seed
(SPEC.md G2) because every draw comes from a seeded ``default_rng``.

The policy honours the SAME family mask (SPEC.md 18.2) and task mask
(SPEC.md 72.1) as ``MetaRLPolicy``, so it is a drop-in for the bandit / RL
``propose(env_state) -> spec_dict`` contract (SPEC.md 17) and can be
driven by ``AutoRefineEnv`` or the Gym adapter unchanged.

Defaults are deterministic (seeded); the posterior starts at Beta(1, 1)
(uniform) for every arm. This is a NEW policy (opt-in via
``--policy thompson``); it does not touch the bit-exact bandit / RL
streams (A1-A18 / A57 pins stay green because it is a separate code path).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .catalog import (
    ACTIONS,
    apply_action,
    relevant_actions,
    relevant_families,
)


class ThompsonPolicy:
    """Beta-Bernoulli Thompson sampling over the catalog actions.

    Implement the improver contract: ``propose(env_state) -> spec_dict``;
    credit the previous step via ``env_state["last"]``. Deterministic given
    ``seed`` (G2).
    """

    def __init__(self, seed: int, alpha: float = 1.0) -> None:
        self.rng = np.random.default_rng(seed)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}")
        self.alpha = float(alpha)  # concentration scale (1.0 = plain Beta(1+w,1+l))
        self.n_actions = len(ACTIONS)
        # Beta posteriors: alpha_i = 1 + wins, beta_i = 1 + losses (prior (1, 1))
        self.wins = np.zeros(self.n_actions)
        self.losses = np.zeros(self.n_actions)
        self._last: int | None = None
        self.n_updates = 0

    def _allowed_actions(self, family: str, task_name: str | None) -> tuple[int, ...]:
        """SPEC.md 18.2 family mask ∩ SPEC.md 72.1 task mask (same rule as
        ``MetaRLPolicy._allowed_actions``): a ``model_family -> F`` action is
        allowed only when ``F`` is offered by the task; the set is never
        empty (every family's relevant set includes the model_family field)."""
        allowed = tuple(relevant_actions(family))
        if task_name is None:
            return allowed
        offered = set(relevant_families(task_name))
        return tuple(
            i for i in allowed
            if ACTIONS[i][0] != "model_family" or ACTIONS[i][1] in offered
        )

    def _credit(self, env_state: dict[str, Any]) -> None:
        """Credit the arm proposed last with the previous step's outcome."""
        if self._last is None:
            return
        last = env_state.get("last")
        if not last:
            return
        accepted = bool(last.get("accepted"))
        if accepted:
            self.wins[self._last] += 1.0
        else:
            self.losses[self._last] += 1.0
        self.n_updates += 1
        self._last = None

    def propose(self, env_state: dict[str, Any]) -> dict:
        """Consume the previous step's outcome, then sample the next action."""
        self._credit(env_state)
        best = env_state.get("best_spec") or {}
        family = str(best.get("model_family", "mlp"))
        task_name = env_state.get("task")
        allowed = list(self._allowed_actions(family, task_name))
        a_idx = np.array(allowed, dtype=int)
        # sample one posterior draw per allowed arm, take the argmax; ties
        # broken by the earliest allowed action (deterministic given the seed)
        alpha_i = self.alpha * (1.0 + self.wins[a_idx])
        beta_i = self.alpha * (1.0 + self.losses[a_idx])
        theta = self.rng.beta(alpha_i, beta_i)
        a = int(a_idx[int(np.argmax(theta))])
        self._last = a
        return apply_action(env_state["best_spec"], a)

    # --- diagnostics (pure, no RNG draw) ------------------------------------
    def posterior_mean(self) -> np.ndarray:
        """Alpha/(alpha+beta) per arm — a deterministic belief readout."""
        a = self.alpha * (1.0 + self.wins)
        b = self.alpha * (1.0 + self.losses)
        return a / (a + b)