"""Meta-loop: the improver tunes its own knobs from run history (A89.1).

The search loop has always had *knobs* (UCB alpha, mutation mode, the
acceptance gate z, the screening fraction, the per-episode budget split)
that a human sets once and trusts. The meta-loop closes that loop: a small
UCB bandit over the knob space proposes one knob mutation per episode,
the episode runs end-to-end on the task, and the mutated knob is credited
with a win when the episode beats the best score so far — the exact
win/credit discipline of ``BanditPolicy`` (SPEC.md 17), one level up.

Determinism (G2): every draw is a seeded ``default_rng``; the knob order,
the UCB tie-break (lexicographic), and the per-episode seed derivation are
all fixed, so the same (task, seed, episodes, budget) reproduces the same
best config bit-for-bit.

Legacy safety (99.2): ``LEGACY_META`` is the exact pre-v0.85 behavior —
``ucb_alpha 1.0`` / ``mutation_mode "local"`` (the ``BanditPolicy``
defaults), ``gate_z 0.0`` (the strict v1 score gate), ``screen_frac 1.0``
(no screening), ``budget_split 1.0`` (full budget). ``meta_loop``'s first
episode runs exactly that config, so a 1-episode meta run IS the legacy
run; every knob is opt-in by virtue of the meta command existing at all.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Budget
from .bandit import BanditPolicy

# The meta-knob space: one row per knob — (name, value space, legacy value).
# Values are the catalog style of SPEC.md 36.2 (a finite offered tuple per
# knob); the legacy value is the pre-v0.85 behavior and is always in the
# space (so a meta run can always return to legacy).
META_KNOBS: dict[str, tuple] = {
    "ucb_alpha": (0.5, 1.0, 2.0),          # BanditPolicy's exploration weight
    "mutation_mode": ("local", "uniform"),  # BanditPolicy's move style
    "gate_z": (0.0, 1.0, 2.0),             # acceptance gate z (0 = strict)
    "screen_frac": (0.25, 0.5, 1.0),       # two-stage screening fraction
    "budget_split": (0.5, 0.75, 1.0),      # per-episode budget fraction
}
LEGACY_META: dict[str, Any] = {
    "ucb_alpha": 1.0,
    "mutation_mode": "local",
    "gate_z": 0.0,
    "screen_frac": 1.0,
    "budget_split": 1.0,
}
META_KNOB_NAMES = tuple(META_KNOBS)


class MetaBandit:
    """UCB bandit over the meta-knob space (one level above ``BanditPolicy``).

    Same discipline as SPEC.md 17's field bandit: every knob is tried once
    (cold start, infinite UCB) before exploitation; the mutated knob is
    credited with a win when the episode improved on the best score so far.
    Deterministic given a seed (G2).
    """

    def __init__(self, seed: int, alpha: float = 1.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.trials: dict[str, int] = {k: 0 for k in META_KNOB_NAMES}
        self.wins: dict[str, float] = {k: 0.0 for k in META_KNOB_NAMES}
        self._last: str | None = None

    def observe(self, improved: bool) -> None:
        """Credit the knob mutated in the previous proposal with its outcome."""
        if self._last is None:
            return
        self.trials[self._last] += 1
        if improved:
            self.wins[self._last] += 1.0
        self._last = None

    def _ucb(self, knob: str) -> float:
        t = self.trials[knob]
        if t == 0:
            return float("inf")  # cold start: try every knob exactly once
        total = max(1, sum(self.trials.values()))
        return (self.wins[knob] / t
                + math.sqrt(self.alpha * math.log(total) / t))

    def _next_value(self, knob: str, current: Any) -> Any:
        """The deterministic successor of `current` in the knob's space.

        Ordered (numeric) knobs take the nearest value strictly above the
        current one (wrapping to the lowest) — a neighborhood move in value
        space, mirroring SPEC.md 18.1's local-move spirit; categorical knobs
        take the next value in declared order (wrapping)."""
        values = META_KNOBS[knob]
        i = values.index(current)
        nxt = values[(i + 1) % len(values)]
        if nxt == current and len(values) > 1:  # single-distinct-value guard
            nxt = values[(i + 2) % len(values)]
        return nxt

    def propose(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Mutate exactly one knob of `cfg` (deterministic choice + value)."""
        # deterministic tie-break: highest UCB, then lexicographically largest
        knob = max(META_KNOB_NAMES, key=lambda k: (self._ucb(k), k))
        out = dict(cfg)
        out[knob] = self._next_value(knob, cfg[knob])
        self._last = knob
        return out

    def belief(self) -> dict[str, dict[str, float]]:
        """A pure readout: per-knob (trials, win_rate) — the meta belief."""
        return {k: {"trials": self.trials[k],
                    "win_rate": (self.wins[k] / self.trials[k]
                                 if self.trials[k] else 0.0)}
                for k in META_KNOB_NAMES}


def episode_config(cfg: dict[str, Any], base_budget: int) -> dict[str, Any]:
    """Map a meta config onto (policy kwargs, env kwargs, episode budget).

    The mapping is the single source of truth for what each knob means to a
    concrete episode:

    * ``ucb_alpha`` / ``mutation_mode`` -> the ``BanditPolicy`` ctor;
    * ``gate_z`` -> the acceptance gate: ``z_accept = gate_z`` with
      ``ci_blocks = 8`` (bootstrap SE) when ``gate_z > 0`` — the SPEC.md
      18.6 rule (accept ⟺ Δeff > z·SE); ``gate_z = 0`` is the strict
      legacy gate exactly (``ci_blocks = 0``);
    * ``screen_frac`` -> the env's two-stage screening (1.0 = off, legacy);
    * ``budget_split`` -> the per-episode experiment budget fraction.
    """
    split = max(0.05, float(cfg["budget_split"]))
    budget_n = max(1, int(round(base_budget * split)))
    z = float(cfg["gate_z"])
    return {
        "policy_kwargs": {"alpha": float(cfg["ucb_alpha"]),
                          "mode": str(cfg["mutation_mode"])},
        "env_kwargs": {"ci_blocks": 8 if z > 0.0 else 0,
                       "z_accept": z,
                       "screen_frac": float(cfg["screen_frac"])},
        "budget": budget_n,
    }


def _run_episode(task: str, seed: int, cfg: dict[str, Any], base_budget: int,
                 runs_dir: str | Path, dataset_episodes: int | None = None,
                 wall_seconds: float = 900.0,
                 train_seconds: float = 30.0) -> float:
    """Run one bandit episode under the meta config; return its best score."""
    from .meta_env import AutoRefineEnv  # local: keep the import graph flat
    m = episode_config(cfg, base_budget)
    env = AutoRefineEnv(
        task=task, seed=seed,
        budget=Budget(m["budget"], wall_seconds, train_seconds),
        runs_dir=runs_dir,
        dataset_episodes=dataset_episodes,
        policy="bandit", target=None,  # the meta-loop scores; it does not gate
        **m["env_kwargs"],
    )
    policy = BanditPolicy(seed=seed, **m["policy_kwargs"])
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return float(env.best_score)


def meta_loop(task: str, seed: int = 7, episodes: int = 3, budget: int = 6,
              runs_dir: str | Path = "runs",
              dataset_episodes: int | None = None,
              verbose: bool = False) -> dict[str, Any]:
    """Run the meta-loop: episode 0 is the exact legacy config; each later
    episode mutates one knob (UCB over the meta-knobs) and is credited on
    whether it improved on the best score so far.

    Returns ``{"best_config", "best_score", "history"}`` where each history
    row is ``{"episode", "config", "score", "improved", "run_dir"}``.
    Deterministic given (task, seed, episodes, budget) (G2); `episodes`
    must be >= 1 (1 = a single legacy run).
    """
    if int(episodes) < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes!r}")
    bandit = MetaBandit(seed)
    cfg = dict(LEGACY_META)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_cfg = dict(cfg)
    for ep in range(int(episodes)):
        if ep > 0:
            cfg = bandit.propose(cfg)
        # per-episode seed: stable, distinct, independent of the base seed
        ep_seed = int(seed) * 1_000_003 + ep  # G2: fixed derivation
        score = _run_episode(task, ep_seed, cfg, int(budget), runs_dir,
                             dataset_episodes=dataset_episodes)
        improved = score > best_score + 1e-12
        bandit.observe(improved)
        history.append({
            "episode": ep,
            "config": dict(cfg),
            "score": score,
            "improved": bool(improved),
            "run_dir": str(Path(runs_dir) / f"meta-{ep}"),
        })
        if verbose:
            print(f"meta ep{ep}: {cfg} -> {score:.2f} "
                  f"({'improved' if improved else 'no'})")
        if improved:
            best_score = score
            best_cfg = dict(cfg)
    return {
        "task": task,
        "seed": int(seed),
        "episodes": int(episodes),
        "budget": int(budget),
        "best_config": best_cfg,
        "best_score": best_score,
        "belief": bandit.belief(),
        "history": history,
    }