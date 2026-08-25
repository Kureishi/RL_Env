"""Optional Gymnasium adapter (SPEC.md 15).

    from autorefine.gym import AutoRefineGymEnv
    env = AutoRefineGymEnv(task="cartpole-v1", seed=7)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

`gymnasium` stays an *optional* dependency (SPEC.md 3): importing this module
without it raises an actionable ImportError. Observation is a fixed-length
numeric encoding of the AutoRefineEnv state; the action space is the discrete
mutation catalog (improver/catalog.py).
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "autorefine.gym requires the optional 'gymnasium' package "
        "(kept out of the core dependency list on purpose). "
        "Install it with: pip install gymnasium"
    ) from exc

from .config import Budget
from .improver.catalog import CATALOG_FIELDS, FIELD_CATALOG, ACTIONS, apply_action
from .improver.meta_env import AutoRefineEnv
from .pareto import ParetoFrontier

HISTORY_SLOTS = 8   # env history digest is capped at 8 entries
FRONTIER_SLOTS = 4  # encode up to 4 cheapest Pareto frontier points

OBS_DIM = (
    3                          # best_score, experiments_left, seconds_left
    + len(CATALOG_FIELDS)      # one catalog-value index per spec field
    + 2 * HISTORY_SLOTS        # (score, delta) per history slot
    + 2 * FRONTIER_SLOTS       # (score, seconds) per frontier slot
)


def _obs_vector(state: dict) -> np.ndarray:
    v: list[float] = []
    best = state.get("best_spec") or {}
    v += [
        float(state.get("best_score", 0.0)),
        float(state.get("experiments_left", 0)),
        float(state.get("seconds_left", 0.0)),
    ]
    for field in CATALOG_FIELDS:
        values = FIELD_CATALOG[field]
        cur = best.get(field, "mlp" if field == "model_family" else None)
        if field == "architecture":
            idx = next((i for i, val in enumerate(values) if tuple(val) == tuple(cur)), 0)
        else:
            idx = values.index(cur) if cur in values else 0
        v.append(idx / max(len(values) - 1, 1))
    digest = list(state.get("history_digest") or [])
    for i in range(HISTORY_SLOTS):
        if i < len(digest):
            v += [digest[i][1], digest[i][2]]
        else:
            v += [0.0, 0.0]
    frontier = list(state.get("pareto_frontier") or [])
    for i in range(FRONTIER_SLOTS):
        if i < len(frontier):
            v += [frontier[i]["score"], frontier[i]["seconds"]]
        else:
            v += [0.0, 0.0]
    out = np.zeros(OBS_DIM)
    out[: len(v)] = np.asarray(v, dtype=np.float64)
    return out


class AutoRefineGymEnv(gym.Env):
    """Gymnasium wrapper around AutoRefineEnv (SPEC.md 15)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "cartpole-v1",
        seed: int = 7,
        budget: Budget | None = None,
        runs_dir: str = "runs",
    ) -> None:
        super().__init__()
        self._task = task
        self._budget = budget or Budget()
        self._runs_dir = runs_dir
        self._env = AutoRefineEnv(
            task=task, seed=seed, budget=self._budget, runs_dir=runs_dir
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._state: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None and int(seed) != self._env.seed:
            self._env = AutoRefineEnv(
                task=self._task, seed=int(seed), budget=self._budget, runs_dir=self._runs_dir
            )
        self._state = self._env.reset()
        return _obs_vector(self._state), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        spec = apply_action(self._state["best_spec"], int(action))
        state, reward, done, info = self._env.step(spec)
        self._state = state
        return _obs_vector(state), float(reward), bool(done), False, dict(info)
