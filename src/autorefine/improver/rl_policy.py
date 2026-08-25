"""Meta-RL improver: a small policy trained on AutoRefineEnv itself (SPEC.md 15).

Drops into the loop with the same `propose(env_state) -> spec_dict` contract
as SearchPolicy. Actions are the discrete mutation catalog (catalog.py): each
action sets one spec field of the current best spec to a catalog value.

The policy is a softmax over catalog actions with a linear state embedding
(one-hot of the best spec's catalog values + budget/score scalars) and is
trained with REINFORCE: sampled actions, return-to-go, running-mean baseline,
gradient step at the end of each episode. Deterministic given a seed
(SPEC.md G2, S3) — the RNG stream and the (seeded) environment together pin
the whole meta-training run.
"""
from __future__ import annotations

import numpy as np

from .catalog import (
    ACTIONS,
    CATALOG_FIELDS,
    FIELD_CATALOG,
    apply_action,
    index_of_value,
    relevant_actions,
)


class MetaRLPolicy:
    def __init__(self, seed: int, lr: float = 0.25, baseline_decay: float = 0.9,
                 task_names: list[str] | None = None) -> None:
        rng = np.random.default_rng(seed)
        n_hot = sum(len(v) for v in FIELD_CATALOG.values())
        self.state_dim = n_hot + 3
        self.n_actions = len(ACTIONS)
        scale = np.sqrt(3.0 / self.state_dim)
        self.w = rng.normal(0.0, scale, (self.state_dim, self.n_actions))
        self.b = np.zeros(self.n_actions)
        self.lr = float(lr)
        self.baseline_decay = float(baseline_decay)
        self.rng = np.random.default_rng(seed + 1)
        self.baseline = 0.0
        self.n_updates = 0
        self.last_episode_return = 0.0
        self._traj: list[tuple] = []
        # SPEC.md 20.2: task conditioning — shared w/b (the transfer channel)
        # plus a per-task bias row B (task specificity):
        #   logits = x·w + b + B[task_idx]
        # task_names=None keeps the v0.5 single-task policy bit-identical.
        self.task_names: list[str] | None = (
            list(task_names) if task_names is not None else None
        )
        if self.task_names is not None:
            if not self.task_names:
                raise ValueError("task_names must be non-empty")
            if len(set(self.task_names)) != len(self.task_names):
                raise ValueError(f"task_names must be unique, got {task_names!r}")
            self.task_index = {t: i for i, t in enumerate(self.task_names)}
            self.B = np.zeros((len(self.task_names), self.n_actions))
        else:
            self.task_index = {}
            self.B = None

    def _task_idx(self, env_state: dict) -> int | None:
        """SPEC.md 20.2: the task row for this env state (state["task"])."""
        if self.B is None:
            return None
        name = env_state.get("task", self.task_names[0])
        return self.task_index.get(name, 0)

    # --- state embedding ------------------------------------------------------
    def embed(self, env_state: dict) -> np.ndarray:
        v = np.zeros(self.state_dim)
        best = env_state.get("best_spec") or {}
        off = 0
        for field in CATALOG_FIELDS:
            values = FIELD_CATALOG[field]
            i = index_of_value(field, best.get(field, "mlp" if field == "model_family" else None))
            if i is not None:
                v[off + i] = 1.0
            off += len(values)
        v[off] = float(np.tanh(env_state.get("best_score", 0.0)))
        v[off + 1] = float(min(env_state.get("experiments_left", 0) / 30.0, 1.0))
        v[off + 2] = 1.0 if env_state.get("done") else 0.0
        return v

    def _probs(self, x: np.ndarray, family: str = "mlp",
               task_idx: int | None = None) -> np.ndarray:
        """Softmax over catalog actions; SPEC.md 18.2 masks the actions the
        best spec's family ignores to -inf (the action space is len(ACTIONS)
        — 54 in v0.5, SPEC.md 19.4 — so weight shapes and saved policy
        state remain loadable). SPEC.md 20.2: adds the task's bias row."""
        logits = x @ self.w + self.b
        if task_idx is not None:
            logits = logits + self.B[task_idx]
        rel = np.zeros(self.n_actions, dtype=bool)
        rel[list(relevant_actions(family))] = True
        logits = np.where(rel, logits, -np.inf)
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()

    # --- policy contract (same as SearchPolicy) --------------------------------
    def propose(self, env_state: dict) -> dict:
        """Sample an action and return the candidate spec (dict).

        The best spec's family at proposal time is stored in the trajectory
        so `_update` re-computes the same masked probabilities (G2)."""
        best = env_state.get("best_spec") or {}
        family = str(best.get("model_family", "mlp"))
        tidx = self._task_idx(env_state)  # SPEC.md 20.2
        x = self.embed(env_state)
        p = self._probs(x, family, tidx)
        a = int(self.rng.choice(self.n_actions, p=p))
        self._traj.append((x, a, 0.0, family, tidx))
        return apply_action(env_state["best_spec"], a)

    def observe(self, reward: float, done: bool) -> None:
        """Record a step's reward; flush the gradient when the episode ends.

        The full entry is preserved (family mask, SPEC.md 18.2, and task
        index, SPEC.md 20.2) so `_update` re-computes exactly the masked,
        task-conditioned probabilities the action was sampled under."""
        if self._traj:
            entry = list(self._traj[-1])
            entry[2] = float(reward)
            self._traj[-1] = tuple(entry)
        if done:
            self._update()

    # --- REINFORCE -------------------------------------------------------------
    def _update(self) -> None:
        traj = self._traj
        T = len(traj)
        if T == 0:
            return
        # return-to-go (gamma = 1: the loop's own reward already is the delta)
        adv = [0.0] * T
        g = 0.0
        for t in range(T - 1, -1, -1):
            g += traj[t][2]
            adv[t] = g - self.baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * (g / T)
        self.last_episode_return = g

        gw = np.zeros_like(self.w)
        gb = np.zeros(self.n_actions)
        gb_task = np.zeros_like(self.B) if self.B is not None else None
        for t, entry in enumerate(traj):
            x, a, _r = entry[0], entry[1], entry[2]
            family = entry[3] if len(entry) > 3 else "mlp"  # SPEC.md 18.2 masking
            tidx = entry[4] if len(entry) > 4 else None       # SPEC.md 20.2
            p = self._probs(x, family, tidx)
            delta = -p
            delta[a] += 1.0
            gw += adv[t] * np.outer(x, delta)
            gb += adv[t] * delta
            if tidx is not None:
                gb_task[tidx] += adv[t] * delta
        self.w -= self.lr * (gw / T)
        self.b -= self.lr * (gb / T)
        if self.B is not None:
            # only the rows of tasks that appeared in this episode move
            for i in range(len(self.task_names)):
                self.B[i] -= self.lr * (gb_task[i] / T)
        self.n_updates += 1
        self._traj = []

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self.w.copy(), self.b.copy()

    def task_bias(self) -> np.ndarray | None:
        """SPEC.md 20.2: the per-task bias matrix B (None in single-task mode)."""
        return self.B.copy() if self.B is not None else None


def train_policy(env, policy: MetaRLPolicy, n_episodes: int, verbose: bool = False) -> dict:
    """Drive `env` for `n_episodes` full runs, updating `policy` after each
    episode (SPEC.md 15: the improver learning from the loop itself)."""
    returns = []
    for ep in range(n_episodes):
        state = env.reset()
        while not env.done:
            action = policy.propose(state)
            state, reward, done, info = env.step(action)
            policy.observe(reward, done)
        returns.append(policy.last_episode_return)
        if verbose:
            print(f"  episode {ep + 1}/{n_episodes}: return {policy.last_episode_return:+.4f}")
    return {
        "episodes": n_episodes,
        "episode_returns": returns,
        "policy_updates": policy.n_updates,
        "last_return": returns[-1] if returns else None,
    }


def train_multi_policy(envs: list, policy: MetaRLPolicy,
                       episodes_per_task: int, verbose: bool = False) -> dict:
    """SPEC.md 20.2: multi-task meta-RL — round-robin episodes over several
    tasks with ONE shared policy (transfer via the shared w/b; task
    specificity in B). Deterministic given (envs, policy seed, counts): the
    order is the `task_names` order, one per round."""
    if not envs:
        raise ValueError("envs must be non-empty")
    if policy.B is None:
        raise ValueError(
            "train_multi_policy needs MetaRLPolicy(task_names=[...])")
    if len(envs) != len(policy.task_names):
        raise ValueError(
            f"expected {len(policy.task_names)} envs (one per task_names entry), "
            f"got {len(envs)}")
    if episodes_per_task < 1:
        raise ValueError(f"episodes_per_task must be >= 1, got {episodes_per_task!r}")
    returns: dict[str, list[float]] = {t: [] for t in policy.task_names}
    total = episodes_per_task * len(envs)
    for ep in range(total):
        i = ep % len(envs)  # deterministic round-robin
        task = policy.task_names[i]
        env = envs[i]
        state = env.reset()
        while not env.done:
            action = policy.propose(state)
            state, reward, done, info = env.step(action)
            policy.observe(reward, done)
        returns[task].append(policy.last_episode_return)
        if verbose:
            print(f"  ep {ep + 1}/{total} [{task}]: "
                  f"return {policy.last_episode_return:+.4f}")
    return {
        "tasks": list(policy.task_names),
        "episodes_per_task": int(episodes_per_task),
        "episode_returns": returns,
        "last_returns": {t: r[-1] for t, r in returns.items() if r},
        "policy_updates": policy.n_updates,
    }
