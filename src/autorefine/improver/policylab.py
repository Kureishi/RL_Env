"""Search-policy laboratory: mine, pretrain, and mix improver policies.

Three self-improvement surfaces over the machinery that already exists:

* **A2 — offline policy mining** (``mine_policy``): read the
  ``experiments.jsonl`` logs of finished runs and fit a ``MetaRLPolicy``
  by AWR-weighted maximum likelihood — each logged (state, action) pair is
  weighted by ``exp(advantage / beta)`` (the SPEC.md 84.1/84.3 AWR
  reweighting), so a policy can be learned from *past* runs with zero new
  training. Reuses ``awr.awr_weights`` for the weight law.

* **A3 — pretrained search policies** (``pretrain``): drive
  ``MetaRLPolicy`` over a set of tasks (the SPEC.md 20.2 multi-task
  channel) and persist the result with ``save_policy`` — a bundled
  starting point for ``run --pretrained FILE``.

* **A4 — population search** (``PopulationPolicy``): several improver
  policies (bandit / rl / random) compete for the same budget; the member
  that proposed the previous step is credited on acceptance and the
  selection weights adapt — a deterministic, seed-fixed policy mixture.

All three are deterministic given their seeds (SPEC.md G2) and opt-in
(new code paths; the bit-exact bandit / RL streams are untouched, 99.2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..awr import awr_weights
from .bandit import BanditPolicy
from .catalog import (
    ACTIONS,
    apply_action,
)
from .policy import SearchPolicy
from .rl_policy import MetaRLPolicy, save_policy, train_policy


# --- A2: offline policy mining ------------------------------------------------
def _transitions_from_run(run_dir: str | Path) -> tuple[list[dict[str, Any]], str | None]:
    """Recover (state, action, reward) transitions from one run's log.

    Walks ``experiments.jsonl`` in order, reconstructing the best-spec state
    before each experiment (baseline row, then every accepted experiment);
    the action is the catalog index whose ``apply_action`` from that state
    equals the candidate spec; the reward is the raw holdout delta — the
    same signal the environment gives an RL policy (meta_env's step reward).
    Rows that do not map to a single catalog action (duplicate / multi-field
    mutations) are skipped — they carry no catalog signal. Returns
    ``(transitions, task_name)``.
    """
    run_dir = Path(run_dir)
    log = run_dir / "experiments.jsonl"
    if not log.exists():
        raise FileNotFoundError(
            f"policy mine: {log} not found — pass a finished run directory")
    task_name: str | None = None
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        task_name = summary.get("task") or \
            (summary.get("run_config") or {}).get("task")
    except (OSError, json.JSONDecodeError):
        pass
    best_spec: dict | None = None
    best_score = 0.0
    n_done = 0
    budget = 30  # the embed normalizes by 30 — a stable proxy for the run
    transitions: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "baseline":
            best_spec = row.get("spec")
            best_score = float(row.get("holdout_score", 0.0))
            continue
        if kind != "experiment" or best_spec is None:
            continue
        cand = row.get("spec")
        cand_score = float(row.get("holdout_score", 0.0))
        n_done += 1
        action_index = _action_between(best_spec, cand)
        if action_index is not None:
            family = str(best_spec.get("model_family", "mlp"))
            env_state = {
                "best_spec": dict(best_spec),
                "best_score": best_score,
                "experiments_left": max(1, budget - n_done),
                "done": False,
                "task": task_name,
            }
            transitions.append({
                "env_state": env_state,
                "action": int(action_index),
                "reward": cand_score - best_score,
                "family": family,
            })
        if bool(row.get("accepted")):
            best_spec = dict(cand)
            best_score = cand_score
    return transitions, task_name


def _action_between(best_spec: dict, cand_spec: dict) -> int | None:
    """The catalog action index that maps `best_spec` -> `cand_spec`, or None
    (the candidate is not a single catalog mutation of the best spec — a
    multi-field mutation or a duplicate; it carries no catalog signal)."""
    for i, (_field, _value) in enumerate(ACTIONS):
        if apply_action(best_spec, i) == dict(cand_spec):
            return i
    return None


def mine_policy(runs_dirs: list[str | Path], seed: int = 7, beta: float = 1.0,
                lr: float = 0.5, n_epochs: int = 1) -> MetaRLPolicy:
    """A2 (A89.2): fit a ``MetaRLPolicy`` from finished runs' logs.

    Each (state, action) transition is weighted by AWR — ``w =
    exp(advantage / beta)`` with advantages centered per run (the
    SPEC.md 84.3 law via ``awr.awr_weights``) — and the policy's w/b are
    updated by weighted log-likelihood ascent over the family-masked
    softmax (the exact ``MetaRLPolicy._probs`` surface, so mined policies
    propose the same way trained ones do). Deterministic given `seed`
    (G2); raises ``ValueError`` when no transition in the given runs maps
    to a catalog action.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    if int(n_epochs) < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs!r}")
    all_trans: list[dict[str, Any]] = []
    for rd in runs_dirs:
        trans, _task = _transitions_from_run(rd)
        all_trans.extend(trans)
    if not all_trans:
        raise ValueError(
            "policy mine: no catalog-mapped transitions in the given runs "
            "(need finished runs with accepted/rejected experiments)")
    rewards = np.array([t["reward"] for t in all_trans], dtype=np.float64)
    mean_r = float(rewards.mean())
    # AWR law (SPEC.md 84.3): exp(advantage / beta) with a per-run baseline;
    # a constant-advantage run still yields positive weights (the exp shift)
    adv = rewards - mean_r
    weights = awr_weights(adv, beta=beta)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    policy = MetaRLPolicy(seed=seed)
    for _epoch in range(int(n_epochs)):
        gw = np.zeros_like(policy.w)
        gb = np.zeros(policy.n_actions)
        for t, w_t in zip(all_trans, weights):
            p = policy._probs(policy.embed(t["env_state"]), t["family"],
                              None, t["env_state"].get("task"))
            delta = -p
            delta[t["action"]] += 1.0
            gw += float(w_t) * np.outer(policy.embed(t["env_state"]), delta)
            gb += float(w_t) * delta
        n = float(len(all_trans))
        policy.w += lr * (gw / n)
        policy.b += lr * (gb / n)
    policy.n_updates = int(n_epochs)
    return policy


# --- A3: pretrained search policies -------------------------------------------
PRETRAIN_TASKS = ("parity-v1", "sine-v1", "sequence-motif-v1")


def pretrain(tasks: tuple[str, ...] = PRETRAIN_TASKS, seed: int = 7,
             episodes: int = 3, budget: int = 6,
             runs_dir: str | Path = "runs",
             out_path: str | Path | None = None,
             dataset_episodes: int | None = None) -> tuple[MetaRLPolicy, Path | None]:
    """A3 (A89.3): pretrain one task-conditioned ``MetaRLPolicy`` over
    `tasks` (the SPEC.md 20.2 transfer channel) and optionally persist it.

    Deterministic given (tasks, seed, episodes, budget) (G2). The returned
    policy has `task_names` set (so its per-task bias rows B exist);
    `out_path=None` skips persistence (the caller saves). Returns
    ``(policy, saved_path)``.
    """
    if not tasks:
        raise ValueError("pretrain: tasks must be non-empty")
    from .meta_env import AutoRefineEnv  # local: keep the import graph flat
    from ..config import Budget
    policy = MetaRLPolicy(seed=seed, task_names=list(tasks))
    for i, task in enumerate(tasks):
        env = AutoRefineEnv(
            task=task, seed=int(seed) * 1_000_003 + i,
            budget=Budget(int(budget), 900.0, 30.0),
            runs_dir=Path(runs_dir) / f"pretrain-{task}",
            dataset_episodes=dataset_episodes,
            policy="rl", target=None,
        )
        train_policy(env, policy, int(episodes))
    saved = save_policy(policy, out_path) if out_path is not None else None
    return policy, saved


# --- A4: population search ------------------------------------------------------
_POPULATION_MEMBERS = ("bandit", "rl", "random")


def _make_member(name: str, seed: int):
    if name == "bandit":
        return BanditPolicy(seed=seed)
    if name == "rl":
        return MetaRLPolicy(seed=seed)
    if name == "random":
        return SearchPolicy(seed=seed)
    raise ValueError(
        f"population member {name!r} not in {_POPULATION_MEMBERS}")


class PopulationPolicy:
    """A4 (A89.4): a competing set of improver policies over one budget.

    Each step, a member is drawn with probability proportional to
    ``1 + acceptances`` (credit assigned to the member that proposed the
    previous step, read from the environment's ``last`` info — the same
    feedback channel every policy uses, SPEC.md 17), and that member's
    ``propose(env_state)`` returns the candidate. All members share the
    environment's step feedback, so each keeps its own beliefs up to date
    while the mixture re-allocates the budget toward what works.

    Deterministic given `seed` (G2); `members` defaults to the full trio
    (bandit / rl / random).
    """

    def __init__(self, seed: int = 7,
                 members: tuple[str, ...] = _POPULATION_MEMBERS) -> None:
        if not members or len(set(members)) != len(members):
            raise ValueError(
                f"members must be a non-empty unique set, got {members!r}")
        self.rng = np.random.default_rng(seed)
        self.names: tuple[str, ...] = tuple(members)
        self.members: dict[str, Any] = {
            name: _make_member(name, int(seed) * 1_000_003 + i)
            for i, name in enumerate(self.names)
        }
        self.accepts: dict[str, int] = {name: 0 for name in self.names}
        self.proposals: dict[str, int] = {name: 0 for name in self.names}
        self._last_member: str | None = None

    def propose(self, env_state: dict[str, Any]) -> dict[str, Any]:
        last = env_state.get("last")
        if self._last_member is not None and last is not None:
            if bool(last.get("accepted")):
                self.accepts[self._last_member] += 1
        w = np.array([1.0 + self.accepts[name] for name in self.names],
                     dtype=np.float64)
        p = w / w.sum()
        name = self.names[int(self.rng.choice(len(self.names), p=p))]
        self._last_member = name
        self.proposals[name] += 1
        return self.members[name].propose(env_state)

    def allocation(self) -> dict[str, float]:
        """A pure readout: the current selection probability per member."""
        w = np.array([1.0 + self.accepts[name] for name in self.names],
                     dtype=np.float64)
        p = w / w.sum()
        return {name: float(p[i]) for i, name in enumerate(self.names)}