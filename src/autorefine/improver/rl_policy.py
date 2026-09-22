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

v0.53 (SPEC.md 67, A57) — self-improving RL: the trained policy is no
longer "train and forget".
  * 67.1 persistence — `save_policy` / `load_policy` round-trip the full
    state (weights, biases, task bias B, hyper-parameters, exploration) as
    a tagged .npz; a load reconstructs a bit-exact policy.
  * 67.2 exploration — opt-in ε-greedy over the family-relevant actions
    (`epsilon`, `epsilon_decay`); the default `epsilon=0.0` keeps the exact
    single-draw proposal path (A6 stays bit-identical).
  * 67.3 checkpointing — `train_policy(..., best=True)` snapshots the
    best-episode weights (opt-in; the default return dict is unchanged).

v0.58 (SPEC.md 72, A62) — task-aware action masking: the policy offers
only the families the task offers (SPEC.md 25.5, the bandit's principle).
A `model_family -> convnet` action on a flat task (csv, parity, …) is an
*invalid* spec — the env logs a rejection with reward 0.0 and no budget
spent (SPEC.md 25.4), while every valid experiment below baseline yields a
*negative* reward. With the REINFORCE running-mean baseline, the free 0.0
steps carry positive advantage and the policy learns to propose invalid
specs — the v0.58 learning collapse. Masking the non-offered family actions
removes the exploit at its source (policy side; the env contract for
gym/external proposers is unchanged). An env_state without a `task` key
keeps the exact SPEC.md 18.2 family mask (72.5 pin safety).

v0.59 (SPEC.md 73, A63) — no-op step exclusion: free no-op steps
(duplicate re-proposals, the done-after-dedup edge — the env signals them
with `info["candidate_score"] is None`, the same contract class 72 closed
for invalid specs) carry positive advantage under the REINFORCE running-
mean baseline and reinforce the duplicated action (the sin20 collapse:
32 free duplicates swamped one real step). `observe(reward, done, no_op=True)`
drops the step's proposed action from the trajectory so no-ops are
invisible to the gradient (policy side; the env contract is unchanged
again, 72.1). The default `no_op=False` keeps the exact pre-v0.59 update
(pin-safe).

v0.59 also corrects the REINFORCE sign (SPEC.md 73.8): `_update`
accumulates the expected-return GRADIENT (SUM_t A_t * grad_log_pi) but
applied it with `-=` — a descent. Good actions were pushed down and bad
ones up (a controlled test drove an always-rewarded action's probability
to 0.0); with the corrected `+=` the policy ascends and the sin20 run
converges onto good candidates instead of locking onto bad actions. This
is the second, deeper half of the sin20 collapse fix; 73.6's acceptance
re-derives the affected pins (A6's suite stays green — its pins are
reproducibility/coverage, not value-locked — and the new test pins the
ascent behavior).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .catalog import (
    ACTIONS,
    CATALOG_FIELDS,
    FIELD_CATALOG,
    apply_action,
    index_of_value,
    relevant_actions,
    relevant_families,
)

# SPEC.md 67.1.2 (v0.53, A57): the npz format tag. A static string — NOT
# autorefine.__version__ (importing autorefine from improver would be
# circular: __init__ imports this module).
FORMAT_TAG = "autorefine.meta_rl/1"


class MetaRLPolicy:
    def __init__(self, seed: int, lr: float = 0.25, baseline_decay: float = 0.9,
                 task_names: list[str] | None = None,
                 epsilon: float = 0.0, epsilon_decay: float = 1.0) -> None:
        # SPEC.md 67.2 (v0.53, A57): opt-in ε-greedy exploration. Defaults
        # (epsilon=0.0, epsilon_decay=1.0) short-circuit to the exact
        # single-draw path — A6's bit-identical proposal stream (67.2.3).
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(
                f"epsilon must be in [0, 1], got {epsilon!r}")
        if not 0.0 < float(epsilon_decay) <= 1.0:
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {epsilon_decay!r}")
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)

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
        # D2 (SPEC.md 29.2): the last propose()'s (action_index, probs) — one
        # attribute so the opt-in train trace can read it (propose()'s RNG
        # stream and returned spec are unchanged, so A1–A18 stay bit-identical)
        self.last_proposal: tuple[int, np.ndarray] | None = None
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

    def _allowed_actions(self, family: str, task_name: str | None) -> tuple[int, ...]:
        """SPEC.md 72.1: the SPEC.md 18.2 family mask ∩ the SPEC.md 25.5
        task mask. A `model_family -> F` action is allowed only when `F`
        is a family the task offers (`relevant_families`): flat tasks
        (csv / parity / sine / cartpole / text) offer mlp/tree/boost, so on
        them the policy cannot propose the `knn` / `convnet` family
        actions — the convnet one is an invalid spec (a free 0.0-reward
        rejection, SPEC.md 25.4) that REINFORCE would learn to exploit
        (72.3, the collapse). `task_name=None` (a legacy env_state without
        the `task` key) keeps the exact 18.2 mask — 72.5 pin safety.
        The set is never empty: every family's relevant set includes the
        `model_family` field, and `mlp` is always offered."""
        if task_name is None:
            return tuple(relevant_actions(family))
        offered = set(relevant_families(task_name))
        return tuple(
            i for i in relevant_actions(family)
            if ACTIONS[i][0] != "model_family" or ACTIONS[i][1] in offered
        )

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
               task_idx: int | None = None,
               task_name: str | None = None) -> np.ndarray:
        """Softmax over catalog actions; SPEC.md 18.2 masks the actions the
        best spec's family ignores to -inf (the action space is len(ACTIONS)
        — 54 in v0.5, SPEC.md 19.4 — so weight shapes and saved policy
        state remain loadable). SPEC.md 20.2: adds the task's bias row.
        SPEC.md 72.1: the family mask is further intersected with the
        task mask (`task_name`; the 72.5 legacy mask when None)."""
        logits = x @ self.w + self.b
        if task_idx is not None:
            logits = logits + self.B[task_idx]
        rel = np.zeros(self.n_actions, dtype=bool)
        rel[list(self._allowed_actions(family, task_name))] = True
        logits = np.where(rel, logits, -np.inf)
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()

    # --- policy contract (same as SearchPolicy) --------------------------------
    def propose(self, env_state: dict) -> dict:
        """Sample an action and return the candidate spec (dict).

        The best spec's family at proposal time — and the env state's task
        name (SPEC.md 72.1) — are stored in the trajectory so `_update`
        re-computes the same masked probabilities (G2)."""
        best = env_state.get("best_spec") or {}
        family = str(best.get("model_family", "mlp"))
        tidx = self._task_idx(env_state)  # SPEC.md 20.2
        task_name = env_state.get("task")  # SPEC.md 72.1 (72.5 mask when None)
        x = self.embed(env_state)
        p = self._probs(x, family, tidx, task_name)
        if self.epsilon > 0.0:  # SPEC.md 67.2.1 (v0.53, A57): ε-greedy
            # one uniform draw over the allowed actions (the same
            # family ∩ task mask the softmax sees, 72.1), else the usual
            # softmax sample. The epsilon=0.0 default never reaches here
            # (67.2.3 pin safety).
            if self.rng.random() < self.epsilon:
                allowed = list(self._allowed_actions(family, task_name))
                a = int(allowed[int(self.rng.choice(len(allowed)))])
            else:
                a = int(self.rng.choice(self.n_actions, p=p))
        else:
            a = int(self.rng.choice(self.n_actions, p=p))
        # D2 (SPEC.md 29.2): expose what was sampled (pure bookkeeping; set
        # *after* the RNG draw so the stream and the returned spec are unchanged)
        self.last_proposal = (a, p)
        self._traj.append((x, a, 0.0, family, tidx, task_name))
        return apply_action(env_state["best_spec"], a)

    def probabilities(self, env_state: dict) -> np.ndarray:
        """D2 (SPEC.md 29.2): the full action-probability vector (length
        `n_actions`) that `propose()` would sample from at this state — the
        exact masked softmax (family mask per SPEC.md 18.2, task bias per
        SPEC.md 20.2, task mask per SPEC.md 72.1). **Pure**: no sampling,
        no RNG draw, no state change (G2); masked actions read ~0 and the
        allowed actions sum to 1."""
        best = env_state.get("best_spec") or {}
        family = str(best.get("model_family", "mlp"))
        tidx = self._task_idx(env_state)
        task_name = env_state.get("task")  # SPEC.md 72.1
        return self._probs(self.embed(env_state), family, tidx, task_name)

    def observe(self, reward: float, done: bool, no_op: bool = False) -> None:
        """Record a step's reward; flush the gradient when the episode ends.

        The full entry is preserved (family mask, SPEC.md 18.2; task index,
        SPEC.md 20.2; task name, SPEC.md 72.1) so `_update` re-computes
        exactly the masked, task-conditioned probabilities the action was
        sampled under.

        SPEC.md 73.2 (v0.59, A63): `no_op=True` marks a *free* no-op step —
        the env's `info["candidate_score"] is None` (duplicate /
        duplicate_stall / stopped / invalid-spec rejection: no experiment
        happened, reward 0.0, no budget). The action proposed for that
        step is dropped from the trajectory (it was appended at
        `propose()` time), so it carries no advantage and the reward
        backfill is skipped; a pure-no-op episode flushes as a zero-return,
        zero-gradient update (73.2.3). Screened rejections and real
        experiments set `candidate_score` and keep the default path —
        exact pre-v0.59 behavior (pin-safe default)."""
        if no_op:
            if self._traj:
                self._traj.pop()
        elif self._traj:
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
            # SPEC.md 73.2.3 (v0.59, A63): pure-no-op episode — no decision
            # points to update, but the episode still ends: flush as a
            # zero-return, zero-gradient update (dead code pre-v0.59). One
            # ε decay per episode keeps the 67.2.2 "one factor per
            # episode" invariant, and `last_episode_return` is reset so a
            # following real episode can't leak the previous one.
            self.last_episode_return = 0.0
            self.n_updates += 1
            self._traj = []
            self.epsilon = max(0.0, self.epsilon * self.epsilon_decay)
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
            task_name = entry[5] if len(entry) > 5 else None  # SPEC.md 72.1
            p = self._probs(x, family, tidx, task_name)
            delta = -p
            delta[a] += 1.0
            gw += adv[t] * np.outer(x, delta)
            gb += adv[t] * delta
            if tidx is not None:
                gb_task[tidx] += adv[t] * delta
        # SPEC.md 73.8 (v0.59, A63): REINFORCE ASCENT. gw/gb (and gb_task)
        # accumulate SUM_t A_t * grad_log_pi, the gradient of the expected
        # return w.r.t. w/b (A_t = return-to-go - running-mean baseline;
        # the env's reward is positive-for-good, meta_env.py:620). Maximize
        # => w += lr * grad. The pre-v0.59 `-=` was a sign inversion (a
        # descent on expected return, present since the first commit): it
        # pushed good actions' probabilities DOWN and bad ones UP, which is
        # the root cause of the sin20 learning collapse (73.1) beyond the
        # no-op channel 73.2 closes. Controlled check (always +10 reward
        # on one action): pi(action) -> 0.0 under the old sign, -> 1.0 now.
        self.w += self.lr * (gw / T)
        self.b += self.lr * (gb / T)
        if self.B is not None:
            # only the rows of tasks that appeared in this episode move
            for i in range(len(self.task_names)):
                self.B[i] += self.lr * (gb_task[i] / T)
        self.n_updates += 1
        self._traj = []
        # SPEC.md 67.2.2 (v0.53, A57): geometric ε decay, one factor per
        # episode (0 stays 0; the default decay 1.0 is a no-op)
        self.epsilon = max(0.0, self.epsilon * self.epsilon_decay)

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self.w.copy(), self.b.copy()

    def task_bias(self) -> np.ndarray | None:
        """SPEC.md 20.2: the per-task bias matrix B (None in single-task mode)."""
        return self.B.copy() if self.B is not None else None


def train_policy(env, policy: MetaRLPolicy, n_episodes: int, verbose: bool = False,
                 trace: list | None = None, best: bool = False) -> dict:
    """Drive `env` for `n_episodes` full runs, updating `policy` after each
    episode (SPEC.md 15: the improver learning from the loop itself).

    D2 (SPEC.md 29.2): when `trace` (a list) is given, append one record per
    step — `{"task", "action", "probs", "reward"}` — for the policy view. The
    return dict is **unchanged** when `trace is None` (the default), so every
    existing caller stays bit-identical (G2).

    SPEC.md 67.3 (v0.53, A57): `best=True` (opt-in; the D2-trace precedent)
    also tracks the strictly-greater best episode and returns a `"best"`
    snapshot — `{episode, return, w, b, B}` — in the result. The key is
    added **only** when `best=True`, so the default return dict is
    bit-identical (67.5). Multi-task `train_multi_policy` is deliberately
    unextended: a per-task "best" is ambiguous there (67.3.2)."""
    returns = []
    best_snap: dict | None = None
    for ep in range(n_episodes):
        state = env.reset()
        while not env.done:
            step_task = state.get("task")  # the env_state the proposal is made under
            action = policy.propose(state)
            state, reward, done, info = env.step(action)
            # SPEC.md 73.2 (v0.59, A63): free no-op steps (candidate_score
            # unset) are invisible to the update; the D2 trace still
            # records them (informative, 73.4).
            policy.observe(reward, done, no_op=info.get("candidate_score") is None)
            if trace is not None:
                a, p = policy.last_proposal
                trace.append({
                    "task": step_task,
                    "action": int(a),
                    "probs": [float(v) for v in p],
                    "reward": float(reward),
                })
        returns.append(policy.last_episode_return)
        if best:  # SPEC.md 67.3.1 (v0.53, A57): strictly-greater snapshot
            r = policy.last_episode_return
            if best_snap is None or r > best_snap["return"]:
                best_snap = {
                    "episode": ep,
                    "return": float(r),
                    "w": policy.w.copy(),
                    "b": policy.b.copy(),
                    "B": policy.B.copy() if policy.B is not None else None,
                }
        if verbose:
            print(f"  episode {ep + 1}/{n_episodes}: return {policy.last_episode_return:+.4f}")
    out = {
        "episodes": n_episodes,
        "episode_returns": returns,
        "policy_updates": policy.n_updates,
        "last_return": returns[-1] if returns else None,
    }
    if best:  # SPEC.md 67.3.1: the key exists only in opt-in mode (67.5)
        out["best"] = best_snap
    return out


# --- policy bytes (68.2.1, v0.54, A58) + v0.53 (67.1, A57) persistence --------

def _policy_arrays(policy: MetaRLPolicy) -> dict:
    """The full trainable state as an npz array dict (67.1.1) — ONE home:
    both `save_policy` (file) and `policy_to_bytes` (memory) build the
    identical arrays, so the file and bytes formats cannot drift (68.2.1)."""
    arrays = {
        "format": np.asarray(FORMAT_TAG),
        "w": np.asarray(policy.w, dtype=np.float64),
        "b": np.asarray(policy.b, dtype=np.float64),
        "lr": np.float64(policy.lr),
        "baseline_decay": np.float64(policy.baseline_decay),
        "n_updates": np.int64(policy.n_updates),
        "epsilon": np.float64(policy.epsilon),
        "epsilon_decay": np.float64(policy.epsilon_decay),
    }
    if policy.B is not None:
        arrays["B"] = np.asarray(policy.B, dtype=np.float64)
        arrays["task_names"] = np.asarray(",".join(policy.task_names))
    return arrays


def policy_to_bytes(policy: MetaRLPolicy) -> bytes:
    """SPEC.md 68.2.1 (v0.54, A58): the full trainable state (67.1.1) as
    an in-memory tagged npz — the same arrays, the same `FORMAT_TAG` as
    `save_policy` (one home, `_policy_arrays`), so a policy can be
    carried in memory (the app's download / upload round) without a file."""
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **_policy_arrays(policy))
    return buf.getvalue()


def _policy_from_npz(z, seed: int) -> MetaRLPolicy:
    """Reconstruct a policy from an opened npz view (67.1.2 validation).
    Shared by `load_policy` (file) and `policy_from_bytes` (memory) so the
    two paths can never disagree (68.2.1)."""
    if "format" not in z or str(z["format"]) != FORMAT_TAG:
        raise ValueError(
            f"not an AutoRefine policy file (expected format tag "
            f"{FORMAT_TAG!r}, got "
            f"{None if 'format' not in z else str(z['format'])!r})")
    w = z["w"].astype(np.float64)
    b = z["b"].astype(np.float64)
    if w.shape[1] != len(ACTIONS) or b.shape[0] != len(ACTIONS):
        raise ValueError(
            f"incompatible action dimension: weights {w.shape}, "
            f"bias {b.shape} vs {len(ACTIONS)} catalog actions")
    lr = float(z["lr"])
    baseline_decay = float(z["baseline_decay"])
    n_updates = int(z["n_updates"])
    epsilon = float(z["epsilon"])
    epsilon_decay = float(z["epsilon_decay"])
    B = z["B"].astype(np.float64) if "B" in z else None
    task_names = (
        [t for t in str(z["task_names"]).split(",") if t]
        if "task_names" in z else None
    )
    policy = MetaRLPolicy(seed, lr=lr, baseline_decay=baseline_decay,
                          task_names=task_names,
                          epsilon=epsilon, epsilon_decay=epsilon_decay)
    if B is not None:
        if policy.B is None:
            raise ValueError(
                "file carries a task bias B but no task_names")
        if B.shape != (len(policy.task_names), len(ACTIONS)):
            raise ValueError(
                f"task bias B has shape {B.shape}, expected "
                f"({len(policy.task_names)}, {len(ACTIONS)})")
        policy.B = B
    policy.w = w
    policy.b = b
    policy.n_updates = n_updates
    return policy


def policy_from_bytes(data, seed: int = 0) -> MetaRLPolicy:
    """SPEC.md 68.2.1 (v0.54, A58): the in-memory counterpart of
    `load_policy` — reconstruct a bit-exact policy (67.1.3) from bytes
    written by `policy_to_bytes`. The 67.1.2 validation (format tag,
    weight shape, B/task_names consistency) applies to arbitrary bytes:
    garbage in is a `ValueError`. `seed` is the reconstruction seed (it
    fixes `rng` for continued sampling); the saved weights are
    authoritative."""
    import io
    if isinstance(data, (str, os.PathLike)):
        raise ValueError(
            "policy_from_bytes expects bytes (use load_policy for a path)")
    with np.load(io.BytesIO(bytes(data)), allow_pickle=False) as z:
        return _policy_from_npz(z, seed)


def save_policy(policy: MetaRLPolicy, path) -> Path:
    """SPEC.md 67.1 (v0.53, A57): persist a trained policy to `path` (npz).

    67.1.1 one home — the full trainable state: `w`, `b`, the per-task bias
    `B` (+ the comma-joined `task_names`) in multi-task mode, the
    hyper-parameters (`lr`, `baseline_decay`), `n_updates`, and the
    exploration schedule (`epsilon`, `epsilon_decay`). Tagged with
    `FORMAT_TAG` so a foreign file is rejected on load (67.1.2)."""
    p = Path(path)
    # 68.2.1 (v0.54, A58): the array dict comes from `_policy_arrays` —
    # one home with `policy_to_bytes`, so file and bytes cannot drift.
    np.savez_compressed(p, **_policy_arrays(policy))
    return p


def load_policy(path, seed: int = 0) -> MetaRLPolicy:
    """SPEC.md 67.1 (v0.53, A57): reconstruct a policy bit-exactly from a
    file written by `save_policy`.

    67.1.2 contract — a `ValueError` on a missing / mismatched format tag or
    an incompatible weight shape (the catalog length, SPEC.md 19.4); a
    missing file raises `FileNotFoundError`. The reconstruction re-enters
    `MetaRLPolicy.__init__` (hyper-parameter validation included) and then
    overwrites `w` / `b` / `B` / `n_updates` with the saved arrays, so the
    loaded policy is bit-exact: same-seed `propose` streams are identical to
    the original's (67.1.3)."""
    p = Path(path)
    # 68.2.1 (v0.54, A58): the reconstruction + 67.1.2 validation come from
    # `_policy_from_npz` — one home with `policy_from_bytes`.
    with np.load(p, allow_pickle=False) as z:
        return _policy_from_npz(z, seed)


def train_multi_policy(envs: list, policy: MetaRLPolicy,
                       episodes_per_task: int, verbose: bool = False,
                       trace: list | None = None) -> dict:
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
            # SPEC.md 73.2 (v0.59, A63): no-op exclusion (73.2.4, per-task)
            policy.observe(reward, done, no_op=info.get("candidate_score") is None)
            if trace is not None:  # D2 (SPEC.md 29.2), per-step record
                a, p = policy.last_proposal
                trace.append({
                    "task": task,
                    "action": int(a),
                    "probs": [float(v) for v in p],
                    "reward": float(reward),
                })
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
