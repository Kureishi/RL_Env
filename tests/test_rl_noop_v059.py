"""RL no-op step exclusion — the duplicate-channel collapse fix (v0.59,
SPEC.md 73, A63).

SPEC.md 73: free no-op steps (duplicate / duplicate_stall / stopped /
invalid rejections — the env's `info["candidate_score"] is None`
contract) used to carry positive advantage under the REINFORCE
running-mean baseline and reinforce the duplicated action (the sin20
collapse: 32 free duplicates swamping one real step, 73.1). `observe`
now takes `no_op=True` and drops that step's proposed action from the
trajectory — the no-op is invisible to the gradient (73.2); a
pure-no-op episode flushes as a zero-return, zero-gradient update
(73.2.3). The default `no_op=False` keeps the exact pre-v0.59 path
(73.4 pin safety).

House rules: pure / deterministic (G2), stdlib + numpy, no cross-test
imports (the scripted env is synthesized here — duck-typed, exactly
what `train_policy` consumes), tiny budgets for speed.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

import autorefine
from autorefine.cli import build_parser
from autorefine.config import DEFAULT_SPEC
from autorefine.improver.rl_policy import MetaRLPolicy, train_policy

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


# --- fixtures (synthesized here — no cross-test imports) ----------------------

def _env_state() -> dict:
    """A minimal proposal env_state (the keys `propose`/`embed` read)."""
    return {
        "best_spec": DEFAULT_SPEC.to_dict(),
        "best_score": 77.5,
        "experiments_left": 10,
        "done": False,
    }


class _ScriptedEnv:
    """Duck-typed env for `train_policy` — exactly the surface it
    consumes: `reset()`, `step(action) -> (state, reward, done, info)`,
    `.done`. Steps are `(reward, done, candidate_score)`; a
    `candidate_score` of `None` is the env's free-no-op contract signal
    (73.2.1) and is left absent from `info` — a real experiment /
    screened rejection always carries one (including 0.0)."""

    def __init__(self, steps: list[tuple[float, bool, float | None]]) -> None:
        self._steps = list(steps)
        self.done = False
        self.state = _env_state()

    def reset(self) -> dict:
        self._steps = list(self._steps)
        self.done = False
        self.state = _env_state()
        return self.state

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        reward, done, score = self._steps.pop(0)
        self.done = done
        self.state = {**self.state, "done": done}
        info = {"reason": "no_op" if score is None else "candidate"}
        if score is not None:
            info["candidate_score"] = score
        return self.state, reward, done, info


def _weights(p: MetaRLPolicy) -> tuple[bytes, bytes]:
    return p.w.tobytes(), p.b.tobytes()


# --- 73.2.2 unit: no-op drops the step (A63) ----------------------------------

def test_noop_observe_drops_the_proposed_action():
    """A63 (73.2.2): `observe(reward, done, no_op=True)` drops the
    step's proposed action from the trajectory — a pure-no-op episode
    ends with `last_episode_return == 0.0`, `n_updates == 1`, and
    bit-identical weights (zero gradient)."""
    policy = MetaRLPolicy(seed=7)
    w0, b0 = _weights(policy)
    env = _ScriptedEnv([(0.0, False, None), (0.0, True, None)])
    out = train_policy(env, policy, n_episodes=1)
    assert policy.last_episode_return == 0.0
    assert policy.n_updates == 1
    assert _weights(policy) == (w0, b0), "a no-op episode must move nothing"
    assert out["last_return"] == 0.0


def test_legacy_two_arg_observe_still_updates():
    """A63 (73.4 pin safety): the default 2-argument `observe(reward,
    done)` path is unchanged — a real step still backfills its reward
    and applies a (nonzero) gradient."""
    policy = MetaRLPolicy(seed=7)
    w0, b0 = _weights(policy)
    policy.propose(_env_state())
    policy.observe(-5.0, True)
    assert policy.n_updates == 1
    assert policy.last_episode_return == -5.0
    assert _weights(policy) != (w0, b0), "the legacy path must still learn"


# --- 73.2 behavioral pin: no-ops are invisible to the update (A63) -------------

def test_train_policy_ignores_free_noop_steps():
    """A63 (73.2, the key pin): one real step + 5 free no-ops (the sin20
    collapse shape, 73.1) leaves the weights bit-identical to one real
    step alone — same policy seed, same first action, same single-step
    gradient. The 5 no-ops contribute nothing to the update."""
    real = (-23.75, False, 53.75)          # reward, not-done, candidate_score set
    no_op = (0.0, False, None)             # free duplicate: candidate_score absent
    env_mixed = _ScriptedEnv([real, no_op, no_op, no_op, no_op, (0.0, True, None)])
    env_pure = _ScriptedEnv([(-23.75, True, 53.75)])

    p1 = MetaRLPolicy(seed=11)
    p2 = MetaRLPolicy(seed=11)
    o1 = train_policy(env_mixed, p1, n_episodes=1)
    o2 = train_policy(env_pure, p2, n_episodes=1)
    assert _weights(p1) == _weights(p2), "no-ops must be invisible to the update"
    assert p1.n_updates == p2.n_updates == 1
    assert p1.last_episode_return == p2.last_episode_return == -23.75
    assert o1["last_return"] == o2["last_return"] == -23.75


# --- 73.8 REINFORCE ascent: the sign is maximization (A63) ---------------------

def _force_action(policy: MetaRLPolicy, state: dict, want_idx: int) -> None:
    """Append `propose`'s trajectory entry, then pin the recorded action
    index to `want_idx` — a scripted decision point (the same hook the
    73.2 pins rely on: `propose` appends, the test rewrites the action)."""
    policy.propose(state)
    entry = list(policy._traj[-1])
    entry[1] = want_idx
    policy._traj[-1] = tuple(entry)


def test_reinforce_update_is_ascent_not_descent():
    """A63 (73.8, the sign pin): `_update` must ASCEND on expected return.
    A scripted policy that always proposes the same action and always
    earns +10 for it must converge on that action (pi -> 1.0, argmax =
    the target). The pre-v0.59 `-=` sign did the opposite (pi -> 0.0,
    argmax drifted AWAY — the controlled anti-learning evidence behind
    the sin20 collapse, 73.1). The mirror case (always -10) must
    converge OFF the action."""
    state = _env_state()

    up = MetaRLPolicy(seed=3, lr=0.5)
    for _ in range(20):
        for t in range(4):
            _force_action(up, state, 5)
            up.observe(10.0, done=(t == 3))
    p_up = up.probabilities(state)
    assert p_up[5] > 0.999, f"ascend: pi(target) must rise, got {p_up[5]:.6f}"
    assert int(np.argmax(p_up)) == 5, "ascend: argmax must be the rewarded action"

    down = MetaRLPolicy(seed=3, lr=0.5)
    for _ in range(20):
        for t in range(4):
            _force_action(down, state, 5)
            down.observe(-10.0, done=(t == 3))
    p_down = down.probabilities(state)
    assert p_down[5] < 0.001, f"descend: pi(target) must fall, got {p_down[5]:.6f}"


# --- 73.3 fit parser: ε flags (A63) ---------------------------------------------

def test_fit_parser_exposes_rl_epsilon_flags():
    """A63 (73.3 + 73.9.2): `fit` exposes `--rl-epsilon` / `--rl-epsilon-decay`.
    The fit default is 0.1 / 1.0 (73.9: ε=0 deadlocks the one-shot loop in
    the duplicate-stall attractor — the third collapse shape); a non-default
    parse flows through to `args` (the shared `_drive` rl branch reads it)."""
    parser = build_parser()
    args = parser.parse_args([
        "fit", "--data", "d.csv", "--policy", "rl",
        "--rl-epsilon", "0.3", "--rl-epsilon-decay", "0.9",
    ])
    assert args.rl_epsilon == 0.3
    assert args.rl_epsilon_decay == 0.9
    defaults = parser.parse_args(["fit", "--data", "d.csv", "--policy", "rl"])
    assert defaults.rl_epsilon == 0.1
    assert defaults.rl_epsilon_decay == 1.0


# --- 73.6 pins / 33.1 (A63) ------------------------------------------------------

def test_version_spec_cites_and_index_row():
    """A63 (73.6 + 33.1): the version steps to `0.61.0` in both sources;
    SPEC §73 cites A63 and the index row resolves to this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert py["project"]["version"] == autorefine.__version__ == "0.61.0"
    assert '"0.61.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 73.6 Acceptance (A63)" in spec
    row = next(l for l in spec.splitlines() if l.startswith("| M62 "))
    assert "73" in row and "A63" in row and "tests/test_rl_noop_v059.py" in row
    assert "A63" in Path(__file__).read_text(encoding="utf-8")
