"""RL task-aware action masking — the learning-collapse fix (v0.58,
SPEC.md 72, A62).

SPEC.md 72: the RL policy offers only the families the task offers
(25.5, the bandit's principle). On a flat task the `model_family ->
convnet` action is an invalid spec — a free 0.0-reward rejection (25.4)
that REINFORCE's running-mean baseline turns into positive advantage,
so the policy learns to propose invalid specs (the v0.58 collapse:
invalid counts 10 → 28,756 across 8 episodes). The mask removes the
exploit at its source; the env contract, the persistence format, and
the grid-task action set are unchanged (72.2).

House rules: pure / deterministic (G2), stdlib + numpy, no cross-test
imports (fixtures synthesized here), tiny envs (quadrant-XOR CSV, one
episode) for speed.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

import autorefine
from autorefine import AutoRefineEnv, Budget
from autorefine.config import DEFAULT_SPEC
from autorefine.improver.catalog import ACTIONS, relevant_actions
from autorefine.improver.rl_policy import MetaRLPolicy, train_policy
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

# The tiny per-episode budget: baseline + one scored candidate.
EXP = 2
WALL = 900.0
TRAIN = 30.0


# --- fixtures (synthesized here — no cross-test imports) ----------------------

def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    (Same shape as tests/test_rl_dashboard_v054.py's fixture.)"""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _family_action_indices(name: str) -> list[int]:
    """The catalog action indices of `model_family -> name` (one each)."""
    return [i for i, (f, v) in enumerate(ACTIONS)
            if f == "model_family" and v == name]


def _env_state(task: str | None = "csv") -> dict:
    """A minimal proposal env_state (the keys `propose`/`probabilities`
    read); `task=None` omits the key entirely (72.4.3 legacy)."""
    state = {
        "best_spec": DEFAULT_SPEC.to_dict(),
        "best_score": 50.0,
        "experiments_left": 10,
        "done": False,
    }
    if task is not None:
        state["task"] = task
    return state


def _csv_env(tmp_path: Path, seed: int) -> AutoRefineEnv:
    csv = _write_csv(tmp_path)
    task = TASKS["csv"](seed=seed, path=str(csv))
    return AutoRefineEnv(
        task="csv", seed=seed, budget=Budget(EXP, WALL, TRAIN),
        runs_dir=str(tmp_path / "runs"),
        dataset_episodes=int(task.default_dataset_size),
        task_config={"path": str(csv)},
    )


# --- 72.4 mask semantics (A62) -------------------------------------------------

def test_flat_task_masks_non_offered_family_actions():
    """A62 (72.4.1): on the flat `csv` task the policy puts zero mass on
    the `model_family -> knn/convnet` actions, positive mass on the
    offered `mlp/tree/boost` family actions, sums to 1, and never
    proposes a spec outside the offered families (60 seeded draws)."""
    policy = MetaRLPolicy(seed=7)
    p = policy.probabilities(_env_state("csv"))
    for fam in ("knn", "convnet"):
        for i in _family_action_indices(fam):
            assert p[i] == 0.0, f"model_family -> {fam} must be masked"
    for fam in ("mlp", "tree", "boost"):
        for i in _family_action_indices(fam):
            assert p[i] > 0.0, f"model_family -> {fam} must be offered"
    assert abs(float(p.sum()) - 1.0) < 1e-9
    for _ in range(60):
        spec = policy.propose(_env_state("csv"))
        assert spec["model_family"] in ("mlp", "tree", "boost"), spec


def test_grid_task_keeps_all_five_families():
    """A62 (72.4.2): the `image` task is grid_capable (25.3/25.5) — all
    five `model_family` actions stay unmasked for an `mlp` best spec."""
    policy = MetaRLPolicy(seed=7)
    p = policy.probabilities(_env_state("image"))
    for fam in ("mlp", "tree", "boost", "knn", "convnet"):
        for i in _family_action_indices(fam):
            assert p[i] > 0.0, f"model_family -> {fam} must be unmasked"
    assert abs(float(p.sum()) - 1.0) < 1e-9


def test_legacy_env_state_keeps_182_family_mask():
    """A62 (72.4.3): an env_state without the `task` key keeps the exact
    18.2 family mask (72.5 pin safety): an `mlp` best spec exposes all
    five family actions, byte-identical to pre-v0.58 semantics."""
    policy = MetaRLPolicy(seed=7)
    p = policy.probabilities(_env_state(None))
    for fam in ("mlp", "tree", "boost", "knn", "convnet"):
        for i in _family_action_indices(fam):
            assert p[i] > 0.0, f"model_family -> {fam} must be unmasked"
    allowed = set(policy._allowed_actions("mlp", None))
    assert allowed == set(relevant_actions("mlp"))
    assert allowed == set(policy._allowed_actions("mlp", None))  # pure (72.6)


def test_allowed_actions_never_empty_across_tasks():
    """A62 (72.1.1): `_allowed_actions` is never empty for any
    (family, task) pair — `mlp` is always offered — so the softmax is
    always well-defined; it is a pure function of its two arguments."""
    policy = MetaRLPolicy(seed=3)
    tasks = (None, "csv", "image", "audio", "parity-v1", "sine-v1",
             "cartpole-v1", "text")
    for family in ("mlp", "tree", "boost", "knn", "convnet"):
        for task in tasks:
            a1 = policy._allowed_actions(family, task)
            a2 = policy._allowed_actions(family, task)
            assert a1 and a1 == a2, (family, task)
            assert set(a1) <= set(relevant_actions(family)), (family, task)


# --- 72.3 the collapse is gone (behavioral pin) --------------------------------

def test_full_episode_logs_no_invalid_specs_and_is_finite(tmp_path):
    """A62 (72.3): one full `train_policy` episode on a tiny flat-CSV env
    logs zero `invalid_spec` rejections, keeps the weights finite, and
    returns a finite episode return — the pre-v0.58 collapse (the free
    0.0 convnet steps) cannot occur."""
    env = _csv_env(tmp_path, 11)
    policy = MetaRLPolicy(seed=11)
    out = train_policy(env, policy, n_episodes=1)
    assert env.done
    entries = env.memory.load_experiments()
    assert entries, "the episode must log its baseline + candidates"
    assert not [e for e in entries if e.get("kind") == "invalid_spec"], \
        "the policy must not propose a spec the flat task cannot train"
    assert np.isfinite(policy.w).all() and np.isfinite(policy.b).all()
    assert out["last_return"] is not None and np.isfinite(out["last_return"])


def _episode_fingerprint(tmp: Path, tag: str, seed: int) -> tuple:
    """Run one full episode and fingerprint the outcome (72.3, G2): the
    action stream, the reward stream, and the trained weights."""
    (tmp / tag).mkdir(parents=True, exist_ok=True)
    csv = _write_csv(tmp / tag)
    task = TASKS["csv"](seed=seed, path=str(csv))
    env = AutoRefineEnv(
        task="csv", seed=seed, budget=Budget(EXP, WALL, TRAIN),
        runs_dir=str(tmp / tag / "runs"),
        dataset_episodes=int(task.default_dataset_size),
        task_config={"path": str(csv)},
    )
    policy = MetaRLPolicy(seed=seed)
    trace: list[dict] = []
    train_policy(env, policy, n_episodes=1, trace=trace)
    return (
        [t["action"] for t in trace],
        [t["reward"] for t in trace],
        policy.w.tobytes(), policy.b.tobytes(), policy.n_updates,
    )


def test_full_episode_is_bit_reproducible_for_a_seed(tmp_path):
    """A62 (72.3, G2): the same seed reproduces the identical action
    stream and the identical weights — determinism is preserved by the
    mask (no new RNG draws; 72.6)."""
    fa = _episode_fingerprint(tmp_path, "a", 23)
    fb = _episode_fingerprint(tmp_path, "b", 23)
    assert fa == fb


# --- 72.5 pins / 33.1 (A62) -----------------------------------------------------

def test_version_spec_cites_and_index_row():
    """A62 (72.5 + 33.1): the version steps to `0.58.0` in both sources;
    SPEC §72 cites A62 and the index row resolves to this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert py["project"]["version"] == autorefine.__version__ == "0.60.0"
    assert '"0.60.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 72.7 Acceptance (A62)" in spec
    row = next(l for l in spec.splitlines() if l.startswith("| M61 "))
    assert "72" in row and "A62" in row and "tests/test_rl_taskmask_v058.py" in row
    assert "A62" in Path(__file__).read_text(encoding="utf-8")
