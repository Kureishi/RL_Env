"""Environment & loop design tests (SPEC.md 20, acceptance A10).

Covers the three v0.6 points:
  - curriculum / adaptive difficulty (SPEC.md 20.1)
  - multi-task meta-RL + the v0.6 stall guard (SPEC.md 20.2)
  - task-level default dataset size (SPEC.md 20.3)
All runs are deterministic given their seeds (SPEC.md G2) and use tmp_path
for artifacts (the repo's runs/ demo dirs are never touched).
"""
import numpy as np
import pytest

from autorefine import (
    AutoRefineEnv,
    BanditPolicy,
    Budget,
    DEFAULT_SPEC,
    MetaRLPolicy,
    ParityCurriculum,
    SearchPolicy,
    TASKS,
    train_multi_policy,
    train_policy,
)
from autorefine.config import ModelSpec
from autorefine.tasks import Parity4V1, ParityTask, parity_ceiling

# 3 experiments/episode keeps every meta-RL test in the small-budget regime
# (same style as tests/test_rl_policy.py); the stall guard bounds the rest.
BUD = Budget(max_experiments=3, max_wall_seconds=300.0, max_train_seconds=30.0)


def _drive(env, policy):
    state = env.reset()
    while not env.done:
        state = env.step(policy.propose(state))[0]
    return state


# --- SPEC.md 20.3: task-level default dataset size --------------------------

def test_tasks_expose_default_dataset_size():
    """Every task declares default_dataset_size (SPEC.md 20.3); the fitting
    tasks moved off 60 points, the episode tasks keep their v1 value.
    SPEC.md 22.1: the csv task is data-driven, so its size is an instance
    attribute (class-level None) derived from the file."""
    for name, cls in TASKS.items():
        if name == "csv":
            assert cls.default_dataset_size is None
            continue
        size = cls.default_dataset_size
        assert isinstance(size, int) and size > 0, name
    assert TASKS["cartpole-v1"].default_dataset_size == 60
    assert TASKS["gridnav-v1"].default_dataset_size == 60
    assert TASKS["sine-v1"].default_dataset_size == 2048
    assert TASKS["parity-v1"].default_dataset_size == 4096


def test_env_uses_task_default_and_override_wins(tmp_path):
    """AutoRefineEnv falls back to task.default_dataset_size; an explicit
    dataset_episodes still wins (parameter name unchanged, SPEC.md 20.3)."""
    assert AutoRefineEnv(task="parity-v1", seed=7, runs_dir=tmp_path).dataset_size == 4096
    assert AutoRefineEnv(task="sine-v1", seed=7, runs_dir=tmp_path).dataset_size == 2048
    assert AutoRefineEnv(task="cartpole-v1", seed=7, runs_dir=tmp_path).dataset_size == 60
    env = AutoRefineEnv(task="parity-v1", seed=7, dataset_episodes=60, runs_dir=tmp_path)
    assert env.dataset_size == 60
    assert env.dataset_episodes == 60


def test_default_dataset_size_improves_parity_baseline(tmp_path):
    """A10: the parity baseline at the 4096-point default beats the 60-point
    legacy baseline (the §18.7 run's 45.0) (SPEC.md 20.3)."""
    env60 = AutoRefineEnv(task="parity-v1", seed=7, dataset_episodes=60,
                          budget=Budget(1, 300, 30), runs_dir=tmp_path / "60")
    env60.reset()
    env_def = AutoRefineEnv(task="parity-v1", seed=7,
                            budget=Budget(1, 300, 30), runs_dir=tmp_path / "def")
    env_def.reset()
    assert np.isfinite(env60.baseline_score) and np.isfinite(env_def.baseline_score)
    assert env_def.baseline_score > env60.baseline_score


def test_default_dataset_size_improves_sine_baseline(tmp_path):
    """A10: same for sine (2048-point default vs 60 points) (SPEC.md 20.3)."""
    env60 = AutoRefineEnv(task="sine-v1", seed=7, dataset_episodes=60,
                          budget=Budget(1, 300, 30), runs_dir=tmp_path / "60")
    env60.reset()
    env_def = AutoRefineEnv(task="sine-v1", seed=7,
                            budget=Budget(1, 300, 30), runs_dir=tmp_path / "def")
    env_def.reset()
    assert np.isfinite(env60.baseline_score) and np.isfinite(env_def.baseline_score)
    assert env_def.baseline_score > env60.baseline_score


# --- SPEC.md 20.1: ParityTask parameterization + Bayes ceiling --------------

def test_parity_task_parameterized():
    """ParityTask is parameterized by (n_bits, p_flip); Parity4V1 keeps the
    v0.3 identity (SPEC.md 20.1)."""
    t = ParityTask(7, n_bits=5, p_flip=0.10)
    assert t.name == "parity-5-p0.10"
    assert t.state_dim == 5
    x, y = t.make_dataset(100)
    assert x.shape == (100, 5) and y.shape == (100,)
    x1, y1 = t.make_dataset(100)
    x2, y2 = t.make_dataset(100)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)  # G2
    x3, _ = ParityTask(8, n_bits=5, p_flip=0.10).make_dataset(100)
    assert not np.array_equal(x1, x3)
    p4 = Parity4V1(seed=7)
    assert p4.name == "parity-v1" and p4.state_dim == 4
    assert p4.n_bits == 4 and p4.p_flip == 0.08
    assert ParityTask.default_dataset_size == 4096
    with pytest.raises(ValueError):
        ParityTask(7, n_bits=0)
    with pytest.raises(ValueError):
        ParityTask(7, n_bits=9)
    with pytest.raises(ValueError):
        ParityTask(7, p_flip=-0.1)
    with pytest.raises(ValueError):
        ParityTask(7, p_flip=0.5)


def test_parity_ceiling_formula_and_monotone():
    """parity_ceiling matches the Bayes formula and is monotone in both
    difficulty axes (SPEC.md 20.1)."""
    assert parity_ceiling(4, 0.08) == pytest.approx(100.0 * (1.0 + 0.84 ** 4) / 2.0)
    assert parity_ceiling(4, 0.08) == pytest.approx(74.894, abs=1e-3)
    assert parity_ceiling(1, 0.0) == pytest.approx(100.0)  # clean bit: always right
    assert parity_ceiling(5, 0.08) < parity_ceiling(4, 0.08)
    assert parity_ceiling(4, 0.16) < parity_ceiling(4, 0.08)


# --- SPEC.md 20.1: ParityCurriculum ladder ----------------------------------

def test_curriculum_ladder_order_and_exhaustion():
    """The ladder sweeps p_flip first, then raises n_bits (SPEC.md 20.1)."""
    cur = ParityCurriculum(seed=7)
    assert (cur.n_bits, cur.p_flip) == (4, 0.08)
    assert cur.levels_left() == 9
    seq = [(cur.n_bits, cur.p_flip)]
    while cur.step_up_if_saturated(100.0):  # above every trigger threshold
        seq.append((cur.n_bits, cur.p_flip))
    assert seq == [
        (4, 0.08), (4, 0.10), (4, 0.12), (4, 0.14), (4, 0.16),
        (5, 0.08), (5, 0.10), (5, 0.12), (5, 0.14), (5, 0.16),
    ]
    assert cur.level == 9 and cur.levels_left() == 0
    assert cur.task().name == "parity-5-p0.16"
    # low scores never trigger a step-up
    cur2 = ParityCurriculum(seed=7)
    assert not cur2.step_up_if_saturated(1.0)
    assert cur2.level == 0
    # the trigger threshold is exact: score >= trigger_fraction * ceiling
    thr = 0.9 * parity_ceiling(4, 0.08)
    cur3 = ParityCurriculum(seed=7)
    assert not cur3.step_up_if_saturated(thr - 1e-9)
    assert cur3.step_up_if_saturated(thr + 1e-9)
    # construction validation
    for bad in (dict(p_flip_step=0.0), dict(n_bits_start=0),
                dict(trigger_fraction=0.0),
                dict(p_flip_start=0.2, max_p_flip=0.1)):
        with pytest.raises(ValueError):
            ParityCurriculum(seed=7, **bad)


def test_curriculum_run_steps_up_rebaselines_and_is_reproducible(tmp_path):
    """A10: a curriculum run steps up, re-baselines on the new difficulty,
    records the ladder in state/summary, and is bit-reproducible (G2)."""
    def run(tag):
        env = AutoRefineEnv(
            task="parity-v1", seed=7, budget=Budget(4, 300, 30),
            runs_dir=tmp_path / tag,
            curriculum=ParityCurriculum(seed=7, trigger_fraction=0.3),
        )
        _drive(env, BanditPolicy(seed=7))
        return env

    env1 = run("r1")
    assert env1.curriculum.level >= 1
    assert env1.task_name != "parity-v1"  # the active task is a ladder level
    assert env1.curriculum.level == len(env1.curriculum_events)
    summary1 = env1.memory.load_summary()
    assert summary1["finished_reason"] == "budget_exhausted"
    assert summary1["task"] == env1.task_name
    assert "curriculum" in summary1
    rows = summary1["curriculum"]["levels"]
    assert len(rows) == env1.curriculum.level
    for row in rows:
        for key in ("level", "difficulty", "n_bits", "p_flip", "ceiling",
                    "best_before_step", "new_baseline_score"):
            assert key in row
    # the log carries the first baseline, the experiments and the step-ups
    entries1 = env1.memory.load_experiments()
    assert [e["kind"] for e in entries1][0] == "baseline"
    assert [e["kind"] for e in entries1].count("experiment") == 4
    assert [e["kind"] for e in entries1].count("curriculum") == env1.curriculum.level
    # bit-reproducible: same seed => same kinds, hashes, scores, acceptances
    env2 = run("r2")
    entries2 = env2.memory.load_experiments()
    assert [
        (e["kind"], e.get("spec_hash"), e.get("holdout_score"), e.get("accepted"))
        for e in entries1
    ] == [
        (e["kind"], e.get("spec_hash"), e.get("holdout_score"), e.get("accepted"))
        for e in entries2
    ]
    assert env2.task_name == env1.task_name
    assert env2.curriculum.level == env1.curriculum.level


def test_curriculum_off_by_default(tmp_path):
    """Without curriculum=, behavior is exactly v0.5 (SPEC.md 20.1)."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path)
    state = env.reset()
    assert "curriculum" not in state
    assert env.task_name == "parity-v1"
    assert env.curriculum is None
    _drive(env, SearchPolicy(seed=7))
    summary = env.memory.load_summary()
    assert "curriculum" not in summary
    assert summary["task"] == "parity-v1"


def test_env_state_carries_task(tmp_path):
    """The env state exposes the active task name (SPEC.md 20.2)."""
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(1, 300, 30),
                        runs_dir=tmp_path)
    state = env.reset()
    assert state["task"] == "sine-v1"


# --- SPEC.md 20.2: multi-task meta-RL ----------------------------------------

def test_multi_task_policy_matches_single_task_at_init():
    """B is zero at init, so a multi-task policy proposes exactly like the
    v0.5 single-task policy on the same mlp-family state (SPEC.md 20.2)."""
    single = MetaRLPolicy(seed=3)
    multi = MetaRLPolicy(seed=3, task_names=["sine-v1", "parity-v1"])
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 50.0,
             "experiments_left": 5, "done": False, "task": "parity-v1"}
    for _ in range(5):
        assert ModelSpec.from_dict(single.propose(state)).to_dict() == \
            ModelSpec.from_dict(multi.propose(state)).to_dict()


def test_multi_task_policy_conditions_on_task(tmp_path):
    """A10: after one episode per task, both B rows move and differ
    (transfer via shared w/b, specificity via B) (SPEC.md 20.2)."""
    envs = [
        AutoRefineEnv(task="sine-v1", seed=7, budget=BUD, runs_dir=tmp_path / "s"),
        AutoRefineEnv(task="parity-v1", seed=7, budget=BUD, runs_dir=tmp_path / "p"),
    ]
    pol = MetaRLPolicy(seed=11, task_names=["sine-v1", "parity-v1"])
    B0 = pol.task_bias()
    assert B0.shape == (2, pol.n_actions) and np.all(B0 == 0.0)
    train_policy(envs[0], pol, 1)
    train_policy(envs[1], pol, 1)
    B = pol.task_bias()
    assert np.any(B[0] != 0.0) and np.any(B[1] != 0.0)
    assert not np.allclose(B[0], B[1])
    assert pol.n_updates == 2


def test_multi_task_row_isolation(tmp_path):
    """A10: only the active task's B row moves (SPEC.md 20.2)."""
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=BUD, runs_dir=tmp_path)
    pol = MetaRLPolicy(seed=11, task_names=["sine-v1", "parity-v1"])
    train_policy(env, pol, 1)  # sine only
    B = pol.task_bias()
    assert np.all(B[1] == 0.0)  # parity row untouched by sine episodes
    assert np.any(B[0] != 0.0)


def test_multi_task_training_deterministic_across_tasks(tmp_path):
    """A10: sine -> parity -> cartpole with one shared policy is
    bit-reproducible (G2) (SPEC.md 20.2)."""
    def run(tag):
        envs = [
            AutoRefineEnv(task=t, seed=7, budget=BUD, runs_dir=tmp_path / tag)
            for t in ("sine-v1", "parity-v1", "cartpole-v1")
        ]
        pol = MetaRLPolicy(seed=99, task_names=["sine-v1", "parity-v1", "cartpole-v1"])
        rep = train_multi_policy(envs, pol, 1)
        return rep, pol

    rep1, p1 = run("m1")
    rep2, p2 = run("m2")
    assert rep1["tasks"] == ["sine-v1", "parity-v1", "cartpole-v1"]
    assert rep1["policy_updates"] == 3
    assert rep1["last_returns"] == rep2["last_returns"]  # exact (G2)
    assert np.array_equal(p1.w, p2.w) and np.array_equal(p1.b, p2.b)
    assert np.array_equal(p1.B, p2.B)


def test_multi_task_policy_and_train_multi_validate(tmp_path):
    """Validation of the multi-task API surface (SPEC.md 20.2)."""
    with pytest.raises(ValueError):
        MetaRLPolicy(seed=1, task_names=[])
    with pytest.raises(ValueError):
        MetaRLPolicy(seed=1, task_names=["a", "a"])
    envs = [AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(1, 300, 30),
                          runs_dir=tmp_path)]
    with pytest.raises(ValueError):
        train_multi_policy(envs, MetaRLPolicy(seed=1), 1)  # single-task policy
    two = MetaRLPolicy(seed=1, task_names=["sine-v1", "parity-v1"])
    with pytest.raises(ValueError):
        train_multi_policy(envs, two, 1)  # env count != task_names count
    with pytest.raises(ValueError):
        train_multi_policy([], two, 1)  # empty envs
    one = MetaRLPolicy(seed=1, task_names=["sine-v1"])
    with pytest.raises(ValueError):
        train_multi_policy(envs, one, 0)  # episodes_per_task < 1


# --- SPEC.md 20.2: stall guard ------------------------------------------------

def test_stall_guard_binds_duplicate_loop_and_resets(tmp_path):
    """A10: 32 consecutive duplicate rejections end the episode with
    'duplicate_stall'; a fresh (non-duplicate) proposal resets the streak;
    duplicates stay free (R3) until the cap (SPEC.md 20.2)."""
    env = AutoRefineEnv(task="sine-v1", seed=7,
                        budget=Budget(100, 300, 30), runs_dir=tmp_path / "stall")
    env.reset()
    spec_a = {**DEFAULT_SPEC.to_dict(), "batch_size": 64}
    spec_b = {**DEFAULT_SPEC.to_dict(), "batch_size": 16}
    # first proposal of A is a real experiment (consumes 1)
    state, _r, done, info = env.step(spec_a)
    assert done is False and info["accepted"] in (True, False)
    # 31 consecutive duplicates: all free, episode still open
    for _ in range(31):
        state, r, done, info = env.step(spec_a)
        assert done is False
        assert r == 0.0 and info["reason"] == "duplicate"
    assert env.bm.used_experiments == 1
    # a fresh proposal resets the streak
    state, _r, done, info = env.step(spec_b)
    assert done is False
    assert env.bm.used_experiments == 2
    # ... and 32 consecutive duplicates then stall the episode
    for i in range(31):
        state, r, done, info = env.step(spec_a)
        assert done is False and info["reason"] == "duplicate"
    state, r, done, info = env.step(spec_a)
    assert done is True
    assert info["reason"] == "duplicate_stall"
    assert env.done_reason == "duplicate_stall"
    # the duplicates never touched the budget
    assert env.bm.used_experiments == 2
    summary = env.memory.load_summary()
    assert summary["finished_reason"] == "duplicate_stall"
