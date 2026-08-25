"""Gymnasium adapter tests (SPEC.md 15; gymnasium is optional, skipped if absent)."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from autorefine import Budget
from autorefine.gym import OBS_DIM, AutoRefineGymEnv
from autorefine.improver.catalog import ACTIONS


def test_spaces_are_correct(tmp_path):
    env = AutoRefineGymEnv(task="sine-v1", seed=7, runs_dir=str(tmp_path))
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (OBS_DIM,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == len(ACTIONS)


def test_reset_and_step_contract(tmp_path):
    env = AutoRefineGymEnv(
        task="sine-v1", seed=7,
        budget=Budget(max_experiments=2, max_wall_seconds=300, max_train_seconds=30),
        runs_dir=str(tmp_path),
    )
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)

    terminated = False
    steps = 0
    while not terminated:
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
        assert isinstance(reward, float)
        assert terminated in (True, False) and truncated is False
        assert env.action_space.contains(action)
        steps += 1
        if steps > 50:
            raise AssertionError("loop did not terminate")
    assert env._env.done  # inner environment reports done too
    assert (env._env.run_dir / "summary.json").exists()


def test_reset_with_new_seed_recreates_inner_env(tmp_path):
    env = AutoRefineGymEnv(task="sine-v1", seed=7, runs_dir=str(tmp_path))
    assert env._env.seed == 7
    env.reset(seed=42)
    assert env._env.seed == 42


def test_step_before_reset_raises(tmp_path):
    env = AutoRefineGymEnv(task="sine-v1", seed=7, runs_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        env.step(0)
