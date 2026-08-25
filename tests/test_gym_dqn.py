"""DQN-style agent through the Gymnasium adapter (SPEC.md 21.1, A11).

The agent (examples/gym_dqn.py) is driven only through the gymnasium API —
these tests prove the §15 adapter contract with an external-library-style
agent. Skipped when gymnasium is not installed (it stays optional, §3).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from gym_dqn import DQNAgent  # noqa: E402

from autorefine import Budget
from autorefine.gym import AutoRefineGymEnv


def _episode(tmp_path, seed: int, experiments: int = 2):
    """One full episode driven purely through the gymnasium API."""
    env = AutoRefineGymEnv(
        task="sine-v1", seed=seed,
        budget=Budget(experiments, 300, 30), runs_dir=str(tmp_path / f"seed{seed}"),
    )
    obs, _ = env.reset(seed=seed)
    assert env.observation_space.contains(obs)
    agent = DQNAgent(obs.shape[0], env.action_space.n, seed=seed,
                     batch_size=2, buffer_size=64)  # small batch: updates fire in-loop
    actions = []
    total = 0.0
    while True:
        action = agent.act(obs)
        assert env.action_space.contains(action)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        agent.step_learning(obs, action, reward, next_obs, terminated)
        actions.append(action)
        total += reward
        obs = next_obs
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        if terminated:
            break
        if len(actions) > 40:
            raise AssertionError("episode did not terminate within 40 steps")
    return env, agent, actions, total


def test_dqn_episode_through_adapter(tmp_path):
    env, agent, actions, total = _episode(tmp_path, seed=7)
    assert actions and all(0 <= a < env.action_space.n for a in actions)
    assert np.isfinite(total)
    assert agent.updates >= 1  # replay actually ran
    assert env._env.done  # inner env reports done too (SPEC.md 15)
    assert (env._env.run_dir / "summary.json").exists()


def test_replay_updates_move_weights():
    agent = DQNAgent(10, 8, seed=7, batch_size=4, buffer_size=32)
    before = (agent.w1.copy(), agent.b1.copy(), agent.w2.copy(), agent.b2.copy())
    rng = np.random.default_rng(0)
    for i in range(8):
        obs = rng.normal(size=10)
        agent.step_learning(obs, i % 8, float(i % 3), rng.normal(size=10), i == 7)
    assert agent.updates >= 1
    for name, ref in zip(("w1", "b1", "w2", "b2"), before):
        after = getattr(agent, name)
        assert np.isfinite(after).all()
        assert not np.array_equal(after, ref)  # the update actually moved weights


def test_same_seed_same_action_stream(tmp_path):
    _env1, _agent1, actions1, total1 = _episode(tmp_path, seed=11)
    _env2, _agent2, actions2, total2 = _episode(tmp_path / "b", seed=11)
    assert actions1 == actions2  # G2
    assert total1 == total2


def test_agent_is_deterministic_in_isolation():
    a1 = DQNAgent(16, 8, seed=3)
    a2 = DQNAgent(16, 8, seed=3)
    obs = np.linspace(0.0, 1.0, 16)
    for _ in range(5):
        assert a1.act(obs) == a2.act(obs)
    assert np.array_equal(a1.q(obs), a2.q(obs))
