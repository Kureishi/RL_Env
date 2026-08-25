"""Meta-RL improver tests (SPEC.md 15: an improver trained on the loop)."""
import numpy as np
import pytest

from autorefine import AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec
from autorefine.improver.catalog import ACTIONS, apply_action
from autorefine.improver.rl_policy import MetaRLPolicy, train_policy


def test_policy_construction_and_state_dim():
    p = MetaRLPolicy(seed=0)
    assert p.n_actions == len(ACTIONS)
    assert p.w.shape == (p.state_dim, p.n_actions)
    assert np.allclose(p.b, 0.0)


def test_propose_is_deterministic_given_seed():
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 100.0,
             "experiments_left": 12, "done": False}
    a, b = MetaRLPolicy(seed=3), MetaRLPolicy(seed=3)
    # two same-seed policies sample the same action sequence
    for _ in range(5):
        sa = a.propose(state)
        sb = b.propose(state)
        assert ModelSpec.from_dict(sa).to_dict() == ModelSpec.from_dict(sb).to_dict()
        assert a._traj and a._traj[-1][1] == b._traj[-1][1]


def test_every_catalog_action_yields_a_valid_spec():
    """From both an mlp and a tree best spec, every action stays valid."""
    best_mlp = DEFAULT_SPEC.to_dict()
    best_tree = {**DEFAULT_SPEC.to_dict(), "model_family": "tree",
                 "architecture": [2]}
    for i in range(len(ACTIONS)):
        ModelSpec.from_dict(apply_action(best_mlp, i))
        ModelSpec.from_dict(apply_action(best_tree, i))


def test_embed_is_finite_and_deterministic():
    p = MetaRLPolicy(seed=0)
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 123.4,
             "experiments_left": 7, "done": False}
    e1, e2 = p.embed(state), p.embed(state)
    assert np.array_equal(e1, e2)
    assert np.isfinite(e1).all()


def test_train_policy_updates_weights_and_completes(tmp_path):
    """One episode over a tiny budget: env finishes, weights move, valid run."""
    env = AutoRefineEnv(task="sine-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path / "rl")
    policy = MetaRLPolicy(seed=7)
    w0, b0 = policy.weights()
    report = train_policy(env, policy, n_episodes=1)
    assert env.done
    assert report["policy_updates"] == 1
    w1, b1 = policy.weights()
    assert not np.array_equal(w0, w1) or not np.array_equal(b0, b1)
    assert report["episode_returns"] == [report["last_return"]]
    assert env.best_score >= env.baseline_score


def test_train_policy_is_reproducible(tmp_path):
    """Same env seed + policy seed => identical episode returns (G2)."""
    returns = []
    for i in range(2):
        env = AutoRefineEnv(task="sine-v1", seed=11,
                            budget=Budget(2, 300, 30), runs_dir=tmp_path / f"rp-{i}")
        policy = MetaRLPolicy(seed=11)
        returns.append(train_policy(env, policy, n_episodes=1)["last_return"])
    assert returns[0] == pytest.approx(returns[1], rel=1e-12)


def test_proposals_during_episodes_are_all_valid(tmp_path):
    env = AutoRefineEnv(task="gridnav-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path / "gv")
    policy = MetaRLPolicy(seed=5)
    state = env.reset()
    n = 0
    while not env.done:
        spec_dict = policy.propose(state)
        ModelSpec.from_dict(spec_dict)  # must always be a valid spec
        state, reward, done, info = env.step(spec_dict)
        policy.observe(reward, done)
        n += 1
    assert n >= 1
    assert policy.n_updates == 1
