"""Meta-environment extension tests (SPEC.md 15: tasks, families, pareto)."""
import json

import pytest

from autorefine import AutoRefineEnv, Budget, ModelSpec, SearchPolicy
from autorefine.config import SpecError


def _run(env, policy):
    state = env.reset()
    while not env.done:
        action = policy.propose(state)
        state, reward, done, info = env.step(action)
    return state


SMALL = Budget(max_experiments=5, max_wall_seconds=600.0, max_train_seconds=30.0)


def test_unknown_task_rejected(tmp_path):
    with pytest.raises(ValueError):
        AutoRefineEnv(task="nope-v1", runs_dir=tmp_path)


def test_sine_task_full_run_improves_and_writes_pareto(tmp_path):
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=SMALL, runs_dir=tmp_path / "sine")
    state = _run(env, SearchPolicy(seed=7))
    assert env.done
    assert state["best_score"] >= state["baseline_score"]
    summary = json.loads((env.run_dir / "summary.json").read_text())
    assert summary["final_best_score"] >= summary["baseline_score"]
    # SPEC.md 15: pareto memory surfaced in the summary
    assert isinstance(summary["pareto_frontier"], list) and summary["pareto_frontier"]
    assert "best_score_at_1s" in summary
    assert "efficiency_at_1s" in summary
    for p in summary["pareto_frontier"]:
        assert set(p) == {"score", "train_seconds", "spec_hash"}


def test_gridnav_task_full_run_improves(tmp_path):
    env = AutoRefineEnv(task="gridnav-v1", seed=7, budget=SMALL, runs_dir=tmp_path / "gn")
    state = _run(env, SearchPolicy(seed=7))
    assert env.done
    assert state["best_score"] >= state["baseline_score"]
    # reference policy is near-perfect; the baseline (un-tuned MLP) should
    # already learn most of the tabular policy, and the loop never regresses
    assert state["baseline_score"] > 50.0


def test_tree_family_spec_can_be_stepped(tmp_path):
    """SPEC.md 15: a candidate with model_family='tree' trains and scores."""
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(3, 300, 30),
                        runs_dir=tmp_path / "tree")
    state = env.reset()
    tree = ModelSpec(
        architecture=(3,), optimizer="sgd", learning_rate=1e-3, batch_size=32,
        weight_decay=0.0, train_steps=500, input_noise=0.0, activation="tanh",
        model_family="tree",
    )
    state, reward, done, info = env.step(tree.to_dict())
    assert info["candidate_score"] is not None and info["candidate_score"] >= 0.0
    assert env.best_spec.model_family in ("mlp", "tree")


def test_reset_after_done_restores_budget(tmp_path):
    """R5 + SPEC.md 15: after done, reset() starts a fresh episode — the
    budget is per-episode and the env must be driveable again (meta-RL)."""
    env = AutoRefineEnv(task="sine-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        action = policy.propose(state)
        state, reward, done, info = env.step(action)
    assert env.done and env.bm.used_experiments == 2

    state = env.reset()  # episode 2
    assert env.done is False
    assert env.bm.used_experiments == 0
    assert env.bm.experiments_left == 2
    action = policy.propose(state)
    state, reward, done, info = env.step(action)  # must not raise 'environment is done'
    assert env.bm.used_experiments <= 2


def test_old_style_spec_json_still_loads(tmp_path):
    """Back-compat: a pre-extension spec dict (no model_family) is valid."""
    d = {
        "architecture": [16, 8], "optimizer": "momentum", "learning_rate": 1e-3,
        "batch_size": 32, "weight_decay": 1e-4, "train_steps": 400,
        "input_noise": 0.02, "activation": "tanh",
    }
    spec = ModelSpec.from_dict(d)
    assert spec.model_family == "mlp"
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**d, "model_family": "bogus"})
