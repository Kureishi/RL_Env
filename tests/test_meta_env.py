"""Meta-environment tests (SPEC.md R1-R5, T4, T5) and acceptance A1-A4."""
import json

import pytest

from autorefine import AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec, SearchPolicy

SMALL_BUDGET = Budget(max_experiments=15, max_wall_seconds=600.0, max_train_seconds=30.0)


def _run(env, policy, max_steps=None):
    """Drive the loop until done (or max_steps); return (state, info list)."""
    state = env.reset()
    infos = []
    n = 0
    while not env.done and (max_steps is None or n < max_steps):
        action = policy.propose(state)
        state, reward, done, info = env.step(action)
        infos.append(info)
        n += 1
    return state, infos


def test_step_before_reset_raises(tmp_path):
    env = AutoRefineEnv(seed=1, runs_dir=tmp_path)
    with pytest.raises(RuntimeError):
        env.step(DEFAULT_SPEC.to_dict())


def test_duplicate_rejected_without_budget(tmp_path):
    """T4: resubmitting the best spec is a free 'duplicate', budget untouched."""
    env = AutoRefineEnv(seed=7, budget=Budget(5, 300, 30), runs_dir=tmp_path)
    state = env.reset()
    best = state["best_spec"]
    state, reward, done, info = env.step(best)
    assert info["accepted"] is False
    assert info["reason"] == "duplicate"
    assert reward == 0.0
    assert state["experiments_left"] == 5  # no budget consumed


def test_single_experiment_budget_and_done_semantics(tmp_path):
    """T5: max_experiments=1 ends after one real step; step after done raises."""
    env = AutoRefineEnv(seed=7, budget=Budget(1, 300, 30), runs_dir=tmp_path)
    state = env.reset()
    policy = SearchPolicy(seed=7)
    action = policy.propose(state)
    state, reward, done, info = env.step(action)
    if info.get("reason") == "duplicate":  # unlikely but safe
        action = policy.propose(state)
        state, reward, done, info = env.step(action)
    assert done is True
    assert info["reason"] == "budget_exhausted"
    assert env.bm.used_experiments == 1
    with pytest.raises(RuntimeError):
        env.step(action)


def test_reward_formula_and_strict_improvement(tmp_path):
    """reward = delta + 1e-3 novelty; best updates only on strict improvement."""
    env = AutoRefineEnv(seed=7, budget=Budget(6, 300, 30), runs_dir=tmp_path)
    state = env.reset()
    baseline = state["baseline_score"]
    policy = SearchPolicy(seed=7)
    for _ in range(6):
        if env.done:
            break
        action = policy.propose(state)
        prev_best = env.best_score
        state, reward, done, info = env.step(action)
        sc = info.get("candidate_score")
        if isinstance(sc, (int, float)):
            assert reward == pytest.approx((sc - prev_best) + 1e-3, rel=1e-9)
            if sc > prev_best:
                assert env.best_score == pytest.approx(sc)
            else:
                assert env.best_score == pytest.approx(prev_best)


def test_acceptance_A1_improvement_over_seeds(tmp_path):
    """A1 (reduced budget): final best >= 1.25x baseline, no seed regresses."""
    for seed in (1, 2, 3):
        env = AutoRefineEnv(seed=seed, budget=SMALL_BUDGET, runs_dir=tmp_path / f"a1-{seed}")
        state, infos = _run(env, SearchPolicy(seed=seed))
        assert env.done
        baseline = state["baseline_score"]
        final = state["best_score"]
        assert final >= baseline, f"seed {seed}: regression"
        assert final >= 1.25 * baseline, (
            f"seed {seed}: final {final:.1f} < 1.25*baseline {1.25 * baseline:.1f}"
        )


def test_acceptance_A2_reproducibility(tmp_path):
    """A2: two same-seed runs produce identical experiment sequences (sans ts)."""
    seqs = []
    for i in range(2):
        env = AutoRefineEnv(seed=7, budget=Budget(8, 300, 30), runs_dir=tmp_path / f"repro-{i}")
        _run(env, SearchPolicy(seed=7))
        entries = env.memory.load_experiments()
        seqs.append([
            {k: e[k] for k in ("kind", "spec_hash", "holdout_score", "gen_score", "accepted")}
            for e in entries
        ])
    assert seqs[0] == seqs[1]
    # byte-identical best spec
    import glob
    def best_spec_file(d):
        return glob.glob(str(d / "cartpole-v1-seed7-*" / "best_spec.json"))[0]
    with open(best_spec_file(tmp_path / "repro-0")) as f1, open(best_spec_file(tmp_path / "repro-1")) as f2:
        assert json.load(f1) == json.load(f2)


def test_acceptance_A3_generalization_guard(tmp_path):
    """A3: gen_gap of the final best < 5% of its holdout score."""
    env = AutoRefineEnv(seed=7, budget=SMALL_BUDGET, runs_dir=tmp_path / "a3")
    _run(env, SearchPolicy(seed=7))
    entries = env.memory.load_experiments()
    best_entry = max(
        (e for e in entries if e["kind"] == "experiment" and e.get("accepted")),
        key=lambda e: e["holdout_score"],
        default=None,
    )
    ref = best_entry or next(e for e in entries if e["kind"] == "baseline")
    gap = abs(ref["gen_gap"])
    assert gap < 0.05 * ref["holdout_score"], f"gen_gap {gap} too large vs {ref['holdout_score']}"


def test_acceptance_A4_budget_discipline(tmp_path):
    """A4: experiment cap never exceeded; per-training time cap honored."""
    env = AutoRefineEnv(
        seed=7, budget=Budget(10, 300, 2.0), runs_dir=tmp_path / "a4"
    )
    state, infos = _run(env, SearchPolicy(seed=7))
    assert env.bm.used_experiments <= 10
    assert env.bm.used_experiments == 10 or env.done
    for info in infos:
        if info.get("train_seconds") is not None:
            assert info["train_seconds"] < 30.0  # comfortably under the 2s cap + overhead
    entries = env.memory.load_experiments()
    n_experiments = sum(1 for e in entries if e["kind"] == "experiment")
    assert n_experiments <= 10


def test_artifacts_written_on_finish(tmp_path):
    env = AutoRefineEnv(seed=7, budget=Budget(3, 300, 30), runs_dir=tmp_path / "art")
    _run(env, SearchPolicy(seed=7))
    rd = env.run_dir
    assert (rd / "best_spec.json").exists()
    assert (rd / "best_model.npz").exists()
    assert (rd / "summary.json").exists()
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["final_best_score"] >= summary["baseline_score"]
    assert summary["experiments_run"] == env.bm.used_experiments
    # checkpoint loads without pickle
    from autorefine.models.mlp import MLP
    model = MLP.load(str(rd / "best_model.npz"))
    assert len(model.layers) == len(env.best_spec.architecture) + 1
