"""BanditPolicy tests (SPEC.md 17: the UCB field-bandit improver;
SPEC.md 18.1/18.2: local mutation mode + family-conditioned proposals);
acceptance A7 (SPEC.md 12: the v0.3 bandit extension coverage)."""
import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget, DEFAULT_SPEC, ModelSpec
from autorefine.improver.actions import FIELD_NAMES
from autorefine.improver.catalog import relevant_fields


def test_propose_is_deterministic_given_seed():
    a, b = BanditPolicy(seed=3), BanditPolicy(seed=3)
    state_a, state_b = (
        {"best_spec": DEFAULT_SPEC.to_dict(), "last": None} for _ in range(2)
    )
    for _ in range(10):
        sa = a.propose(state_a)
        sb = b.propose(state_b)
        assert ModelSpec.from_dict(sa).to_dict() == ModelSpec.from_dict(sb).to_dict()
        state_a = {"best_spec": sa,
                   "last": {"accepted": True, "fields": ["learning_rate"]}}
        state_b = {"best_spec": sb,
                   "last": {"accepted": True, "fields": ["learning_rate"]}}


def test_proposals_are_always_valid_specs():
    pol = BanditPolicy(seed=0)
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "last": None}
    for _ in range(20):
        spec = pol.propose(state)
        ModelSpec.from_dict(spec)  # must never raise
        assert spec != state["best_spec"]  # mutate_spec_dict guarantees a change
        state = {"best_spec": spec,
                 "last": {"accepted": False, "fields": ["batch_size"]}}


def _changed_fields(prev: dict, nxt: dict) -> list[str]:
    """Fields whose value actually changed (architecture tuple/list-normalized),
    mirroring what the env reports in state["last"]["fields"]."""
    norm = lambda d: {
        k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()
    }
    p, n = norm(prev), norm(nxt)
    return [f for f in FIELD_NAMES if p[f] != n[f]]


def test_cold_start_tries_every_relevant_field():
    """SPEC.md 18.2/19.4: UCB is inf for untried *relevant* fields, so while
    the current best family has an untried relevant field, every proposal
    credits an untried one (no premature exploitation). A family switch
    (model_family is relevant to all families) temporarily deactivates the
    other families' fields; they are re-explored as soon as the bandit
    returns to that family, so every field is still tried. Trials/wins
    history is kept across family switches (no state reset).

    v0.5 (SPEC.md 19.4): the space grew to 14 fields, 4 of which
    (lr_schedule, early_stopping_patience, init_scale, gradient_clipping)
    are mlp-only, so the bandit must return to mlp to credit them. Shared
    fields (architecture, train_steps, input_noise, model_family) are
    legitimately re-tried in each family context, so the v0.3 "bounded
    exploitation" cap (max<=2) no longer holds; instead the per-step
    invariant above (never exploit while an untried relevant field exists)
    and the final all-fields-tried check capture the cold-start property."""
    pol = BanditPolicy(seed=0)
    best = DEFAULT_SPEC.to_dict()
    state = {"best_spec": best, "last": None}
    # window comfortably above the 25 proposals seed 0 needs to credit all 15
    # fields (v0.11, SPEC.md 25.2: knn_k joins the space) across the family
    # switches (probe-verified; deterministic, G2)
    for _ in range(28):
        rel = relevant_fields(best.get("model_family", "mlp"))
        # the previous step's fields are credited inside propose(), so they
        # are no longer "untried" at selection time
        credited = set((state.get("last") or {}).get("fields") or [])
        untried = [f for f in rel if pol.trials[f] == 0 and f not in credited]
        spec = pol.propose(state)
        ModelSpec.from_dict(spec)  # always valid
        fields = _changed_fields(best, spec)
        assert fields  # a mutation always changes at least one field
        if untried:
            # the selected field (always in the diff: guaranteed-different)
            # must have been an untried relevant field (UCB inf > finite)
            assert set(fields) & set(untried)
        best = spec
        state = {"best_spec": best, "last": {"accepted": False, "fields": fields}}
    assert min(pol.trials.values()) >= 1  # every field was tried


def test_cold_start_fixed_family_tries_every_field():
    """SPEC.md 18.2, stable-family case: if the best spec's family is held
    at mlp, cold start still tries every field once before exploitation
    (the v0.3 invariant, now scoped to relevant_fields('mlp') = all 10)."""
    pol = BanditPolicy(seed=0)
    best = DEFAULT_SPEC.to_dict()
    state = {"best_spec": best, "last": None}
    for _ in range(len(FIELD_NAMES)):
        rel = relevant_fields(best.get("model_family", "mlp"))
        assert set(rel) == set(FIELD_NAMES)  # family pinned to mlp
        if min(pol.trials[f] for f in rel) == 0:
            assert max(pol.trials[f] for f in rel) <= 1
        spec = pol.propose(state)
        ModelSpec.from_dict(spec)  # always valid
        fields = _changed_fields(best, spec)
        assert fields
        # pin the family back to mlp so the next proposal sees mlp relevance
        if spec.get("model_family") != "mlp":
            spec = {**spec, "model_family": "mlp", "architecture": [16, 8]}
        best = spec
        state = {"best_spec": best, "last": {"accepted": False, "fields": fields}}
    # one final proposal credits the last field: all 10 now have a trial
    pol.propose(state)
    assert min(pol.trials.values()) >= 1  # every field was tried
    assert max(pol.trials.values()) <= 2  # nothing exploited more than once


def test_feedback_credits_fields():
    """Accepted steps increment the credited field's trials/wins (G2)."""
    pol = BanditPolicy(seed=0)
    state = {"best_spec": DEFAULT_SPEC.to_dict(),
             "last": {"accepted": True, "fields": ["learning_rate"]}}
    pol.propose(state)
    assert pol.trials["learning_rate"] == 1 and pol.wins["learning_rate"] == 1.0
    assert pol.trials["batch_size"] == 0
    # a rejected step still counts as a trial but not a win
    pol2 = BanditPolicy(seed=0)
    pol2.propose({"best_spec": DEFAULT_SPEC.to_dict(),
                  "last": {"accepted": False, "fields": ["batch_size"]}})
    assert pol2.trials["batch_size"] == 1 and pol2.wins["batch_size"] == 0.0


def test_full_parity_run_completes_never_below_baseline(tmp_path):
    """SPEC.md 17 end-to-end: bandit drives parity-v1 to done, final >= baseline."""
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(5, 300, 30), runs_dir=tmp_path / "bandit")
    pol = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        spec_dict = pol.propose(state)
        ModelSpec.from_dict(spec_dict)
        state, reward, done, info = env.step(spec_dict)
    assert env.done
    assert env.best_score >= env.baseline_score
    summary = env.memory.load_summary()
    assert summary["finished_reason"] == "budget_exhausted"
