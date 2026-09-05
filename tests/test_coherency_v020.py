"""v0.20 coherency II — SPEC.md 34, A24:

- 34.1 (C4) spec-space coherency — the spec space is one object with
  three surfaces (the law in `config.py`, the `FIELD_CATALOG` catalog,
  the `FIELD_SAMPLERS` samplers); a violation of any cross-check below
  is a suite failure, not a runtime surprise:
    1. exact-value fields equal the config constants (order included);
    2. range-field catalog values lie inside the config ranges;
    3. every catalog architecture is legal for at least one non-knn
       family (the knn accept-any escape hatch, 25.2, is deliberately
       excluded);
    4. a fixed-seed battery of every sampler draw lands in the field's
       registered space (field classes exhaustive over FIELD_NAMES);
    5. the field sets are one set (FIELD_NAMES == CATALOG_FIELDS; the
       25.7 SEARCH_FIELDS exclusion; ORDERED_FIELDS; FAMILY_FIELDS).
- 34.2 (C5) summary contract — the exact top-level `summary.json` key
  set is pinned for the legacy run (17 keys) and the full-flag run
  (those 17 + task_config/curriculum/screening/ensemble); the internal
  dicts pin their exact key sets.

House rules: no cross-test imports (fixtures duplicated); tiny budgets
(scripted flat-score fake task, 4-experiment budgets); determinism (G2)
— the sampler battery is fixed-seed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget
from autorefine.config import (
    ACTIVATIONS,
    BATCH_SIZES,
    CONV_FILTERS,
    DEFAULT_SPEC,
    EARLY_STOPPING_RANGE,
    GRADIENT_CLIP_RANGE,
    HIDDEN_LAYER_SIZES,
    INPUT_NOISE_RANGE,
    INIT_SCALE_RANGE,
    KNN_K_VALUES,
    LABEL_SMOOTHING_RANGE,
    LEARNING_RATE_RANGE,
    LR_SCHEDULES,
    MODEL_FAMILIES,
    OPTIMIZERS,
    TRAIN_STEPS_RANGE,
    WEIGHT_DECAY_RANGE,
)
from autorefine.improver.actions import (
    FIELD_NAMES,
    FIELD_SAMPLERS,
    ORDERED_FIELDS,
    SEARCH_FIELDS,
)
from autorefine.improver.catalog import CATALOG_FIELDS, FAMILY_FIELDS, FIELD_CATALOG
from autorefine.tasks import TASKS

# ---------------------------------------------------------------------------
# 34.1 (C4): the field classes — exhaustive over FIELD_NAMES by contract 4;
# a new field that appears in none of them fails the sampler test.
# ---------------------------------------------------------------------------

EXACT_FIELDS = {
    "optimizer": OPTIMIZERS,
    "activation": ACTIVATIONS,
    "batch_size": BATCH_SIZES,
    "model_family": MODEL_FAMILIES,
    "lr_schedule": LR_SCHEDULES,
    "knn_k": KNN_K_VALUES,
}

RANGE_FIELDS = {
    "learning_rate": LEARNING_RATE_RANGE,
    "weight_decay": WEIGHT_DECAY_RANGE,
    "train_steps": TRAIN_STEPS_RANGE,
    "input_noise": INPUT_NOISE_RANGE,
    "label_smoothing": LABEL_SMOOTHING_RANGE,
    "early_stopping_patience": EARLY_STOPPING_RANGE,
    "init_scale": INIT_SCALE_RANGE,
    "gradient_clipping": GRADIENT_CLIP_RANGE,
}


def _legal_non_knn_arch(arch) -> bool:
    """SPEC.md 34.1.3: legal for the mlp, tree/boost, or convnet family
    (the knn family's accept-any-architecture escape hatch, 25.2, is
    deliberately not a validation path here)."""
    a = tuple(int(v) for v in arch)
    if len(a) == 1 and a[0] in (1, 2, 3):        # tree/boost depth (15, 19.2)
        return True
    if len(a) == 2 and all(c in CONV_FILTERS for c in a):  # convnet (25.3)
        return True
    return (0 <= len(a) <= 3                      # mlp; depth 0 = linear (15)
            and all(h in HIDDEN_LAYER_SIZES for h in a))


def test_field_classes_exhaustive():
    """A24 (SPEC.md 34.1.4): the sampler field classes (exact-value,
    range, architecture) cover FIELD_NAMES exactly — a new field that is
    in none of them cannot silently slip past the registered-space
    battery."""
    assert set(EXACT_FIELDS) | set(RANGE_FIELDS) | {"architecture"} \
        == set(FIELD_NAMES)


def test_catalog_exact_fields_equal_config_constants():
    """A24 (SPEC.md 34.1.1): the catalog's exact-value fields equal the
    config constants (order included) — the law and the catalog cannot
    drift on optimizer/activation/batch_size/model_family/lr_schedule/
    knn_k."""
    for field, constant in EXACT_FIELDS.items():
        assert tuple(FIELD_CATALOG[field]) == tuple(constant), field


def test_catalog_range_fields_within_config_ranges():
    """A24 (SPEC.md 34.1.2): every range-field catalog value lies inside
    the config's allowed range (the catalog offers a subset, never
    outside the law)."""
    for field, (lo, hi) in RANGE_FIELDS.items():
        for value in FIELD_CATALOG[field]:
            assert lo <= value <= hi, (field, value)


def test_catalog_architectures_legal_for_some_family():
    """A24 (SPEC.md 34.1.3): every catalog architecture value is legal
    for at least one non-knn family — mlp widths, tree/boost depths, or
    convnet filter pairs."""
    for arch in FIELD_CATALOG["architecture"]:
        assert _legal_non_knn_arch(arch), arch


def test_samplers_stay_in_registered_space():
    """A24 (SPEC.md 34.1.4): a fixed-seed battery (100 draws per field,
    task=None) of every FIELD_SAMPLERS entry lands inside the field's
    registered space — the bandit/search can never draw outside the law
    or the catalog."""
    rng = np.random.default_rng(2020)
    for field in FIELD_NAMES:
        for _ in range(100):
            value = FIELD_SAMPLERS[field](rng, None)
            if field in EXACT_FIELDS:
                assert value in FIELD_CATALOG[field], (field, value)
            elif field in RANGE_FIELDS:
                lo, hi = RANGE_FIELDS[field]
                assert lo <= value <= hi, (field, value)
            elif field == "architecture":
                assert _legal_non_knn_arch(value), value
            else:  # pragma: no cover — exhaustiveness is pinned above
                raise AssertionError(f"unclassified sampler field {field!r}")


def test_field_sets_are_one_set():
    """A24 (SPEC.md 34.1.5): bandit field names, catalog fields, the
    25.7 legacy SEARCH_FIELDS exclusion, the ordered fields, and the
    FAMILY_FIELDS rows all agree — one field set, three surfaces."""
    assert set(FIELD_NAMES) == set(CATALOG_FIELDS)
    assert set(SEARCH_FIELDS) == set(FIELD_NAMES) - {"knn_k"}  # 25.7
    assert len(SEARCH_FIELDS) == 14  # the documented v0.10 legacy space
    assert set(ORDERED_FIELDS) <= set(FIELD_NAMES)
    for family, fields in FAMILY_FIELDS.items():
        assert family in MODEL_FAMILIES, family
        assert set(fields) <= set(CATALOG_FIELDS), family


# ---------------------------------------------------------------------------
# fakes (SPEC.md 34.2): a flat-score task — every `train` call returns a
# model the task scores at one constant (55.0), so no candidate ever
# improves and the whole run is deterministic; only the summary key sets
# are asserted. The model satisfies the `save_best` contract
# (`model.save(path)`).
# ---------------------------------------------------------------------------

FAKE_TASK = "fake-coh2-v1"
FLAT_SCORE = 55.0


class _FakeModel:
    """A stand-in model: the task scores it from `.score_val`."""

    def __init__(self, score_val: float) -> None:
        self.score_val = score_val

    def save(self, path: str) -> None:
        np.savez(path, a=np.zeros(1))  # the model.save contract


class _FakeTask:
    """Minimal AutoRefineEnv task (name/head/n_outputs/make_dataset/score)."""

    name = FAKE_TASK
    head = "softmax"
    n_outputs = 2
    default_dataset_size = 16

    def __init__(self, seed: int = 7, **_kw) -> None:
        pass  # accepts task_config kwargs (22.1) without storing them

    def make_dataset(self, size: int):
        return (np.zeros((size, 2)), np.zeros(size, dtype=np.int64))

    def score(self, model, _split: str, _n: int) -> float:
        # _EnsembleModel (the 19.3 final-evaluation stand-in) has no
        # score_val — the fake scores it 0.0; only the key sets are
        # asserted in this pin
        return float(getattr(model, "score_val", 0.0))


class _FakeCurriculum:
    """Two-level fake ladder (level 0 → 1 at `best >= 50`); implements the
    curriculum contract the env reads (task() + level metadata +
    step_up_if_saturated)."""

    def __init__(self) -> None:
        self.level = 0
        self.n_bits = 4
        self.p_flip = 0.08
        self.ceiling = 100.0

    def level_description(self) -> str:
        return f"fake-level-{self.level}"

    def levels_left(self) -> int:
        return 0 if self.level >= 1 else 1

    def task(self) -> _FakeTask:
        return _FakeTask(seed=7)

    def step_up_if_saturated(self, best_score: float) -> bool:
        if self.level >= 1 or float(best_score) < 50.0:
            return False
        self.level = 1
        self.ceiling = 60.0
        return True


def _install_flat_train(monkeypatch, score: float) -> None:
    """`autorefine.improver.meta_env.train` → one constant score for every
    call (screen, full, or re-baseline)."""
    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        return SimpleNamespace(model=_FakeModel(score), train_seconds=0.001,
                               loss_history=[], time_capped=False)
    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _drive_to_done(env: AutoRefineEnv) -> None:
    policy = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


# SPEC.md 34.2: the summary key contract — the always-present 17 keys and
# the 4 conditional ones.
LEGACY_KEYS = {
    "task", "seed", "finished_reason",
    "baseline_score", "final_best_score", "improvement_factor",
    "baseline_effective", "final_best_effective", "effective_improvement_factor",
    "search_quality", "experiments_run", "wall_seconds",
    "best_spec", "mutation_win_rate",
    "pareto_frontier", "best_score_at_1s", "efficiency_at_1s",
}
FULL_EXTRA_KEYS = {"task_config", "curriculum", "screening", "ensemble"}


def test_summary_contract_legacy_run(tmp_path, monkeypatch):
    """A24 (SPEC.md 34.2): the legacy run (no optional flags) writes
    exactly the 17 always-present keys — no conditionals leak in, nothing
    is missing; the internal dicts pin their exact key sets."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(task=FAKE_TASK, seed=7,
                            budget=Budget(4, 300, 30), runs_dir=tmp_path)
        _drive_to_done(env)
        assert env.done
        summary = env.memory.load_summary()
        assert set(summary) == LEGACY_KEYS
        assert set(summary["search_quality"]) == {
            "ci_blocks", "z_accept", "efficiency_weight",
            "gen_gap_penalty", "block_size"}
        # best_spec carries the full spec dict (15 fields, SPEC.md 5.1)
        assert set(summary["best_spec"]) == set(DEFAULT_SPEC.to_dict())
        # every pareto point carries exactly the 15 contract's fields
        for point in summary["pareto_frontier"]:
            assert set(point) == {"score", "train_seconds", "spec_hash"}
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_summary_contract_full_flag_run(tmp_path, monkeypatch):
    """A24 (SPEC.md 34.2): the full-flag run (task_config + curriculum +
    screening + ensemble) writes exactly those 17 keys plus the 4
    conditional ones, each with its exact internal key set."""
    _install_flat_train(monkeypatch, FLAT_SCORE)
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7, budget=Budget(4, 300, 30),
            runs_dir=tmp_path,
            task_config={"split_frac": 0.2},
            curriculum=_FakeCurriculum(),
            screen_frac=0.25,
            ensemble_top_k=2,
        )
        _drive_to_done(env)
        assert env.done
        summary = env.memory.load_summary()
        assert set(summary) == LEGACY_KEYS | FULL_EXTRA_KEYS
        assert set(summary["task_config"]) == {"split_frac"}
        assert set(summary["curriculum"]) == {
            "levels", "final_difficulty", "final_ceiling", "levels_left"}
        # flat score ⇒ every candidate is screen-rejected (55 !> 55), and
        # a screen rejection returns before the step-up check — so zero
        # levels; the `curriculum` key is still present (the key contract
        # is per configuration; the level-row contract is A23's pin)
        assert summary["curriculum"]["levels"] == []
        assert set(summary["screening"]) == {"frac", "baseline_screen_score"}
        assert set(summary["ensemble"]) == {
            "top_k", "member_scores", "ensemble_score"}
    finally:
        TASKS.pop(FAKE_TASK, None)
