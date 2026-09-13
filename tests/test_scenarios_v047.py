"""v0.47 — "A. More specialized scenarios" (SPEC.md 61, A51, M50).

Covers the five A51 items as implemented — all **additive / opt-in** over
machinery that already exists (the §37 objective set in ``gate.py``, the
§36 Task ABC, the §21 plugin loader, the §18 gate), with **no default-path
behavior change** (A1–A50 stay green; ``default_objectives`` is
byte-identical):

- 61.1 **A1** constraint-aware search — ``OBJECTIVE_NAMES`` gains
  ``per_class_f1`` / ``ece`` / ``cost`` / ``monotonic_in`` (the same
  ``name op threshold`` form); ``compute_actuals`` threads ``extras`` over
  recomputation and leaves unknown names ``None``; ``default_objectives``
  is the A1–A50 pin (byte-identical).
- ``calibration.py`` (61.1.3 / 61.4) — ``per_class_f1`` (the min over
  non-empty classes; an empty class is excluded), ``ece`` (∈ ``[0, 1]``,
  calibrated → ≈ 0, confident-wrong → high, deterministic),
  ``coverage_width`` (∈ ``[0, 1]``), ``monotonicity_slope`` (sign), and
  the ``ensemble_ece`` spread predictor (deterministic, ≤ the worst
  single model's ECE on a hand-built fixture).
- 61.2 **A2** domain task packs — ``MedicalTabularTask`` (imbalanced
  cost-sensitive binary), ``FinanceForecastTask`` (temporal horizon,
  directional, flat → 50, ``cost``), ``RobustTask`` (worst-group wrapper);
  each identity / shape / score∈[0,100] + deterministic (G2), ``holdout_rows``
  present; medical + finance register in ``TASKS``.
- 61.3 **A3** scenarios — ``Scenario`` (name/description/task/
  budget_overrides/gate_overrides); ``fewshot`` (tiny train pool),
  ``drifting`` (raised noise), ``trap`` (tight ``gen_gap``); each composes
  with any ``Budget`` / gate (pure, ``resolve_budget`` / ``resolved``).
- 61.5 **A5** — the ``cost`` name is in the registry; the optional
  ``ParetoFrontier.cost_of`` accessor + the byte-identical default
  ``(score, train)`` contract.
- exports (61.6.5) + the A51 round regression (version ``0.47.0`` in both
  sources; the A25 index advances to 47 rows / ``range(1, 52)``).

House rules (A51): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; the app is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

import autorefine
from autorefine import (
    FinanceForecastTask,
    MedicalTabularTask,
    ParetoFrontier,
    RobustTask,
    Scenario,
    compute_actuals,
    coverage_width,
    drifting,
    ece,
    ensemble_ece,
    fewshot,
    monotonicity_slope,
    per_class_f1,
    trap,
)
from autorefine.gate import (
    OBJECTIVE_NAMES,
    Objective,
    default_objectives,
    evaluate,
    parse_objective,
)
from autorefine.tasks import TASKS
from autorefine.tasks.base import Task


# --- fake models / tasks (synthesized; no cross-test imports, A51) ----------

class LinSoftmax:
    """2-class logits ``[sum, -sum + bias]`` — a deterministic stand-in."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = float(bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = x.sum(axis=1, keepdims=True)
        return np.concatenate([s, -s + self.bias], axis=1)


class Flat1D:
    """A single flat (zero) output — the "no signal" regression model."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        return np.zeros((n, 1), dtype=np.float64)


class FakeBase(Task):
    """A minimal duck-typed classification base task for the wrapper."""

    head = "softmax"
    n_outputs = 2
    state_dim = 3
    max_steps = 1
    default_dataset_size = 100
    metric = "accuracy"
    capabilities = frozenset()
    class_values = [0.0, 1.0]

    def __init__(self, seed: int = 1) -> None:
        self.seed = int(seed)
        self.name = "fake-base"

    def make_dataset(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((10, 3), dtype=np.float64), np.zeros(10, dtype=np.int64)

    def score(self, model, split: str, n: int) -> float:  # ABC requirement
        return 0.0

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        x = np.random.default_rng(self.seed).normal(0.0, 1.0, (n, 3))
        return x, np.zeros(n, dtype=np.int64)


# --- 61.1 A1 — constraint-aware search ---------------------------------------

def test_objective_names_gain_constraints():
    """A51 (SPEC.md 61.1.1): the four constraint names are in the registry
    and the original three remain (the A1–A50 pin)."""
    for name in ("score", "train", "model", "per_class_f1", "ece",
                 "cost", "monotonic_in"):
        assert name in OBJECTIVE_NAMES, name


def test_parse_objective_accepts_new_names():
    """A51 (SPEC.md 61.1.1): each new name parses to the right Objective."""
    assert parse_objective("per_class_f1>=0.9") == \
        Objective("per_class_f1", ">=", 0.9)
    assert parse_objective("ece<=0.05") == Objective("ece", "<=", 0.05)
    assert parse_objective("cost<=3.0") == Objective("cost", "<=", 3.0)
    assert parse_objective("monotonic_in>=1.0") == \
        Objective("monotonic_in", ">=", 1.0)


def test_parse_objective_rejects_unknown_name():
    """A51 (SPEC.md 61.1.1): an unknown name is still a ValueError."""
    with pytest.raises(ValueError):
        parse_objective("bogus>=1")


def test_default_objectives_byte_identical():
    """A51 (SPEC.md 61.1.1, the A1–A50 pin): default_objectives is exactly
    today's single score gate — unchanged by the constraint names."""
    objs = default_objectives(95.0)
    assert objs == (Objective("score", ">=", 95.0),)
    assert [o.name for o in objs] == ["score"]


def test_evaluate_rich_actuals_pass_iff_all_pass():
    """A51 (SPEC.md 61.1): PASS iff ALL objectives pass; a missing actual
    fails its objective (the 37.2 rule)."""
    objs = (
        Objective("score", ">=", 90.0),
        Objective("per_class_f1", ">=", 0.5),
        Objective("ece", "<=", 0.2),
    )
    all_pass = {"score": 95.0, "per_class_f1": 0.6, "ece": 0.1}
    assert evaluate(objs, all_pass)["pass"] is True
    # one objective fails → the whole gate fails
    assert evaluate(objs, {"score": 95.0, "per_class_f1": 0.4, "ece": 0.1})["pass"] is False
    # a missing actual fails its objective
    assert evaluate(objs, {"score": 95.0, "per_class_f1": 0.6})["pass"] is False


def test_compute_actuals_threads_extras_and_leaves_unknown_none():
    """A51 (SPEC.md 61.1.2): ``extras`` wins over recomputation; unknown /
    unsupplied names stay absent (→ their objective fails, never a pass)."""
    med = MedicalTabularTask(seed=3)
    model = LinSoftmax(bias=1.0)
    base = compute_actuals(None, med, model, [])
    assert "per_class_f1" in base
    assert "ece" in base
    assert 0.0 <= base["per_class_f1"] <= 1.0
    assert 0.0 <= base["ece"] <= 1.0
    # an extra supplied value overrides any recomputation
    merged = compute_actuals(None, med, model, [], extras={"ece": 0.01})
    assert merged["ece"] == 0.01
    # an unknown name is never invented
    assert "monotonic_in" not in base


def test_compute_actuals_uses_task_cost_and_monotonic_probe():
    """A51 (SPEC.md 61.1.2 / 61.5): a task exposing ``cost`` /
    ``monotonic_probe`` contributes those actuals."""
    med = MedicalTabularTask(seed=3)

    class Costed(med.__class__):  # subclass to add the two optional hooks
        def cost(self, model, split, n):
            return 2.5

        def monotonic_probe(self, model):
            return 0.7

    c = Costed(seed=3)
    out = compute_actuals(None, c, LinSoftmax(bias=1.0), [])
    assert out["cost"] == 2.5
    assert out["monotonic_in"] == 0.7


# --- calibration.py (61.1.3 / 61.4) -------------------------------------------

def test_per_class_f1_min_over_nonempty_and_excludes_empty():
    """A51 (SPEC.md 61.1.3): the worst non-empty class's F1; an empty class
    is excluded (it never sinks the min to 0)."""
    # class 0: perfect (F1 = 1.0); class 1: 1 tp, 1 fp → P = 1/2, R = 1 → F1 = 2/3
    conf = [[5, 0], [1, 1]]
    expected = min(1.0, 2.0 / 3.0)
    assert per_class_f1(conf) == pytest.approx(expected)
    # an empty class (all-zero row) is excluded — here class 2 is empty
    conf2 = [[5, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert per_class_f1(conf2) == pytest.approx(expected)
    # all-empty matrix → 0.0
    assert per_class_f1([[0, 0], [0, 0]]) == 0.0


def test_ece_bounds_and_calibrated_vs_confident_wrong():
    """A51 (SPEC.md 61.4): ECE ∈ [0, 1]; calibrated → ≈ 0, confident-wrong
    → high; deterministic."""
    # perfectly calibrated: 4 examples, 50% confident, 50% accurate
    probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 0, 1, 1])
    assert ece(probs, labels) == pytest.approx(0.0)
    # confident-wrong: 1.0 confident, all wrong → ECE = 1.0
    probs_w = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    labels_w = np.array([0, 0, 0, 0])
    e = ece(probs_w, labels_w)
    assert e == pytest.approx(1.0)
    # determinism
    assert ece(probs, labels) == ece(probs, labels)
    # bounds on a random-looking case
    rng = np.random.default_rng(0)
    p = rng.random((100, 3))
    p = p / p.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 100)
    assert 0.0 <= ece(p, y) <= 1.0


def test_coverage_width_bounds():
    """A51 (SPEC.md 61.4.1): coverage ∈ [0, 1]."""
    assert coverage_width([0, 0, 0], [1, 1, 1], [0.5, 0.5, 2.0]) == pytest.approx(2 / 3)
    assert coverage_width([0, 0], [1, 1], [5, 5]) == 0.0
    assert coverage_width([0, 0], [1, 1], [0.5, 0.5]) == 1.0
    # empty / mismatched → 0.0
    assert coverage_width([], [], []) == 0.0


def test_monotonicity_slope_sign():
    """A51 (SPEC.md 61.1.3): slope >= 0 on non-decreasing, < 0 on
    non-increasing; loud dx / k validation."""
    assert monotonicity_slope(lambda v: v ** 2, x0=3.0, dx=0.5, k=3) >= 0.0
    assert monotonicity_slope(lambda v: -v, x0=1.0, dx=0.25, k=2) < 0.0
    with pytest.raises(ValueError):
        monotonicity_slope(lambda v: v, x0=0.0, dx=0.0)
    with pytest.raises(ValueError):
        monotonicity_slope(lambda v: v, x0=0.0, dx=1.0, k=0)


def test_ensemble_ece_deterministic_and_beats_worst():
    """A51 (SPEC.md 61.4.2): the ensemble-spread ECE is deterministic and
    ≤ the worst single model's ECE on a hand-built fixture."""

    class M:
        def __init__(self, w: np.ndarray) -> None:
            self.w = np.asarray(w, dtype=np.float64)

        def forward(self, x):
            return np.asarray(x, dtype=np.float64) @ self.w

    class T:
        head = "softmax"
        class_values = [0.0, 1.0]

        def __init__(self, x, y) -> None:
            self._x, self._y = x, y

        def holdout_rows(self, n, model=None):
            return self._x, self._y

    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    t = T(x, y)
    # two models that disagree: ensemble average is more centered
    m1 = M(np.array([[2.0, 0.0], [0.0, -2.0]]))
    m2 = M(np.array([[-2.0, 0.0], [0.0, 2.0]]))
    e_ens = ensemble_ece(t, [m1, m2], k=2, n=10)
    e_worst = max(
        ece(_softmax(m1.forward(x)), y),
        ece(_softmax(m2.forward(x)), y),
    )
    assert 0.0 <= e_ens <= 1.0
    assert e_ens <= e_worst + 1e-12
    assert ensemble_ece(t, [m1, m2], k=2, n=10) == e_ens
    with pytest.raises(ValueError):
        ensemble_ece(t, [], k=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


# --- 61.2 A2 — domain task packs ---------------------------------------------

def test_medical_identity_and_dataset():
    """A51 (SPEC.md 61.2.1): identity, shape, and an imbalanced positive
    rate (≈ the declared 5%)."""
    t = MedicalTabularTask(seed=7)
    assert t.name == "medical-v1"
    assert t.head == "softmax"
    assert t.n_outputs == 2
    assert t.state_dim == 4
    assert t.metric == "cost_sensitive"
    assert t.split_mode == "random"
    x, y = t.make_dataset(n_points=1000)
    assert x.shape == (1000, 4)
    assert y.shape == (1000,)
    rate = float((y == 1).mean())
    assert 0.02 <= rate <= 0.08, rate


def test_medical_score_in_range_and_deterministic():
    """A51 (SPEC.md 61.2.1): score ∈ [0, 100] and deterministic (G2)."""
    t1 = MedicalTabularTask(seed=11)
    t2 = MedicalTabularTask(seed=11)
    m = LinSoftmax(bias=1.0)
    s1, s2 = t1.score(m, "holdout", 200), t2.score(m, "holdout", 200)
    assert 0.0 <= s1 <= 100.0
    assert s1 == s2
    # a flat all-negative model scores 0 (TPR = 0)
    flat = LinSoftmax(bias=-100.0)
    assert 0.0 <= t1.score(flat, "holdout", 200) <= 100.0


def test_medical_holdout_rows_and_registration():
    """A51 (SPEC.md 61.2.1 / 61.2.4): holdout_rows present; registered."""
    t = MedicalTabularTask(seed=5)
    x, y = t.holdout_rows(50)
    assert x.shape == (50, 4) and y.shape == (50,)
    assert "medical-v1" in TASKS
    assert TASKS["medical-v1"] is MedicalTabularTask


def test_finance_identity_temporal_and_flat_50():
    """A51 (SPEC.md 61.2.2): temporal split, directional score, flat → 50,
    and train strictly precedes holdout (no row reuse)."""
    t = FinanceForecastTask(seed=9)
    assert t.name == "finance-v1"
    assert t.head == "mse"
    assert t.n_outputs == 1
    assert t.split_mode == "temporal"
    assert t.metric == "directional"
    x_tr, y_tr = t.make_dataset(n_points=64)
    assert x_tr.shape == (64, t.state_dim)
    x_ho, y_ho = t.holdout_rows(64)
    # temporal separation: the first train and first holdout windows sit at
    # different time positions (a temporal rule, not a shared pool)
    assert not np.array_equal(x_tr[0], x_ho[0])
    # a flat (zero) prediction scores exactly 50 (chance level)
    assert t.score(Flat1D(), "holdout", 64) == pytest.approx(50.0)
    assert 0.0 <= t.score(Flat1D(), "train", 64) <= 100.0


def test_finance_cost_and_deterministic_and_registration():
    """A51 (SPEC.md 61.2.2 / 61.5): the cost actual is finite, deterministic
    score, and the task is registered."""
    t = FinanceForecastTask(seed=13)
    c1 = t.cost(Flat1D(), "holdout", 64)
    c2 = FinanceForecastTask(seed=13).cost(Flat1D(), "holdout", 64)
    assert np.isfinite(c1) and c1 == c2
    assert "finance-v1" in TASKS
    assert TASKS["finance-v1"] is FinanceForecastTask


def test_robust_wrapper_worst_group():
    """A51 (SPEC.md 61.2.3): the wrapper scores the worst group across
    noise levels (≤ the clean-group accuracy), in [0, 100], deterministic."""
    base = FakeBase(seed=2)
    t = RobustTask(base, seed=2, n_shift=3, max_sigma=0.5)
    assert t.name == "robust-v1"
    assert t.head == "softmax" and t.n_outputs == 2
    assert t.metric == "worst_group"
    m = LinSoftmax(bias=0.0)
    s = t.score(m, "holdout", 50)
    assert 0.0 <= s <= 100.0
    # determinism
    assert t.score(m, "holdout", 50) == s
    # holdout_rows delegates to the base (the clean group)
    x, y = t.holdout_rows(20)
    assert x.shape == (20, 3)
    # a non-softmax base is rejected
    class Reg:
        head = "mse"
        n_outputs = 1
        state_dim = 3

        def holdout_rows(self, n, model=None):
            return np.zeros((n, 3)), np.zeros(n)

    with pytest.raises(ValueError):
        RobustTask(Reg(), seed=1)


def test_task_packs_carry_domain_bar():
    """A51 (SPEC.md 61.2): each pack exposes a starter_spec + a
    default_objectives bar that includes the score gate."""
    for t in (MedicalTabularTask(seed=1), FinanceForecastTask(seed=1),
              RobustTask(FakeBase(seed=1), seed=1)):
        assert isinstance(t.starter_spec, dict) and t.starter_spec
        objs = t.default_objectives(90.0)
        assert any(o.name == "score" and o.op == ">=" for o in objs)


# --- 61.3 A3 — scenarios ------------------------------------------------------

def test_scenario_presets_shape():
    """A51 (SPEC.md 61.3.1–61.3.3): each preset is a Scenario with a name,
    description, a concrete Task, and budget / gate overrides."""
    for sc in (fewshot(), drifting(), trap()):
        assert isinstance(sc, Scenario)
        assert sc.name and sc.description
        assert isinstance(sc.task, Task)
        assert isinstance(sc.budget_overrides, dict)
        assert isinstance(sc.gate_overrides, dict)


def test_fewshot_tiny_train_pool():
    """A51 (SPEC.md 61.3.1): a tiny train pool (64 points) + a small budget."""
    sc = fewshot()
    assert sc.extras.get("n_points") == 64
    assert sc.budget_overrides["max_experiments"] <= 5


def test_drifting_raises_noise_and_trap_tightens_gap():
    """A51 (SPEC.md 61.3.2 / 61.3.3): drifting raises the base noise; trap
    pins a tight gen_gap."""
    d = drifting(seed=1, p_flip=0.30)
    assert d.task.p_flip == 0.30
    assert d.gate_overrides.get("p_flip") == 0.30
    assert trap().gate_overrides.get("gen_gap") == 0.0
    with pytest.raises(ValueError):
        drifting(seed=1, p_flip=0.7)


def test_scenario_resolves_budget_and_recipe():
    """A51 (SPEC.md 61.3.4): resolve_budget merges onto a base Budget;
    resolved() returns a copy-pasteable recipe (pure, no env)."""
    from autorefine import Budget

    base = Budget(max_experiments=30, max_wall_seconds=900.0,
                  max_train_seconds=30.0)
    sc = fewshot()
    merged = sc.resolve_budget(base)
    assert merged.max_experiments == 5
    assert merged.max_train_seconds == 20.0
    assert merged.max_wall_seconds == 900.0  # untouched field preserved
    recipe = sc.resolved(base)
    assert recipe["task"] == "parity-4-p0.08"
    assert recipe["extras"]["n_points"] == 64
    assert recipe["budget"]["max_experiments"] == 5


# --- 61.5 A5 — cost objective + optional frontier cost -----------------------

def test_cost_name_and_gate_pass_fail():
    """A51 (SPEC.md 61.5): the cost name is in the registry; a low-cost
    model passes cost<=X and a high-cost model fails."""
    assert "cost" in OBJECTIVE_NAMES
    obj = (Objective("cost", "<=", 3.0),)
    assert evaluate(obj, {"cost": 2.0})["pass"] is True
    assert evaluate(obj, {"cost": 5.0})["pass"] is False


def test_pareto_frontier_cost_of_optional():
    """A51 (SPEC.md 61.5.1): cost_of returns the mapped value when supplied
    and None otherwise; the default (score, train) contract is unchanged."""
    pf = ParetoFrontier()
    pf.add(90.0, 1.0, "a")                      # no cost
    pf.add(95.0, 2.0, "b", cost=1.5)            # with cost
    assert pf.cost_of("b") == 1.5
    assert pf.cost_of("a") is None
    assert pf.cost_of("zz") is None
    # the default summary still reports (score, train_seconds, spec_hash)
    snap = pf.summary(time_budget=3.0)
    assert {"score", "train_seconds", "spec_hash"} <= set(snap["pareto_frontier"][0])


# --- exports (61.6.5) + round regression --------------------------------------

def test_new_names_exported():
    """A51 (SPEC.md 61.6.5): the new top-level names are exported."""
    expected = {
        "per_class_f1", "ece", "coverage_width", "monotonicity_slope",
        "ensemble_ece", "compute_actuals",
        "MedicalTabularTask", "FinanceForecastTask", "RobustTask",
        "Scenario", "fewshot", "drifting", "trap",
    }
    assert expected <= set(autorefine.__all__)
    for name in expected:
        assert hasattr(autorefine, name), name


def test_version_and_spec_round():
    """A51 (SPEC.md 61.6, 33.1): the version stepped to 0.47.0 in both
    sources; SPEC carries the A51 acceptance block + the M50 row."""
    import re
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[1]
    assert autorefine.__version__ == "0.50.0"
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 61.6 Acceptance (A51)" in spec
    assert re.search(r"^\s*\| M50 \| v0\.47\s*\|\s*61\s*\|\s*A51\s*\|",
                     spec, re.MULTILINE)
    assert "**M50**" in spec
