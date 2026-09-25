"""Modeling capability v0.61 (SPEC.md 75, A65).

SPEC.md 75: four additive, **value-level** model-space extensions (no new
`ModelSpec` fields — the 15-field registry, the A24/A26 pins, and the
bandit draw-space stay byte-identical):
  * 75.2.1 — an `adamw` optimizer with decoupled weight decay (`wd == 0`
    is bit-identical to `adam`);
  * 75.2.2 — three LR schedules `step` / `exponential` / `cyclic`;
  * 75.2.3 — a Gaussian-process family (fixed-seed random-Fourier-features
    one-vs-rest);
  * 75.2.4 — a generalized additive model family (smoothing splines).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here). The value-level
approach keeps the A1–A4 bandit stream, the §18.7 bit-exact sequence, and
the A26 catalog pin green — re-derived below (83 actions, 15 fields,
`model_family` = all seven families).
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np

import autorefine
from autorefine.config import (
    LR_SCHEDULES,
    MODEL_FAMILIES,
    OPTIMIZERS,
    ModelSpec,
)
from autorefine.improver.catalog import ACTIONS
from autorefine.improver.specspace import SPEC_FIELDS
from autorefine.models.gam import GAM
from autorefine.models.gp import GP
from autorefine.models.optimizers import Adam, AdamW
from autorefine.tasks import SineRegressionV1
from autorefine.trainer import _scheduled_lr, train, train_from_task

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


# --- 75.2.1 adamw: decoupled decay + wd=0 bit-identity (A65) ------------------

def test_adamw_decoupled_formula_single_step():
    """A65 (75.2.1): one `AdamW.step` with `wd > 0` equals the decoupled
    formula — the adaptive step first, then `param -= lr*wd*param` applied to
    the *post-step* parameter — and stores the same moments as `Adam`."""
    p0 = np.array([1.0, -2.0, 3.0])
    g = np.array([0.5, 0.1, -0.3])
    lr, wd = 0.01, 0.05
    p = p0.copy()
    state: dict = {}
    AdamW(lr, wd).step(p, g, state, 1)

    b1, b2, eps = AdamW.b1, AdamW.b2, AdamW.eps
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    mhat = m / (1 - b1 ** 1)
    vhat = v / (1 - b2 ** 1)
    p1 = p0 - lr * mhat / (np.sqrt(vhat) + eps)
    p2 = p1 - lr * wd * p1  # decoupled decay on the post-step parameter
    assert np.array_equal(p, p2)
    assert np.array_equal(state["m"], m)
    assert np.array_equal(state["v"], v)


def test_adamw_wd0_bitidentical_to_adam():
    """A65 (75.2.1): `AdamW(lr, wd=0).step` is bit-identical to
    `Adam(lr).step` — parameters **and** moment state — across many steps and
    differing gradients (the value identity that keeps `wd == 0` runs
    equivalent to a pre-v0.61 `adam` run)."""
    rng = np.random.default_rng(0)
    a = Adam(0.01)
    w = AdamW(0.01, 0.0)
    pa = rng.normal(size=8)
    pw = pa.copy()
    sa: dict = {}
    sw: dict = {}
    for t in range(1, 12):
        g = rng.normal(size=8)
        a.step(pa, g, sa, t)
        w.step(pw, g, sw, t)
    assert np.array_equal(pa, pw)
    assert np.array_equal(sa["m"], sw["m"])
    assert np.array_equal(sa["v"], sw["v"])


def test_make_optimizer_wd_consumption():
    """A65 (75.2.1): `make_optimizer("adamw", lr, wd)` consumes the decay and
    `make_optimizer("adam", lr)` (2-arg) ignores it (unchanged)."""
    from autorefine.models.optimizers import make_optimizer

    assert make_optimizer("adamw", 0.01, 0.05).wd == 0.05
    assert make_optimizer("adam", 0.01).name == "adam"
    assert "adamw" in OPTIMIZERS


# --- 75.2.2 the new schedules: value identity + finiteness (A65) --------------

def test_step_schedule_plateaus_and_monotone():
    """A65 (75.2.2): `step` is three flat plateaus (base -> 0.1x -> 0.01x),
    monotonically non-increasing across the run, finite at every `t`."""
    base, T = 0.003, 300
    vals = [_scheduled_lr("step", base, t, T) for t in range(1, T + 1)]
    assert all(math.isfinite(v) for v in vals)
    assert vals[0] == base                    # first third is the base rate
    assert vals[-1] == base * 0.01            # last third is 0.01x
    assert all(vals[i] >= vals[i + 1] - 1e-15 for i in range(len(vals) - 1))
    # three (and only three) flat plateaus across the whole run
    assert len(set(vals)) == 3


def test_exponential_schedule_decays_to_floor():
    """A65 (75.2.2): `exponential` decays geometrically `base -> base*0.1`
    over the run (finite, monotonically non-increasing, hitting the floor)."""
    base, T = 0.003, 300
    vals = [_scheduled_lr("exponential", base, t, T) for t in range(1, T + 1)]
    assert all(math.isfinite(v) for v in vals)
    assert vals[-1] == base * 0.1             # the geometric floor at t == T
    assert all(vals[i] >= vals[i + 1] - 1e-15 for i in range(len(vals) - 1))
    # value identity: base * 0.1 ** (t/T) at every step
    for t in (1, T // 4, T // 2, 3 * T // 4, T):
        assert _scheduled_lr("exponential", base, t, T) \
            == base * (0.1 ** (t / T))


def test_cyclic_schedule_oscillates():
    """A65 (75.2.2): `cyclic` is a triangle wave — `base` at the midpoint,
    `base*0.1` at both ends — finite for every `t in [1, T]`."""
    base, T = 0.003, 300
    vals = [_scheduled_lr("cyclic", base, t, T) for t in range(1, T + 1)]
    assert all(math.isfinite(v) for v in vals)
    assert max(vals) == base                  # peak at the midpoint
    assert min(vals) == base * 0.1            # floor at both ends
    assert _scheduled_lr("cyclic", base, T, T) == base * 0.1   # frac == 1
    assert _scheduled_lr("cyclic", base, T // 2, T) == base    # frac == 0.5
    # it genuinely oscillates (not monotone): rises then falls
    assert vals[T // 4] < vals[T // 2] and vals[T // 2] > vals[3 * T // 4]


def test_new_schedules_registered_and_valid():
    """A65 (75.2.2): the three new schedules are in the config law and are a
    valid `ModelSpec.lr_schedule` for the mlp family (no new field)."""
    for s in ("step", "exponential", "cyclic"):
        assert s in LR_SCHEDULES
        spec = ModelSpec(
            architecture=(16, 8), optimizer="adam", learning_rate=0.003,
            batch_size=32, weight_decay=0.0, train_steps=200, input_noise=0.0,
            activation="tanh", label_smoothing=0.0, lr_schedule=s,
            early_stopping_patience=0, init_scale=1.0, gradient_clipping=0.0,
            knn_k=5)
        spec.validate()  # must not raise


# --- 75.2.3 GP + 75.2.4 GAM: determinism, round-trip, quality (A65) -----------

def _sine_fixture():
    task = SineRegressionV1(seed=7)
    X, y = task.make_dataset(300)
    return task, X, y, task.n_outputs, task.head


def _constant_model(c: float):
    class _Const:
        def forward(self, x):
            n = np.asarray(x).shape[0]
            return np.full((n, 1), float(c))
    return _Const()


def test_gp_deterministic_and_beats_constant_baseline():
    """A65 (75.2.3): the GP family is bit-reproducible on a fixed dataset
    (fixed internal seed) and beats the constant-mean baseline on sine-v1."""
    task, X, y, n_out, head = _sine_fixture()
    assert head == "mse" and n_out == 1
    g1 = GP(n_out=n_out, head=head).fit(X, y)
    g2 = GP(n_out=n_out, head=head).fit(X, y)
    assert np.array_equal(g1.forward(X[:50]), g2.forward(X[:50]))

    const = _constant_model(float(np.mean(y)))
    base_score = task.score(const, "holdout", 200)
    gp_score = task.score(g1, "holdout", 200)
    assert base_score <= 0.0  # a mean predictor has R^2 <= 0 on fresh data
    assert gp_score > base_score  # the GP actually learns the sine


def test_gam_deterministic_and_beats_constant_baseline():
    """A65 (75.2.4): the GAM family is bit-reproducible, idempotent under
    re-fit (knot list does not accumulate), and beats the constant-mean
    baseline on sine-v1."""
    task, X, y, n_out, head = _sine_fixture()
    m1 = GAM(n_out=n_out, head=head, weight_decay=0.0).fit(X, y)
    m2 = GAM(n_out=n_out, head=head, weight_decay=0.0).fit(X, y)
    assert np.array_equal(m1.forward(X[:50]), m2.forward(X[:50]))
    # re-fitting the same object must not accumulate knots (idempotent fit)
    m1.fit(X, y)
    assert np.array_equal(m1.forward(X[:50]), m2.forward(X[:50]))

    const = _constant_model(float(np.mean(y)))
    assert task.score(m1, "holdout", 200) > task.score(const, "holdout", 200)


def _roundtrip(path: Path, model, X):
    model.save(str(path))
    loaded = type(model).load(str(path))
    return np.array_equal(loaded.forward(X), model.forward(X))


def test_gp_save_load_roundtrip_forward_identical():
    """A65 (75.2.3): `GP.save` / `GP.load` round-trips forward-identically
    (plain-array checkpoint, `.npz`)."""
    _task, X, _y, n_out, head = _sine_fixture()
    m = GP(n_out=n_out, head=head).fit(X, _y)
    with _tmp_npz("gp") as path:
        assert _roundtrip(path, m, X)


def test_gam_save_load_roundtrip_forward_identical():
    """A65 (75.2.4): `GAM.save` / `GAM.load` round-trips forward-identically,
    including a fit with knots and a zero-smoothing fit (empty-knot path)."""
    _task, X, y, n_out, head = _sine_fixture()
    with _tmp_npz("gam") as path:
        for wd in (1e-3, 0.0):
            m = GAM(n_out=n_out, head=head, weight_decay=wd).fit(X, y)
            assert _roundtrip(path, m, X)


# --- 75.2.5 bandit/catalog re-derivation (A65) --------------------------------

def test_catalog_actions_and_registry_rederived():
    """A65 (75.2.5): the new *values* grow the action catalog to 83 (77
    historical + 6) while the registry stays 15 fields and `model_family`
    spans all seven families (A26 re-derived; the bandit draw-space is
    untouched so the A1–A4 stream stays green)."""
    assert len(ACTIONS) == 83
    assert len(SPEC_FIELDS) == 15
    assert SPEC_FIELDS["model_family"].families == MODEL_FAMILIES
    assert MODEL_FAMILIES == ("mlp", "tree", "boost", "knn", "convnet",
                              "gp", "gam")


def test_gp_gam_specs_valid_and_trainable_end_to_end():
    """A65 (75.2.3/75.2.4): a `gp` / `gam` spec is a valid `ModelSpec` and
    trains end-to-end via `train_from_task` to a finite loss (the spec
    surface reaches the new families; the task scoring path is unchanged)."""
    task = SineRegressionV1(seed=7)
    base = ModelSpec(
        architecture=(16, 8), optimizer="adam", learning_rate=0.003,
        batch_size=32, weight_decay=0.001, train_steps=300, input_noise=0.02,
        activation="tanh", label_smoothing=0.05, lr_schedule="constant",
        early_stopping_patience=0, init_scale=1.3, gradient_clipping=0.0,
        knn_k=5).to_dict()
    for fam in ("gp", "gam"):
        spec = ModelSpec.from_dict({**base, "model_family": fam})
        spec.validate()
        result = train_from_task(task, spec, seed=11)
        assert np.isfinite(result.final_loss)


# --- 33.1 version + SPEC index (A65) -------------------------------------------

def test_version_and_spec_cite_a65():
    """A65 (33.1): the version steps to `0.63.0` in both sources; SPEC §75
    cites A65; the A25 index row for M64 lands and points at this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.63.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.63.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 75. Modeling capability v0.61" in spec
    assert "Acceptance (A65)" in spec
    assert ("| M64 | v0.61   | 75     | A65 | tests/test_modeling_v061.py |"
            in spec)


# --- small helper: a temp .npz path (stdlib, no fixture import) ---------------

def _tmp_npz(tag: str):
    import contextlib
    import tempfile

    @contextlib.contextmanager
    def _ctx():
        d = tempfile.mkdtemp(prefix=f"autorefine_{tag}_")
        p = Path(d) / "model.npz"
        try:
            yield p
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    return _ctx()
