"""Modeling capability v0.64 — fine-pattern capture (SPEC.md 78, A68).

SPEC.md 78: three additive **field-level** `ModelSpec` extensions, each a
`0`/`1.0` **legacy default** so every pre-v0.64 spec dict, the A1–A4
proposal stream, and the §18.7 bit-exact sequence stay green (78.2.4):
  * 78.2.1 — `fourier_features` (a deterministic spectral input expansion
    `[x, cos(k·x̃), sin(k·x̃)]`, `x̃ = x/max(|x|, 1e-6)`, block-per-k layout;
    consumed by mlp / knn / gp; `K = 0` returns `X` unmodified);
  * 78.2.2 — `gam_interactions` (GAM two-way interaction terms, ranked by
    the univariate hinge strengths; `P = 0` is the legacy additive fit);
  * 78.2.3 — `gp_length_scale` (the RBF length scale as a one-argument
    change to the fixed-seed RFF draw; `1.0` is the legacy unit scale).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here). The field-level
additions grow the registry 15→18 fields and the catalog 83→95 actions
(A26/A8 re-derived) — the legacy values, the A24 one-field-set invariant,
and the lexicographic bandit cold-start stay green (78.2.4).
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.config import (
    FOURIER_FEATURES,
    GAM_INTERACTIONS,
    GP_LENGTH_SCALES,
    MODEL_FAMILIES,
    SpecError,
    ModelSpec,
    spec_n_params,
)
from autorefine.improver.actions import (
    EXCLUDED_FROM_SEARCH,
    FIELD_NAMES,
    FIELD_SAMPLERS,
    SEARCH_FIELDS,
)
from autorefine.improver.catalog import ACTIONS
from autorefine.improver.specspace import SPEC_FIELDS
from autorefine.models.gam import GAM
from autorefine.models.gp import GP
from autorefine.tasks import SineRegressionV1
from autorefine.trainer import fourier_expand, train_from_task

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


# --- 78.2.1 fourier_features: spectral expansion (A68) -----------------------

def test_fourier_expand_k0_is_identity():
    """A68 (78.2.1): `fourier_expand(X, 0) is X` — the `K = 0` default is a
    no-op that returns the array **unmodified** (no dtype coercion), the
    bit-identity guard that keeps every pre-v0.64 run untouched."""
    X = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])
    assert fourier_expand(X, 0) is X


def test_fourier_expand_block_per_k_layout():
    """A68 (78.2.1): `K > 0` gives shape `(n, d·(2K+1))` with the
    block-per-k column layout — `[x | cos k=1 (all dims) | sin k=1 | …]` —
    over the per-dimension normalized coordinate `x_t = x / max(|x|, 1e-6)`
    (a pure function: no RNG, exact cos/sin values)."""
    X = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])
    E = fourier_expand(X, 2)
    assert E.shape == (3, 2 * (2 * 2 + 1))
    scale = np.maximum(np.abs(X).max(axis=0), 1e-6)  # per-dim scale [3, 4]
    xt = X / scale
    # block-per-k: [x(2) | cos k=1 (2) | sin k=1 (2) | cos k=2 (2) | sin k=2 (2)]
    assert np.allclose(E[:, 0:2], X)
    assert np.allclose(E[:, 2:4], np.cos(xt))
    assert np.allclose(E[:, 4:6], np.sin(xt))
    assert np.allclose(E[:, 6:8], np.cos(2 * xt))
    assert np.allclose(E[:, 8:10], np.sin(2 * xt))


def test_mlp_fourier_trains_finite_and_nparams_reflects_width():
    """A68 (78.2.1): an `mlp` spec with `fourier_features > 0` trains to a
    finite loss, and `spec_n_params` reflects the expanded input width —
    `K = 0` is exactly the legacy formula, `K > 0` multiplies the input dim
    by `(2K+1)` (the honest size axis)."""
    task = SineRegressionV1(seed=7)
    spec = ModelSpec(
        architecture=(16,), optimizer="adam", learning_rate=0.003,
        batch_size=32, weight_decay=0.001, train_steps=200, input_noise=0.02,
        activation="tanh", label_smoothing=0.0, lr_schedule="constant",
        early_stopping_patience=0, init_scale=1.0, gradient_clipping=0.0,
        knn_k=5, fourier_features=16, gam_interactions=0, gp_length_scale=1.0)
    spec.validate()
    result = train_from_task(task, spec, seed=11)
    assert np.isfinite(result.final_loss)

    # spec_n_params: d=4, arch=(16,), n_out=2
    d, o = 4, 2
    base = dict(architecture=(16,), optimizer="momentum", learning_rate=1e-3,
                batch_size=32, weight_decay=1e-4, train_steps=400,
                input_noise=0.02, activation="tanh", model_family="mlp")
    k0 = spec_n_params(base, state_dim=d, n_out=o)
    k16 = spec_n_params({**base, "fourier_features": 16}, state_dim=d, n_out=o)
    assert k0 == d * 16 + 16 + 16 * o + o            # the legacy formula
    assert k16 == d * 33 * 16 + 16 + 16 * o + o     # d*(2*16+1) input dim
    assert k16 > k0


# --- 78.2.2 gam_interactions: two-way interaction terms (A68) -----------------

def _xor(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)  # XOR on (col0, col1)
    return X, y


def test_gam_p0_bitidentical_to_additive():
    """A68 (78.2.2): `GAM` with `n_interactions = 0` (the default) is
    **bit-identical** to the legacy additive fit — two fresh fits agree on
    `forward` and no interaction pair is selected (the pin guard, 78.2.2)."""
    X, y = _xor(400, seed=0)
    a = GAM(n_out=2, head="softmax").fit(X, y)
    b = GAM(n_out=2, head="softmax", n_interactions=0).fit(X, y)
    assert a.interactions == [] and b.interactions == []
    assert np.array_equal(a.forward(X), b.forward(X))


def test_gam_interactions_break_xor_structural_limit():
    """A68 (78.2.2): on a 400-row 2-D XOR the additive model hits its
    structural limit (< 0.7) while `n_interactions > 0` recovers the target
    (≥ 0.99) with the true `(0, 1)` pair selected first — the capability
    delta (40 rows is too sparse for the 5-knot additive model to fail, so
    the fixture is 400 rows, SPEC 78.3.3(b))."""
    X, y = _xor(400, seed=0)
    additive = (GAM(n_out=2, head="softmax").fit(X, y).forward(X).argmax(1)
                == y).mean()
    inter = GAM(n_out=2, head="softmax", n_interactions=4).fit(X, y)
    interacting = (inter.forward(X).argmax(1) == y).mean()
    assert additive < 0.7, additive
    assert interacting >= 0.99, interacting
    assert inter.interactions[0] == (0, 1), inter.interactions


def test_gam_mse_interaction_recovers_product():
    """A68 (78.2.2): an `mse` head with `n_interactions > 0` recovers an
    `x₀·x₁`-dominated target (rmse ≈ 0) — the interaction term is exactly
    the two-way product structure an additive model cannot express."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] ** 2
    gm = GAM(n_out=1, head="mse", n_interactions=2).fit(X, y)
    resid = gm.forward(X)[:, 0] - y
    assert np.sqrt((resid ** 2).mean()) < 5e-3
    assert len(gm.interactions) == 2


@contextlib.contextmanager
def _tmp(tag: str):
    d = tempfile.mkdtemp(prefix=f"autorefine_{tag}_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gam_save_load_roundtrip_and_legacy_load():
    """A68 (78.2.2): `save`/`load` round-trips are forward-identical and
    preserve the interaction pairs; a pre-v0.64 checkpoint (the two new keys
    stripped) loads additive-only (back-compat, 78.2.2)."""
    X, y = _xor(60, seed=2)
    gm = GAM(n_out=2, head="softmax", n_interactions=2).fit(X, y)
    with _tmp("gam64") as d:
        p = d / "gam.npz"
        gm.save(str(p))
        gl = GAM.load(str(p))
        assert np.array_equal(gl.forward(X), gm.forward(X))
        assert gl.interactions == gm.interactions
        with np.load(str(p), allow_pickle=False) as z:
            keys = {k: z[k] for k in z.files
                    if k not in ("n_interactions", "pairs")}
        p2 = d / "legacy.npz"
        np.savez(str(p2), **keys)
        gl2 = GAM.load(str(p2))
        assert gl2.interactions == []


# --- 78.2.3 gp_length_scale: RBF bandwidth (A68) -------------------------------

def test_gp_length_scale1_bitidentical_to_legacy():
    """A68 (78.2.3): `GP` with `length_scale = 1.0` (the default) is
    **bit-identical** to the legacy unit-scale GP — only the RFF `normal`
    scale argument changes, and `1.0 / 1.0 == 1.0` (the pin guard)."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    p1 = GP(n_out=2, head="softmax").fit(X, y)
    p2 = GP(n_out=2, head="softmax", length_scale=1.0).fit(X, y)
    assert np.array_equal(p1.forward(X), p2.forward(X))


def test_gp_length_scale_sensitivity_and_validation():
    """A68 (78.2.3): `length_scale != 1.0` moves the output (the sensitivity
    — the RBF bandwidth), and `length_scale <= 0` is a `SpecError`."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    p1 = GP(n_out=2, head="softmax").fit(X, y)
    p3 = GP(n_out=2, head="softmax", length_scale=0.5).fit(X, y)
    assert not np.array_equal(p3.forward(X), p1.forward(X))
    with pytest.raises(SpecError):
        GP(n_out=2, head="softmax", length_scale=0.0)
    with pytest.raises(SpecError):
        GP(n_out=2, head="softmax", length_scale=-1.0)


def test_gp_save_load_roundtrip_and_legacy_load():
    """A68 (78.2.3): `save`/`load` round-trips are forward-identical and
    preserve the length scale; a pre-v0.64 checkpoint (the new key stripped)
    loads at the legacy `1.0` (back-compat, 78.2.3)."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    p3 = GP(n_out=2, head="softmax", length_scale=0.5).fit(X, y)
    with _tmp("gp64") as d:
        p = d / "gp.npz"
        p3.save(str(p))
        pl = GP.load(str(p))
        assert pl.length_scale == 0.5
        assert np.array_equal(pl.forward(X), p3.forward(X))
        with np.load(str(p), allow_pickle=False) as z:
            keys = {k: z[k] for k in z.files if k != "length_scale"}
        p2 = d / "legacy.npz"
        np.savez(str(p2), **keys)
        pl2 = GP.load(str(p2))
        assert pl2.length_scale == 1.0


# --- 78.2.4 pins: field-set invariants + registry growth (A68) ---------------

def test_registry_grows_to_18_fields_95_actions():
    """A68 (78.2.4 / A26 re-derived): the registry grows 15→18 fields and the
    catalog 83→95 actions; the three new fields span their consuming families
    (A26, the catalog view)."""
    assert len(ACTIONS) == 95
    assert len(SPEC_FIELDS) == 18
    assert len(FIELD_NAMES) == 18
    new_fields = {"fourier_features", "gam_interactions", "gp_length_scale"}
    assert new_fields <= set(FIELD_NAMES)
    assert SPEC_FIELDS["model_family"].families == MODEL_FAMILIES
    assert MODEL_FAMILIES == ("mlp", "tree", "boost", "knn", "convnet",
                              "gp", "gam")


def test_one_set_invariant_and_search_fields_stay_14():
    """A68 (78.2.4 / A24 re-derived): the field sets are one set — the three
    new fields join `EXCLUDED_FROM_SEARCH` (alongside `knn_k`), so the 14-field
    SearchPolicy proposal space is unchanged, while `FIELD_NAMES`/
    `FIELD_SAMPLERS`/`FAMILY_FIELDS` all see the 18 fields (the A24
    one-field-set invariant)."""
    assert set(SEARCH_FIELDS) == set(FIELD_NAMES) - set(EXCLUDED_FROM_SEARCH)
    assert len(SEARCH_FIELDS) == 14  # the documented v0.10 legacy space
    for field in FIELD_NAMES:
        assert field in FIELD_SAMPLERS, field
    assert set(EXCLUDED_FROM_SEARCH) == {"knn_k", "fourier_features",
                                         "gam_interactions", "gp_length_scale"}


def test_new_fields_registered_in_legal_spaces():
    """A68 (78.2.4): each new field is validated for every family but consumed
    only by its named family (the 25.2 `knn_k` pattern) — the registered spaces
    are the config constants, and every value round-trips through
    `to_dict`/`from_dict`."""
    assert tuple(FOURIER_FEATURES) == (0, 16, 32, 64)
    assert tuple(GAM_INTERACTIONS) == (0, 4, 8, 16)
    assert tuple(GP_LENGTH_SCALES) == (0.25, 0.5, 1.0, 2.0)
    for K, P, LS in ((16, 4, 0.5), (32, 8, 0.25), (64, 16, 2.0)):
        spec = ModelSpec(
            architecture=(16,), optimizer="momentum", learning_rate=1e-3,
            batch_size=32, weight_decay=1e-4, train_steps=400, input_noise=0.02,
            activation="tanh", model_family="mlp", fourier_features=K,
            gam_interactions=P, gp_length_scale=LS)
        spec.validate()
        rt = ModelSpec.from_dict(spec.to_dict())
        assert (rt.fourier_features, rt.gam_interactions, rt.gp_length_scale) \
            == (K, P, LS)


def test_legacy_spec_json_still_loads():
    """A68 (78.2.4 / 78.2.1–3): a pre-v0.64 spec dict (no new keys) loads with
    the legacy defaults (`fourier_features=0`, `gam_interactions=0`,
    `gp_length_scale=1.0`) — back-compat, the §17/§19.1/§25.2 pattern."""
    legacy = dict(architecture=(16,), optimizer="momentum", learning_rate=1e-3,
                  batch_size=32, weight_decay=1e-4, train_steps=400,
                  input_noise=0.02, activation="tanh")
    spec = ModelSpec.from_dict(legacy)
    spec.validate()
    assert spec.fourier_features == 0
    assert spec.gam_interactions == 0
    assert spec.gp_length_scale == 1.0


# --- 33.1 version + SPEC index (A68) ------------------------------------------

def test_version_and_spec_cite_a68():
    """A68 (33.1): the version steps to `0.64.0` in both sources; SPEC §78
    cites A68; the A25 index row for M67 points at this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.64.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.64.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 78. Modeling capability v0.64" in spec
    assert "Acceptance (A68)" in spec
    assert ("| M67 | v0.64   | 78     | A68 | tests/test_modeling_v064.py |"
            in spec)
