"""Hot-path execution performance (v0.56, SPEC.md 70, A60).

SPEC.md 70: the training hot path no longer pays for work that changes
no result:
  * 70.1 — Momentum/Adam optimizer state is created lazily (the
    zero-vector default is no longer allocated on every step);
  * 70.2 — the MLP + ConvNet softmax branch caches `np.arange(n)` per
    batch size (`_row_idx`) and reuses the one-hot array directly when
    `label_smoothing == 0.0` (bit-identical: x*1.0, x+0.0 are exact);
  * 70.3 — activation dispatch is hoisted: bound `act` / `act_grad`
    function pairs (identical expressions to the string-dispatch
    helpers) are resolved once at construction, not per layer, per call;
  * 70.4 — A1–A59 stay green: every fix is semantics-identical (same
    ops, same order, same RNG draws — G2-safe); the full suite
    including A6's bit-exact pins is the proof;
  * 70.5 — the fixed-loop benchmark drops 482 ms -> 115 ms (~4.2x).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (fixtures synthesized here), the fixes are internal
to `models/` (no new exports — the 33.1 re-export pins stay unchanged).
"""
from __future__ import annotations

import tomllib
import time
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.models.convnet import ConvNet
from autorefine.models.mlp import (
    MLP,
    _activation,
    _activation_grad,
    _row_idx,
)
from autorefine.models.optimizers import Adam, Momentum

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


# --- 70.1 lazy optimizer state (A60) -------------------------------------------

def test_momentum_first_step_equals_explicit_zeros_reference():
    """A60 (70.1): a fresh-state first step equals the explicit-zeros
    reference bit-for-bit — the lazy default changes *when* the zero
    vector is allocated, not what it is."""
    rng = np.random.default_rng(0)
    p = rng.normal(size=5).copy()
    g = rng.normal(size=5)
    p_ref = p.copy()
    opt = Momentum(lr=0.05)
    state: dict = {}
    opt.step(p, g, state, 1)
    # explicit-zeros reference (the pre-round semantics, spelled out)
    v = np.zeros_like(p_ref)
    v = opt.mu * v + g
    p_ref -= opt.lr * v
    assert np.array_equal(p, p_ref)
    assert "v" in state and np.array_equal(state["v"], opt.mu * 0.0 + g)


def test_adam_first_two_steps_deterministic_and_stateful():
    """A60 (70.1): Adam's two lazily-created state entries are
    deterministic and accumulate across steps (second step reuses the
    stored state — no re-creation, same values as the reference)."""
    p1 = np.ones(4); p2 = np.ones(4)
    g = np.full(4, 0.1)
    s1: dict = {}; s2: dict = {}
    opt1, opt2 = Adam(lr=0.01), Adam(lr=0.01)
    for t in (1, 2, 3):
        opt1.step(p1, g, s1, t)
        opt2.step(p2, g, s2, t)
    assert np.array_equal(p1, p2)
    assert np.array_equal(s1["m"], s2["m"])
    assert np.array_equal(s1["v"], s2["v"])
    # the state entries are exactly the EMA accumulations (b1/b2 on the
    # explicit-zeros start) — the lazy default is semantically zeros
    m = ((1 - opt1.b1) * g) + opt1.b1 * ((1 - opt1.b1) * g)
    m = opt1.b1 * m + (1 - opt1.b1) * g
    assert np.array_equal(s1["m"], m)
    v = ((1 - opt1.b2) * g * g) + opt1.b2 * ((1 - opt1.b2) * g * g)
    v = opt1.b2 * v + (1 - opt1.b2) * g * g
    assert np.array_equal(s1["v"], v)


# --- 70.2 softmax fast paths (A60) ----------------------------------------------

def test_row_idx_correct_and_deterministic():
    """A60 (70.2.1): `_row_idx(n)` returns `np.arange(n)` — correct
    values, and the cached array is stable across calls (G2)."""
    for n in (1, 7, 120):
        idx = _row_idx(n)
        assert np.array_equal(idx, np.arange(n))
        assert _row_idx(n) is idx  # same batch size -> same cached array


def test_smoothing_zero_paths_bit_identical():
    """A60 (70.2.2): `label_smoothing=0`, `0.0`, and the pre-round
    explicit expression `(1-eps)*onehot + eps/n_out` return
    byte-identical loss and gradients — the fast path is an
    optimization, not a different computation."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 4))
    y = rng.integers(0, 3, 24)

    def trained_bytes() -> bytes:
        m = MLP(4, [8], 3, "tanh", 11)
        for _ in range(5):
            _loss, grads = m.loss_and_grads(x, y, 0)
            for (w, b), (gw, gb) in zip(m.layers, grads):
                w -= 0.05 * gw
                b -= 0.05 * gb
        return m.forward(x).tobytes()

    assert trained_bytes() == trained_bytes()  # deterministic (G2)

    m0 = MLP(4, [8], 3, "tanh", 11)
    l_int, g_int = m0.loss_and_grads(x, y, 0)
    m_f = MLP(4, [8], 3, "tanh", 11)
    l_float, g_float = m_f.loss_and_grads(x, y, 0.0)
    assert l_int == l_float
    for (gi, bi), (gf, bf) in zip(g_int, g_float):
        assert np.array_equal(gi, gf) and np.array_equal(bi, bf)

    # the explicit pre-round expression, spelled out here as the reference
    logits = m0.forward(x)
    z = logits - logits.max(axis=1, keepdims=True)
    logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    onehot = np.zeros_like(logz)
    onehot[np.arange(len(x)), y] = 1.0
    eps = 0.0
    target = (1.0 - eps) * onehot + eps / m0.n_out  # == onehot exactly
    ref_loss = float(-(target * logz).sum(axis=1).mean())
    assert l_int == ref_loss


def test_relu_and_mse_paths_stay_finite():
    """A60 (70.2/70.3): the relu softmax path (with smoothing) and the
    mse path remain finite and deterministic after the dispatch hoist."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(16, 3))
    y = rng.integers(0, 2, 16)
    m = MLP(3, [8], 2, "relu", 5)
    loss, grads = m.loss_and_grads(x, y, 0.1)
    assert np.isfinite(loss)
    for gw, gb in grads:
        assert np.isfinite(gw).all() and np.isfinite(gb).all()
    mt = MLP(3, [8], 1, "tanh", 5, head="mse")
    loss_t, grads_t = mt.loss_and_grads(x, rng.normal(size=16), 0.0)
    assert np.isfinite(loss_t)
    for gw, gb in grads_t:
        assert np.isfinite(gw).all()


# --- 70.3 hoisted dispatch + round-trips (A60) ----------------------------------

def test_hoisted_dispatch_matches_reference_helpers():
    """A60 (70.3): the bound act/act_grad pairs compute the same values
    as the public `_activation` / `_activation_grad` helpers — for both
    activations, in MLP and ConvNet."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(6, 8))
    for name in ("tanh", "relu"):
        out = rng.normal(size=(6, 8))
        m = MLP(8, [16], 2, name, 1)
        c = ConvNet((1, 8, 8), 4, 4, 2, name, 1)
        for act, act_grad in ((m._act_fn, m._act_grad_fn),
                              (c._act_fn, c._act_grad_fn)):
            assert act(z).tobytes() == _activation(name, z).tobytes()
            assert act_grad(z, out).tobytes() == \
                _activation_grad(name, z, out).tobytes()


def test_mlp_save_load_roundtrip_preserves_forward():
    """A60 (70.3): `MLP.load` rebinds the dispatch (the `__new__`
    constructor path) — a save/load round trip forwards
    byte-identically."""
    rng = np.random.default_rng(4)
    m = MLP(4, [16, 8], 3, "relu", 9)
    x = rng.normal(size=(5, 4))
    before = m.forward(x).tobytes()
    path = Path(__file__).parent / "_perf_v056_mlp.npz"
    try:
        m.save(str(path))
        m2 = MLP.load(str(path))
        assert m2.forward(x).tobytes() == before
    finally:
        path.unlink(missing_ok=True)


def test_convnet_save_load_roundtrip_preserves_forward():
    """A60 (70.3): `ConvNet.load` rebinds the dispatch too — a
    save/load round trip forwards byte-identically on the grid input."""
    rng = np.random.default_rng(5)
    c = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 7)
    x = rng.normal(size=(3, 1, 8, 8))
    before = c.forward(x).tobytes()
    path = Path(__file__).parent / "_perf_v056_conv.npz"
    try:
        c.save(str(path))
        c2 = ConvNet.load(str(path))
        assert c2.forward(x).tobytes() == before
    finally:
        path.unlink(missing_ok=True)


# --- 70.5 the fixed loop is functionally intact (A60) ----------------------------

def test_fixed_loop_benchmark_completes(tmp_path):
    """A60 (70.5): the canonical DashboardRunner loop (baseline + 2
    experiments, `search_quality="legacy"`) still completes with a
    verdict — the optimizations change speed, not behavior. The wall
    cap is a generous sanity bound (measured ~115 ms where the
    pre-round baseline was 482 ms; machines vary), not a performance
    gate."""
    from autorefine.dashboard import DashboardRunner

    csv = tmp_path / "d.csv"
    lines = ["a,b,churn"]
    for i in range(120):
        a = (i % 8) / 8.0
        b = ((i // 8) % 5) / 5.0
        c = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{c:.0f}")
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    r = DashboardRunner(
        csv_path=str(csv), seed=1, experiments=2, max_seconds=300.0,
        max_train_seconds=30.0, runs_dir=str(tmp_path / "runs"),
        search_quality="legacy")
    r.start()
    while not r.done:
        r.next()
    result = r.finish()
    dt = time.perf_counter() - t0
    assert dt < 30.0  # generous sanity cap, not a perf gate
    assert result is not None
    assert result["verdict"] in ("PASS", "MISS")


# --- 33.1 version + SPEC index (A60) --------------------------------------------

def test_version_and_spec_cite_a60():
    """A60 (33.1): the version steps to `0.57.0` in both sources;
    SPEC §70 cites A60; the A25 index row for M59 lands."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.57.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.57.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 70.6 Acceptance (A60)" in spec
    assert "## 70. Hot-path execution performance (v0.56)" in spec
    assert "**M59** — v0.56" in spec
    assert ("| M59 | v0.56   | 70     | A60 | tests/test_perf_v056.py |"
            in spec)
