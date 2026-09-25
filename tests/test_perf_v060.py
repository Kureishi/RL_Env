"""Training hot path, round 2 (v0.60, SPEC.md 74, A64).

SPEC.md 74: the training hot path drops its remaining pure-Python
overhead, every fix **bit-identical** (the 70.4/G2 precedent — same
ops, same per-element order, fewer temporaries):
  * 74.2.1 — `_train_neural` hoists loop-invariant `ModelSpec` reads;
  * 74.2.2 — the constant schedule short-circuits the per-step
    `_scheduled_lr` call (value identity proven by its own early
    return);
  * 74.2.3 — Momentum/Adam moment updates run in-place (1–2 fewer
    temporary allocations per step);
  * 74.2.4 — the MLP/ConvNet backward chains multiply in-place; MLP
    dispatches on a hoisted `_is_mse` flag (kept in sync by `load`);
  * 74.3   — the 11-training pin battery is byte-stable against the
    locked reference; A1–A63 (incl. A6's bit-exact sequence) pass
    unmodified.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (fixtures synthesized here), the fixes are internal
to `trainer.py` + `models/` (no new exports — the 33.1 re-export pins
stay unchanged).
"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import numpy as np

import autorefine
from autorefine.config import ModelSpec
from autorefine.models.convnet import ConvNet
from autorefine.models.mlp import MLP, _activation, _activation_grad
from autorefine.models.optimizers import Adam, Momentum
from autorefine.trainer import _scheduled_lr, train
from autorefine.tasks import Parity4V1, SineRegressionV1

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


def _base_spec(**kw) -> ModelSpec:
    base = dict(architecture=(16, 8), optimizer="adam",
                learning_rate=0.003, batch_size=32, weight_decay=0.001,
                train_steps=300, input_noise=0.02, activation="tanh",
                label_smoothing=0.05, lr_schedule="constant",
                early_stopping_patience=0, init_scale=1.3,
                gradient_clipping=0.0, knn_k=5)
    base.update(kw)
    return ModelSpec(**base)


# --- 74.2.2 constant-schedule value identity (A64) ----------------------------

def test_scheduled_lr_constant_returns_base_for_every_step():
    """A64 (74.2.2): `_scheduled_lr("constant", base, t, total)` is
    exactly `base` for every `t` and every `total` (incl. `total <= 0`)
    — the value identity the trainer's short-circuit (one assignment
    before the loop instead of a per-step call) relies on."""
    for base in (0.0, 0.003, 0.5):
        for total in (0, 1, 300):
            for t in range(1, total + 1):
                assert _scheduled_lr("constant", base, t, total) == base
            # total <= 0: the early return the short-circuit mirrors
            assert _scheduled_lr("constant", base, 1, total) == base


def test_constant_schedule_training_is_bit_stable():
    """A64 (74.2.1/74.2.2): a fixed-seed constant-schedule training is
    bit-stable across two consecutive runs — the hoisted invariants +
    short-circuit change speed, not the stream."""
    ds = Parity4V1(seed=7).make_dataset(60)
    sp = _base_spec(learning_rate=0.01, train_steps=200)
    r1 = train(ds, sp, 7, n_out=2, head="softmax")
    r2 = train(ds, sp, 7, n_out=2, head="softmax")
    assert repr(r1.final_loss) == repr(r2.final_loss)
    assert r1.steps_run == r2.steps_run
    for (w1, b1), (w2, b2) in zip(r1.model.layers, r2.model.layers):
        assert w1.tobytes() == w2.tobytes()
        assert b1.tobytes() == b2.tobytes()


# --- 74.2.3 in-place moment updates (A64) -------------------------------------

def test_momentum_inplace_equals_outplace_reference():
    """A64 (74.2.3): the in-place moment update `v *= mu; v += grad`
    equals the pre-round out-of-place expression `v = mu * v + grad`
    bit-for-bit — parameters AND state, across multiple steps."""
    rng = np.random.default_rng(0)
    g1, g2, g3 = (rng.normal(size=5) for _ in range(3))
    p_new = rng.normal(size=5).copy()
    p_ref = p_new.copy()
    opt = Momentum(lr=0.05)
    v_ref = np.zeros_like(p_ref)
    state: dict = {}
    for t, g in ((1, g1), (2, g2), (3, g3)):
        opt.step(p_new, g, state, t)
        # the pre-round expression, spelled out as the reference
        v_ref = opt.mu * v_ref + g
        p_ref -= opt.lr * v_ref
        assert np.array_equal(p_new, p_ref)
        assert np.array_equal(state["v"], v_ref)


def test_adam_inplace_moments_equal_reference():
    """A64 (74.2.3): Adam's in-place moment updates equal the pre-round
    out-of-place expressions `m = b1*m + (1-b1)*grad`,
    `v = b2*v + (1-b2)*grad*grad` bit-for-bit — moments AND params,
    across multiple steps."""
    p_new = np.ones(4)
    p_ref = np.ones(4)
    g = np.full(4, 0.1)
    opt = Adam(lr=0.01)
    state: dict = {}
    m_ref = np.zeros(4)
    v_ref = np.zeros(4)
    for t in (1, 2, 3):
        opt.step(p_new, g, state, t)
        # the pre-round expressions, spelled out as the reference
        m_ref = opt.b1 * m_ref + (1 - opt.b1) * g
        v_ref = opt.b2 * v_ref + (1 - opt.b2) * g * g
        mhat = m_ref / (1 - opt.b1 ** t)
        vhat = v_ref / (1 - opt.b2 ** t)
        p_ref -= opt.lr * mhat / (np.sqrt(vhat) + opt.eps)
        assert np.array_equal(p_new, p_ref)
        assert np.array_equal(state["m"], m_ref)
        assert np.array_equal(state["v"], v_ref)


# --- 74.2.4 in-place backward chains (A64) ------------------------------------

def _mlp_reference(m: MLP, x: np.ndarray, y: np.ndarray,
                   label_smoothing: float) -> tuple[float, list]:
    """Hand-spelled backprop with OUT-OF-PLACE chain multiplications —
    the pre-round expression, used as the bit-identity reference for
    the in-place `d *= act_grad(...)` chain (74.2.4)."""
    act = m.activation
    a = x
    cache = []
    for w, b in m.layers[:-1]:
        z = a @ w + b
        h = _activation(act, z)
        cache.append((a, z, h))
        a = h
    w, b = m.layers[-1]
    logits = a @ w + b
    cache.append((a, logits, logits))
    n = x.shape[0]
    if m.head == "mse":
        target = np.asarray(y, dtype=np.float64).reshape(n, m.n_out)
        diff = logits - target
        loss = float((diff * diff).mean())
        dz = 2.0 * diff / n
    else:
        z = logits - logits.max(axis=1, keepdims=True)
        logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        onehot = np.zeros_like(logz)
        onehot[np.arange(n), y] = 1.0
        eps = float(label_smoothing)
        target = onehot if eps == 0.0 else (1.0 - eps) * onehot + eps / m.n_out
        loss = float(-(target * logz).sum(axis=1).mean())
        dz = (np.exp(logz) - target) / n
    grads = []
    d = dz
    for i in range(len(m.layers) - 1, -1, -1):
        gw = cache[i][0].T @ d
        gb = d.sum(axis=0)
        if i > 0:
            z_prev = cache[i - 1][1]
            h_prev = cache[i - 1][2]
            d = (d @ m.layers[i][0].T) * \
                _activation_grad(act, z_prev, h_prev)  # out-of-place
        grads.append((gw, gb))
    grads.reverse()
    return loss, grads


def test_mlp_grads_match_outplace_reference():
    """A64 (74.2.4): `MLP.loss_and_grads` (in-place chain) equals the
    hand-spelled out-of-place reference byte-for-byte — loss plus every
    `gw`/`gb` — for both activations and both heads."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 4))
    y_cls = rng.integers(0, 3, 20)
    y_reg = rng.normal(size=(20, 2))
    for act in ("tanh", "relu"):
        for eps in (0.0, 0.1):
            m = MLP(4, [8, 6], 3, act, 11)
            loss, grads = m.loss_and_grads(x, y_cls, eps)
            rloss, rgrads = _mlp_reference(m, x, y_cls, eps)
            assert loss == rloss
            for (gw, gb), (rgw, rgb) in zip(grads, rgrads):
                assert gw.tobytes() == rgw.tobytes()
                assert gb.tobytes() == rgb.tobytes()
        m = MLP(4, [8, 6], 2, act, 11, head="mse")
        loss, grads = m.loss_and_grads(x, y_reg, 0.0)
        rloss, rgrads = _mlp_reference(m, x, y_reg, 0.0)
        assert loss == rloss
        for (gw, gb), (rgw, rgb) in zip(grads, rgrads):
            assert gw.tobytes() == rgw.tobytes()
            assert gb.tobytes() == rgb.tobytes()


def test_is_mse_flag_synced_after_load():
    """A64 (74.2.4): the `_is_mse` dispatch flag is set in `__init__`
    AND re-synced by `load()` — an mse checkpoint loads as mse, a
    softmax checkpoint as softmax, and both forward byte-identically
    through the round trip."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 4))
    for head, y in (("softmax", rng.integers(0, 3, 6)),
                    ("mse", rng.normal(size=(6, 1)))):
        m = MLP(4, [8], 3 if head == "softmax" else 1, "tanh", 9, head=head)
        assert m._is_mse == (head == "mse")
        before = m.forward(x).tobytes()
        path = Path(__file__).parent / f"_perf_v060_{head}.npz"
        try:
            m.save(str(path))
            m2 = MLP.load(str(path))
            assert m2.head == head
            assert m2._is_mse == (head == "mse")
            assert m2.forward(x).tobytes() == before
        finally:
            path.unlink(missing_ok=True)


def test_convnet_grads_byte_stable():
    """A64 (74.2.4): `ConvNet.loss_and_grads` — with its three in-place
    `* act_grad` multiplies — is byte-stable across fresh same-seed
    instances (loss + every grad array)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 1, 8, 8))
    y = rng.integers(0, 2, 4)
    outs = []
    for _ in range(2):
        c = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 7)
        loss, grads = c.loss_and_grads(x, y, 0.05)
        outs.append((repr(loss), [g.tobytes() for g in
                                  [gw for gw, _ in grads] +
                                  [gb for _, gb in grads]]))
    assert outs[0] == outs[1]


# --- 74.3.1 the 11-training pin battery (A64) ---------------------------------

# Locked reference (SPEC.md 74.3.1; scratchpad pin_capture.py ->
# pins_AFTER.json, verified bit-identical vs pins_BEFORE.json).
# (hash of weights, repr(final_loss), steps_run)
_BATTERY_REFERENCE = {
    "parity-adam-7":    ("f7a07ff7d4859c37", "0.6532037759681817", 300),
    "parity-mom-sched": ("2c064ad2feacab5e", "0.5231799996533362", 76),
    "parity-sgd-clip":  ("f2e1f05e10bf20d9", "0.6735606920969781", 300),
    "parity-adam-relu": ("95f5729ddb16b6a5", "0.5180021187174498", 300),
    "parity-cosine":    ("ec697ef6f2ae4f3d", "0.725489855880329", 300),
    "parity-boost-7":   ("e3b0c44298fc1c14", "0.6937420830787159", 4),
    "parity-tree-7":    ("0571ea2138e20df9", "0.6973295916778232", 4),
    "parity-knn-7":     ("4fc82b26aecb47d2", "0.7054064019651031", 1),
    "sine-adam-13":     ("79b5a1b3714d2e23", "0.9877146479150769", 300),
    "sine-momentum-13": ("5172cf8b141b6219", "0.9411544472291459", 300),
    "sine-sgd-13":      ("5c323b9c77cc39d5", "1.1045520841998009", 300),
}


def _hash_model(model) -> str:
    h = hashlib.sha256()
    for w, b in getattr(model, "layers", []):
        h.update(w.tobytes())
        h.update(b.tobytes())
    if hasattr(model, "trees"):
        for t in model.trees:
            for a in ("feat", "thr", "left", "right", "leafv"):
                h.update(getattr(t, a).tobytes())
    if hasattr(model, "k"):
        h.update(str(model.k).encode())
    return h.hexdigest()[:16]


def test_training_battery_matches_locked_reference():
    """A64 (74.3.1): the 11-training pin battery — all four 74.2 edit
    surfaces (constant/cosine/warmup_cosine schedules, early stopping,
    gradient clipping, boost/tree/knn families, softmax AND mse heads)
    — matches the locked reference bit-for-bit. A float-order change
    anywhere in the optimized path breaks this test."""
    parity = Parity4V1(seed=7)
    ds_p = parity.make_dataset(60)
    sine = SineRegressionV1(seed=13)
    ds_s = sine.make_dataset(200)

    def run(spec: ModelSpec, ds, seed: int, n_out: int, head: str):
        res = train(ds, spec, seed, n_out=n_out, head=head)
        return (_hash_model(res.model), repr(res.final_loss), res.steps_run)

    got = {
        "parity-adam-7": run(_base_spec(), ds_p, 7, 2, "softmax"),
        "parity-mom-sched": run(
            _base_spec(optimizer="momentum", lr_schedule="warmup_cosine",
                       early_stopping_patience=40), ds_p, 11, 2, "softmax"),
        "parity-sgd-clip": run(
            _base_spec(optimizer="sgd", learning_rate=0.05,
                       gradient_clipping=1.0, input_noise=0.0), ds_p, 3, 2,
            "softmax"),
        "parity-adam-relu": run(_base_spec(activation="relu"), ds_p, 42, 2,
                                "softmax"),
        "parity-cosine": run(_base_spec(lr_schedule="cosine"), ds_p, 5, 2,
                             "softmax"),
        "parity-boost-7": run(
            _base_spec(model_family="boost", train_steps=400,
                       input_noise=0.0, architecture=(2,)), ds_p, 7, 2,
            "softmax"),
        "parity-tree-7": run(
            _base_spec(model_family="tree", train_steps=400,
                       architecture=(2,), input_noise=0.0), ds_p, 7, 2,
            "softmax"),
        "parity-knn-7": run(
            _base_spec(model_family="knn", knn_k=11, train_steps=200), ds_p,
            7, 2, "softmax"),
    }
    sine_kw = dict(train_steps=300, input_noise=0.0, weight_decay=0.0,
                   label_smoothing=0.0)
    got["sine-adam-13"] = run(_base_spec(optimizer="adam", **sine_kw), ds_s,
                              13, 1, "mse")
    got["sine-momentum-13"] = run(
        _base_spec(optimizer="momentum", **sine_kw), ds_s, 13, 1, "mse")
    got["sine-sgd-13"] = run(
        _base_spec(optimizer="sgd", learning_rate=0.01, **sine_kw), ds_s, 13,
        1, "mse")

    for tag, ref in _BATTERY_REFERENCE.items():
        assert got[tag] == ref, f"{tag}: got {got[tag]}, want {ref}"


# --- 33.1 version + SPEC index (A64) -------------------------------------------

def test_version_and_spec_cite_a64():
    """A64 (33.1): the version steps to `0.63.0` in both sources;
    SPEC §74 cites A64; the A25 index row for M63 lands."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.63.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.63.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 74.5 Acceptance (A64)" in spec
    assert "## 74. Training hot path, round 2" in spec
    assert "**M63** — v0.60" in spec
    assert ("| M63 | v0.60   | 74     | A64 | tests/test_perf_v060.py |"
            in spec)
