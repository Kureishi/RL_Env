"""Trainer/model tests (SPEC.md T2, T3, T6 + gradient correctness);
acceptance A9 (SPEC.md 12: the v0.5 trainer/model-space coverage)."""
import numpy as np
import pytest

from autorefine.config import ModelSpec, SpecError
from autorefine.models.mlp import MLP
from autorefine.tasks.cartpole import CartPoleV1
from autorefine.trainer import train

VALID = ModelSpec(
    architecture=(16, 8),
    optimizer="momentum",
    learning_rate=1e-3,
    batch_size=32,
    weight_decay=1e-4,
    train_steps=200,
    input_noise=0.02,
    activation="tanh",
)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(architecture=(24,)),               # hidden size outside allowed set
        dict(architecture=(16, 16, 16, 16)),    # depth 4 > max 3
        dict(optimizer="rmsprop"),
        dict(learning_rate=1e-5),                # below range
        dict(learning_rate=1e-1 + 1e-9),         # above range
        dict(batch_size=24),
        dict(weight_decay=2e-2),
        dict(train_steps=199),
        dict(train_steps=5001),
        dict(input_noise=0.11),
        dict(activation="swish"),
    ],
)
def test_invalid_spec_rejected(kwargs):
    """T3: out-of-range values raise ValueError (SpecError), never clipped."""
    base = VALID.to_dict()
    base.update(kwargs)
    with pytest.raises(SpecError):
        ModelSpec.from_dict(base)


def test_depth_zero_is_a_valid_linear_spec():
    """SPEC.md 15: mlp depth 0 is a linear model and must be accepted."""
    base = VALID.to_dict()
    base["architecture"] = []
    ModelSpec.from_dict(base)  # must not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(model_family="bogus"),
        dict(model_family="tree", architecture=()),          # tree needs (depth,)
        dict(model_family="tree", architecture=(16, 8)),     # tree depth must be 1..3
        dict(model_family="tree", architecture=(4,)),        # depth 4 > 3
    ],
)
def test_model_family_validation(kwargs):
    """SPEC.md 15: model_family values and family-specific architecture rules."""
    base = VALID.to_dict()
    base.update(kwargs)
    with pytest.raises(SpecError):
        ModelSpec.from_dict(base)


def test_tree_family_trains_deterministically_and_saves():
    """SPEC.md 15 tree family: deterministic fit, loadable npz checkpoint."""
    import tempfile
    import pathlib

    from autorefine.models.trees import TreeEnsemble

    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    tree = ModelSpec(
        architecture=(2,), optimizer="sgd", learning_rate=1e-3, batch_size=32,
        weight_decay=0.0, train_steps=1000, input_noise=0.0, activation="tanh",
        model_family="tree",
    )
    r1 = train(ds, tree, seed=7, n_out=task.n_outputs, head=task.head)
    r2 = train(ds, tree, seed=7, n_out=task.n_outputs, head=task.head)
    X = ds[0][:100]
    assert np.array_equal(r1.model.forward(X), r2.model.forward(X))
    assert r1.model.n_trees == 10  # train_steps // 100
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "t.npz"
        r1.model.save(str(p))
        m = TreeEnsemble.load(str(p))
        assert np.allclose(m.forward(X), r1.model.forward(X))


def test_linear_depth_zero_trains():
    """SPEC.md 15: a depth-0 mlp trains (linear model) and predicts."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    lin = ModelSpec(
        architecture=(), optimizer="adam", learning_rate=1e-2, batch_size=64,
        weight_decay=0.0, train_steps=200, input_noise=0.0, activation="relu",
    )
    r = train(ds, lin, seed=7, n_out=task.n_outputs, head=task.head)
    out = r.model.forward(ds[0][:16])
    assert out.shape == (16, task.n_outputs)
    assert np.isfinite(out).all()


def test_valid_boundary_values_accepted():
    for d in (
        dict(learning_rate=1e-4, train_steps=200, input_noise=0.0),
        dict(learning_rate=1e-1, train_steps=5000, input_noise=0.1, weight_decay=1e-2),
    ):
        base = VALID.to_dict()
        base.update(d)
        ModelSpec.from_dict(base)  # must not raise


def test_training_is_deterministic():
    """T2: same dataset + spec + seed => bit-identical weights."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    r1 = train(ds, VALID, seed=7)
    r2 = train(ds, VALID, seed=7)
    for (w1, b1), (w2, b2) in zip(r1.model.layers, r2.model.layers):
        assert np.array_equal(w1, w2)
        assert np.array_equal(b1, b2)


def test_time_cap_aborts_cleanly():
    """T6: a heavy spec under a tiny time cap stops cleanly with partial progress."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    heavy = ModelSpec(
        architecture=(128, 128, 128), optimizer="adam", learning_rate=3e-3,
        batch_size=128, weight_decay=0.0, train_steps=5000, input_noise=0.0,
        activation="tanh",
    )
    result = train(ds, heavy, seed=7, time_limit_seconds=0.15, time_check_every=64)
    assert result.time_capped
    assert result.steps_run < heavy.train_steps
    assert result.train_seconds < 5.0  # no runaway


def test_gradient_check_finite_differences():
    """Analytic backprop vs central finite differences (both activations)."""
    rng = np.random.default_rng(0)
    for act in ("tanh", "relu"):
        m = MLP(4, (8,), 2, act, seed=0)
        X = rng.normal(size=(6, 4))
        y = rng.integers(0, 2, 6)
        loss, grads = m.loss_and_grads(X, y)
        eps = 1e-6
        for i, (w, b) in enumerate(m.layers):
            for name, p, g in (("w", w, grads[i][0]), ("b", b, grads[i][1])):
                num = np.zeros_like(p)
                for idx in np.ndindex(p.shape):
                    orig = p[idx]
                    p[idx] = orig + eps
                    lp, _ = m.loss_and_grads(X, y)
                    p[idx] = orig - eps
                    lm, _ = m.loss_and_grads(X, y)
                    p[idx] = orig
                    num[idx] = (lp - lm) / (2 * eps)
                denom = np.abs(num) + np.abs(g) + 1e-8
                assert np.max(np.abs(num - g) / denom) < 1e-5, f"{act} layer{i} {name}"


def test_label_smoothing_validation_and_back_compat():
    """SPEC.md 17: label_smoothing is range-checked; pre-v0.3 JSON still loads."""
    base = VALID.to_dict()
    for bad in (-1e-9, 0.15 + 1e-9):
        d = dict(base, label_smoothing=bad)
        with pytest.raises(SpecError):
            ModelSpec.from_dict(d)
    for good in (0.0, 0.15):  # boundaries accepted
        ModelSpec.from_dict(dict(base, label_smoothing=good))
    # pre-v0.3 spec JSON has no label_smoothing field at all
    old = {k: v for k, v in base.items() if k != "label_smoothing"}
    assert ModelSpec.from_dict(old).label_smoothing == 0.0


def test_label_smoothing_trains_deterministically():
    """SPEC.md 17: smoothing > 0 changes training but stays deterministic."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    sm = dict(VALID.to_dict(), label_smoothing=0.05)
    spec = ModelSpec.from_dict(sm)
    r1 = train(ds, spec, seed=7, n_out=task.n_outputs, head=task.head)
    r2 = train(ds, spec, seed=7, n_out=task.n_outputs, head=task.head)
    for (w1, b1), (w2, b2) in zip(r1.model.layers, r2.model.layers):
        assert np.array_equal(w1, w2)
        assert np.array_equal(b1, b2)
    # plain CE (0.0) and smoothed CE differ in the loss value
    m = r1.model
    l0, _ = m.loss_and_grads(ds[0][:32], ds[1][:32], label_smoothing=0.0)
    l1, _ = m.loss_and_grads(ds[0][:32], ds[1][:32], label_smoothing=0.05)
    assert l1 != l0


def test_gradient_check_with_label_smoothing():
    """SPEC.md 17: smoothed-CE analytic gradients match finite differences."""
    rng = np.random.default_rng(0)
    m = MLP(4, (8,), 2, "tanh", seed=0, head="softmax")
    X = rng.normal(size=(6, 4))
    y = rng.integers(0, 2, 6)
    loss, grads = m.loss_and_grads(X, y, label_smoothing=0.1)
    assert np.isfinite(loss)
    eps = 1e-6
    for i, (w, b) in enumerate(m.layers):
        for name, p, g in (("w", w, grads[i][0]), ("b", b, grads[i][1])):
            num = np.zeros_like(p)
            for idx in np.ndindex(p.shape):
                orig = p[idx]
                p[idx] = orig + eps
                lp, _ = m.loss_and_grads(X, y, label_smoothing=0.1)
                p[idx] = orig - eps
                lm, _ = m.loss_and_grads(X, y, label_smoothing=0.1)
                p[idx] = orig
                num[idx] = (lp - lm) / (2 * eps)
            denom = np.abs(num) + np.abs(g) + 1e-8
            assert np.max(np.abs(num - g) / denom) < 1e-5, f"layer{i} {name}"


def test_loss_decreases_with_sane_hyperparams():
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    spec = ModelSpec(
        architecture=(64, 32), optimizer="adam", learning_rate=5e-3,
        batch_size=64, weight_decay=0.0, train_steps=2000, input_noise=0.0,
        activation="tanh",
    )
    result = train(ds, spec, seed=7)
    assert result.final_loss < 0.1
    assert result.final_loss < np.log(2.0)  # better than a coin flip
