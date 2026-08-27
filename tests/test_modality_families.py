"""v0.11 modality-aware model families (SPEC.md 25, M14, A15).

knn (25.2): the non-parametric family on every task — "training" memorizes
the standardized train split, forward is a k-NN query; deterministic ties;
round-trip save/load with plain arrays only.
convnet (25.3): the small NumPy convnet over grid-structured features —
the image spatial model and the audio *temporal* model (log-mel
spectrogram); deterministic, gradient-sane, round-trippable.
Loop safety (25.4): `step()` rejects invalid specs (convnet on a flat task)
without crashing or spending budget, stays usable, and logs the rejection.
Contracts: the audio flat path stays bit-identical to v0.10 (state_dim 26,
the §24 tests stay green); the image grid is a pure reshape of the flat
features; the §19.3 ensemble contract filter is deterministic.
"""
from pathlib import Path

import numpy as np
import pytest

from autorefine import (
    AutoRefineEnv,
    Budget,
    DEFAULT_SPEC,
    CsvTask,
    ModelSpec,
)
from autorefine.cli import main as cli_main
from autorefine.config import SpecError
from autorefine.improver.meta_env import _EnsembleModel
from autorefine.models.convnet import ConvNet
from autorefine.models.knn import KNN
from autorefine.tasks import AudioTask, ImageTask, log_mel_features, log_mel_frames
from autorefine.trainer import train

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "examples" / "data" / "image_shapes"
SAMPLE_TONE = REPO_ROOT / "examples" / "data" / "tone_clips"


def _cls_csv(path: Path) -> Path:
    """Linearly separable 2-D classification (k-NN friendly)."""
    rng = np.random.default_rng(3)
    n = 100
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = (x1 + x2 > 0).astype(int)
    lines = ["a,b,churn"]
    lines += [f"{x1[i]:.5f},{x2[i]:.5f},{int(y[i])}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _spec(**over) -> ModelSpec:
    base = {**DEFAULT_SPEC.to_dict(), **over}
    return ModelSpec.from_dict(base)


# --- KNN protocol (SPEC.md 25.2) ---------------------------------------------

def test_knn_softmax_determinism_ties_roundtrip(tmp_path):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    a = KNN(k=2, n_out=2, head="softmax").fit(X, y)
    b = KNN(k=2, n_out=2, head="softmax").fit(X, y)
    q = np.array([[0.5, 0.5]])
    assert np.array_equal(a.forward(q), b.forward(q))  # G2 determinism
    # all four points are equidistant from q: the stable tie-break (25.2)
    # keeps dataset order, so the k=2 nearest are rows 0,1 (both class 0)
    out = a.forward(q)[0]
    assert np.argmax(out) == 0
    assert out[0] == 1.0 + 1e-6 and out[1] == 1e-6  # vote fractions + eps
    # a genuine vote tie (k=4, two of each class) resolves to the lower
    # class index (argmax convention, 25.2)
    c = KNN(k=4, n_out=2, head="softmax").fit(X, np.array([0, 1, 0, 1]))
    assert np.argmax(c.forward(q)[0]) == 0
    # round-trip through plain arrays (load enforces allow_pickle=False)
    p = str(tmp_path / "knn.npz")
    a.save(p)
    assert np.array_equal(KNN.load(p).forward(q), a.forward(q))


def test_knn_mse_head_and_validation(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(20, 3))
    y = rng.uniform(size=20)
    # k=1 over the memorized set: the nearest neighbor of a train row is
    # itself → the mean of one target is the target (exact)
    m = KNN(k=1, n_out=1, head="mse").fit(X, y)
    out = m.forward(X)
    assert out.shape == (20, 1)
    assert np.array_equal(out[:, 0], y)
    # k=3 round-trip (deterministic mean of the k nearest targets)
    m3 = KNN(k=3, n_out=1, head="mse").fit(X, y)
    q = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    p = str(tmp_path / "knn_mse.npz")
    m3.save(p)
    assert np.array_equal(KNN.load(p).forward(q), m3.forward(q))
    with pytest.raises(SpecError):
        KNN(k=0, n_out=1, head="mse")
    with pytest.raises(SpecError):
        KNN(k=1, n_out=1, head="bogus")


# --- ConvNet protocol (SPEC.md 25.3) ------------------------------------------

def test_convnet_determinism_roundtrip_flat_robustness(tmp_path):
    a = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 3, "softmax")
    b = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 3, "softmax")
    for (wa, ba), (wb, bb) in zip(a.layers, b.layers):
        assert np.array_equal(wa, wb) and np.array_equal(ba, bb)  # G2
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 1, 8, 8))
    assert np.array_equal(a.forward(x), b.forward(x))
    # forward also accepts flat (n, C*H*W) input (25.3 robustness)
    assert np.array_equal(a.forward(x.reshape(2, 64)), a.forward(x))
    p = str(tmp_path / "convnet.npz")
    a.save(p)
    al = ConvNet.load(p)  # load enforces allow_pickle=False
    assert np.array_equal(al.forward(x), a.forward(x))


def test_convnet_gradient_sanity_finite_difference():
    """Analytic head gradients match per-element central finite differences
    (SPEC.md 25.7 gradient sanity); the loss is finite and sane."""
    m = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 3, "softmax")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 1, 8, 8))
    y = np.array([0, 1])
    loss0, grads = m.loss_and_grads(x, y, 0.0)
    assert np.isfinite(loss0) and 0.0 < loss0 < 5.0
    gw, gb = grads[-1]
    assert np.all(np.isfinite(gw)) and np.all(np.isfinite(gb))
    e = 1e-6
    w, b = m.layers[-1]
    saved = m.layers[-1]
    try:
        num_w = np.zeros_like(gw)
        for i in range(gw.shape[0]):
            for j in range(gw.shape[1]):
                pert = np.zeros_like(w)
                pert[i, j] = e
                m.layers[-1] = (w + pert, b)
                lp = m.loss_and_grads(x, y, 0.0)[0]
                m.layers[-1] = (w - pert, b)
                lm = m.loss_and_grads(x, y, 0.0)[0]
                num_w[i, j] = (lp - lm) / (2 * e)
        num_b = np.zeros_like(gb)
        for j in range(gb.shape[0]):
            pert = np.zeros_like(b)
            pert[j] = e
            m.layers[-1] = (w, b + pert)
            lp = m.loss_and_grads(x, y, 0.0)[0]
            m.layers[-1] = (w, b - pert)
            lm = m.loss_and_grads(x, y, 0.0)[0]
            num_b[j] = (lp - lm) / (2 * e)
    finally:
        m.layers[-1] = saved
    assert np.allclose(gw, num_w, atol=1e-5)
    assert np.allclose(gb, num_b, atol=1e-5)


# --- Integration: trainer branches (SPEC.md 25.7) -----------------------------

def test_knn_branch_on_csv_fixture(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    t = CsvTask(seed=7, path=str(p), label="churn")
    x, y = t.make_dataset()
    res = train((x, y), _spec(model_family="knn", knn_k=5), 7,
                n_out=t.n_outputs, head=t.head)
    assert isinstance(res.model, KNN)
    assert res.steps_run == 1  # one blocking step (25.2)
    assert t.score(res.model, "holdout", 5) >= 80.0


def test_convnet_branch_on_image_grid_fixture():
    im = ImageTask(seed=7, path=str(SAMPLE_IMAGE))
    gx, gy = im.grid_dataset(None)
    assert gx.ndim == 4 and gx.shape[1:] == im.feature_grid  # (n, C, H, W)
    res = train((gx, gy), _spec(model_family="convnet", architecture=[4, 8]), 7,
                n_out=im.n_outputs, head=im.head)
    assert isinstance(res.model, ConvNet)
    assert im.score(res.model, "holdout", 4) >= 80.0  # A15 gate


def test_convnet_branch_on_audio_grid_fixture():
    a = AudioTask(seed=7, path=str(SAMPLE_TONE))
    gx, gy = a.grid_dataset(None)
    assert gx.ndim == 4 and gx.shape[1:] == a.feature_grid  # (n, 1, T, bands)
    res = train((gx, gy), _spec(model_family="convnet", architecture=[4, 8]), 7,
                n_out=a.n_outputs, head=a.head)
    assert isinstance(res.model, ConvNet)
    assert a.score(res.model, "holdout", 4) >= 80.0  # A15 gate (tone sample)


def test_convnet_on_flat_task_is_clean_spec_error(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    t = CsvTask(seed=7, path=str(p), label="churn")
    x, y = t.make_dataset()
    with pytest.raises(SpecError, match="grid"):
        train((x, y), _spec(model_family="convnet", architecture=[4, 8]), 7,
              n_out=t.n_outputs, head=t.head)


# --- Loop safety: step() invalid-spec rejection (SPEC.md 25.4) ----------------

def test_step_rejects_invalid_spec_budget_intact_env_usable(tmp_path):
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(3, 300, 30),
        runs_dir=tmp_path / "runs",
    )
    state = env.reset()
    used_before = env.bm.used_experiments
    conv = {**state["best_spec"], "model_family": "convnet", "architecture": [4, 8]}
    state, _r, _d, info = env.step(conv)
    assert info["accepted"] is False
    assert info["reason"] == "invalid_spec"
    assert info["candidate_score"] is None
    assert env.bm.used_experiments == used_before  # no budget spent (25.4)
    # not added to the dedup set: resubmitting is rejected again as
    # invalid_spec, not as a free duplicate
    state, _r, _d, info = env.step(conv)
    assert info["reason"] == "invalid_spec"
    assert env.bm.used_experiments == used_before
    # the env stays usable: a valid spec trains and scores normally
    state, _r, _d, info = env.step({**state["best_spec"], "learning_rate": 3e-4})
    assert info["accepted"] is not None
    assert isinstance(info["candidate_score"], (int, float))
    # and the rejection is in the experiment log
    entries = env.memory.load_experiments()
    assert any(e.get("kind") == "invalid_spec" for e in entries)


# --- Contracts (SPEC.md 25.1/25.3/19.3) ---------------------------------------

def test_audio_flat_path_bit_identical_to_v010():
    """state_dim stays 26 and the flat feature is exactly the v0.10
    time-mean of the spectrogram frames (25.1; §24 tests stay green)."""
    import math
    sr = 16000
    sig = 0.5 * np.sin(2 * math.pi * 440.0 * np.arange(sr) / sr)
    assert np.array_equal(log_mel_features(sig, sr),
                          log_mel_frames(sig, sr).mean(axis=0))
    a = AudioTask(seed=7, path=str(SAMPLE_TONE))
    assert a.state_dim == 26
    x, _ = a.make_dataset()
    assert x.shape[1] == 26


def test_image_grid_is_pure_reshape_of_flat():
    im = ImageTask(seed=7, path=str(SAMPLE_IMAGE))
    x, y = im.make_dataset()
    gx, gy = im.grid_dataset(None)
    assert gx.shape[1:] == im.feature_grid
    assert np.array_equal(gx.reshape(x.shape[0], -1), x)  # flat stays exact
    assert np.array_equal(gy, y)


def test_ensemble_contract_filter_deterministic():
    """The §19.3 ensemble keeps only members sharing the top member's input
    contract (grid vs flat); the result is deterministic (SPEC.md 25.3)."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(12, 4))
    y = (X[:, 0] > 0).astype(int)
    flat_a = KNN(k=3, n_out=2, head="softmax").fit(X, y)
    flat_b = KNN(k=5, n_out=2, head="softmax").fit(X, y)
    grid = ConvNet((1, 8, 8), 4, 4, 2, "tanh", 3, "softmax")
    xg = rng.normal(size=(2, 1, 8, 8))
    xf = X[:2]
    # top member is the grid model → the flats are filtered out
    e = _EnsembleModel([grid, flat_a, flat_b])
    assert e.wants_grid is True and e.members == [grid]
    assert np.array_equal(e.forward(xg), grid.forward(xg))
    # top member is flat → the grid is filtered out, flats average
    f = _EnsembleModel([flat_a, flat_b, grid])
    assert f.wants_grid is False and f.members == [flat_a, flat_b]
    expected = (flat_a.forward(xf) + flat_b.forward(xf)) / 2
    assert np.array_equal(f.forward(xf), expected)
    # deterministic: the same construction gives the same forward
    f2 = _EnsembleModel([flat_a, flat_b, grid])
    assert np.array_equal(f2.forward(xf), f.forward(xf))


# --- End-to-end (SPEC.md 25.7) -------------------------------------------------

def test_fit_cli_bundled_image_sample_passes_gate(tmp_path, capsys):
    rc = cli_main(["fit", "--data", str(SAMPLE_IMAGE), "--seed", "7",
                   "--experiments", "4", "--max-train-seconds", "10",
                   "--target", "90", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0  # the §22.1 gate: PASS on the bundled image sample (A15)
    assert "PASS" in out


def test_improver_trains_knn_and_convnet_on_tone(tmp_path):
    """A15: the improver trains a knn spec to ≥ 90 on the tone sample, and
    a convnet spec (the audio temporal model) reaches ≥ 80 there."""
    env = AutoRefineEnv(
        task="audio", seed=7, budget=Budget(4, 300, 60),
        runs_dir=tmp_path / "runs",
        task_config={"path": str(SAMPLE_TONE)},
        dataset_episodes=32,  # 80% of 40 bundled clips (SPEC.md 20.3)
    )
    state = env.reset()
    # The gate is on the score the improver trains each spec to (the
    # candidate_score), independent of acceptance (a tied best scores 100
    # on this easy sample, so acceptance can be a no-delta tie).
    state, _r, _d, info = env.step(
        {**state["best_spec"], "model_family": "knn", "knn_k": 5})
    assert isinstance(info["candidate_score"], (int, float))
    assert info["candidate_score"] >= 90.0
    state, _r, _d, info = env.step(
        {**state["best_spec"], "model_family": "convnet", "architecture": [4, 8]})
    assert isinstance(info["candidate_score"], (int, float))
    assert info["candidate_score"] >= 80.0
