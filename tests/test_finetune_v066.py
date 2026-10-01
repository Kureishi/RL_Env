"""Fine-tuning — start from a trained model (SPEC.md 80, A70).

SPEC.md 80 (v0.66): a saved ``*.npz`` model is loadable, its spec
reconstructed and validated against the task, and the loop scores it as
the never-retrained baseline while warm-starting compatible candidates
from its weights:

  * 80.2 — ``finetune.py``: family detection by npz key set, spec
    reconstruction through ``ModelSpec.from_dict``, ``validate_for_task``
    (head / n_out / input contract), fail-loud ``ValueError`` semantics,
    pins × upload mutual exclusion;
  * 80.3 — warm start: ``can_warm_start`` (parametric families only),
    ``copy_weights`` (atomic — a mismatch leaves the destination
    untouched), the trainer's ``warm_start`` kwarg;
  * 80.1 — the env's final ``initial_model`` kwarg (baseline row
    ``from_model: true``, ``train_seconds 0.0``), the ``RunConfig``
    field (round-trip, ``--from-model`` only when set), and the
    untouched default path (no upload = no new keys).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import json
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import (AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec,
                        SearchPolicy)
from autorefine.finetune import (PARAMETRIC, can_warm_start, copy_weights,
                                 load_checkpoint, validate_for_task)
from autorefine.models.convnet import ConvNet
from autorefine.models.knn import KNN
from autorefine.models.mlp import MLP
from autorefine.runconfig import RunConfig, fit_recipe, runconfig_to_flags
from autorefine.steering import SteeringState
from autorefine.tasks.parity import Parity4V1

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

# A valid mlp spec with the checkpoint's architecture — the DEFAULT_SPEC
# (16, 8) / tanh baseline, the natural warm-start target of _save_mlp.
_MLP_SPEC = ModelSpec(
    architecture=(16, 8), optimizer="momentum", learning_rate=1e-3,
    batch_size=32, weight_decay=1e-4, train_steps=400, input_noise=0.0,
    activation="tanh", model_family="mlp")


def _save_mlp(path: Path, in_dim=4, hidden=(16, 8), activation="tanh",
              n_out=2, head="softmax") -> Path:
    """A synthetic but format-faithful trained MLP checkpoint (80.2)."""
    m = MLP(in_dim, list(hidden), n_out=n_out, activation=activation,
            seed=7, head=head)
    m.save(str(path))
    return path


def _parity_task() -> Parity4V1:
    return Parity4V1(seed=7)  # head softmax, n_out 2, state_dim 4


def _read_rows(env: AutoRefineEnv) -> list[dict]:
    lines = (env.run_dir / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _mlp_checkpoint() -> "autorefine.finetune.Checkpoint":
    with tempfile.TemporaryDirectory() as td:
        return load_checkpoint(_save_mlp(Path(td) / "m.npz"))


# --- 80.2.2 save/load round-trips (A70) ---------------------------------------

def test_mlp_checkpoint_roundtrip():
    """A70 (80.2.2): a saved mlp loads with its family, spec fields
    (architecture / activation / spectral order), head and output count."""
    with tempfile.TemporaryDirectory() as td:
        p = _save_mlp(Path(td) / "mlp.npz")
        cp = load_checkpoint(p)
        assert cp.family == "mlp"
        assert cp.head == "softmax" and cp.n_out == 2
        assert cp.spec.model_family == "mlp"
        assert tuple(cp.spec.architecture) == (16, 8)
        assert cp.spec.activation == "tanh"
        assert cp.spec.fourier_features == 0
        assert cp.model is not None and len(cp.model.layers) == 3


def test_convnet_checkpoint_roundtrip():
    """A70 (80.2.2): a saved convnet loads with family "convnet" and the
    filter architecture (c1, c2) in the reconstructed spec."""
    with tempfile.TemporaryDirectory() as td:
        net = ConvNet((1, 8, 8), 8, 16, n_out=2, activation="tanh",
                      seed=7, head="softmax")
        p = Path(td) / "conv.npz"
        net.save(str(p))
        cp = load_checkpoint(p)
        assert cp.family == "convnet"
        assert cp.spec.model_family == "convnet"
        assert tuple(cp.spec.architecture) == (8, 16)
        assert cp.spec.activation == "tanh"
        assert cp.n_out == 2


def test_missing_file_raises_value_error():
    """A70 (80.2.4): a missing file is a loud ValueError before anything
    else runs."""
    with pytest.raises(ValueError, match="no such model file"):
        load_checkpoint("/nonexistent/model.npz")


# --- 80.2.3 validate_for_task (A70) -------------------------------------------

def test_validate_accepts_parity_mlp():
    """A70 (80.2.3): a parity-v1-compatible mlp checkpoint is accepted."""
    with tempfile.TemporaryDirectory() as td:
        cp = load_checkpoint(_save_mlp(Path(td) / "ok.npz"))
        ok, reason = validate_for_task(cp, _parity_task())
        assert ok and reason == ""


def test_validate_rejects_head_mismatch():
    """A70 (80.2.3): a regression (mse) head against a classification task
    is rejected with a user-facing reason."""
    with tempfile.TemporaryDirectory() as td:
        cp = load_checkpoint(_save_mlp(Path(td) / "mse.npz", head="mse"))
        ok, reason = validate_for_task(cp, _parity_task())
        assert not ok
        assert "head" in reason.lower()


def test_validate_rejects_output_count_mismatch():
    """A70 (80.2.3): 3 outputs against the task's 2 are rejected."""
    with tempfile.TemporaryDirectory() as td:
        cp = load_checkpoint(_save_mlp(Path(td) / "out3.npz", n_out=3))
        ok, reason = validate_for_task(cp, _parity_task())
        assert not ok
        assert "output" in reason.lower()


def test_validate_rejects_input_width_mismatch():
    """A70 (80.2.3): an input width that is not state_dim·(2K+1) is
    rejected with a dimension reason."""
    with tempfile.TemporaryDirectory() as td:
        cp = load_checkpoint(_save_mlp(Path(td) / "dim6.npz", in_dim=6))
        ok, reason = validate_for_task(cp, _parity_task())
        assert not ok
        assert "dimension" in reason.lower()


# --- 80.3 warm start (A70) ------------------------------------------------------

def test_can_warm_start_identical_spec():
    """A70 (80.3.1): an identical mlp spec is a valid warm-start target."""
    cp = _mlp_checkpoint()
    assert can_warm_start(cp, _MLP_SPEC, n_out=2, head="softmax",
                          in_dim=4) is True
    assert can_warm_start(cp, _MLP_SPEC, n_out=2, head="softmax",
                          in_dim=8) is False  # a known incompatible width


def test_can_warm_start_rejects_activation_and_architecture():
    """A70 (80.3.1): a different activation or hidden architecture is not
    warm-start compatible — the candidate trains fresh."""
    cp = _mlp_checkpoint()
    bad_act = replace(_MLP_SPEC, activation="relu")
    assert can_warm_start(cp, bad_act, n_out=2, head="softmax") is False
    bad_arch = replace(_MLP_SPEC, architecture=(64, 32))
    assert can_warm_start(cp, bad_arch, n_out=2, head="softmax") is False


def test_can_warm_start_rejects_non_parametric_family():
    """A70 (80.3.3): non-parametric families (knn) never warm-start — they
    memorize training rows and have no layer weights to copy — in either
    direction (an mlp seed for a knn spec, a knn seed for an mlp spec)."""
    cp = _mlp_checkpoint()
    knn_spec = replace(_MLP_SPEC, model_family="knn")
    assert can_warm_start(cp, knn_spec, n_out=2, head="softmax") is False
    assert "knn" not in PARAMETRIC

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(0)
        k = KNN(k=5, n_out=2, head="softmax")
        k.fit(rng.random((10, 4)), rng.integers(0, 2, 10))
        p = Path(td) / "knn.npz"
        k.save(str(p))
        kcp = load_checkpoint(p)
        assert can_warm_start(kcp, _MLP_SPEC, n_out=2, head="softmax") is False


def test_copy_weights_success_and_atomic_mismatch():
    """A70 (80.3.2): matching shapes copy; a shape mismatch copies nothing
    and leaves the freshly built destination exactly as built."""
    src = MLP(4, [16, 8], n_out=2, activation="tanh", seed=1, head="softmax")
    dst = MLP(4, [16, 8], n_out=2, activation="tanh", seed=2, head="softmax")
    before = [(w.copy(), b.copy()) for w, b in dst.layers]
    assert not np.array_equal(src.layers[0][0], before[0][0])  # distinct inits
    assert copy_weights(dst, src) is True
    for (dw, db), (sw, sb) in zip(dst.layers, src.layers):
        assert np.array_equal(dw, sw) and np.array_equal(db, sb)

    # mismatch: different hidden width — dst untouched (80.3.2 atomicity)
    dst2 = MLP(4, [32, 8], n_out=2, activation="tanh", seed=3, head="softmax")
    before2 = [(w.copy(), b.copy()) for w, b in dst2.layers]
    assert copy_weights(dst2, src) is False
    for (dw, db), (bw, bb) in zip(dst2.layers, before2):
        assert np.array_equal(dw, bw) and np.array_equal(db, bb)


# --- 80.1.1/80.1.5 env-level (A70) ---------------------------------------------

def _drive(env: AutoRefineEnv, policy: SearchPolicy) -> None:
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


def test_env_run_with_initial_model(tmp_path):
    """A70 (80.1.1): with an uploaded parity-v1 model the run completes;
    the baseline row is from-model (scored, never retrained: train_seconds
    0.0, no loss history) and the summary carries the initial_model key."""
    with tempfile.TemporaryDirectory() as td:
        p = _save_mlp(Path(td) / "seed_model.npz")
        env = AutoRefineEnv(
            task="parity-v1", seed=7,
            budget=Budget(max_experiments=3, max_wall_seconds=120.0,
                          max_train_seconds=15.0),
            runs_dir=tmp_path / "ft", initial_model=p)
        _drive(env, SearchPolicy(seed=7))
        assert env.done
        base = _read_rows(env)[0]
        assert base["kind"] == "baseline"
        assert base["from_model"] is True
        assert base["train_seconds"] == 0.0
        assert base["loss_history"] == []
        assert base["accepted"] is True
        summary = json.loads(
            (env.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["initial_model"]["path"] == str(p)
        assert summary["initial_model"]["family"] == "mlp"
        assert isinstance(summary["initial_model"]["baseline_score"], float)
        assert summary["final_best_score"] >= summary["baseline_score"]


def test_env_default_run_has_no_finetune_keys(tmp_path):
    """A70 (80.1.5): with no upload the legacy key set is intact — no
    from_model on the baseline row, no initial_model in the summary."""
    env = AutoRefineEnv(
        task="parity-v1", seed=7,
        budget=Budget(max_experiments=2, max_wall_seconds=60.0,
                      max_train_seconds=10.0),
        runs_dir=tmp_path / "legacy")
    _drive(env, SearchPolicy(seed=7))
    assert env.done
    assert "from_model" not in _read_rows(env)[0]
    summary = json.loads(
        (env.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "initial_model" not in summary


# --- 80.2.4/80.2.5 failure semantics (A70) --------------------------------------

def test_mismatched_checkpoint_raises(tmp_path):
    """A70 (80.2.4): a task-incompatible model is a construction-time
    ValueError naming the task and the mismatch, before any reset."""
    with tempfile.TemporaryDirectory() as td:
        p = _save_mlp(Path(td) / "mse.npz", head="mse")
        with pytest.raises(ValueError) as exc:
            AutoRefineEnv(task="parity-v1", seed=7, runs_dir=tmp_path / "bad",
                          initial_model=p)
        msg = str(exc.value)
        assert "parity-v1" in msg and "head" in msg


def test_pins_and_upload_are_mutually_exclusive(tmp_path):
    """A70 (80.2.5): steering pins × uploaded model is a construction-time
    ValueError (a pin would rewrite the baseline the model replaces)."""
    with tempfile.TemporaryDirectory() as td:
        p = _save_mlp(Path(td) / "ok.npz")  # a *valid* parity-v1 model
        with pytest.raises(ValueError, match="mutually exclusive"):
            AutoRefineEnv(
                task="parity-v1", seed=7, runs_dir=tmp_path / "pin",
                steering=SteeringState(pins=[("knn_k", 5)]),
                initial_model=p)


# --- 80.1.2 RunConfig (A70) ------------------------------------------------------

def _base_config(**over) -> RunConfig:
    return RunConfig(task="parity-v1", seed=7, **over)


def test_runconfig_roundtrip_initial_model():
    """A70 (80.1.2): initial_model round-trips through to_dict/from_dict;
    an absent key (a pre-v0.66 config) loads as None."""
    cfg = _base_config(initial_model="m.npz")
    d = cfg.to_dict()
    assert d["initial_model"] == "m.npz"
    assert RunConfig.from_dict(d).initial_model == "m.npz"

    legacy = dict(d)
    del legacy["initial_model"]
    assert RunConfig.from_dict(legacy).initial_model is None

    assert _base_config().initial_model is None


def test_recipe_and_flags_emit_from_model_only_when_set():
    """A70 (80.1.2): `fit_recipe` appends `--from-model PATH` and
    `runconfig_to_flags` emits the `from-model` key only when set — a
    fresh run's recipe is unchanged."""
    set_cfg = _base_config(initial_model="m.npz")
    fresh = _base_config()

    recipe = fit_recipe(set_cfg)
    assert "--from-model" in recipe
    assert "m.npz" in recipe

    assert runconfig_to_flags(set_cfg)["from-model"] == "m.npz"
    assert "from-model" not in runconfig_to_flags(fresh)
    assert "--from-model" not in fit_recipe(fresh)


# --- 33.1 version + SPEC index (A70) ---------------------------------------------

def test_version_and_spec_cite_a70():
    """A70 (33.1): the version steps to `0.67.0` in both sources; SPEC §80
    cites A70; the A25 index row for M69 points at this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.67.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(encoding="utf-8")
    assert '"0.67.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 80. Fine-tuning" in spec
    assert "Acceptance (A70)" in spec
    assert ("| M69 | v0.66   | 80     | A70 | tests/test_finetune_v066.py |"
            in spec)
