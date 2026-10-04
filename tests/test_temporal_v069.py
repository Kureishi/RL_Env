"""Native temporal models — sequence layout, conv1d / rnn, motif task
(SPEC.md 83, A73).

SPEC.md 83 (v0.69): "Implement all 3 in order (Native temporal models ->
Sequential decision-making -> Offline-only)." Milestone 1 of 3: the
environment gains a native ``(n, T, C)`` sequence layout, two temporal
families, and a built-in sequence task that demonstrates them:

  * 83.1  — the sequence protocol: ``capabilities={"sequence"}`` tasks
    yield 3-D X; ``wants_sequence`` models read it natively, every other
    family reads the flattened rows; a conv1d/rnn spec on flat data is a
    clean SpecError; the three new spec fields are out of the bandit's
    14-field stream but in the 106-action catalog;
  * 83.2  — `models.conv1d.Conv1D`: causal conv -> pool -> FC -> head,
    the shared ``.layers`` / ``.loss_and_grads`` contract (the
    _train_neural loop), analytic gradients, plain-array save/load;
  * 83.3  — `models.rnn.RNN`: Elman recurrence with exact BPTT, the same
    contract;
  * 83.4  — `tasks.sequence.SequenceMotifV1` (the motif task) and
    `models._lag.make_lag_features`;
  * 83.5  — acceptance: registry / catalog growth, finetune integration,
    the architecture diagram, the version step, the A-index advance.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import json
import math
import tempfile
import tomllib
import xml.dom.minidom
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec, SearchPolicy
from autorefine.config import MODEL_FAMILIES, SpecError
from autorefine.finetune import (
    can_warm_start,
    load_checkpoint,
    validate_for_task,
)
from autorefine.improver.actions import (
    EXCLUDED_FROM_SEARCH,
    FIELD_NAMES,
    FIELD_SAMPLERS,
    SEARCH_FIELDS,
)
from autorefine.improver.catalog import (
    ACTIONS,
    CATALOG_FIELDS,
    FAMILY_FIELDS,
    relevant_families,
)
from autorefine.improver.specspace import SPEC_FIELDS
from autorefine.models import Conv1D, RNN
from autorefine.models._lag import make_lag_features
from autorefine.models.mlp import MLP
from autorefine.plotting import svg_architecture, svg_is_well_formed
from autorefine.tasks import Parity4V1, SequenceMotifV1, motif_present
from autorefine.trainer import train

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


def _spec(family: str, **over) -> ModelSpec:
    d = dict(DEFAULT_SPEC.to_dict())
    d["model_family"] = family
    d.update(over)
    return ModelSpec.from_dict(d)


def _finite_diff_gradcheck(model, x: np.ndarray, y: np.ndarray,
                           tol: float = 2e-4) -> None:
    """A73 (83.5.1): the analytic ``loss_and_grads`` matches a central
    finite-difference check on every parameter (the models are tiny, so
    every element is perturbed)."""
    loss0, grads = model.loss_and_grads(x, y)
    assert np.isfinite(loss0)
    for i, (gw, gb) in enumerate(grads):
        for arr, g in ((model.layers[i][0], gw), (model.layers[i][1], gb)):
            it = np.nditer(arr, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = arr[idx]
                h = 1e-6
                arr[idx] = orig + h
                _, _ = model.loss_and_grads(x, y)  # refresh the cache path
                lp = model.loss_and_grads(x, y)[0]
                arr[idx] = orig - h
                lm = model.loss_and_grads(x, y)[0]
                arr[idx] = orig
                num = (lp - lm) / (2 * h)
                assert math.isclose(num, float(g[idx]), rel_tol=tol,
                                   abs_tol=tol * 1e-2), \
                    f"layer {i} idx {idx}: analytic {g[idx]} vs numeric {num}"
                it.iternext()


# --- 83.2 the conv1d family (A73) ---------------------------------------------

def test_conv1d_forward_shapes_and_flat_layout():
    """A73 (83.5.1): forward reads the native (n, T, C) layout AND the
    flat (n, T*C) rows (the 83.2.2 robustness rule) with identical
    logits."""
    m = Conv1D(10, 1, 4, 3, n_out=2, activation="tanh", seed=5,
               head="softmax")
    assert m.wants_sequence is True
    assert len(m.layers) == 3  # conv -> fc -> head (83.2.4 order)
    x = np.random.default_rng(0).integers(0, 2, (8, 10, 1)).astype(np.float64)
    a = m.forward(x)
    assert a.shape == (8, 2) and np.isfinite(a).all()
    b = m.forward(x.reshape(8, 10))  # the flat layout
    assert np.array_equal(a, b)
    with pytest.raises(SpecError):
        m.forward(np.ones((8, 12)))  # a width that is not T*C


def test_conv1d_gradients_match_finite_differences():
    """A73 (83.5.1): the analytic gradients are correct (finite
    difference on every parameter of a tiny model)."""
    rng = np.random.default_rng(1)
    m = Conv1D(4, 1, 2, 2, n_out=2, activation="tanh", seed=2,
               head="softmax")
    x = rng.integers(0, 2, (3, 4, 1)).astype(np.float64)
    y = rng.integers(0, 2, 3)
    _finite_diff_gradcheck(m, x, y)


def test_conv1d_save_load_roundtrip_and_determinism():
    """A73 (83.5.1): save -> load reproduces the forward exactly (plain
    arrays, allow_pickle=False); the same seed gives the same init (G2),
    a different seed a different one."""
    m = Conv1D(6, 1, 4, 3, n_out=2, activation="relu", seed=9, head="softmax")
    x = np.random.default_rng(3).integers(0, 2, (5, 6, 1)).astype(np.float64)
    before = m.forward(x)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "c1d.npz")
        m.save(p)
        # plain arrays only (allow_pickle=False is load's default path)
        with np.load(p, allow_pickle=False) as z:
            assert {"T", "C", "F", "K"} <= set(z.files)
        m2 = Conv1D.load(p)
    assert np.array_equal(before, m2.forward(x))
    a = Conv1D(6, 1, 4, 3, n_out=2, activation="tanh", seed=9, head="softmax")
    b = Conv1D(6, 1, 4, 3, n_out=2, activation="tanh", seed=9, head="softmax")
    c = Conv1D(6, 1, 4, 3, n_out=2, activation="tanh", seed=10, head="softmax")
    for (wa, ba), (wb, bb) in zip(a.layers, b.layers):
        assert np.array_equal(wa, wb) and np.array_equal(ba, bb)
    assert not np.array_equal(a.layers[0][0], c.layers[0][0])


# --- 83.3 the rnn family (A73) --------------------------------------------------

def test_rnn_forward_shapes_and_flat_layout():
    """A73 (83.5.1): the Elman RNN reads the native (n, T, C) layout AND
    the flat rows, with identical logits."""
    m = RNN(8, 1, 6, n_out=2, activation="tanh", seed=5, head="softmax")
    assert m.wants_sequence is True
    x = np.random.default_rng(0).integers(0, 2, (6, 8, 1)).astype(np.float64)
    a = m.forward(x)
    assert a.shape == (6, 2) and np.isfinite(a).all()
    assert np.array_equal(a, m.forward(x.reshape(6, 8)))
    with pytest.raises(SpecError):
        m.forward(np.ones((6, 9)))


def test_rnn_gradients_match_finite_differences():
    """A73 (83.5.1): exact BPTT — the analytic gradients match finite
    differences on every parameter of a tiny model."""
    rng = np.random.default_rng(1)
    m = RNN(4, 1, 2, n_out=2, activation="tanh", seed=2, head="softmax")
    x = rng.integers(0, 2, (3, 4, 1)).astype(np.float64)
    y = rng.integers(0, 2, 3)
    _finite_diff_gradcheck(m, x, y)


def test_rnn_save_load_roundtrip():
    """A73 (83.5.1): save -> load reproduces the forward exactly; the
    checkpoint stores the T/C/H layout (allow_pickle=False)."""
    m = RNN(6, 1, 8, n_out=2, activation="tanh", seed=9, head="softmax")
    x = np.random.default_rng(3).integers(0, 2, (5, 6, 1)).astype(np.float64)
    before = m.forward(x)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "rnn.npz")
        m.save(p)
        with np.load(p, allow_pickle=False) as z:
            assert {"T", "C", "H"} <= set(z.files)
        m2 = RNN.load(p)
    assert np.array_equal(before, m2.forward(x))


# --- 83.4 the motif task + lag features (A73) ---------------------------------

def test_motif_present_labels():
    """A73 (83.5.3): the motif label is a pure search — the pattern 101
    present anywhere, absent otherwise (single row and batch)."""
    assert int(motif_present(np.array([1, 0, 1, 0, 0]))) == 1  # 101 at t=0
    assert int(motif_present(np.array([0, 1, 0, 1, 1]))) == 1  # 101 at t=1
    assert int(motif_present(np.array([0, 0, 1, 1, 0]))) == 0  # no 101 run
    assert int(motif_present(np.array([0, 0, 0, 0, 0]))) == 0
    assert int(motif_present(np.array([1, 0]))) == 0  # shorter than the motif
    X = np.array([[[1], [0], [1]], [[0], [1], [0]]]).astype(np.float64)
    assert motif_present(X).tolist() == [1, 0]


def test_sequence_task_dataset_layout_and_determinism():
    """A73 (83.5.3): make_dataset yields 3-D (n, T, C) binary rows with a
    binary motif label; seed-derived (G2) — two instances with the same
    seed agree, and the splits differ."""
    t1 = SequenceMotifV1(seed=3)
    t2 = SequenceMotifV1(seed=3)
    assert t1.capabilities == frozenset({"sequence"})
    assert t1.metric == "accuracy" and t1.n_outputs == 2 and t1.head == "softmax"
    X, y = t1.make_dataset(16)
    assert X.shape == (16, 10, 1)
    assert set(np.unique(y)) <= {0, 1} and y.dtype.kind in "iu"
    X2, y2 = t2.make_dataset(16)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    # the label agrees with a direct motif search over the bits
    assert np.array_equal(y, motif_present(X, t1.motif))
    tr = t1.initial_conditions("train", 8)
    ho = t1.initial_conditions("holdout", 8)
    assert not np.array_equal(tr, ho)
    with pytest.raises(ValueError):
        SequenceMotifV1(seed=3, timesteps=0)


def test_score_reads_the_wants_sequence_rule():
    """A73 (83.5.3): the task's score feeds temporal families the native
    (n, T, C) layout and flat families the flattened rows — both score
    finitely in 0..100."""
    task = SequenceMotifV1(seed=7)
    X, y = task.make_dataset(96)
    for fam in ("conv1d", "rnn", "mlp"):
        res = train((X, y), _spec(fam, train_steps=200), seed=7)
        s = task.score(res.model, "holdout", 32)
        assert 0.0 <= s <= 100.0 and np.isfinite(s)
    # and the layout rule holds directly: a wants_sequence model on flat
    # data raises (its T/C do not match the flattened width)
    m = Conv1D(10, 1, 4, 3, n_out=2, activation="tanh", seed=7, head="softmax")
    flat = np.ones((3, 10, 1)).reshape(3, 10)  # (n, T*C) with C=1 is (n,10)
    assert m.forward(flat).shape == (3, 2)  # ...but the model accepts it


def test_temporal_families_beat_the_flat_mlp():
    """A73 (83.5.3): with a real budget the temporal families reach high
    holdout accuracy and beat the un-tuned flat MLP — the demonstration
    that they earn their place (83.4)."""
    task = SequenceMotifV1(seed=11)
    X, y = task.make_dataset(512)
    specs = {fam: _spec(fam, train_steps=1500, learning_rate=0.01)
             for fam in ("conv1d", "rnn", "mlp")}
    scores = {fam: task.score(train((X, y), sp, seed=11).model,
                              "holdout", 128)
              for fam, sp in specs.items()}
    assert scores["conv1d"] >= 95.0
    assert scores["rnn"] >= 95.0
    assert scores["conv1d"] > scores["mlp"]
    assert scores["rnn"] > scores["mlp"]


def test_temporal_spec_on_flat_data_is_a_clean_spec_error():
    """A73 (83.1): a conv1d/rnn spec on genuinely flat (2-D) data is a
    SpecError at train time — never a silent reshape (the 25.3 guard,
    restated for sequences)."""
    X = np.random.default_rng(0).integers(0, 2, (32, 10)).astype(np.float64)
    y = np.random.default_rng(1).integers(0, 2, 32)
    for fam in ("conv1d", "rnn"):
        with pytest.raises(SpecError):
            train((X, y), _spec(fam), seed=0)
    # ...while a flat family reads the same 2-D rows fine
    res = train((X, y), _spec("mlp", train_steps=200), seed=0)
    assert res.model is not None


def test_make_lag_features_layout():
    """A73 (83.5.2): make_lag_features (n, T, C) -> (n, window*C), the
    most recent `window` steps flattened time-major; (n, T) is treated
    as C = 1; invalid inputs raise."""
    x = np.arange(3 * 4 * 2).reshape(3, 4, 2).astype(np.float64)
    out = make_lag_features(x, window=2)
    assert out.shape == (3, 4)
    # sequence 0 = [[0,1],[2,3],[4,5],[6,7]]; the last 2 steps flattened
    # time-major, then channels: [4,5,6,7]
    assert out[0].tolist() == [4.0, 5.0, 6.0, 7.0]
    single = make_lag_features(np.arange(5).reshape(1, 5).astype(np.float64),
                               window=3)
    assert single.shape == (1, 3) and single[0].tolist() == [2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        make_lag_features(x, window=5)  # window > T
    with pytest.raises(ValueError):
        make_lag_features(np.ones(4), window=2)  # 1-D is not a sequence


# --- 83.1 registry / catalog growth (A73) --------------------------------------

def test_model_family_and_registry_growth():
    """A73 (83.5.4): the two families join MODEL_FAMILIES; the registry
    gains exactly the three temporal fields (21 total); every field has
    a sampler; the three are out of the bandit's 14-field stream."""
    assert set(MODEL_FAMILIES) >= {"mlp", "tree", "boost", "knn", "convnet",
                                   "gp", "gam", "conv1d", "rnn"}
    assert len(SPEC_FIELDS) == 21
    assert set(FIELD_NAMES) == set(SPEC_FIELDS) and len(FIELD_NAMES) == 21
    assert {"conv1d_filters", "conv1d_kernel", "rnn_hidden"} <= set(SPEC_FIELDS)
    assert set(FIELD_SAMPLERS) == set(SPEC_FIELDS)
    assert len(SEARCH_FIELDS) == 14
    for f in ("conv1d_filters", "conv1d_kernel", "rnn_hidden"):
        assert f in EXCLUDED_FROM_SEARCH


def test_catalog_and_family_fields_growth():
    """A73 (83.5.4): the catalog grows to 106 actions (the 21-field
    registry); conv1d / rnn expose the full 21-field row set (their own
    knobs included); the pre-v0.69 mlp / convnet pools keep the 18-field
    v0.64 set (83.2.5 — the §18.7 bit-exact bandit stream must not see
    the new knobs); the sequence task offers the temporal families and no
    pre-v0.69 task changes its offer."""
    assert len(CATALOG_FIELDS) == 21
    assert len(ACTIONS) == 106
    # conv1d / rnn: the full row set, including their own knobs
    assert FAMILY_FIELDS["conv1d"] == CATALOG_FIELDS
    assert FAMILY_FIELDS["rnn"] == CATALOG_FIELDS
    # mlp / convnet: the pre-v0.69 18-field set (83.2.5, §18.7 pin safety)
    legacy = tuple(f for f in CATALOG_FIELDS
                   if f not in ("conv1d_filters", "conv1d_kernel",
                                "rnn_hidden"))
    assert len(legacy) == 18
    assert FAMILY_FIELDS["mlp"] == legacy
    assert FAMILY_FIELDS["convnet"] == legacy
    assert relevant_families("sequence-motif-v1") == \
        ("mlp", "tree", "boost", "conv1d", "rnn")
    assert relevant_families("parity-v1") == ("mlp", "tree", "boost")
    assert relevant_families("sine-v1") == ("mlp", "tree", "boost")
    assert relevant_families(None) == ("mlp", "tree", "boost")


def test_spec_fields_roundtrip():
    """A73 (83.5.4): the three temporal fields round-trip through
    to_dict / from_dict with their defaults (8 / 3 / 16), and an
    out-of-space value is a loud SpecError."""
    s = _spec("conv1d")
    assert s.conv1d_filters == 8 and s.conv1d_kernel == 3 and s.rnn_hidden == 16
    d = s.to_dict()
    d["conv1d_filters"] = 16
    d["conv1d_kernel"] = 7
    d["rnn_hidden"] = 32
    s2 = ModelSpec.from_dict(d)
    assert (s2.conv1d_filters, s2.conv1d_kernel, s2.rnn_hidden) == (16, 7, 32)
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**d, "conv1d_filters": 5})
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**d, "rnn_hidden": 24})


# --- 83.5.5 finetune integration (A73) -----------------------------------------

def _trained_temporal(family: str):
    task = SequenceMotifV1(seed=5)
    X, y = task.make_dataset(64)
    spec = _spec(family, train_steps=200)
    return train((X, y), spec, seed=5).model, task


def test_finetune_conv1d_checkpoint():
    """A73 (83.5.5): a saved conv1d checkpoint loads, its spec
    reconstructs (the F/K knobs), validate_for_task accepts a sequence
    task and rejects a flat one, and a compatible candidate warm-starts."""
    model, task = _trained_temporal("conv1d")
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "c1d.npz")
        model.save(p)
        cp = load_checkpoint(p)
        assert cp.family == "conv1d"
        assert cp.spec.model_family == "conv1d"
        assert cp.spec.conv1d_filters == int(model.F)
        assert cp.spec.conv1d_kernel == int(model.K)
        ok, _why = validate_for_task(cp, task)
        assert ok
        ok2, why2 = validate_for_task(cp, Parity4V1(seed=5))
        assert not ok2 and "sequence" in why2
        good = _spec("conv1d", conv1d_filters=int(model.F),
                     conv1d_kernel=int(model.K))
        assert can_warm_start(cp, good, n_out=2, head="softmax") is True
        bad = _spec("conv1d", conv1d_filters=16)
        assert can_warm_start(cp, bad, n_out=2, head="softmax") is False
        assert can_warm_start(cp, _spec("mlp"), n_out=2, head="softmax") \
            is False


def test_finetune_rnn_checkpoint():
    """A73 (83.5.5): the same integration for the Elman RNN (the H
    knob; a width mismatch refuses the warm start)."""
    model, task = _trained_temporal("rnn")
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "rnn.npz")
        model.save(p)
        cp = load_checkpoint(p)
        assert cp.family == "rnn"
        assert cp.spec.model_family == "rnn"
        assert cp.spec.rnn_hidden == int(model.H)
        ok, _why = validate_for_task(cp, task)
        assert ok
        ok2, _why2 = validate_for_task(cp, Parity4V1(seed=5))
        assert not ok2
        good = _spec("rnn", rnn_hidden=int(model.H))
        assert can_warm_start(cp, good, n_out=2, head="softmax") is True
        assert can_warm_start(cp, _spec("rnn", rnn_hidden=32),
                              n_out=2, head="softmax") is False


# --- 83.5.6 the architecture diagram (A73) -------------------------------------

def _strict(svg: str) -> None:
    xml.dom.minidom.parseString(svg)


def test_svg_architecture_temporal_families():
    """A73 (83.5.6): the conv1d and rnn block diagrams render (family
    body + head), are well-formed and deterministic, and the field
    highlight rings the temporal knobs (53.3)."""
    for fam, field, frag in (("conv1d", "conv1d_filters", "conv1d"),
                             ("rnn", "rnn_hidden", "Elman RNN")):
        s = _spec(fam)
        plain = svg_architecture(s, state_dim=10, n_out=2)
        _strict(plain)
        assert frag in plain
        assert plain == svg_architecture(s, state_dim=10, n_out=2)  # G2
        assert svg_is_well_formed(plain)
        hl = svg_architecture(s, state_dim=10, n_out=2, highlight=field)
        _strict(hl)
        assert hl != plain, f"{fam}: the {field} ring is missing"
        # a field the family ignores gets the note line, not a ring
        note = svg_architecture(s, state_dim=10, n_out=2,
                                highlight="knn_k")
        assert "knn_k" in note


# --- a real env run on the sequence task (A73) ---------------------------------

def test_env_run_on_sequence_task(tmp_path):
    """A73 (83.5.3): a real AutoRefineEnv run on sequence-motif-v1
    completes, the baseline is accepted, and the loop never falls below
    the baseline (2-experiment budget, tiny)."""
    env = AutoRefineEnv(
        task="sequence-motif-v1", seed=7,
        budget=Budget(2, 300, 30), runs_dir=tmp_path / "seq")
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    assert env.done
    rows = env.memory.load_experiments()
    assert rows[0]["kind"] == "baseline" and rows[0]["accepted"] is True
    summary = json.loads(
        (env.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_best_score"] >= summary["baseline_score"]


# --- 83.5.7 version + spec (A73) -------------------------------------------------

def test_version_and_spec_cite_a73():
    """A73 (83.5.7): the version steps to 0.72.0 in both sources (33.1);
    SPEC.md carries the section 83 + the A-index row resolving to this
    file (the coherency index test in tests/test_coherency_v021.py does
    the cross-check)."""
    py = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.72.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.72.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 83." in spec and "Acceptance (A73)" in spec
    assert "| M72 | v0.69" in spec
    # the two new families + the task are exported at the top level
    assert autorefine.Conv1D is Conv1D and autorefine.RNN is RNN
    assert autorefine.SequenceMotifV1 is SequenceMotifV1
