"""v0.22 generalization — SPEC.md 36, A26:

- 36.1 (G1) Task ABC — `Task` is a nominal ABC (not the former structural
  Protocol) with exactly two abstract methods (`make_dataset`, `score`);
  every registered task lists it as its base and is instantiable with a
  sane identity contract (instance-based — `ParityTask`'s `name`/
  `state_dim` are `n_bits`-derived properties); `metric` is declared
  (not inferred) per task — class defaults for the data tasks, instance
  values set in `__init__` alongside `head` (softmax CSV → `accuracy`,
  mse CSV → `r2`); `capabilities` is a `frozenset` with exactly the
  declared members (`interactive` cartpole/gridnav; `grid`+`media`
  image/audio; empty otherwise); no core source performs
  `isinstance(…, Task)` (the duck-typed-fake guarantee); and the reading
  sites are metric-driven — the app preview caption renders the
  declared metric per data, and the `fit` gate line names it.
- 36.2 (G2) ModelSpec field registry — `specspace.SPEC_FIELDS` is
  exactly the 15 catalog fields (the A24 field set) in catalog order;
  every row has a valid `spec_ref`/`space`/`validator` (accepts every
  space value, rejects an out-of-space value)/`families`/`kind`; and
  the deriving surfaces agree in values — `FIELD_CATALOG`/
  `CATALOG_FIELDS` and the 77-action `ACTIONS` (A8) in catalog
  (registry) order, `FAMILY_FIELDS` (membership from the rows;
  historical tuples), `FIELD_NAMES` (registry membership, v0.21 legacy
  order — the A1–A4 proposal stream), `ORDERED_FIELDS` (the
  `kind == "ordered"` rows), `FIELD_SAMPLERS` (exhaustive over the
  registry, both directions), and `SEARCH_FIELDS` (the registry minus
  `knn_k`, 25.7, legacy order).
- 36.3 (regression) — version stepped to `0.22.0` in both sources
  (33.1) with the round assertion advanced (v0.22 ⇒ `0.22.0`).

House rules: no cross-test imports (fixtures duplicated from
test_data_cli.py / test_media_tasks.py); source scans are read-only
(G2-safe); the source scan is AST-based so docstrings are inert.
"""
import ast
import csv
import importlib.util
import math
import re
import tomllib
import wave
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.config import MODEL_FAMILIES
from autorefine.improver import SPEC_FIELDS, SPEC_FIELD_NAMES, SpecField
from autorefine.improver.actions import (
    FIELD_NAMES,
    FIELD_SAMPLERS,
    LEGACY_FIELD_ORDER,
    ORDERED_FIELDS,
    SEARCH_FIELDS,
)
from autorefine.improver.catalog import (
    ACTIONS,
    CATALOG_FIELDS,
    FAMILY_FIELDS,
    FIELD_CATALOG,
)
from autorefine.tasks import (
    AudioTask,
    CartPoleV1,
    CsvTask,
    GridNavV1,
    ImageTask,
    Parity4V1,
    SineRegressionV1,
    TASKS,
    Task,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "autorefine"

# SPEC.md 36.2: the v0.21 catalog order (A24 field set; A26 pins the
# registry against these historical values).
HISTORICAL_FIELDS = (
    "architecture", "model_family", "optimizer", "learning_rate",
    "batch_size", "weight_decay", "train_steps", "input_noise",
    "activation", "label_smoothing", "lr_schedule",
    "early_stopping_patience", "init_scale", "gradient_clipping", "knn_k",
)
# SPEC.md 19.4: the historical ordered (neighborhood-move) fields.
HISTORICAL_ORDERED = (
    "learning_rate", "batch_size", "weight_decay", "train_steps",
    "input_noise", "label_smoothing", "early_stopping_patience",
    "init_scale", "gradient_clipping",
)
# SPEC.md 19.2: the tree/boost historical 4-field tuples (A24 order).
HISTORICAL_TREE_FIELDS = ("architecture", "train_steps", "input_noise",
                          "model_family")

# SPEC.md 36.1.3: the declared metric set (new metrics such as F1 are a
# task property, not a head string); logloss joined in v0.32 (SPEC.md 46.1).
DECLARED_METRICS = frozenset(
    {"accuracy", "r2", "mean_steps", "success", "logloss"})


# --- fixtures (duplicated; no cross-test imports) ---------------------------

def _cls_csv(path: Path) -> Path:
    """Non-linear 2-D classification (quadrant XOR) → softmax head."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = ((x1 > 0) == (x2 > 0)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "x1", "x2", "churn"])
        for i in range(n):
            w.writerow(["A" if i % 2 else "B", f"{x1[i]:.5f}",
                        f"{x2[i]:.5f}", int(y[i])])
    return path


def _reg_csv(path: Path) -> Path:
    """Near-linear regression (fractional labels → mse head)."""
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = 2.5 * x - 1.0 + 0.05 * rng.standard_normal(n)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y"])
        for i in range(n):
            w.writerow([f"{x[i]:.5f}", f"{y[i]:.5f}"])
    return path


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 1.0,
               amp: float = 0.6, phase: float = 0.0) -> None:
    """16-bit PCM mono WAV (SPEC.md 24.4) — stdlib only."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = amp * np.sin(2 * math.pi * freq * t + phase)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _tone_dir(tmp_path: Path, n: int = 10, seed: int = 0) -> Path:
    """Two tone classes (220 vs 440 Hz), one subfolder per class (24.2)."""
    d = tmp_path / "tones"
    rng = np.random.default_rng(seed)
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _write_wav(sub / f"c{i:02d}.wav", freq,
                       amp=0.5 + 0.3 * rng.random(),
                       phase=rng.random() * 2 * math.pi)
    return d


def _make_image(path: Path, kind: str, size: int = 64) -> None:
    pytest.importorskip("PIL", reason="image fixtures need autorefine[image] (SPEC.md 24.1)")
    from PIL import Image
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    if kind == "ring":
        a = (np.abs(np.hypot(xx - cx, yy - cy) - size * 0.30)
             < size * 0.07).astype(np.uint8) * 255
    else:
        a = (np.minimum(np.abs(xx - cx), np.abs(yy - cy))
             < size * 0.07).astype(np.uint8) * 255
    Image.fromarray(a, mode="L").save(str(path))


def _shape_dir(tmp_path: Path, n: int = 10) -> Path:
    d = tmp_path / "shapes"
    for name, kind in (("ring", "ring"), ("cross", "cross")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _make_image(sub / f"s{i:02d}.png", kind)
    return d


def _pil_available() -> bool:
    return importlib.util.find_spec("PIL") is not None


# --- 36.1 (G1): Task ABC -----------------------------------------------------

def test_task_is_nominal_abc_with_two_abstract_methods():
    """A26 (SPEC.md 36.1.1): Task is a nominal ABC (abc.ABCMeta, not the
    former structural Protocol) with exactly the two abstract methods
    `make_dataset` and `score`; `Task()` is a construction error, not a
    first-step crash."""
    import abc as _abc
    assert type(Task) is _abc.ABCMeta
    assert Task.__abstractmethods__ == frozenset({"make_dataset", "score"})
    with pytest.raises(TypeError):
        Task()


def test_every_task_is_a_task_subclass():
    """A26 (SPEC.md 36.1): every `TASKS` value lists Task as its base —
    the nominal interface the §23 plugin loader targets (frozen
    interface, no duck-typing guess). v0.31 (SPEC.md 45.2): the text
    modality advances the count 7 -> 8 in place."""
    assert len(TASKS) == 8
    for name, cls in TASKS.items():
        assert issubclass(cls, Task), name


def test_tasks_instantiable_with_sane_identity(tmp_path):
    """A26 (SPEC.md 36.1.2): every registered task is instantiable (data
    fixtures for the data-derived tasks) and its instance exposes the
    identity contract with sane values. Instance-based deliberately —
    `ParityTask`'s `name`/`state_dim` are `n_bits`-derived properties,
    and `CsvTask`/`ImageTask`/`AudioTask` set identity per-instance from
    their data."""
    instances = [
        CartPoleV1(seed=7),
        SineRegressionV1(seed=7),
        GridNavV1(seed=7),
        Parity4V1(seed=7),
        CsvTask(seed=7, path=_cls_csv(tmp_path / "cls.csv")),
        CsvTask(seed=7, path=_reg_csv(tmp_path / "reg.csv")),
        AudioTask(seed=7, path=_tone_dir(tmp_path)),
    ]
    if _pil_available():
        instances.append(ImageTask(seed=7, path=_shape_dir(tmp_path)))
    for inst in instances:
        assert isinstance(inst, Task)
        assert isinstance(inst.name, str) and inst.name
        assert isinstance(inst.state_dim, int) and inst.state_dim > 0
        assert isinstance(inst.n_outputs, int) and inst.n_outputs > 0
        assert inst.head in ("softmax", "mse")
        assert isinstance(inst.max_steps, int) and inst.max_steps > 0
        assert (isinstance(inst.default_dataset_size, int)
                and inst.default_dataset_size > 0)


def test_metrics_and_capabilities_declared(tmp_path):
    """A26 (SPEC.md 36.1.3/36.1.4): `metric` is declared per task, not
    inferred from n_outputs/label dtype — class defaults for the data
    tasks (matching their head introspection defaults), instance values
    set in `__init__` alongside `head` (a softmax CSV is `accuracy`, an
    mse CSV is `r2`); `capabilities` is a `frozenset` with exactly the
    declared members."""
    # class-level (registry) metric defaults
    assert CartPoleV1.metric == "mean_steps"
    assert SineRegressionV1.metric == "r2"
    assert GridNavV1.metric == "success"
    assert Parity4V1.metric == "accuracy"
    # data tasks: class defaults match the head introspection defaults
    assert CsvTask.metric == "r2"          # head default "mse" (22.1)
    assert ImageTask.metric == "accuracy"  # head default "softmax" (24)
    assert AudioTask.metric == "accuracy"  # head default "softmax" (24)
    # instance-level (data-derived), set alongside head
    cls = CsvTask(seed=7, path=_cls_csv(tmp_path / "cls.csv"))
    assert cls.head == "softmax" and cls.metric == "accuracy"
    reg = CsvTask(seed=7, path=_reg_csv(tmp_path / "reg.csv"))
    assert reg.head == "mse" and reg.metric == "r2"
    tone = AudioTask(seed=7, path=_tone_dir(tmp_path))
    assert tone.head == "softmax" and tone.metric == "accuracy"
    # capabilities — exactly the declared members, frozenset type
    assert CartPoleV1.capabilities == frozenset({"interactive"})
    assert SineRegressionV1.capabilities == frozenset()
    assert GridNavV1.capabilities == frozenset({"interactive"})
    assert Parity4V1.capabilities == frozenset()
    assert ImageTask.capabilities == frozenset({"grid", "media"})
    assert AudioTask.capabilities == frozenset({"grid", "media"})
    for inst in (cls, reg):  # flat data tasks: no declared capabilities
        assert type(inst.capabilities) is frozenset and not inst.capabilities
    assert type(tone.capabilities) is frozenset
    assert tone.capabilities == frozenset({"grid", "media"})
    # metric semantics: the declared set + head agreement on every task
    for inst in [CartPoleV1(seed=7), SineRegressionV1(seed=7),
                 GridNavV1(seed=7), Parity4V1(seed=7), cls, reg, tone]:
        assert inst.metric in DECLARED_METRICS
        if inst.head == "mse":
            assert inst.metric == "r2"
        else:
            assert inst.metric in ("accuracy", "mean_steps", "success")


def test_no_isinstance_task_in_core_sources():
    """A26 (SPEC.md 36.1): no core source performs `isinstance(…, Task)`
    — the duck-typed-fake guarantee (the fake tasks in the existing test
    files stay valid without listing Task as a base). AST scan:
    docstrings/comments are inert, so base.py's own documentation text
    about the guarantee is allowed."""
    for py in sorted(SRC.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if fname != "isinstance" or len(node.args) < 2:
                continue
            ids = {n.id for n in ast.walk(node.args[1])
                   if isinstance(n, ast.Name)}
            ids |= {n.attr for n in ast.walk(node.args[1])
                    if isinstance(n, ast.Attribute)}
            assert "Task" not in ids, f"{py.name}: isinstance(…, Task)"


def test_app_preview_caption_is_metric_driven(tmp_path):
    """A26 (SPEC.md 36.1.5): the app data-preview caption reads the
    task's declared `metric` — a softmax CSV renders the `accuracy`
    metric text, a regression CSV the `r2` text (previously a
    `head == "softmax"` guess)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional (SPEC.md 23)")
    import autorefine.dashboard_app as app_mod
    app_mod.st.cache_data.clear()  # isolate from any earlier app render
    cls_p = _cls_csv(tmp_path / "cls.csv")
    reg_p = _reg_csv(tmp_path / "reg.csv")
    d1 = app_mod._preview_data(str(cls_p), cls_p.stat().st_mtime_ns,
                               cls_p.stat().st_size)
    d2 = app_mod._preview_data(str(reg_p), reg_p.stat().st_mtime_ns,
                               reg_p.stat().st_size)
    assert "accuracy" in d1["caption"]
    assert "r2" in d2["caption"]


def test_fit_gate_line_names_the_metric():
    """A26 (SPEC.md 36.1.5): the `fit` gate line names the task's
    declared metric — read (`getattr`), not inferred from the head."""
    cli_src = (SRC / "cli.py").read_text(encoding="utf-8")
    assert 'getattr(probe, "metric"' in cli_src
    assert "on {metric}" in cli_src  # the gate line's metric slot


# --- 36.2 (G2): ModelSpec field registry -------------------------------------

def test_registry_is_15_fields_in_catalog_order():
    """A26 (SPEC.md 36.2): the registry is exactly the 15 catalog fields
    (the A24 field set) in catalog order; every value is a SpecField
    row whose `name` matches its key; the names tuple agrees."""
    assert len(SPEC_FIELDS) == 15
    assert tuple(SPEC_FIELDS) == HISTORICAL_FIELDS
    assert SPEC_FIELD_NAMES == tuple(SPEC_FIELDS)
    for name, row in SPEC_FIELDS.items():
        assert isinstance(row, SpecField), name
        assert row.name == name, name


def _out_of_law(row: SpecField):
    """A26 (36.2): a value the row's validator must reject — outside the
    config law (the validator accepts the whole law, of which the
    catalog's `space` is the offered subset; the A24 split). Derived
    from the row itself."""
    if row.kind == "sequence":
        return (999, 999)  # no family legal: not width/depth/filter pair
    if all(isinstance(v, str) for v in row.space):
        return "not-a-registered-value"  # enum: outside the registered set
    return 1e12  # range/int: outside any config law (all laws are << 1e12)


def test_registry_rows_are_valid():
    """A26 (SPEC.md 36.2): every row has a non-empty `spec_ref` matching
    `SPEC.md <n>`, a non-empty `space`, a validator that accepts every
    `space` value and rejects an out-of-law value (the validator's law
    is the config range, of which `space` is the offered subset), a
    non-empty `families` subset of the five model families, and a
    declared `kind` (sequence/ordered/categorical)."""
    for name, row in SPEC_FIELDS.items():
        assert row.spec_ref and re.fullmatch(r"\d+(\.\d+)?", row.spec_ref), name
        assert row.space, name
        for value in row.space:
            row.validator(value)  # must not raise — accepts its space
        with pytest.raises(ValueError):
            row.validator(_out_of_law(row))  # rejects out-of-law
        assert row.families and set(row.families) <= set(MODEL_FAMILIES), name
        assert row.kind in ("sequence", "ordered", "categorical"), name


def test_catalog_is_the_registry_view():
    """A26 (SPEC.md 36.2): `FIELD_CATALOG` is the registry's name→space
    view and `CATALOG_FIELDS` its name order; the `ACTIONS` catalog is
    field-major over that order with exactly 77 actions in the
    historical (A8) order."""
    assert FIELD_CATALOG == {f.name: f.space for f in SPEC_FIELDS.values()}
    assert CATALOG_FIELDS == tuple(SPEC_FIELDS)
    assert len(ACTIONS) == 77
    assert ACTIONS == tuple((field, value)
                            for field in CATALOG_FIELDS
                            for value in FIELD_CATALOG[field])


def test_family_fields_derivable_from_registry():
    """A26 (SPEC.md 36.2): for the family-specific families (tree,
    boost, knn), `FAMILY_FIELDS` membership is exactly the rows that
    declare that family, in historical tuple order (A24, SPEC.md 19.2);
    the neural families expose the full row set (v0.11 semantics:
    `knn_k` validated-but-ignored outside knn, so the derived knn set
    does not apply to them). `model_family` is the only field
    affecting all five families."""
    # family-specific families: membership derives exactly from the rows
    for fam in ("tree", "boost", "knn"):
        derived = {f.name for f in SPEC_FIELDS.values() if fam in f.families}
        assert set(FAMILY_FIELDS[fam]) == derived, fam
    assert FAMILY_FIELDS["tree"] == HISTORICAL_TREE_FIELDS
    assert FAMILY_FIELDS["boost"] == HISTORICAL_TREE_FIELDS  # SPEC.md 19.2
    assert FAMILY_FIELDS["knn"] == ("knn_k", "model_family")  # SPEC.md 25.2
    # neural families: the FULL row set (v0.11 semantics — `knn_k`
    # validated-but-ignored outside knn), not the knn-derived set
    assert FAMILY_FIELDS["mlp"] == CATALOG_FIELDS
    assert FAMILY_FIELDS["convnet"] == CATALOG_FIELDS
    all_five = [f.name for f in SPEC_FIELDS.values()
                if set(f.families) == set(MODEL_FAMILIES)]
    assert all_five == ["model_family"]


def test_bandit_search_views_are_consistent():
    """A26 (SPEC.md 36.2): `set(FIELD_NAMES) == set(SPEC_FIELDS) ==
    set(CATALOG_FIELDS)` — with `FIELD_NAMES` keeping the v0.21 legacy
    (sampler) order, NOT the catalog order: the order is behavioral,
    since `SearchPolicy` draws `rng.choice(SEARCH_FIELDS)` by index and
    it IS the A1–A4 proposal stream (SPEC.md 36.2 item 2); `CATALOG_FIELDS`
    stays in catalog (registry) order. `ORDERED_FIELDS` is exactly the
    `kind == "ordered"` rows in registry order (the historical 9);
    `FIELD_SAMPLERS` is exhaustive over the registry in both directions
    (a registry field with a missing sampler, or a sampler for an
    unregistered field, fails the suite); `SEARCH_FIELDS` keeps the
    `knn_k` exclusion (SPEC.md 25.7) — 14 fields."""
    assert set(FIELD_NAMES) == set(SPEC_FIELDS) == set(CATALOG_FIELDS)
    assert FIELD_NAMES == LEGACY_FIELD_ORDER
    assert ORDERED_FIELDS == tuple(
        f.name for f in SPEC_FIELDS.values() if f.kind == "ordered")
    assert ORDERED_FIELDS == HISTORICAL_ORDERED
    assert set(FIELD_SAMPLERS) == set(SPEC_FIELDS)
    # `SEARCH_FIELDS` keeps the legacy order too (it is the A1–A4
    # proposal stream) minus `knn_k` (SPEC.md 25.7).
    assert SEARCH_FIELDS == tuple(f for f in LEGACY_FIELD_ORDER
                                  if f != "knn_k")
    assert len(SEARCH_FIELDS) == 14  # the documented v0.10 legacy space


# --- 36.3 (regression) -------------------------------------------------------

def test_version_round_v022():
    """A26 (SPEC.md 36.3, 33.1): the version stepped to `0.39.0` in both
    sources with the round (v0.39 ⇒ `0.39.0`, both together — the round
    assertion advanced in place per SPEC.md 33.1)."""
    py = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"
