"""ModelSpec field registry (SPEC.md 36.2, v0.22 G2).

One row per ModelSpec field — (name, space, validator, families, spec_ref,
kind) — in catalog order. The bandit proposal space (FIELD_NAMES /
FIELD_SAMPLERS), the search enumerator (SEARCH_FIELDS), the 77-action
RL/Gym catalog (catalog.py), the policy state embedding, and the app's spec
chips all derive from this table, so a new field added to the registry is
visible to every surface at once — the model-space analogue of the C2 knob
registry (SPEC.md 33.2) and the C4 kind registry (SPEC.md 35.1).

Order and values are byte-identical to the v0.21 literals (A24/A26): the
registry is the single source of truth; the surfaces are views.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import (
    ACTIVATIONS,
    BATCH_SIZES,
    CONV_FILTERS,
    EARLY_STOPPING_RANGE,
    GRADIENT_CLIP_RANGE,
    HIDDEN_LAYER_SIZES,
    INPUT_NOISE_RANGE,
    INIT_SCALE_RANGE,
    KNN_K_VALUES,
    LABEL_SMOOTHING_RANGE,
    LEARNING_RATE_RANGE,
    LR_SCHEDULES,
    MODEL_FAMILIES,
    OPTIMIZERS,
    TRAIN_STEPS_RANGE,
    WEIGHT_DECAY_RANGE,
)


def _v_enum(values: tuple):
    """Validator factory: the value must be one of `values` (SPEC.md 36.2)."""
    def _v(value: Any) -> Any:
        if value not in values:
            raise ValueError(f"{value!r} not in {values}")
        return value
    return _v


def _v_range(lo: float, hi: float):
    """Validator factory: a finite number inside [lo, hi] (SPEC.md 36.2)."""
    def _v(value: Any) -> Any:
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{value!r} is not a number") from None
        if not (lo <= x <= hi):
            raise ValueError(f"{value!r} outside [{lo}, {hi}]")
        return x
    return _v


def _v_int_range(lo: int, hi: int):
    """Validator factory: an int inside [lo, hi] (SPEC.md 36.2)."""
    def _v(value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{value!r} is not an int")
        if not (lo <= value <= hi):
            raise ValueError(f"{value!r} outside [{lo}, {hi}]")
        return value
    return _v


def _v_architecture(value: Any) -> tuple:
    """The value must be legal for at least one non-knn family (SPEC.md 36.2):
    an mlp depth-0..3 tuple of allowed widths, a tree/boost depth 1..3, or
    a convnet (c1, c2) filter pair (the A24 legal-architecture rule)."""
    try:
        a = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{value!r} is not an architecture tuple") from None
    if len(a) == 1 and a[0] in (1, 2, 3):
        return a  # tree/boost depth
    if len(a) == 2 and all(c in CONV_FILTERS for c in a):
        return a  # convnet filter pair
    if 0 <= len(a) <= 3 and all(h in HIDDEN_LAYER_SIZES for h in a):
        return a  # mlp (depth 0 = linear)
    raise ValueError(
        f"architecture {value!r} is legal for no family "
        f"(mlp/tree/boost/convnet)")


@dataclass(frozen=True)
class SpecField:
    """One registry row (SPEC.md 36.2).

    `space` is the offered value tuple (always inside the config.py law);
    `validator` accepts every value in `space` and rejects the others;
    `families` are the model families the field affects (`model_family` is
    the only one affecting all five); `spec_ref` is the SPEC.md section
    where the field is defined; `kind` is "sequence" (architecture),
    "ordered" (numeric neighborhood-move fields), or "categorical"
    (local move == uniform resample).
    """
    name: str
    space: tuple
    validator: Callable[[Any], Any]
    families: tuple[str, ...]
    spec_ref: str
    kind: str


_CONVNET_ARCHS = tuple((c1, c2) for c1 in CONV_FILTERS for c2 in CONV_FILTERS)
_ARCH_SPACE = ((16, 8), (32, 16), (64, 32), (16, 32, 16), (1,), (2,), (3,)) \
    + _CONVNET_ARCHS

_NEURAL = ("mlp", "convnet")  # the neural families' shared fields

# Order = the v0.21 FIELD_CATALOG order (byte-identical; A26)
SPEC_FIELDS: dict[str, SpecField] = {
    "architecture": SpecField(
        "architecture", _ARCH_SPACE, _v_architecture,
        ("mlp", "tree", "boost", "convnet"), "5.1", "sequence"),
    "model_family": SpecField(
        "model_family", MODEL_FAMILIES, _v_enum(MODEL_FAMILIES),
        ("mlp", "tree", "boost", "knn", "convnet"), "15", "categorical"),
    "optimizer": SpecField(
        "optimizer", OPTIMIZERS, _v_enum(OPTIMIZERS),
        _NEURAL, "5.1", "categorical"),
    "learning_rate": SpecField(
        "learning_rate", (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        _v_range(*LEARNING_RATE_RANGE), _NEURAL, "5.1", "ordered"),
    "batch_size": SpecField(
        "batch_size", BATCH_SIZES, _v_enum(BATCH_SIZES),
        _NEURAL, "5.1", "ordered"),
    "weight_decay": SpecField(
        "weight_decay", (0.0, 1e-4, 1e-3),
        _v_range(*WEIGHT_DECAY_RANGE), _NEURAL, "5.1", "ordered"),
    "train_steps": SpecField(
        "train_steps", (200, 400, 1000, 2000, 5000),
        _v_int_range(*TRAIN_STEPS_RANGE),
        ("mlp", "tree", "boost", "convnet"), "5.1", "ordered"),
    "input_noise": SpecField(
        "input_noise", (0.0, 0.02, 0.05, 0.1),
        _v_range(*INPUT_NOISE_RANGE),
        ("mlp", "tree", "boost", "convnet"), "5.1", "ordered"),
    "activation": SpecField(
        "activation", ACTIVATIONS, _v_enum(ACTIVATIONS),
        _NEURAL, "5.1", "categorical"),
    "label_smoothing": SpecField(
        "label_smoothing", (0.0, 0.05, 0.1),
        _v_range(*LABEL_SMOOTHING_RANGE), _NEURAL, "17", "ordered"),
    "lr_schedule": SpecField(
        "lr_schedule", LR_SCHEDULES, _v_enum(LR_SCHEDULES),
        _NEURAL, "19.1", "categorical"),
    "early_stopping_patience": SpecField(
        "early_stopping_patience", (0, 10, 25, 50),
        _v_int_range(*EARLY_STOPPING_RANGE), _NEURAL, "19.1", "ordered"),
    "init_scale": SpecField(
        "init_scale", (0.5, 1.0, 2.0),
        _v_range(*INIT_SCALE_RANGE), _NEURAL, "19.1", "ordered"),
    "gradient_clipping": SpecField(
        "gradient_clipping", (0.0, 1.0, 5.0),
        _v_range(*GRADIENT_CLIP_RANGE), _NEURAL, "19.1", "ordered"),
    "knn_k": SpecField(
        "knn_k", KNN_K_VALUES, _v_enum(KNN_K_VALUES),
        ("knn",), "25.2", "categorical"),
}

SPEC_FIELD_NAMES = tuple(SPEC_FIELDS)
