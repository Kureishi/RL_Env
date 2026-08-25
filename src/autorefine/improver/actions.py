"""Mutation operators over the ModelSpec space (SPEC.md 8, 15).

Each field has a sampler drawn from the field's allowed distribution
(SPEC.md 5.1). Deterministic given the RNG stream.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import (
    ACTIVATIONS,
    BATCH_SIZES,
    HIDDEN_LAYER_SIZES,
    INPUT_NOISE_RANGE,
    LABEL_SMOOTHING_RANGE,
    LEARNING_RATE_RANGE,
    MODEL_FAMILIES,
    OPTIMIZERS,
    TRAIN_STEPS_RANGE,
    WEIGHT_DECAY_RANGE,
)
from .catalog import FIELD_CATALOG, index_of_value_nearest

# SPEC.md 18.1: ordered numeric fields that get a local (neighborhood) move
# (v0.5, SPEC.md 19.4: the three ordered new fields join; lr_schedule stays
# categorical — local == uniform resample there)
ORDERED_FIELDS = (
    "learning_rate", "batch_size", "weight_decay",
    "train_steps", "input_noise", "label_smoothing",
    "early_stopping_patience", "init_scale", "gradient_clipping",
)


def _sample_architecture(rng: np.random.Generator) -> tuple[int, ...]:
    depth = int(rng.integers(1, 4))
    return tuple(int(s) for s in rng.choice(HIDDEN_LAYER_SIZES, size=depth))


def _sample_learning_rate(rng: np.random.Generator) -> float:
    lo, hi = LEARNING_RATE_RANGE
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


FIELD_SAMPLERS = {
    "architecture": _sample_architecture,
    "optimizer": lambda rng: str(rng.choice(OPTIMIZERS)),
    "learning_rate": _sample_learning_rate,
    "batch_size": lambda rng: int(rng.choice(BATCH_SIZES)),
    "weight_decay": lambda rng: round(float(rng.uniform(*WEIGHT_DECAY_RANGE)), 8),
    "train_steps": lambda rng: int(rng.integers(TRAIN_STEPS_RANGE[0], TRAIN_STEPS_RANGE[1] + 1)),
    "input_noise": lambda rng: round(float(rng.uniform(*INPUT_NOISE_RANGE)), 4),
    "activation": lambda rng: str(rng.choice(ACTIVATIONS)),
    "model_family": lambda rng: str(rng.choice(MODEL_FAMILIES)),
    # SPEC.md 17: new spec fields get a sampler here; validation + dedup follow
    "label_smoothing": lambda rng: round(float(rng.uniform(*LABEL_SMOOTHING_RANGE)), 4),
    # SPEC.md 19.1: v0.5 fields sample catalog values (single source of truth
    # in FIELD_CATALOG; guaranteed-different holds on every field)
    "lr_schedule": lambda rng: str(rng.choice(FIELD_CATALOG["lr_schedule"])),
    "early_stopping_patience": lambda rng: int(rng.choice(FIELD_CATALOG["early_stopping_patience"])),
    "init_scale": lambda rng: float(rng.choice(FIELD_CATALOG["init_scale"])),
    "gradient_clipping": lambda rng: float(rng.choice(FIELD_CATALOG["gradient_clipping"])),
}

FIELD_NAMES = tuple(FIELD_SAMPLERS)


def _values_equal(field: str, a, b) -> bool:
    if field == "architecture":
        return tuple(a) == tuple(b)
    return a == b


def _coerce_for_family(spec_dict: dict, rng: np.random.Generator) -> dict:
    """Keep `architecture` valid for the spec's model_family (SPEC.md 15,
    19.2): tree/boost need (depth 1..3); mlp needs 1..3 layers of allowed sizes."""
    fam = spec_dict.get("model_family", "mlp")
    arch = tuple(spec_dict.get("architecture") or ())
    if fam in ("tree", "boost"):
        if not (len(arch) == 1 and arch[0] in (1, 2, 3)):
            spec_dict["architecture"] = (int(rng.integers(1, 4)),)
    else:
        if not arch or any(h not in HIDDEN_LAYER_SIZES for h in arch):
            spec_dict["architecture"] = _sample_architecture(rng)
    return spec_dict


def _uniform_value(field: str, current, rng: np.random.Generator):
    """Guaranteed-different resample over the field's full range (v1 behavior)."""
    for _ in range(8):
        value = FIELD_SAMPLERS[field](rng)
        if not _values_equal(field, current, value):
            break
    return value


def _local_architecture(spec_dict: dict, current, rng: np.random.Generator):
    """SPEC.md 18.1: `architecture` neighborhood move — 50% depth +/-1 (clamped
    1..3; tree depth is the single element), 50% first-hidden-width +/-1 step
    in HIDDEN_LAYER_SIZES (mlp, depth >= 1); None if neither is possible."""
    fam = spec_dict.get("model_family", "mlp")
    arch = tuple(current or ())
    sizes = HIDDEN_LAYER_SIZES
    if fam in ("tree", "boost"):  # SPEC.md 19.2: boost shares tree's depth move
        depth = arch[0] if len(arch) == 1 and arch[0] in (1, 2, 3) else 2
        if depth == 1:
            return (2,)
        if depth == 3:
            return (1,)
        return (depth + 1 if bool(rng.integers(0, 2)) else depth - 1,)
    # mlp; a depth-0 (linear) spec's only neighborhood is one hidden layer
    if not arch:
        return (sizes[0],)
    d = len(arch)
    if d == 1:
        depth_move = arch + (sizes[min(d, len(sizes) - 1)],)
    elif d == 3:
        depth_move = arch[: d - 1]
    else:
        depth_move = (
            arch + (sizes[min(d, len(sizes) - 1)],)
            if bool(rng.integers(0, 2)) else arch[: d - 1]
        )
    i = min(range(len(sizes)), key=lambda k: abs(int(sizes[k]) - int(arch[0])))
    step = 1 if bool(rng.integers(0, 2)) else -1
    j = min(len(sizes) - 1, max(0, i + step))
    width_move = (sizes[j],) + arch[1:]
    candidates = [m for m in (depth_move, width_move)
                  if not _values_equal("architecture", m, arch)]
    if not candidates:
        return None  # caller falls back to uniform (SPEC.md 18.1)
    if len(candidates) == 2:
        return candidates[bool(rng.integers(0, 2))]
    return candidates[0]


def _local_value(field: str, current, spec_dict: dict, rng: np.random.Generator):
    """SPEC.md 18.1: move to a neighborhood of the current value.

    Ordered fields step +/-1 catalog index (anchor = nearest catalog index,
    log-anchored for learning_rate); `architecture` uses its own move;
    categorical fields have local == uniform (no meaningful neighborhood).
    Always guaranteed-different; falls back to uniform where a neighbor
    does not exist."""
    if field in ORDERED_FIELDS:
        values = FIELD_CATALOG[field]
        n = len(values)
        i = index_of_value_nearest(field, current)
        step = 1 if bool(rng.integers(0, 2)) else -1
        for _ in range(2):  # boundary clamp: if tied with current, flip once
            candidate = values[min(n - 1, max(0, i + step))]
            if not _values_equal(field, current, candidate):
                return candidate
            step = -step
        return _uniform_value(field, current, rng)
    if field == "architecture":
        value = _local_architecture(spec_dict, current, rng)
        if value is not None:
            return value
        return _uniform_value(field, current, rng)
    return _uniform_value(field, current, rng)


def mutate_spec_dict(spec_dict: dict, field: str, rng: np.random.Generator,
                     mode: str = "uniform") -> dict:
    """Return a new spec dict with `field` mutated (guaranteed different).

    `mode` (SPEC.md 18.1): "uniform" (default, v0.3 behavior) resamples the
    full range; "local" moves to a neighborhood of the current value."""
    if field not in FIELD_SAMPLERS:
        raise KeyError(f"unknown spec field {field!r}")
    if mode not in ("uniform", "local"):
        raise ValueError(f"unknown mutation mode {mode!r} (expected 'uniform' or 'local')")
    out = dict(spec_dict)
    current = spec_dict.get(field, "mlp" if field == "model_family" else spec_dict[field])
    value = _uniform_value(field, current, rng) if mode == "uniform" \
        else _local_value(field, current, spec_dict, rng)
    out[field] = value
    return _coerce_for_family(out, rng)


def uniform_random_spec(rng: np.random.Generator) -> dict:
    """A fresh point in the whole allowed spec space (evolutionary restart)."""
    out = {field: FIELD_SAMPLERS[field](rng) for field in FIELD_NAMES}
    return _coerce_for_family(out, rng)
