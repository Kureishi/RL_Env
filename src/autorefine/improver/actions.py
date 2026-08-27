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
    CONV_FILTERS,
    HIDDEN_LAYER_SIZES,
    INPUT_NOISE_RANGE,
    KNN_K_VALUES,
    LABEL_SMOOTHING_RANGE,
    LEARNING_RATE_RANGE,
    OPTIMIZERS,
    TRAIN_STEPS_RANGE,
    WEIGHT_DECAY_RANGE,
)
from .catalog import (
    FIELD_CATALOG,
    index_of_value_nearest,
    relevant_families,
)

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


# SPEC.md 25.5: every sampler takes an optional `task` name so the
# model_family sampler can restrict itself to the task's relevant families
# (relevant_families). `task=None` (default, and all pre-v0.11 call sites)
# keeps the legacy three-family draw, preserving the §18.7 bit-exact pin.
FIELD_SAMPLERS = {
    "architecture": lambda rng, task=None: _sample_architecture(rng),
    "optimizer": lambda rng, task=None: str(rng.choice(OPTIMIZERS)),
    "learning_rate": lambda rng, task=None: _sample_learning_rate(rng),
    "batch_size": lambda rng, task=None: int(rng.choice(BATCH_SIZES)),
    "weight_decay": lambda rng, task=None: round(float(rng.uniform(*WEIGHT_DECAY_RANGE)), 8),
    "train_steps": lambda rng, task=None: int(rng.integers(TRAIN_STEPS_RANGE[0], TRAIN_STEPS_RANGE[1] + 1)),
    "input_noise": lambda rng, task=None: round(float(rng.uniform(*INPUT_NOISE_RANGE)), 4),
    "activation": lambda rng, task=None: str(rng.choice(ACTIVATIONS)),
    # SPEC.md 25.5: task-aware — grid tasks may draw knn/convnet too
    "model_family": lambda rng, task=None: str(rng.choice(relevant_families(task))),
    # SPEC.md 17: new spec fields get a sampler here; validation + dedup follow
    "label_smoothing": lambda rng, task=None: round(float(rng.uniform(*LABEL_SMOOTHING_RANGE)), 4),
    # SPEC.md 19.1: v0.5 fields sample catalog values (single source of truth
    # in FIELD_CATALOG; guaranteed-different holds on every field)
    "lr_schedule": lambda rng, task=None: str(rng.choice(FIELD_CATALOG["lr_schedule"])),
    "early_stopping_patience": lambda rng, task=None: int(rng.choice(FIELD_CATALOG["early_stopping_patience"])),
    "init_scale": lambda rng, task=None: float(rng.choice(FIELD_CATALOG["init_scale"])),
    "gradient_clipping": lambda rng, task=None: float(rng.choice(FIELD_CATALOG["gradient_clipping"])),
    # SPEC.md 25.2: knn family's k samples the catalog values
    "knn_k": lambda rng, task=None: int(rng.choice(KNN_K_VALUES)),
}

FIELD_NAMES = tuple(FIELD_SAMPLERS)

# SPEC.md 25.7 (full suite green; A1-A4 unchanged): the v1 search policy's
# random-field space stays the v0.10 14 fields, so its seeded proposal
# stream (and the A1-A4 acceptance runs) is bit-stable as the spec space
# grows (the same legacy-stability pattern as SPEC.md 19.4's stable action
# indices). The new `knn_k` field is explored where it belongs: the family
# bandit (FAMILY_FIELDS["knn"]), the catalog (77 actions), and the spec
# surface. `knn_k` remains in FIELD_NAMES/FIELD_SAMPLERS, so
# mutate_spec_dict(spec, "knn_k", ...) works for every policy.
SEARCH_FIELDS = tuple(f for f in FIELD_NAMES if f != "knn_k")


def _values_equal(field: str, a, b) -> bool:
    if field == "architecture":
        return tuple(a) == tuple(b)
    return a == b


def _coerce_for_family(spec_dict: dict, rng: np.random.Generator) -> dict:
    """Keep `architecture` valid for the spec's model_family (SPEC.md 15,
    19.2): tree/boost need (depth 1..3); mlp needs 1..3 layers of allowed
    sizes. SPEC.md 25: knn ignores architecture; convnet needs a valid
    (c1, c2) filter pair."""
    fam = spec_dict.get("model_family", "mlp")
    arch = tuple(spec_dict.get("architecture") or ())
    if fam in ("tree", "boost"):
        if not (len(arch) == 1 and arch[0] in (1, 2, 3)):
            spec_dict["architecture"] = (int(rng.integers(1, 4)),)
    elif fam == "knn":
        pass  # SPEC.md 25.2: knn ignores architecture — leave it as-is
    elif fam == "convnet":
        if not (len(arch) == 2 and all(int(h) in CONV_FILTERS for h in arch)):
            spec_dict["architecture"] = (4, 8)  # SPEC.md 25.3: a valid pair
    else:
        if not arch or any(h not in HIDDEN_LAYER_SIZES for h in arch):
            spec_dict["architecture"] = _sample_architecture(rng)
    return spec_dict


def _uniform_value(field: str, current, rng: np.random.Generator,
                   task: str | None = None):
    """Guaranteed-different resample over the field's full range (v1 behavior).

    `task` (SPEC.md 25.5) is forwarded to the field's sampler; only the
    model_family sampler is task-sensitive (knn/convnet appear only for
    grid-capable tasks)."""
    for _ in range(8):
        value = FIELD_SAMPLERS[field](rng, task)
        if not _values_equal(field, current, value):
            break
    if not _values_equal(field, current, value):
        return value
    # All 8 draws tied with the current value (plausible for small catalogs,
    # e.g. activation has 2). Keep the guaranteed-different contract without
    # consuming more rng: deterministically step to the first catalog value
    # that differs (every FIELD_NAMES field is cataloged). Streams that never
    # hit this case are bit-identical to the legacy 8-draw loop.
    for candidate in FIELD_CATALOG[field]:
        if not _values_equal(field, candidate, current):
            return candidate
    raise ValueError(f"could not resample {field!r} to a different value")  # pragma: no cover


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


def _local_value(field: str, current, spec_dict: dict, rng: np.random.Generator,
                 task: str | None = None):
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
        return _uniform_value(field, current, rng, task)
    if field == "architecture":
        value = _local_architecture(spec_dict, current, rng)
        if value is not None:
            return value
        return _uniform_value(field, current, rng, task)
    return _uniform_value(field, current, rng, task)


def mutate_spec_dict(spec_dict: dict, field: str, rng: np.random.Generator,
                     mode: str = "uniform", task: str | None = None) -> dict:
    """Return a new spec dict with `field` mutated (guaranteed different).

    `mode` (SPEC.md 18.1): "uniform" (default, v0.3 behavior) resamples the
    full range; "local" moves to a neighborhood of the current value.

    `task` (SPEC.md 25.5): task name for task-aware sampling (model_family
    offers knn/convnet only on grid-capable tasks); None = legacy behavior."""
    if field not in FIELD_SAMPLERS:
        raise KeyError(f"unknown spec field {field!r}")
    if mode not in ("uniform", "local"):
        raise ValueError(f"unknown mutation mode {mode!r} (expected 'uniform' or 'local')")
    out = dict(spec_dict)
    current = spec_dict.get(field, "mlp" if field == "model_family" else spec_dict[field])
    value = _uniform_value(field, current, rng, task) if mode == "uniform" \
        else _local_value(field, current, spec_dict, rng, task)
    out[field] = value
    return _coerce_for_family(out, rng)


def uniform_random_spec(rng: np.random.Generator, task: str | None = None) -> dict:
    """A fresh point in the v1 spec space (evolutionary restart).

    Draws the legacy 14 fields (SEARCH_FIELDS, SPEC.md 25.7) so the v1
    restart stream stays v0.10 bit-stable; `knn_k` then defaults to 5 via
    ModelSpec.from_dict. `task` (SPEC.md 25.5): task name for task-aware
    family sampling; None keeps the legacy three-family draw."""
    out = {field: FIELD_SAMPLERS[field](rng, task) for field in SEARCH_FIELDS}
    return _coerce_for_family(out, rng)
