"""Discrete mutation catalog shared by the RL improver and the Gym adapter
(SPEC.md 15 "RL improver" / "Gymnasium adapter").

Every action is a (field, value) pair applied to the current best spec, so the
whole improvement loop can be driven by a Discrete action space. `apply_action`
is deterministic and always yields a valid ModelSpec (family/coercion handled
here, mirroring improver/actions.py).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import CONV_FILTERS, HIDDEN_LAYER_SIZES, KNN_K_VALUES, ModelSpec, SpecError
from .specspace import SPEC_FIELDS

# SPEC.md 36.2 (v0.22, G2): the catalog is the registry's name->space view —
# values and order derive from specspace.SPEC_FIELDS (byte-identical to the
# v0.21 literals; A26 pins the derivation). Note (1,)/(2,)/(3,) architecture
# values are valid tree depths while (16, 8) etc. are valid mlp widths —
# apply_action enforces consistency; the 16 (c1, c2) convnet filter pairs
# (SPEC.md 25.3) are part of the registry's architecture space.
FIELD_CATALOG: dict[str, tuple] = {f.name: f.space for f in SPEC_FIELDS.values()}

CATALOG_FIELDS = tuple(SPEC_FIELDS)
ACTIONS: tuple[tuple[str, object], ...] = tuple(
    (field, value) for field in CATALOG_FIELDS for value in FIELD_CATALOG[field]
)


# SPEC.md 18.2: the spec fields that affect each model family's behavior.
# Membership derives from the registry rows (36.2): a field affects a family
# iff the family is in the row's `families`. The neural families (`mlp`,
# `convnet`) expose the full row set (the v0.11 semantics where `knn_k` is
# validated-but-ignored outside knn; convnet trains like mlp, 25.3); the
# family-specific families keep their historical tuple order (A24), with the
# membership itself read from the registry — `tree`/`boost` stay within
# their 4 fields (mutating a field they ignore wastes an experiment:
# fingerprint differs, behavior is identical; SPEC.md 19.2), `knn` only its
# k (SPEC.md 25.2).
_FAMILY_ORDER = {
    "tree": ("architecture", "train_steps", "input_noise", "model_family"),
    "boost": ("architecture", "train_steps", "input_noise", "model_family"),
    "knn": ("knn_k", "model_family"),
}
FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    fam: (CATALOG_FIELDS if fam in ("mlp", "convnet")
          else tuple(f for f in _FAMILY_ORDER[fam]
                     if fam in SPEC_FIELDS[f].families))
    for fam in ("mlp", "tree", "boost", "knn", "convnet")
}


def relevant_families(task_name: str | None) -> tuple[str, ...]:
    """SPEC.md 25.5: the model families the bandit may offer for `task_name`.

    Grid-capable tasks (image, audio) offer all five families, including the
    convnet temporal/spatial model; flat tasks offer the legacy three
    (mlp, tree, boost). Offering only the legacy three on flat tasks keeps the
    §18.7 bit-exact legacy pin green, while knn/convnet remain fully usable
    on flat tasks through the spec surface, `eval`, and the search/RL/gym
    paths (a convnet there is a §25.4 logged rejection, not a crash).
    """
    if task_name is None:
        return ("mlp", "tree", "boost")
    try:
        from ..tasks import TASKS  # lazy: catalog is imported before tasks
    except Exception:  # pragma: no cover - defensive (import order)
        return ("mlp", "tree", "boost")
    cls = TASKS.get(task_name)
    if cls is not None and getattr(cls, "grid_capable", False):
        return ("mlp", "tree", "boost", "knn", "convnet")
    return ("mlp", "tree", "boost")


def relevant_fields(family: str = "mlp") -> tuple[str, ...]:
    """SPEC.md 18.2: the catalog fields that affect `family`'s behavior."""
    return FAMILY_FIELDS.get(family, CATALOG_FIELDS)


def relevant_actions(family: str = "mlp") -> tuple[int, ...]:
    """SPEC.md 18.2: catalog action indices whose field is relevant to `family`."""
    fields = relevant_fields(family)
    return tuple(i for i, (field, _value) in enumerate(ACTIONS) if field in fields)


def index_of_value(field: str, value) -> int | None:
    """Catalog index of the current value of `field` (None if not cataloged).
    Used to one-hot the spec in policy state embeddings."""
    for i, v in enumerate(FIELD_CATALOG[field]):
        if _same(field, v, value):
            return i
    return None


def _same(field: str, a, b) -> bool:
    if field == "architecture":
        return tuple(a) == tuple(b)
    return a == b


def index_of_value_nearest(field: str, value) -> int:
    """SPEC.md 18.1: catalog index of the value closest to `value`.

    `learning_rate` is log-spaced, so distance is measured in log space
    (the neighbor step is then multiplicatively local); all other ordered
    fields use linear distance. Categorical/architecture fields match
    exactly when possible (index 0 otherwise)."""
    values = FIELD_CATALOG[field]
    if field == "architecture":
        for i, v in enumerate(values):
            if tuple(v) == tuple(value):
                return i
        return 0
    if field in ("optimizer", "activation", "model_family", "lr_schedule"):
        for i, v in enumerate(values):
            if v == value:
                return i
        return 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if field == "learning_rate" and v > 0:  # log-spaced catalog
        target = math.log(v)
        return min(range(len(values)),
                   key=lambda i: abs(math.log(float(values[i])) - target))
    return min(range(len(values)), key=lambda i: abs(float(values[i]) - v))


def apply_action(best_spec_dict: dict, action_index: int | np.integer) -> dict:
    """best spec + catalog action -> new valid spec dict (deterministic)."""
    field, value = ACTIONS[int(action_index)]
    out = dict(best_spec_dict)
    out[field] = value
    fam = out.get("model_family", "mlp")
    arch = tuple(out.get("architecture") or ())
    # SPEC.md 19.2: boost shares tree's (depth,) architecture rule
    if fam in ("tree", "boost") and not (len(arch) == 1 and arch[0] in (1, 2, 3)):
        out["architecture"] = (2,)
    elif fam == "knn":
        pass  # SPEC.md 25.2: knn ignores architecture — leave it as-is
    elif fam == "convnet" and not (len(arch) == 2 and all(int(h) in CONV_FILTERS for h in arch)):
        out["architecture"] = (4, 8)  # SPEC.md 25.3: a valid conv filter pair
    elif (not arch or any(h not in HIDDEN_LAYER_SIZES for h in arch)):
        out["architecture"] = (16, 8)
    try:
        ModelSpec.from_dict(out)
        return out
    except SpecError:
        # inconsistent family/architecture pair — force a consistent one
        out["model_family"] = "mlp"
        out["architecture"] = (16, 8)
        ModelSpec.from_dict(out)
        return out
