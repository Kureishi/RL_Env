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

# values per field; note (1,)/(2,)/(3,) architectures are valid tree depths
# while (16, 8) etc. are valid mlp widths — apply_action enforces consistency.
# SPEC.md 25.3: the 16 (c1, c2) convnet filter pairs are appended to the
# architecture values (c1, c2 each in CONV_FILTERS); the mlp/tree values are
# unchanged and keep their relative order.
_CONVNET_ARCHS = tuple((c1, c2) for c1 in CONV_FILTERS for c2 in CONV_FILTERS)
FIELD_CATALOG: dict[str, tuple] = {
    "architecture": ((16, 8), (32, 16), (64, 32), (16, 32, 16), (1,), (2,), (3,)) + _CONVNET_ARCHS,
    "model_family": ("mlp", "tree", "boost", "knn", "convnet"),  # SPEC.md 25.2/25.3
    "optimizer": ("sgd", "momentum", "adam"),
    "learning_rate": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    "batch_size": (16, 32, 64, 128),
    "weight_decay": (0.0, 1e-4, 1e-3),
    "train_steps": (200, 400, 1000, 2000, 5000),
    "input_noise": (0.0, 0.02, 0.05, 0.1),
    "activation": ("tanh", "relu"),
    "label_smoothing": (0.0, 0.05, 0.1),  # SPEC.md 17
    # --- v0.5 model & spec space (SPEC.md 19.1); appended so the 40 old
    # action indices stay stable (SPEC.md 19.4) ---
    "lr_schedule": ("constant", "cosine", "warmup_cosine"),
    "early_stopping_patience": (0, 10, 25, 50),
    "init_scale": (0.5, 1.0, 2.0),
    "gradient_clipping": (0.0, 1.0, 5.0),
    # SPEC.md 25.2: knn family's k (appended; consumed only by the knn family)
    "knn_k": KNN_K_VALUES,
}

CATALOG_FIELDS = tuple(FIELD_CATALOG)
ACTIONS: tuple[tuple[str, object], ...] = tuple(
    (field, value) for field in CATALOG_FIELDS for value in FIELD_CATALOG[field]
)

# SPEC.md 18.2: the spec fields that affect each model family's behavior.
# `tree`/`boost` ignore optimizer/lr/batch/weight-decay/activation/
# label-smoothing and the v0.5 mlp fields (mutating them wastes an experiment:
# fingerprint differs, behavior is identical), so proposals for a tree/boost
# best spec stay within their 4 fields (SPEC.md 19.2: boost mirrors tree).
FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    "mlp": CATALOG_FIELDS,
    "tree": ("architecture", "train_steps", "input_noise", "model_family"),
    "boost": ("architecture", "train_steps", "input_noise", "model_family"),
    # SPEC.md 25.2: knn only exposes its k (architecture is ignored)
    "knn": ("knn_k", "model_family"),
    # SPEC.md 25.3: convnet trains like mlp (shared neural loop)
    "convnet": CATALOG_FIELDS,
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
