"""Core data model: ModelSpec (the object being improved) and Budget.

Everything here must stay plain and JSON-serializable (SPEC.md 5.1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

# --- allowed value spaces (SPEC.md 5.1 table) -----------------------------
HIDDEN_LAYER_SIZES = (8, 16, 32, 64, 128)
OPTIMIZERS = ("sgd", "momentum", "adam")
ACTIVATIONS = ("tanh", "relu")
BATCH_SIZES = (16, 32, 64, 128)
# Model families (SPEC.md 15, 19.2): "mlp" is the neural family; an mlp with a
# depth-0 architecture is a *linear model*. "tree" is a bagged decision-tree
# ensemble; "boost" is a gradient-boosted (residual) decision-tree ensemble
# (SPEC.md 19.2 — a genuinely stronger answer than bagging on parity/sine).
MODEL_FAMILIES = ("mlp", "tree", "boost", "knn", "convnet")

# --- SPEC.md 25: modality-aware families (v0.11) ---------------------------
# knn: allowed k values (SPEC.md 25.2); default 5 keeps pre-v0.11 spec JSON
# loadable (the §17 back-compat pattern)
KNN_K_VALUES = (1, 3, 5, 11, 21)
# convnet: per-layer filter counts (SPEC.md 25.3); architecture = (c1, c2)
CONV_FILTERS = (4, 8, 16, 32)

LEARNING_RATE_RANGE = (1e-4, 1e-1)
WEIGHT_DECAY_RANGE = (0.0, 1e-2)
TRAIN_STEPS_RANGE = (200, 5000)
INPUT_NOISE_RANGE = (0.0, 0.1)
# label smoothing for the softmax head (SPEC.md 17; ignored by mse/tree)
LABEL_SMOOTHING_RANGE = (0.0, 0.15)
# depth 0 is allowed: a depth-0 mlp is a linear model (SPEC.md 15)
DEPTH_RANGE = (0, 3)
# --- SPEC.md 19.1: new spec fields (each defaults to the v0.4/legacy behavior)
# LR schedule for the mlp family; "constant" = fixed learning_rate each step
LR_SCHEDULES = ("constant", "cosine", "warmup_cosine")
# multiplicative scale on the Glorot init bound; 1.0 = plain Glorot
INIT_SCALE_RANGE = (0.5, 2.0)
# global L2-norm cap on per-step gradients; 0.0 = no clipping
GRADIENT_CLIP_RANGE = (0.0, 10.0)
# early stopping patience in steps (mlp only); 0 = disabled (train all steps)
EARLY_STOPPING_RANGE = (0, 50)


class SpecError(ValueError):
    """Raised when a ModelSpec value is outside its allowed space."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SpecError(message)


@dataclass(frozen=True)
class ModelSpec:
    """A fully-specified, trainable model configuration.

    This is the unit of improvement: the improver proposes ModelSpecs, the
    trainer consumes them. Validated on construction; invalid specs raise
    SpecError (a ValueError) and are never silently clipped (SPEC.md 5.1).
    """

    architecture: tuple[int, ...]
    optimizer: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    train_steps: int
    input_noise: float
    activation: str
    # Model family (SPEC.md 15). "mlp" (default) with architecture () is a
    # linear model; "tree" is a bagged decision-tree ensemble, where
    # architecture[0] is the tree depth (1..3) and train_steps/100 is the
    # number of trees (2..50).
    model_family: str = "mlp"
    # Label smoothing for the softmax head (SPEC.md 17); 0.0 = plain CE.
    # Defaulted so pre-v0.3 spec JSON still loads (back-compat, SPEC.md 17).
    label_smoothing: float = 0.0
    # --- v0.5 model & spec space (SPEC.md 19.1); all default to legacy ------
    # LR schedule; "constant" = v0.4 behavior (fixed learning_rate each step)
    lr_schedule: str = "constant"
    # early stopping patience (mlp only; 0 = disabled, train all steps)
    early_stopping_patience: int = 0
    # multiplicative scale on the Glorot init bound (1.0 = plain Glorot)
    init_scale: float = 1.0
    # global L2-norm cap on per-step gradients (0.0 = no clipping)
    gradient_clipping: float = 0.0
    # --- SPEC.md 25.2: knn family's k (default 5 keeps pre-v0.11 spec JSON
    # loadable); validated for every family, consumed only by knn.
    knn_k: int = 5

    def __post_init__(self) -> None:
        # normalize list -> tuple even though the class is frozen
        object.__setattr__(self, "architecture", tuple(self.architecture))
        self.validate()

    def validate(self) -> None:
        arch = self.architecture
        _require(self.model_family in MODEL_FAMILIES,
                 f"model_family {self.model_family!r} not in {MODEL_FAMILIES}")
        if self.model_family in ("tree", "boost"):
            # tree/boost: architecture is (depth,) with depth 1..3 (SPEC.md 15, 19.2)
            _require(len(arch) == 1,
                     f"{self.model_family} family needs architecture=(depth,) of length 1")
            _require(arch[0] in (1, 2, 3),
                     f"{self.model_family} depth {arch[0]} outside 1..3")
        elif self.model_family == "convnet":
            # convnet: architecture = (c1, c2), each a conv filter count (25.3)
            _require(len(arch) == 2,
                     "convnet family needs architecture=(c1, c2) of length 2")
            for c in arch:
                _require(int(c) in CONV_FILTERS,
                         f"convnet filter size {c} not in {CONV_FILTERS}")
        elif self.model_family == "knn":
            # knn: architecture is ignored (SPEC.md 25.2) — accept any shape.
            pass
        else:
            # mlp: depth 0..3; depth 0 is a linear model (SPEC.md 15)
            lo, hi = DEPTH_RANGE
            _require(lo <= len(arch) <= hi,
                     f"architecture depth {len(arch)} outside {lo}..{hi}")
            for h in arch:
                _require(int(h) in HIDDEN_LAYER_SIZES,
                         f"hidden layer size {h} not in {HIDDEN_LAYER_SIZES}")
        _require(self.optimizer in OPTIMIZERS,
                 f"optimizer {self.optimizer!r} not in {OPTIMIZERS}")
        lo, hi = LEARNING_RATE_RANGE
        _require(lo <= self.learning_rate <= hi,
                 f"learning_rate {self.learning_rate} outside {lo}..{hi}")
        _require(int(self.batch_size) in BATCH_SIZES,
                 f"batch_size {self.batch_size} not in {BATCH_SIZES}")
        lo, hi = WEIGHT_DECAY_RANGE
        _require(lo <= self.weight_decay <= hi,
                 f"weight_decay {self.weight_decay} outside {lo}..{hi}")
        lo, hi = TRAIN_STEPS_RANGE
        _require(lo <= int(self.train_steps) <= hi,
                 f"train_steps {self.train_steps} outside {lo}..{hi}")
        lo, hi = INPUT_NOISE_RANGE
        _require(lo <= self.input_noise <= hi,
                 f"input_noise {self.input_noise} outside {lo}..{hi}")
        _require(self.activation in ACTIVATIONS,
                 f"activation {self.activation!r} not in {ACTIVATIONS}")
        lo, hi = LABEL_SMOOTHING_RANGE
        _require(lo <= self.label_smoothing <= hi,
                 f"label_smoothing {self.label_smoothing} outside {lo}..{hi}")
        # --- SPEC.md 19.1: new fields (validated for every family; only the
        # mlp family consumes lr_schedule/early_stopping/init_scale/grad-clip)
        _require(self.lr_schedule in LR_SCHEDULES,
                 f"lr_schedule {self.lr_schedule!r} not in {LR_SCHEDULES}")
        lo, hi = EARLY_STOPPING_RANGE
        _require(lo <= int(self.early_stopping_patience) <= hi,
                 f"early_stopping_patience {self.early_stopping_patience} outside {lo}..{hi}")
        lo, hi = INIT_SCALE_RANGE
        _require(lo <= self.init_scale <= hi,
                 f"init_scale {self.init_scale} outside {lo}..{hi}")
        lo, hi = GRADIENT_CLIP_RANGE
        _require(lo <= self.gradient_clipping <= hi,
                 f"gradient_clipping {self.gradient_clipping} outside {lo}..{hi}")
        # SPEC.md 25.2: knn_k is validated for every family (consumed only by knn)
        _require(int(self.knn_k) in KNN_K_VALUES,
                 f"knn_k {self.knn_k} not in {KNN_K_VALUES}")

    # --- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": list(self.architecture),
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "train_steps": self.train_steps,
            "input_noise": self.input_noise,
            "activation": self.activation,
            "model_family": self.model_family,
            "label_smoothing": self.label_smoothing,
            "lr_schedule": self.lr_schedule,
            "early_stopping_patience": self.early_stopping_patience,
            "init_scale": self.init_scale,
            "gradient_clipping": self.gradient_clipping,
            "knn_k": int(self.knn_k),  # SPEC.md 25.2
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            architecture=tuple(int(h) for h in d["architecture"]),
            optimizer=str(d["optimizer"]),
            learning_rate=float(d["learning_rate"]),
            batch_size=int(d["batch_size"]),
            weight_decay=float(d["weight_decay"]),
            train_steps=int(d["train_steps"]),
            input_noise=float(d["input_noise"]),
            activation=str(d["activation"]),
            # default keeps pre-extension spec JSON (SPEC.md 15) loadable
            model_family=str(d.get("model_family", "mlp")),
            # default keeps pre-v0.3 spec JSON loadable (SPEC.md 17)
            label_smoothing=float(d.get("label_smoothing", 0.0)),
            # defaults keep pre-v0.5 spec JSON loadable (SPEC.md 19.1, back-compat)
            lr_schedule=str(d.get("lr_schedule", "constant")),
            early_stopping_patience=int(d.get("early_stopping_patience", 0)),
            init_scale=float(d.get("init_scale", 1.0)),
            gradient_clipping=float(d.get("gradient_clipping", 0.0)),
            # default keeps pre-v0.11 spec JSON loadable (SPEC.md 25.2, back-compat)
            knn_k=int(d.get("knn_k", 5)),
        )

    def fingerprint(self) -> str:
        """Stable short hash of the canonical spec (SPEC.md R3 dedup key)."""
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]

    def diff_fields(self, other: "ModelSpec") -> tuple[str, ...]:
        """Field names whose values differ from `other` (mutation observability)."""
        a, b = self.to_dict(), other.to_dict()
        return tuple(k for k in a if a[k] != b[k])


def spec_n_params(spec, state_dim: int | None, n_out: int | None,
                  grid: tuple | None = None) -> int | None:
    """SPEC.md 58.2 (v0.44): the parameter count of one spec — the size axis
    of the 3-objective frontier.

    `spec` is a `ModelSpec` or its dict form; `state_dim` and `n_out` come
    from the task (the spec JSON does not carry them — the same contract as
    `plotting.svg_architecture`); `grid` is the task's `feature_grid`
    (`(C, H, W)`, e.g. `(1, 8, 8)`) — required for convnets, ignored
    otherwise. Counts weights + biases exactly (the npz value count, the
    `gate.model_size` semantic, SPEC.md 25.5):

    * **mlp**: `sizes = [state_dim, *architecture, n_out]`; each layer
      contributes `fan_in·fan_out + fan_out` (a depth-0 spec is one linear
      layer — still countable).
    * **convnet**: mirrors `models.convnet.ConvNet.__init__` exactly —
      `conv1: c1·C·9 + c1`, `conv2: c2·c1·9 + c2`,
      `fc: flat·FC_HIDDEN + FC_HIDDEN` (with the same `h1 = H−2`,
      `h1p = max(1, h1//2)` — needs `h1p ≥ 3` — `h2p = max(1, (h1p−2)//2)`,
      `flat = c2·h2p·w2p`), `head: FC_HIDDEN·n_out + n_out`.
    * **tree / boost / knn**: `None` — their "size" is data-dependent
      (bagged tree shapes, k at inference) and not a parameter count; the
      58.2 size axis is incomparable for those candidates (documented).

    `None` (not an error) when the inputs are missing, non-finite, or the
    grid is too small for a valid convnet pipeline (the `ConvNet` would
    raise `SpecError` — the count degrades instead). A leaf function: the
    `FC_HIDDEN` constant is lazy-imported inside the function so this
    module never imports `models` at top level (models import config,
    SPEC.md 3 leaf rule)."""
    d = (spec.to_dict() if isinstance(spec, ModelSpec)
         else spec if isinstance(spec, dict) else None)
    if d is None:
        return None
    family = str(d.get("model_family", "mlp"))
    arch = d.get("architecture") or []
    if (not isinstance(state_dim, (int, float)) or isinstance(state_dim, bool)
            or not isinstance(n_out, (int, float)) or isinstance(n_out, bool)
            or state_dim < 1 or n_out < 1):
        return None
    in_dim, n_outputs = int(state_dim), int(n_out)
    if family == "mlp":
        if any(not isinstance(h, (int, float)) or isinstance(h, bool) or h < 1
               for h in arch):
            return None
        sizes = [in_dim, *[int(h) for h in arch], n_outputs]
        return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                       for i in range(len(sizes) - 1)))
    if family == "convnet":
        if len(arch) != 2:
            return None
        c1, c2 = int(arch[0]), int(arch[1])
        if grid is None:
            return None
        try:
            C, H, W = (int(v) for v in grid)
        except (TypeError, ValueError):
            return None
        if C < 1 or H < 3 or W < 3:
            return None
        h1, w1 = H - 2, W - 2
        h1p, w1p = max(1, h1 // 2), max(1, w1 // 2)  # the convnet's _pool_dim
        if h1p < 3 or w1p < 3:
            return None  # the second conv layer needs pooled dims >= 3
        h2p, w2p = max(1, (h1p - 2) // 2), max(1, (w1p - 2) // 2)
        flat = c2 * h2p * w2p
        from .models.convnet import FC_HIDDEN  # lazy: models import config
        return int(c1 * C * 9 + c1
                   + c2 * c1 * 9 + c2
                   + flat * FC_HIDDEN + FC_HIDDEN
                   + FC_HIDDEN * n_outputs + n_outputs)
    return None  # tree / boost / knn: data-dependent size (58.2)


def default_spec() -> ModelSpec:
    """Intentionally un-tuned starting point (SPEC.md 7 baseline target)."""
    return ModelSpec(
        architecture=(16, 8),
        optimizer="momentum",
        learning_rate=1e-3,
        batch_size=32,
        weight_decay=1e-4,
        train_steps=400,
        input_noise=0.02,
        activation="tanh",
    )


DEFAULT_SPEC = default_spec()


@dataclass(frozen=True)
class Budget:
    """Hard resource caps enforced by BudgetManager (SPEC.md 5.2)."""

    max_experiments: int = 30
    max_wall_seconds: float = 900.0
    max_train_seconds: float = 30.0

    def __post_init__(self) -> None:
        _require(self.max_experiments >= 1, "max_experiments must be >= 1")
        _require(self.max_wall_seconds > 0, "max_wall_seconds must be > 0")
        _require(self.max_train_seconds > 0, "max_train_seconds must be > 0")
