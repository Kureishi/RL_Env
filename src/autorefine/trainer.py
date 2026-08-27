"""Deterministic training of a ModelSpec on a task dataset (SPEC.md 4, 15).

Same (dataset, spec, seed, n_out, head) always yields bit-identical weights
(test T2). The per-training wall-time cap is checked inside the loop, every
256 steps (SPEC.md R4, S2): the trainer aborts cleanly and reports partial
progress.

Model families (SPEC.md 15, 19.2):
  mlp   - NumPy MLP, analytic backprop; architecture () is a linear model.
  tree  - bagged CART ensemble; architecture[0] is the tree depth (1..3) and
          train_steps // 100 is the number of trees (2..50).
  boost - gradient-boosted (residual) CART ensemble (SPEC.md 19.2); same
          surface as tree (depth + rounds), strictly stronger on parity/sine.
The output head is task-defined: "softmax" (classification, n_out classes)
or "mse" (regression, n_out=1).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .config import ModelSpec, SpecError
from .models.convnet import ConvNet
from .models.knn import KNN
from .models.mlp import MLP
from .models.optimizers import make_optimizer
from .models.trees import BoostingEnsemble, TreeEnsemble

Dataset = Tuple[np.ndarray, np.ndarray]  # (features (n, d), targets (n,) or (n, k))
Model = Union[MLP, TreeEnsemble, BoostingEnsemble, KNN, ConvNet]

# SPEC.md 19.1: warmup fraction for the "warmup_cosine" schedule, and the
# internal validation-split fraction used for early stopping (both fixed, so
# the spec surface stays small; deterministic — no RNG).
WARMUP_FRACTION = 0.1
VAL_FRACTION = 0.1


def _scheduled_lr(schedule: str, base_lr: float, t: int, total: int) -> float:
    """Per-step learning rate for the mlp family (SPEC.md 19.1).

    Pure/deterministic (no RNG). "constant" = base_lr (v0.4 behavior);
    "cosine" = cosine decay base_lr -> 0 over `total` steps; "warmup_cosine"
    = linear warmup over the first WARMUP_FRACTION of steps, then cosine decay.
    """
    if schedule == "constant" or total <= 0:
        return base_lr
    frac = min(max(t / total, 0.0), 1.0)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * frac))
    # warmup_cosine
    warmup = max(1, int(round(WARMUP_FRACTION * total)))
    if t <= warmup:
        return base_lr * (t / warmup)
    decay = max(1, total - warmup)
    prog = min(max((t - warmup) / decay, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))


def _clip_grads(grads: list, max_norm: float) -> list:
    """Clip the global L2 norm of the per-step gradients (SPEC.md 19.1).

    Returns new (gw, gb) pairs; a no-op when the norm is within `max_norm`.
    Deterministic (no RNG)."""
    total_sq = 0.0
    for gw, gb in grads:
        total_sq += float((gw * gw).sum()) + float((gb * gb).sum())
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm or total_norm == 0.0:
        return grads
    scale = max_norm / (total_norm + 1e-12)
    return [(gw * scale, gb * scale) for (gw, gb) in grads]


@dataclass
class TrainResult:
    model: Model
    steps_run: int
    final_loss: float
    train_seconds: float
    time_capped: bool


def _time_exceeded(start: float, limit: float | None) -> bool:
    return limit is not None and (time.perf_counter() - start) > limit


def _training_loss(model: Model, X: np.ndarray, y: np.ndarray, head: str) -> float:
    n = X.shape[0]
    out = model.forward(X)
    if head == "mse":
        return float(((out - np.asarray(y, dtype=np.float64).reshape(n, out.shape[1])) ** 2).mean())
    z = out - out.max(axis=1, keepdims=True)
    logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logz[np.arange(n), y].mean())


def _train_neural(
    model: Model, X: np.ndarray, y: np.ndarray, n: int, spec: ModelSpec,
    seed: int, start: float, time_limit_seconds: float | None,
    time_check_every: int, head: str,
) -> TrainResult:
    """Shared neural training loop for the mlp and convnet families.

    SPEC.md 25.3: `ConvNet` exposes the same `.layers` (list of `(w, b)`) +
    `.loss_and_grads(x, y, label_smoothing)` interface as `MLP`, so this loop
    (optimizer, LR schedule, gradient clipping, early stopping, weight decay,
    label smoothing, time cap) is shared verbatim. The mlp path stays
    byte-identical (the T2 pin is untouched).
    """
    opt = make_optimizer(spec.optimizer, spec.learning_rate)
    # per-layer optimizer state: [w_state, b_state]
    opt_states: list[list[dict]] = [[{}, {}] for _ in model.layers]
    schedule = spec.lr_schedule          # SPEC.md 19.1 ("constant" = legacy)
    clip = float(spec.gradient_clipping)  # SPEC.md 19.1 (0.0 = no clipping)
    patience = int(spec.early_stopping_patience)  # SPEC.md 19.1 (0 = off)
    # SPEC.md 19.1: early stopping holds out the LAST VAL_FRACTION of the
    # train split (a deterministic tail split — no RNG). patience == 0 keeps
    # the legacy path byte-identical: no split, no val eval, no restore.
    if patience > 0:
        n_val = min(max(1, int(round(VAL_FRACTION * n))), max(0, n - 1))
        n_tr = n - n_val
        Xv, yv = X[n_tr:], y[n_tr:]
    else:
        n_tr = n
        Xv = yv = None

    rng = np.random.default_rng(seed)
    steps_run = 0
    final_loss = float("inf")
    time_capped = False
    best_val = float("inf")
    best_layers: list[tuple[np.ndarray, np.ndarray]] | None = None
    bad = 0  # consecutive non-improving val steps

    for t in range(1, spec.train_steps + 1):
        idx = rng.integers(0, n_tr, spec.batch_size)
        xb = X[idx]
        if spec.input_noise > 0.0:
            xb = xb + rng.normal(0.0, spec.input_noise, xb.shape)
        yb = y[idx]

        loss, grads = model.loss_and_grads(
            xb, yb, label_smoothing=spec.label_smoothing
        )
        final_loss = loss
        # SPEC.md 19.1: per-step LR schedule ("constant" returns base_lr, so
        # the legacy stream/numerics are unchanged)
        opt.lr = _scheduled_lr(schedule, spec.learning_rate, t, int(spec.train_steps))
        for i, ((gw, gb), (w, _b)) in enumerate(zip(grads, model.layers)):
            gw = gw + spec.weight_decay * w  # weight decay on weights only
            grads[i] = (gw, gb)
        if clip > 0.0:
            grads = _clip_grads(grads, clip)  # SPEC.md 19.1 (no-op at 0.0)
        for i, ((gw, gb), (w, _b)) in enumerate(zip(grads, model.layers)):
            opt.step(w, gw, opt_states[i][0], t)
            opt.step(_b, gb, opt_states[i][1], t)
        steps_run = t

        # SPEC.md 19.1: early stopping on the held-out tail (only when on)
        if patience > 0 and Xv is not None:
            vloss = _training_loss(model, Xv, yv, head)
            if vloss < best_val:
                best_val = vloss
                best_layers = [(w.copy(), b.copy()) for w, b in model.layers]
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if t % time_check_every == 0 and _time_exceeded(start, time_limit_seconds):
            time_capped = True
            break

    # SPEC.md 19.1: restore the best-val weights (classic early stopping)
    if patience > 0 and best_layers is not None:
        model.layers = best_layers

    elapsed = time.perf_counter() - start
    return TrainResult(
        model=model,
        steps_run=steps_run,
        final_loss=final_loss,
        train_seconds=elapsed,
        time_capped=time_capped,
    )


def train(
    dataset: Dataset,
    spec: ModelSpec,
    seed: int,
    time_limit_seconds: float | None = None,
    time_check_every: int = 256,
    n_out: int = 2,
    head: str = "softmax",
) -> TrainResult:
    X, y = dataset
    n = X.shape[0]
    start = time.perf_counter()

    if spec.model_family == "tree":
        # Bagged ensemble: one blocking fit (fast on v1 dataset sizes); the
        # time cap is checked around it rather than inside the growth loop.
        n_trees = max(2, int(spec.train_steps) // 100)
        max_depth = int(spec.architecture[0])
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            model = TreeEnsemble(n_out=n_out, n_trees=0, max_depth=max_depth, head=head, seed=seed)
            return TrainResult(model, 0, float("inf"), 0.0, True)
        model = TreeEnsemble(
            n_out=n_out, n_trees=n_trees, max_depth=max_depth, head=head, seed=seed
        ).fit(X, y, input_noise=spec.input_noise)
        loss = _training_loss(model, X, y, head)
        elapsed = time.perf_counter() - start
        return TrainResult(model, n_trees, loss, elapsed, False)

    if spec.model_family == "boost":
        # Gradient-boosted ensemble (SPEC.md 19.2): same surface as tree —
        # architecture[0] is the per-round depth, train_steps // 100 the
        # number of residual rounds (2..50).
        n_trees = max(2, int(spec.train_steps) // 100)
        max_depth = int(spec.architecture[0])
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            model = BoostingEnsemble(n_out=n_out, n_trees=0, max_depth=max_depth, head=head, seed=seed)
            return TrainResult(model, 0, float("inf"), 0.0, True)
        model = BoostingEnsemble(
            n_out=n_out, n_trees=n_trees, max_depth=max_depth, head=head, seed=seed
        ).fit(X, y, input_noise=spec.input_noise)
        loss = _training_loss(model, X, y, head)
        elapsed = time.perf_counter() - start
        return TrainResult(model, n_trees, loss, elapsed, False)

    if spec.model_family == "knn":
        # Non-parametric family (SPEC.md 25.2): "training" = memorizing the
        # standardized train split; one blocking step (steps_run == 1). The
        # time cap is checked around it, like the tree/boost fits.
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            model = KNN(k=int(spec.knn_k), n_out=n_out, head=head)
            return TrainResult(model, 0, float("inf"), 0.0, True)
        model = KNN(k=int(spec.knn_k), n_out=n_out, head=head).fit(X, y)
        # A k-NN query is O(n * N); on episode tasks the train split is large
        # (tens of thousands of rows), so the reported final loss uses the
        # deterministic first 256 rows (exact when n <= 256, e.g. csv/image/
        # audio fixtures). The memorized model itself is unchanged.
        m = min(n, 256)
        loss = _training_loss(model, X[:m], y[:m], head)
        elapsed = time.perf_counter() - start
        return TrainResult(model, 1, loss, elapsed, False)

    if spec.model_family == "convnet":
        # Grid features (SPEC.md 25.3): a convnet spec on a flat task is a
        # clean SpecError at train time — never a silent fallback.
        if X.ndim != 4:
            raise SpecError(
                "convnet family needs grid data (n, C, H, W); got "
                f"X.shape={X.shape} — offer convnet only for grid_capable "
                "tasks (SPEC.md 25.3)")
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            model = ConvNet(X.shape[1:], int(spec.architecture[0]),
                            int(spec.architecture[1]), n_out, spec.activation,
                            seed, head, spec.init_scale)
            return TrainResult(model, 0, float("inf"), 0.0, True)
        model = ConvNet(
            X.shape[1:], int(spec.architecture[0]), int(spec.architecture[1]),
            n_out, spec.activation, seed, head, spec.init_scale,
        )  # SPEC.md 25.3 (audio temporal model + image spatial model)
        return _train_neural(model, X, y, n, spec, seed, start,
                             time_limit_seconds, time_check_every, head)

    # mlp (the default family): the shared neural training loop (SPEC.md 19.1)
    model = MLP(
        X.shape[1], spec.architecture, n_out=n_out, activation=spec.activation,
        seed=seed, head=head, init_scale=spec.init_scale,  # SPEC.md 19.1
    )
    return _train_neural(model, X, y, n, spec, seed, start,
                         time_limit_seconds, time_check_every, head)


def train_from_task(
    task, spec: ModelSpec, seed: int, time_limit_seconds: float | None = None,
    dataset_episodes: int | None = None,
) -> TrainResult:
    """Convenience: build the train-split dataset, then train (SPEC.md 4).

    SPEC.md 20.3: `dataset_episodes=None` falls back to the task's
    `default_dataset_size` (points for fitting tasks, episodes for episode
    tasks); an explicit int keeps the legacy behavior."""
    size = (int(dataset_episodes) if dataset_episodes is not None
            else int(getattr(task, "default_dataset_size", 60)))
    dataset = task.make_dataset(size)
    return train(
        dataset, spec, seed, time_limit_seconds,
        n_out=task.n_outputs, head=task.head,
    )
