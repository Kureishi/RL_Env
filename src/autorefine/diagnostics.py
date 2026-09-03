"""Final-model holdout diagnostics (SPEC.md 28.2, C2).

One extra forward pass over the task's holdout split turns a single
"96.4%" into per-class accuracy + a confusion matrix. Classification-only:
episode / mse tasks return `None` (SPEC.md 28.5). Pure NumPy — no new core
dependencies; the task protocol is `holdout_rows(n, model)` (SPEC.md 28.2).
"""
from __future__ import annotations

import numpy as np

from .tasks.media import class_label_str


def holdout_diagnostics(task, model, n: int = 200) -> dict | None:
    """`{"n", "correct", "per_class", "confusion", "class_counts",
    "class_labels"}` for one (task, model), or `None` when diagnostics do
    not apply (SPEC.md 28.2):

    * `task.head != "softmax"` (regression / episode tasks), or
    * the task exposes no `holdout_rows` (cartpole/parity/sine/gridnav), or
    * `task.class_values` is falsy, or
    * the holdout split is empty.

    Invariants: each `confusion[i]` row sums to `class_counts[i]`,
    `correct` = the diagonal sum, `per_class[i] = 100 * confusion[i][i] /
    class_counts[i]` (0.0 for an empty class). Deterministic — one forward
    pass, no RNG (SPEC.md 28.5).
    """
    if getattr(task, "head", None) != "softmax":
        return None
    if not callable(getattr(task, "holdout_rows", None)):
        return None
    values = getattr(task, "class_values", None)
    if not values:
        return None
    k = len(values)
    x, y = task.holdout_rows(n, model)
    if len(x) == 0:
        return None
    y_true = np.asarray(y, dtype=np.int64).ravel()
    pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
    y_pred = np.asarray(pred, dtype=np.int64).ravel()
    # keep indices in-range (defensive; a conforming model is in-range)
    y_true = np.clip(y_true, 0, k - 1)
    y_pred = np.clip(y_pred, 0, k - 1)

    conf = np.zeros((k, k), dtype=np.int64)
    np.add.at(conf, (y_true, y_pred), 1)  # deterministic, correct repeats
    counts = [int(conf[i].sum()) for i in range(k)]
    correct = int(sum(conf[i, i] for i in range(k)))
    per_class = [
        100.0 * float(conf[i, i]) / counts[i] if counts[i] else 0.0
        for i in range(k)
    ]
    return {
        "n": int(len(y_true)),
        "correct": correct,
        "per_class": per_class,
        "confusion": [row.tolist() for row in conf],
        "class_counts": counts,
        "class_labels": [class_label_str(c) for c in values],
    }
