"""Final-model holdout diagnostics (SPEC.md 28.2, C2) + the per-input
difficulty ranking (SPEC.md 58.1, v0.44).

One extra forward pass over the task's holdout split turns a single
"96.4%" into per-class accuracy + a confusion matrix. Classification-only:
episode / mse tasks return `None` (SPEC.md 28.5). Pure NumPy — no new core
dependencies; the task protocol is `holdout_rows(n, model)` (SPEC.md 28.2).
"""
from __future__ import annotations

import numpy as np

from .tasks.media import class_label_str


def holdout_difficulty(task, model, n: int = 200) -> dict | None:
    """SPEC.md 58.1 (v0.44): the per-input difficulty ranking — the
    holdout items, easiest -> hardest, so a niche task's "what is it hard
    on" is a single ordered list (the error-analysis pack, 58.1.1).

    One forward pass over `task.holdout_rows(n, model)` (the 28.2
    protocol; slightly broader than `holdout_diagnostics`, which is
    softmax-only — the mse-head case is served too):

    * **softmax head**: `difficulty` is the SIGNED margin
      `logit[y] - max_{j != y} logit[j]` — positive and large = an easy,
      confident correct call; negative = misclassified (the most negative
      = the hardest). `correct = argmax(logits) == y`.
    * **mse head**: `difficulty = max_j |pred_j - y_j|` (small = easy,
      large = hard); `correct = difficulty <= 0.05 * max(1, max_j |y_j|)`
      (a 5% relative tolerance — informational, the difficulty value is
      the ranking signal).

    Returns `{"n", "head", "items": [{index, true, predicted, correct,
    difficulty}]}` — `items` sorted easiest -> hardest (softmax: margin
    DESCENDING, ties by index; mse: |pred - y| ASCENDING, ties by index).
    `None` when the task exposes no `holdout_rows` (episode tasks:
    cartpole/parity/gridnav/sine — same applicability rule as 28.2) or the
    holdout split is empty. Deterministic — one forward pass, no RNG
    (SPEC.md 28.5); `index` is the position in the holdout split (stable,
    so the ranking composes with the 28.3 error gallery)."""
    if not callable(getattr(task, "holdout_rows", None)):
        return None
    x, y = task.holdout_rows(n, model)
    if len(x) == 0:
        return None
    head = getattr(task, "head", None)
    logits = np.asarray(model.forward(x), dtype=np.float64)
    items: list[dict] = []
    if head == "softmax":
        y_true = np.asarray(y, dtype=np.int64).ravel()
        k = logits.shape[1]
        y_true = np.clip(y_true, 0, k - 1)
        for i in range(len(y_true)):
            t = int(y_true[i])
            row = logits[i]
            others = np.delete(row, t)
            margin = float(row[t] - (others.max() if len(others) else 0.0))
            pred = int(row.argmax())
            items.append({"index": int(i), "true": t, "predicted": pred,
                          "correct": bool(pred == t), "difficulty": margin})
        items.sort(key=lambda it: (-it["difficulty"], it["index"]))
    else:  # mse (or any non-softmax head with holdout rows)
        target = np.asarray(y, dtype=np.float64)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        pred = logits
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        diff = np.abs(pred - target)
        for i in range(len(diff)):
            d = float(diff[i].max()) if diff[i].size else 0.0
            tol = 0.05 * max(1.0, float(np.abs(target[i]).max()))
            items.append({"index": int(i),
                          "true": [float(v) for v in target[i]]
                          if target[i].size > 1 else float(target[i][0]),
                          "predicted": [float(v) for v in pred[i]]
                          if pred[i].size > 1 else float(pred[i][0]),
                          "correct": bool(d <= tol), "difficulty": d})
        items.sort(key=lambda it: (it["difficulty"], it["index"]))
    return {"n": int(len(items)), "head": str(head), "items": items}


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
