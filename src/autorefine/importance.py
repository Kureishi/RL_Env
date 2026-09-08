"""Permutation feature importance — pure numpy (SPEC.md 44.2, v0.30).

"Which columns does my winning model actually use" (the follow-up to
"which model won"): shuffle each feature column of the holdout rows,
re-score with the trained model, and take the score drop as the
column's importance. One extra pass over the holdout — zero new data,
zero training (44.2.1). The `diagnostics` (28.2) family member:

  * leaf, import-cycle-free — numpy + stdlib only (3.1);
  * the task is duck-typed against the 28.2 `holdout_rows` protocol;
  * it degrades to `None` instead of failing (44.2.3): episode tasks
    (`max_steps > 1`), tasks without `holdout_rows`, a head other than
    softmax/mse, or non-flat features (convnet/grid models).

Deterministic (G2): every shuffle is a
`SeedSequence([seed, zlib.crc32(b"perm-<j>"), repeat])` draw — same
task/model/seed → bit-identical importances.
"""
from __future__ import annotations

import zlib

import numpy as np


def score_xy(model, x: np.ndarray, y: np.ndarray, head: str) -> float:
    """SPEC.md 44.2.1 (v0.30): the head metric on the given rows —
    softmax → 100·accuracy, mse → 100·max(0, R²) — exactly the
    `CsvTask.score` contract (22.1), so the baseline and every shuffle
    are scored by the same metric the task itself reports."""
    n = len(x)
    if n == 0:
        return 0.0
    pred = np.asarray(model.forward(x), dtype=np.float64)
    if head == "softmax":
        acc = (pred.argmax(axis=1) == y).mean()
        return float(100.0 * acc)
    pred = pred.reshape(-1)
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return max(0.0, 100.0 * r2)


def permutation_importance(task, model, n: int = 200, n_repeats: int = 5,
                           seed: int = 7) -> dict | None:
    """SPEC.md 44.2.1 (v0.30): per-column holdout score drop from
    shuffling, averaged over `n_repeats` deterministic shuffles.

    Rows come from the 28.2 `holdout_rows` protocol (clamped to its
    length); `baseline` is the head metric over those rows; for feature
    column j, `importance_j = baseline − score with column j shuffled`
    (higher = the model uses the column more). Names from
    `task.feature_names` (the csv task's file column names) else `f<j>`.

    Returns `{"head", "n", "baseline",
    "features": [{"name", "importance"}]}` sorted by importance
    descending — or `None` when it does not apply (44.2.3)."""
    if int(getattr(task, "max_steps", 1) or 1) > 1:
        return None  # episode task (44.2.3)
    holdout_rows = getattr(task, "holdout_rows", None)
    if not callable(holdout_rows):
        return None
    head = getattr(task, "head", None)
    if head not in ("softmax", "mse"):
        return None
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        n_repeats = 1
    x, y = holdout_rows(int(n), model)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        return None  # non-flat (convnet/grid) or empty (44.2.3)
    d = int(x.shape[1])
    names = getattr(task, "feature_names", None)
    names = [str(c) for c in names] if names and len(names) == d \
        else [f"f{j}" for j in range(d)]
    baseline = score_xy(model, x, y, head)
    features = []
    for j in range(d):
        salt = zlib.crc32(b"perm-" + str(j).encode("ascii"))
        drops = []
        for r in range(n_repeats):
            seq = np.random.SeedSequence([int(seed), salt, r])  # G2
            rng = np.random.default_rng(seq)
            x_pert = np.array(x, copy=True)
            x_pert[:, j] = x[rng.permutation(len(x)), j]
            drops.append(baseline - score_xy(model, x_pert, y, head))
        features.append({"name": names[j],
                         "importance": float(sum(drops) / len(drops))})
    features.sort(key=lambda f: -f["importance"])
    return {"head": head, "n": int(len(x)), "baseline": float(baseline),
            "features": features}
