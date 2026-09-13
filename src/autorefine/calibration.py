"""Calibration / constraint metrics (SPEC.md 61.1.3 / 61.4, v0.47).

Pure numpy, deterministic (G2), no new core dependencies. These are the
post-hoc metrics behind A1's constraint actuals and A4's calibration
objective — they are scored *over data the model already produced on the
holdout*, never by retraining:

- ``per_class_f1`` (61.1.3) — the *worst non-empty class's* F1 from a
  confusion matrix (the "fairness" reading: the minority class is not
  sacrificed);
- ``ece`` (61.4) — expected calibration error of softmax probabilities
  (a perfectly calibrated model → ≈ 0; confident-wrong → high);
- ``coverage_width`` (61.4.1) — interval coverage, the natural objective
  for an interval predictor;
- ``monotonicity_slope`` (61.1.3) — a central finite-difference slope sign
  estimate for the ``monotonic_in`` constraint;
- ``ensemble_ece`` (61.4.2) — the ensemble-spread predictor: average the
  top-k frontier models' holdout probabilities before scoring ECE.
"""
from __future__ import annotations

import math

import numpy as np


def per_class_f1(confusion) -> float:
    """SPEC.md 61.1.3 (A1): the **minimum** F1 over the *non-empty*
    classes of a confusion matrix — the "worst class" fairness reading.

    ``confusion`` is a ``(k, k)`` array or list-of-lists where
    ``confusion[i][j]`` = the count of true-``i`` predicted-``j`` (the
    same layout as ``diagnostics.holdout_diagnostics["confusion"]``). For
    class ``i``: ``tp = C[i][i]``, ``fp = col_i − tp``, ``fn = row_i − tp``;
    ``F1 = 2PR / (P+R)`` with the standard 0.0 convention when the
    precision/recall denominator is 0. A class that never appears
    (``row_i == 0``) is *excluded* from the minimum (it is not a
    "sacrificed" class — an empty class must not sink the score to 0).

    Returns ``0.0`` when there is no non-empty class (an empty matrix or
    all-zero rows). Deterministic, pure numpy (G2)."""
    c = np.asarray(confusion, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        return 0.0
    k = int(c.shape[0])
    f1s: list[float] = []
    for i in range(k):
        tp = float(c[i, i])
        row = float(c[i].sum())
        if row <= 0.0:
            continue  # empty class — excluded (61.1.3)
        col = float(c[:, i].sum())
        fp = col - tp
        fn = row - tp
        denom = tp + fp + fn  # == 2*tp + fp + fn
        if denom <= 0.0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    if not f1s:
        return 0.0
    return float(min(f1s))


def ece(probs, labels, n_bins: int = 10) -> float:
    """SPEC.md 61.4 (A4): the expected calibration error of ``probs``
    (an ``(n, k)`` array of class probabilities, e.g. softmax output)
    against ``labels`` (an ``(n,)`` integer array).

    Bins by predicted max-probability into ``n_bins`` equal-width bins
    ``[i/n_bins, (i+1)/n_bins)``; within a non-empty bin ``b``:
    ``acc_b = mean(pred == label)`` and ``conf_b = mean(conf)`` (where
    ``conf = max_j probs`` and ``pred = argmax_j probs``). The ECE is
    ``Σ_b (n_b / n) · |acc_b − conf_b|`` — in ``[0, 1]`` (a perfectly
    calibrated model → ≈ 0; a uniform or confident-wrong model → high).

    Deterministic, pure numpy (G2); returns ``0.0`` for an empty input
    or a degenerate single-probability row."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).ravel()
    if p.size == 0 or p.shape[0] != y.shape[0] or p.shape[0] == 0:
        return 0.0
    if p.ndim == 1:
        # single-class probabilities — no calibration to measure
        return 0.0
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    n = int(p.shape[0])
    ece_val = 0.0
    for b in range(int(n_bins)):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # bin b: lo <= conf < hi, but the top bin includes conf == 1.0
        in_bin = (conf >= lo) & ((conf < hi) if b < n_bins - 1 else (conf <= hi))
        if not bool(in_bin.any()):
            continue
        acc = float((pred[in_bin] == y[in_bin]).mean())
        cf = float(conf[in_bin].mean())
        ece_val += (float(in_bin.sum()) / n) * abs(acc - cf)
    return float(min(1.0, max(0.0, ece_val)))


def coverage_width(pred_lo, pred_hi, labels) -> float:
    """SPEC.md 61.4.1 (A4): the interval coverage — the fraction of
    ``labels`` that fall inside ``[pred_lo, pred_hi]`` (elementwise), in
    ``[0, 1]`` (1.0 = every label covered; 0.0 = none). The natural
    objective for an *interval* predictor (coverage at a target width).

    ``pred_lo`` / ``pred_hi`` / ``labels`` are 1-D arrays of the same
    length. Deterministic, pure numpy (G2); returns ``0.0`` for an empty
    or length-mismatched input."""
    lo = np.asarray(pred_lo, dtype=np.float64).ravel()
    hi = np.asarray(pred_hi, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if lo.size == 0 or lo.size != hi.size or lo.size != y.size:
        return 0.0
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean())


def monotonicity_slope(fn, x0: float, dx: float, k: int = 3) -> float:
    """SPEC.md 61.1.3 (A1): a central finite-difference slope sign
    estimate of ``fn`` around ``x0`` — the ``monotonic_in`` constraint
    actual (``>= 0`` when ``fn`` is locally non-decreasing, ``< 0`` when
    non-increasing).

    The slope is the average of the central differences at offsets
    ``dx, 2*dx, ..., k*dx``: ``s_j = (fn(x0 + j*dx) − fn(x0 − j*dx)) /
    (2*j*dx)``, averaged over ``j = 1..k``. ``dx > 0`` and ``k >= 1`` are
    required (a ``ValueError`` otherwise). Deterministic, pure (G2)."""
    if not (dx > 0):
        raise ValueError(f"dx must be > 0, got {dx!r}")
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    slopes = []
    for j in range(1, int(k) + 1):
        h = float(j) * float(dx)
        slopes.append((float(fn(x0 + h)) - float(fn(x0 - h))) / (2.0 * h))
    return float(sum(slopes) / len(slopes))


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def ensemble_ece(task, models, k: int = 3, n: int = 200) -> float:
    """SPEC.md 61.4.2 (A4): the **ensemble-spread** ECE predictor —
    average the holdout softmax probabilities of the top-``k`` frontier
    ``models``, then score the averaged distribution with ``ece``.

    ``models`` is a non-empty sequence of models, each with a
    ``forward(x)`` producing ``(n, c)`` logits; ``task`` must expose
    ``holdout_rows(n, model)`` (the §28.2 protocol) and a ``softmax``
    head. ``k`` clamps to the number of models provided (``k >= 1``
    required). Averaging the top-k frontier specs' probabilities is the
    §37.1 frontier's natural predictor — a pure function over data the
    models already produced (no retraining). Deterministic (G2).

    Returns ``0.0`` when ``task`` has no ``holdout_rows`` or the holdout
    is empty (no calibration to measure)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if not models:
        raise ValueError("ensemble_ece needs at least one model")
    if not callable(getattr(task, "holdout_rows", None)):
        return 0.0
    x, y = task.holdout_rows(int(n), None)
    if len(x) == 0:
        return 0.0
    kk = min(int(k), len(models))
    avg = None
    for m in models[:kk]:
        logits = np.asarray(m.forward(x), dtype=np.float64)
        p = _softmax(logits)
        avg = p if avg is None else avg + p
    avg = avg / kk
    return ece(avg, np.asarray(y).ravel())
