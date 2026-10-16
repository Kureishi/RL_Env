"""Split conformal calibration + OOD familiarity (SPEC.md 93.1/93.2, v0.79,
A83).

The extrapolation gap the environment used to leave open: a finished model
answers "what?" but not "how sure?" — and it gives the same-confident
answer to a row from its own distribution and to a row from nowhere.
This module adds both, calibrated on the task's own holdout split (the
28.2 `holdout_rows` protocol — the same rows the diagnostics score):

* **Conformal (93.1)** — split conformal prediction (Vovk et al.): the
  holdout nonconformity scores (classification: `1 - p(y_true)`;
  regression: `|y - y_hat|`) fix a finite-sample-corrected threshold
  `tau = (1 + 1/m) * nearest-rank(1 - alpha)`. A new row then gets a
  *prediction set* (classification — the classes with
  `p >= 1 - tau`, never empty) or a *prediction interval*
  (regression — `[y_hat - tau, y_hat + tau]`) whose nominal coverage is
  `1 - alpha` under exchangeability.
* **OOD / familiarity (93.2)** — the diagonal-Mahalanobis distance
  `sqrt(mean(z^2))` in the model's input space, thresholded at the
  `(1 - beta)` nearest-rank quantile of the holdout distances; a row
  past the threshold is flagged `ood`, and `familiarity = 1 - min(d/theta, 1)`
  turns the distance into a 0-1 "how much like the training data".

House rules (SPEC.md 3): pure stdlib + numpy; import-cycle-free (no
autorefine modules — the task/model protocol is duck-typed, exactly like
the 42.1 `predict` leaf); deterministic (G2 — nearest-rank quantiles, no
RNG); fail loud (a task without a usable holdout split is a
`ValueError`, never a silent skip).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "Calibration",
    "fit_calibration",
    "conformal_prediction_set",
    "conformal_interval",
    "conformal_distance",
    "is_ood",
    "familiarity",
]


@dataclass(frozen=True)
class Calibration:
    """One fitted calibration object (SPEC.md 93.1.1/93.2.1).

    Attributes
    ----------
    head           - "softmax" (classification) or "mse" (regression).
    alpha          - the nominal miscoverage level (default 0.1).
    n_cal          - the holdout rows the calibration was fit on.
    tau            - the conformal threshold (finite-sample corrected).
    beta           - the OOD specificity level (default 0.95); 1.0 means
                     OOD was not fit (see `ood_threshold`).
    ood_threshold  - the (1 - beta) nearest-rank quantile of the holdout
                     distances; `None` when OOD was not requested.
    class_values   - the task's class values (softmax) or `None` (mse).
    """
    head: str
    alpha: float
    n_cal: int
    tau: float
    beta: float
    ood_threshold: float | None
    class_values: tuple | None


def _check_level(name: str, v: float) -> float:
    v = float(v)
    if not (0.0 < v < 1.0) or not math.isfinite(v):
        raise ValueError(f"{name} must be in (0, 1), got {v!r}")
    return v


def _nearest_rank_quantile(values: np.ndarray, q: float) -> float:
    """The nearest-rank quantile (G2: no interpolation, no RNG) —
    the `ceil(q * m)`-th smallest value of `values`."""
    s = np.sort(np.asarray(values, dtype=np.float64).ravel())
    m = s.shape[0]
    idx = min(max(int(math.ceil(float(q) * m)) - 1, 0), m - 1)
    return float(s[idx])


def _holdout(task, model) -> tuple[np.ndarray, np.ndarray]:
    """The calibration rows — the task's own holdout split (the 28.2
    `holdout_rows` protocol). Fails loud on a task without the protocol
    or with an empty split (93.1.4: a task that cannot be calibrated is
    an error, not a silent skip)."""
    fn = getattr(task, "holdout_rows", None)
    if not callable(fn):
        raise ValueError(
            f"task {getattr(task, 'name', task)!r} exposes no holdout_rows "
            "protocol — conformal calibration needs the task's holdout "
            "split (SPEC.md 93.1.4)")
    x, y = fn(4096, model)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError(
            "the task's holdout split is empty (or has < 2 rows) — "
            "conformal calibration needs calibration data (SPEC.md 93.1.4)")
    return x, y.ravel()


def _forward(task, model, x: np.ndarray) -> np.ndarray:
    out = model.forward(x)
    return np.atleast_2d(np.asarray(out, dtype=np.float64))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_calibration(task, model, alpha: float = 0.1, beta: float = 0.95,
                    n: int | None = None,
                    with_ood: bool = True) -> Calibration:
    """Fit split conformal calibration (93.1) + the OOD threshold (93.2)
    on the task's holdout split. One forward pass, deterministic (G2).

    `alpha` is the nominal miscoverage (the set/interval covers the truth
    with probability >= 1 - alpha under exchangeability); `beta` the OOD
    specificity (a fraction ~1 - beta of holdout rows sit below the
    threshold). `with_ood=False` skips the distance quantile
    (`ood_threshold` is `None`; 93.2.3). Raises `ValueError` on a task
    without a usable holdout split (93.1.4)."""
    alpha = _check_level("alpha", alpha)
    beta = _check_level("beta", beta) if with_ood else 1.0
    head = getattr(task, "head", "mse")
    if head not in ("softmax", "mse"):
        raise ValueError(
            f"conformal calibration supports softmax/mse heads, got "
            f"{head!r} (SPEC.md 93.1.2)")
    x, y = _holdout(task, model)
    m = int(x.shape[0])
    if n is not None:
        m = min(m, max(2, int(n)))
        x, y = x[:m], y[:m]
    out = _forward(task, model, x)
    if head == "softmax":
        k = out.shape[1]
        if out.shape[0] != y.shape[0]:
            raise ValueError("model output rows != holdout rows")
        for i, t in enumerate(y):
            if not (0 <= int(t) < k) or int(t) != t:
                raise ValueError(
                    f"holdout target {t!r} is not a class index in 0..{k - 1} "
                    "(SPEC.md 93.1.3)")
        probs = _softmax(out)
        scores = 1.0 - probs[np.arange(m), y.astype(int)]  # 1 - p(y_true)
        values = tuple(getattr(task, "class_values", ()) or ())
    else:
        pred = out.reshape(-1)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("model output rows != holdout rows")
        scores = np.abs(y - pred)  # |y - y_hat|
        values = None
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite nonconformity scores — the model or "
                         "holdout data is not finite (SPEC.md 93.1.4)")
    tau = (1.0 + 1.0 / m) * _nearest_rank_quantile(scores, 1.0 - alpha)
    if with_ood:
        ood_threshold = _nearest_rank_quantile(
            conformal_distance(x), 1.0 - beta)
    else:
        ood_threshold = None
    return Calibration(head=head, alpha=alpha, n_cal=m, tau=float(tau),
                       beta=beta, ood_threshold=ood_threshold,
                       class_values=values)


def conformal_distance(x) -> float:
    """93.2.1: the OOD distance of ONE model-space row — the diagonal
    Mahalanobis `sqrt(mean(z^2))`. The row must be in the model's input
    space (the task's standardized features — the 42.1.3 rule; a task
    without stats is already in model space)."""
    z = np.asarray(x, dtype=np.float64).ravel()
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("the row must be a non-empty finite vector")
    return float(np.sqrt(np.mean(z * z)))


def is_ood(x, cal: Calibration) -> bool:
    """93.2.2: `True` when the row's distance exceeds the fitted
    threshold. Raises `ValueError` when OOD was not fit (93.2.3)."""
    if cal.ood_threshold is None:
        raise ValueError("this Calibration was fit without OOD "
                         "(fit_calibration(..., with_ood=True))")
    return conformal_distance(x) > cal.ood_threshold


def familiarity(x, cal: Calibration) -> float:
    """93.2.2: `1 - min(d/theta, 1)` — 1.0 at the holdout center, 0.0
    beyond twice the threshold. Raises `ValueError` when OOD was not
    fit (93.2.3)."""
    if cal.ood_threshold is None:
        raise ValueError("this Calibration was fit without OOD "
                         "(fit_calibration(..., with_ood=True))")
    theta = cal.ood_threshold
    d = conformal_distance(x)
    if theta <= 0.0:
        return 0.0 if d > 0.0 else 1.0
    return float(1.0 - min(d / theta, 1.0))


def conformal_prediction_set(task, model, x, cal: Calibration) -> Sequence:
    """93.1.1: the row's prediction SET — the classes whose probability
    is at least `1 - tau`, mapped through the task's `class_values`;
    never empty (the argmax joins when the threshold excludes every
    class). Raises `ValueError` on an mse calibration (93.1.2) or a row
    whose output does not match the calibrated head."""
    if cal.head != "softmax":
        raise ValueError("conformal_prediction_set is for softmax-head "
                         "calibrations (SPEC.md 93.1.2)")
    out = _forward(task, model, np.atleast_2d(np.asarray(x, dtype=np.float64).ravel()))
    if out.shape[0] != 1:
        raise ValueError("expected one row")
    probs = _softmax(out)[0]  # `out` is (1, k); `_softmax` needs the 2-D form
    keep = [i for i, p in enumerate(probs) if p >= 1.0 - cal.tau]
    if not keep:
        keep = [int(out[0].argmax())]
    values = cal.class_values
    if values:
        return [values[i] for i in keep]
    return list(keep)


def conformal_interval(pred, cal: Calibration) -> tuple[float, float]:
    """93.1.1: the regression interval `[y_hat - tau, y_hat + tau]`.
    Raises `ValueError` on a softmax calibration (93.1.2)."""
    if cal.head != "mse":
        raise ValueError("conformal_interval is for mse-head calibrations "
                         "(SPEC.md 93.1.2)")
    p = float(pred)
    if not math.isfinite(p):
        raise ValueError(f"the prediction {p!r} is not finite")
    return (p - cal.tau, p + cal.tau)
