"""Generalized Additive Model family (SPEC.md 75, v0.61).

A non-parametric, interpretable, blocking-fit model in the same
`fit`/`forward`/`save`/`load` contract as the other families (KNN,
SPEC.md 25.2 is the template), so task scoring, the §19.3 ensemble, `eval`,
and the dashboard work unchanged.

Design (deterministic, pure numpy):
  * **Additive form** — ``ŷ = α + Σ_j f_j(x_j)``: one smooth per-feature
    effect ``f_j`` (no interactions), the standard GAM. Interpretability is
    the point: each ``f_j`` is a single-variable curve the user can read.
  * **Basis** — a natural-cubic (truncated-power) spline per feature with
    ``k`` interior knots placed at the data quantiles (deterministic): the
    columns are ``1, x, (x−τ_1)_+², (x−τ_1)_+³, …``.
  * **Smoothing** — a curvature penalty in observation space: the second
    difference of each feature's fitted values, weighted by the spec's
    ``weight_decay`` (so the improver can tune the smoothness directly —
    0.0 = interpolating spline, larger = smoother). The intercept and the
    per-feature linear term have zero second difference, so they are not
    penalized (a valid smoothness penalty, no basis-specific matrix needed).
  * **Classification** — one-vs-rest per class (predictive means → the
    softmax logits, the same contract as KNN / GP).
  * **Regression** — a single additive model to the target (n_out == 1).

Solved as one penalized least-squares per output unit via
`np.linalg.lstsq` (deterministic; no RNG). Checkpoints are plain arrays only
(`allow_pickle=False`, SPEC.md 9).
"""
from __future__ import annotations

import numpy as np

from ..config import SpecError


def _n_knots(n: int) -> int:
    """Interior-knot count for a feature of ``n`` rows (deterministic)."""
    return int(min(6, max(1, n // 8)))


def _spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Truncated-power natural-cubic basis for one feature (SPEC.md 75).

    ``x`` (n,) and ``knots`` (k,) -> (n, 2 + 2k) matrix: ``[1, x, (x−τ)_+²,
    (x−τ)_+³]`` for each interior knot ``τ``. The last two columns are zero
    where ``x <= τ`` (the hinge), so a constant feature yields the harmless
    ``[1, c]`` (absorbed by the intercept)."""
    n = x.shape[0]
    cols = [np.ones(n, dtype=np.float64), x.astype(np.float64)]
    for tau in knots:
        h = np.maximum(x - tau, 0.0)
        cols.append(h * h)
        cols.append(h * h * h)
    return np.column_stack(cols)


def _second_difference(n: int) -> np.ndarray:
    """(n−2)×n second-difference operator ``[1, −2, 1]`` (curvature probe)."""
    if n <= 2:
        return np.zeros((0, n), dtype=np.float64)
    rows = n - 2
    D = np.zeros((rows, n), dtype=np.float64)
    D[:, 0] = 1.0
    D[:, 1:-1] = -2.0
    D[:, -1] = 1.0
    return D


class GAM:
    """Generalized additive model (smooth per-feature effects), SPEC.md 75."""

    def __init__(self, n_out: int, head: str, weight_decay: float = 0.0) -> None:
        if head not in ("softmax", "mse"):
            raise SpecError(f"unknown head {head!r}")
        if weight_decay < 0.0:
            raise SpecError(f"weight_decay {weight_decay!r} must be >= 0")
        self.n_out = int(n_out)
        self.head = head
        self.weight_decay = float(weight_decay)
        self.in_dim = 0
        self.knots = []                # per feature: (k,) array (empty = no knots)
        self.betas = np.zeros((0,), dtype=np.float64)  # (n_out, p) per unit
        self._p = 0                    # design width (1 + sum_j (2 + 2 k_j))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise SpecError(
                "gam needs a non-empty 2-D train split (n, d) to fit, "
                f"got {X.shape}")
        n, d = X.shape
        if y.ndim < 1 or y.shape[0] != n:
            raise SpecError(
                f"gam target must be (n,) or (n, k_out) with n == {n}, "
                f"got y.shape={y.shape}")
        self.in_dim = d

        k = _n_knots(n)
        qs = np.linspace(0.0, 1.0, k + 2)[1:-1]  # k interior knot fractions
        D = _second_difference(n)
        # per-feature basis + its curvature-penalty design (D @ N_j)
        basis, pen, widths = [], [], []
        self.knots = []  # idempotent re-fit (SPEC.md 75)
        for j in range(d):
            col = X[:, j]
            knots = np.quantile(col, qs) if k > 0 else np.zeros(0)
            Nj = _spline_basis(col, knots)
            basis.append(Nj)
            pen.append(D @ Nj)
            widths.append(Nj.shape[1])
            self.knots.append(knots)

        A = np.hstack([np.ones((n, 1), dtype=np.float64)] + basis)   # (n, p)
        B = np.hstack([np.zeros((D.shape[0], 1), dtype=np.float64)] + pen)
        p = A.shape[1]
        self._p = p

        pen_w = float(np.sqrt(self.weight_decay)) if self.weight_decay > 0.0 else 0.0

        if self.head == "mse":
            targets = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
        else:  # softmax: one-vs-rest 0/1 targets (n, n_out)
            labels = np.asarray(y, dtype=np.int64).reshape(-1)
            targets = np.zeros((n, self.n_out), dtype=np.float64)
            targets[np.arange(n), labels] = 1.0

        betas = np.zeros((self.n_out, p), dtype=np.float64)
        for j in range(self.n_out):
            t = targets[:, j]
            if pen_w > 0.0:
                Aa = np.vstack([A, pen_w * B])
                ta = np.concatenate([t, np.zeros(B.shape[0])])
            else:
                Aa, ta = A, t
            sol, *_rest = np.linalg.lstsq(Aa, ta, rcond=None)
            betas[j] = sol
        self.betas = betas
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n, d = x.shape
        A = np.empty((n, self._p), dtype=np.float64)
        A[:, 0] = 1.0
        offset = 1
        for j in range(d):
            Nj = _spline_basis(x[:, j], np.asarray(self.knots[j], dtype=np.float64))
            A[:, offset:offset + Nj.shape[1]] = Nj
            offset += Nj.shape[1]
        out = A @ self.betas.T  # (n, n_out)
        return np.asarray(out, dtype=np.float64)

    def save(self, path: str) -> None:
        arrays = {
            "n_out": np.array(self.n_out),
            "head": np.array(self.head),
            "in_dim": np.array(self.in_dim),
            "weight_decay": np.array(self.weight_decay),
            "p": np.array(self._p),
            "betas": self.betas,
            "n_knots": np.array([k.size if k is not None else 0 for k in self.knots]),
        }
        for j, k in enumerate(self.knots):
            if k is not None and k.size:
                arrays[f"knots{j}"] = k
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "GAM":
        with np.load(path, allow_pickle=False) as z:
            n_out = int(z["n_out"])
            head = str(z["head"])
            in_dim = int(z["in_dim"])
            weight_decay = float(z["weight_decay"])
            p = int(z["p"])
            betas = z["betas"].astype(np.float64)
            n_knots = [int(v) for v in z["n_knots"]]
            knots = [
                (z[f"knots{j}"].astype(np.float64) if n_knots[j] else np.zeros(0))
                for j in range(len(n_knots))
            ]
        m = cls(n_out=n_out, head=head, weight_decay=weight_decay)
        m.in_dim = in_dim
        m.knots = knots
        m._p = p
        m.betas = betas
        return m
