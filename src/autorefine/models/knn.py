"""KNN: k-nearest-neighbors family (SPEC.md 25.2).

Non-parametric: "training" is memorizing the (already standardized) train
split `(X, y)`; `forward(x)` is a k-NN query over it. Same
`forward(x) -> (n, n_out)` contract as the other families, so task scoring,
the §19.3 ensemble, `eval`, and the dashboard work unchanged (SPEC.md 25.1).

  * softmax head: majority vote over the k nearest neighbors → class-vote
    fractions `(n, C)`, plus a deterministic ε = 1e-6 smoothing so a
    cross-entropy training loss stays finite (SPEC.md 25.2).
  * mse head: mean of the k neighbors' targets `(n, 1)` (SPEC.md 25.2).

Deterministic ties: `np.argsort(kind="stable")` (equal distances → lower
dataset index first); vote ties resolve to the lower class index (argmax
convention, SPEC.md 25.2). Checkpoints are plain arrays only
(`allow_pickle=False`, SPEC.md 9).
"""
from __future__ import annotations

import numpy as np

from ..config import SpecError

# SPEC.md 25.2: deterministic vote smoothing (uniform shift → argmax-safe)
VOTE_EPS = 1e-6


class KNN:
    """k-nearest-neighbors over the memorized train split (SPEC.md 25.2)."""

    def __init__(self, k: int, n_out: int, head: str) -> None:
        if k < 1:
            raise SpecError(f"knn_k {k!r} must be >= 1")
        if head not in ("softmax", "mse"):
            raise SpecError(f"unknown head {head!r}")
        self.k = int(k)
        self.n_out = int(n_out)
        self.head = head
        self.X: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.y: np.ndarray = np.zeros((0,), dtype=np.int64)

    # --- "training" = memorize (SPEC.md 25.2) --------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise SpecError(
                "knn needs a non-empty 2-D train split (n, d) to memorize, "
                f"got {X.shape}")
        if y.ndim < 1 or y.shape[0] != X.shape[0]:
            # a scalar/0-d target (or a row-count mismatch) would otherwise
            # surface later as a cryptic indexing error; reject cleanly (25.4)
            raise SpecError(
                f"knn target must be (n,) or (n, k_out) with n == {X.shape[0]}, "
                f"got y.shape={y.shape}")
        self.X = X
        self.y = y
        return self

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        # squared Euclidean distance to every memorized point (numerically
        # clamped so the (a-b)^2 expansion never dips below 0)
        d2 = (x * x).sum(axis=1)[:, None] \
            + (self.X * self.X).sum(axis=1)[None, :] \
            - 2.0 * x @ self.X.T
        d2 = np.maximum(d2, 0.0)
        # SPEC.md 25.2: stable argsort → equal distances keep dataset order
        k = min(self.k, self.X.shape[0])
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        if self.head == "mse":
            near = self.y[order]  # (n, k) or (n, k, k_out)
            return np.asarray(near.mean(axis=1), dtype=np.float64).reshape(n, self.n_out)
        # softmax head: class-vote fractions over the k nearest neighbors
        labels = np.asarray(self.y, dtype=np.int64).reshape(-1)
        near_l = labels[order]  # (n, k)
        votes = np.zeros((n, self.n_out), dtype=np.float64)
        np.add.at(votes, (np.arange(n)[:, None], near_l), 1.0)
        votes /= k
        return votes + VOTE_EPS  # deterministic smoothing (SPEC.md 25.2)

    # --- (de)serialization: plain arrays only (allow_pickle=False) -----------
    def save(self, path: str) -> None:
        np.savez(
            path,
            X=self.X,
            y=self.y,
            k=np.array(self.k),
            n_out=np.array(self.n_out),
            head=np.array(self.head),
        )

    @classmethod
    def load(cls, path: str) -> "KNN":
        with np.load(path, allow_pickle=False) as z:
            X = z["X"].astype(np.float64)
            y = z["y"]
            k = int(z["k"])
            n_out = int(z["n_out"])
            head = str(z["head"])
        m = cls(k=k, n_out=n_out, head=head)
        m.X = X
        m.y = y
        return m
