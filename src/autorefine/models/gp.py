"""Gaussian Process regression family (SPEC.md 75, v0.61).

A non-parametric, blocking-fit model in the same `fit`/`forward`/`save`/
`load` contract as the other families (KNN, SPEC.md 25.2 is the template),
so task scoring, the §19.3 ensemble, `eval`, and the dashboard work
unchanged.

Design (deterministic, pure numpy):
  * **Kernel** — RBF (squared-exponential), length-scale 1.0 (the train
    split is standardized by the task, so unit length-scale is the sane
    default), signal variance 1.0, observation noise 1e-3 (regularization).
  * **Approximation** — random Fourier features (RFF, Rahimi & Recht 2007)
    approximate the RBF kernel, turning the GP into a ridge-regularized
    linear model in an `n_features`-dimensional feature space. This is
    exact-in-the-limit for the kernel and is O(n·m·d + m³) rather than the
    O(n³) of a full GP solve, so it stays tractable on the episode-task
    dataset sizes where a direct GP solve would not.
  * **Classification** — one-vs-rest GP regression per class; the predictive
    means are the softmax logits (the task scores on `argmax`, the same
    contract as KNN's vote fractions and the neural head).
  * **Regression** — a single GP regression to the target (n_out == 1).

The RFF random matrix is drawn from a **fixed** internal seed (0), so a
GP trained on the same dataset is bit-reproducible regardless of the
training seed (non-parametric, like KNN — the seed matters only for the
parametric families). Checkpoints are plain arrays only
(`allow_pickle=False`, SPEC.md 9).
"""
from __future__ import annotations

import numpy as np

from ..config import SpecError

# SPEC.md 75: fixed internal constants (the spec surface stays small — the
# same philosophy as WARMUP_FRACTION / VAL_FRACTION in the trainer).
_N_FEATURES = 128          # RFF dimension (kernel approximation quality)
_SIGNAL_VAR = 1.0          # RBF signal variance
_NOISE = 1e-3             # observation noise (ridge regularizer)
_RFF_SEED = 0             # fixed -> bit-reproducible regardless of train seed


def _rff_matrix(x: np.ndarray, omega: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RFF map: `(n, d) x` -> `(n, m)` features (cosine basis, SPEC.md 75)."""
    # x @ omega.T : (n, m); + b : broadcast (m,)
    proj = x @ omega.T + b
    return np.sqrt(2.0 / omega.shape[0]) * np.cos(proj)


class GP:
    """One-vs-rest Gaussian Process (RFF approximation), SPEC.md 75."""

    def __init__(self, n_out: int, head: str) -> None:
        if head not in ("softmax", "mse"):
            raise SpecError(f"unknown head {head!r}")
        self.n_out = int(n_out)
        self.head = head
        self.in_dim = 0
        self.omega = np.zeros((0, 0), dtype=np.float64)
        self.b = np.zeros(0, dtype=np.float64)
        self.w = np.zeros((0, 0), dtype=np.float64)  # (m, n_out) feature weights

    def _in_dim_from(self, x: np.ndarray) -> int:
        return int(x.shape[1])

    # --- "training" = fit the per-output-unit ridge GP (SPEC.md 75) ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise SpecError(
                "gp needs a non-empty 2-D train split (n, d) to fit, "
                f"got {X.shape}")
        n, d = X.shape
        if y.ndim < 1 or y.shape[0] != n:
            raise SpecError(
                f"gp target must be (n,) or (n, k_out) with n == {n}, "
                f"got y.shape={y.shape}")
        self.in_dim = d

        # fixed-seed random Fourier frequencies/phases for the RBF kernel
        rng = np.random.default_rng(_RFF_SEED)
        omega = rng.normal(0.0, 1.0, size=(_N_FEATURES, d))  # length-scale 1.0
        b = rng.uniform(0.0, 2.0 * np.pi, size=_N_FEATURES)
        self.omega = omega
        self.b = b

        Phi = _rff_matrix(X, omega, b)          # (n, m)
        # ridge: (Phi^T Phi + noise * I) w = Phi^T t  (per output unit)
        m = Phi.shape[1]
        A = Phi.T @ Phi + _NOISE * np.eye(m)

        if self.head == "mse":
            targets = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
        else:  # softmax: one-vs-rest 0/1 targets (n, n_out)
            labels = np.asarray(y, dtype=np.int64).reshape(-1)
            targets = np.zeros((n, self.n_out), dtype=np.float64)
            targets[np.arange(n), labels] = 1.0

        # w_j = A^{-1} Phi^T t_j for every output unit j (n_out solves at once)
        self.w = np.linalg.solve(A, Phi.T @ targets)
        return self

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        out = _rff_matrix(x, self.omega, self.b) @ self.w  # (n, n_out)
        return np.asarray(out, dtype=np.float64)

    # --- (de)serialization: plain arrays only (allow_pickle=False) -----------
    def save(self, path: str) -> None:
        np.savez(
            path,
            omega=self.omega,
            b=self.b,
            w=self.w,
            n_out=np.array(self.n_out),
            head=np.array(self.head),
            in_dim=np.array(self.in_dim),
        )

    @classmethod
    def load(cls, path: str) -> "GP":
        with np.load(path, allow_pickle=False) as z:
            omega = z["omega"].astype(np.float64)
            b = z["b"].astype(np.float64)
            w = z["w"].astype(np.float64)
            n_out = int(z["n_out"])
            head = str(z["head"])
            in_dim = int(z["in_dim"])
        g = cls(n_out=n_out, head=head)
        g.omega = omega
        g.b = b
        g.w = w
        g.in_dim = in_dim
        return g
