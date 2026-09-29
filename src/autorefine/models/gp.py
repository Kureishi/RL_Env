"""Gaussian Process regression family (SPEC.md 75, v0.61).

A non-parametric, blocking-fit model in the same `fit`/`forward`/`save`/
`load` contract as the other families (KNN, SPEC.md 25.2 is the template),
so task scoring, the §19.3 ensemble, `eval`, and the dashboard work
unchanged.

Design (deterministic, pure numpy):
  * **Kernel** — RBF (squared-exponential), signal variance 1.0,
    observation noise 1e-3 (regularization). The length scale defaults to
    1.0 (the train split is standardized by the task, so unit length-scale
    is the sane default); SPEC.md 78 (v0.64) exposes it as the spec field
    `gp_length_scale` (1.0 = the legacy bit-exact draw).
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
from ._fourier import apply_scaled  # SPEC.md 78.2.1 (shared spectral map)

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

    def __init__(self, n_out: int, head: str,
                 length_scale: float = 1.0, fourier_K: int = 0) -> None:
        if head not in ("softmax", "mse"):
            raise SpecError(f"unknown head {head!r}")
        # SPEC.md 78 (v0.64): RBF length scale (1.0 = the legacy unit scale;
        # the RFF draw order is unchanged, only its scale argument)
        if not (length_scale > 0.0):
            raise SpecError(f"gp length_scale {length_scale!r} must be > 0")
        self.n_out = int(n_out)
        self.head = head
        self.length_scale = float(length_scale)
        # SPEC.md 78.2.1: the RFF feature space may live on the spectral
        # expansion. The model owns the map: fit stores K + the train-data
        # scale and expands both the fit rows and every query (K=0 -> off,
        # the legacy bit-exact path).
        self.fourier_K = int(fourier_K)
        self.fourier_scale: np.ndarray | None = None
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
        # SPEC.md 78.2.1: expand the fit space (K=0 -> X, the legacy path).
        # Row count is unchanged, so the `y` shape check below still holds.
        if self.fourier_K > 0:
            self.fourier_scale = np.maximum(np.abs(X).max(axis=0), 1e-6)
            X = apply_scaled(X, self.fourier_K, self.fourier_scale)
        n, d = X.shape
        if y.ndim < 1 or y.shape[0] != n:
            raise SpecError(
                f"gp target must be (n,) or (n, k_out) with n == {n}, "
                f"got y.shape={y.shape}")
        self.in_dim = d

        # fixed-seed random Fourier frequencies/phases for the RBF kernel.
        # SPEC.md 78: the length scale changes only the normal's scale arg
        # (1.0/1.0 == 1.0 → bit-identical legacy draw; the draw order —
        # normal then uniform — is unchanged, so A1–A4 stay green).
        rng = np.random.default_rng(_RFF_SEED)
        omega = rng.normal(0.0, 1.0 / self.length_scale, size=(_N_FEATURES, d))
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
        # SPEC.md 78.2.1: the query lives in the same expanded space as the
        # fit rows (one stored scale -- fit and inference agree)
        if self.fourier_K > 0 and self.fourier_scale is not None:
            x = apply_scaled(x, self.fourier_K, self.fourier_scale)
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
            # SPEC.md 78: length scale (absent in pre-v0.64 checkpoints)
            length_scale=np.array(self.length_scale),
            # SPEC.md 78.2.1: the spectral map (absent in legacy checkpoints)
            fourier_K=np.array(self.fourier_K),
            **({"fourier_scale": np.asarray(self.fourier_scale, dtype=np.float64)}
               if self.fourier_scale is not None else {}),
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
            # SPEC.md 78: pre-v0.64 checkpoints have no length_scale → 1.0
            length_scale = (float(z["length_scale"])
                            if "length_scale" in z else 1.0)
            # SPEC.md 78.2.1: pre-v0.64 checkpoints have no fourier keys → K=0
            fourier_K = (int(z["fourier_K"]) if "fourier_K" in z.files else 0)
            fourier_scale = (z["fourier_scale"].astype(np.float64)
                             if fourier_K > 0 and "fourier_scale" in z.files
                             else None)
        g = cls(n_out=n_out, head=head, length_scale=length_scale,
                fourier_K=fourier_K)
        g.omega = omega
        g.b = b
        g.w = w
        g.in_dim = in_dim
        g.fourier_scale = fourier_scale
        return g
