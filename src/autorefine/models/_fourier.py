"""Shared deterministic spectral input expansion (SPEC.md 78.2.1).

`K <= 0` is a no-op (returns the input unmodified — the legacy bit-exact
path, no dtype coercion). Otherwise the expansion is
`[x, cos(k·x_t), sin(k·x_t)]` for `k = 1..K` over the per-dimension
normalized coordinate `x_t = x / max(|x|, 1e-6)` → shape
`(n, d·(2K+1))` (block-per-k column layout). No RNG; pure math.

The scale is a property of the *training* data: a model that consumes the
expansion (mlp / knn / gp) computes it once at fit time, stores it, and
applies the SAME map to every inference query — so a model's `forward`
reproduces its fit-time feature map (SPEC.md 78.2.1). `apply_scaled` is
that model-side map (one stored scale shared by fit and inference);
`fourier_expand` computes the scale from the input itself.
"""
from __future__ import annotations

import numpy as np


def _data_scale(x: np.ndarray) -> np.ndarray:
    """Per-dimension data scale `max(|x|_j, 1e-6)` (78.2.1)."""
    return np.maximum(np.abs(x).max(axis=0), 1e-6)


def fourier_expand(X: np.ndarray, K: int) -> np.ndarray:
    """Expand `X` to `(n, d·(2K+1))` using its own per-dim data scale.

    `K <= 0` returns `X` unmodified (the legacy bit-exact path)."""
    if K <= 0:
        return X
    x = np.asarray(X, dtype=np.float64)
    return apply_scaled(x, K, _data_scale(x))


def apply_scaled(x: np.ndarray, K: int, scale: np.ndarray) -> np.ndarray:
    """Apply the expansion to `x` with a precomputed per-dim `scale` — the
    model-side map (fit-time and inference-time share one scale)."""
    if K <= 0:
        return x
    x_t = x / scale
    cols = [x]
    for k in range(1, K + 1):
        cols.append(np.cos(k * x_t))
        cols.append(np.sin(k * x_t))
    return np.column_stack(cols)
