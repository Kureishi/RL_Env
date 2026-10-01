"""`make_lag_features`: a small, deterministic lag-window helper (SPEC.md 83.4).

Time-series users often want to feed a *flat* model (mlp / tree / boost /
knn) a recent-window view of a (T, C) sequence instead of a native temporal
family (conv1d / rnn). This is that view, in pure numpy (no new dependency,
SPEC.md 3) and fully deterministic (G2) — a simple slice + reshape, so it can
never perturb a legacy run.
"""
from __future__ import annotations

import numpy as np


def make_lag_features(x: np.ndarray, window: int) -> np.ndarray:
    """Recent-window lag features over a (T, C) sequence layout.

    `x` is either ``(n, T, C)`` (n sequences, T timesteps, C channels) or
    ``(n, T)`` (a single-channel signal — treated as C = 1). Returns
    ``(n, window * C)``: the most recent `window` timesteps of each
    sequence, flattened row-major (time-major, then channels).

    This is the standard "lag-K features" transform for flat time-series
    models: it exposes the last `window` steps as a fixed-width feature
    vector so any family in the registry can consume temporal context. The
    transform is a pure view/reshape — no RNG, no allocation of new values
    beyond the flatten copy — so it is deterministic (G2) and bit-identical
    across calls.

    Raises `ValueError` on a non-2/3-D input or a window outside 1..T.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]  # (n, T) single-channel -> (n, T, 1)
    if x.ndim != 3:
        raise ValueError(
            f"make_lag_features needs (n, T) or (n, T, C); got {x.shape}")
    n, T, C = x.shape
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"window must be an int, got {window!r}")
    window = int(window)
    if not (1 <= window <= T):
        raise ValueError(f"window {window} outside 1..T={T}")
    return x[:, -window:, :].reshape(n, window * C)
