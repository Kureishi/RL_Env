"""Data augmentation for the neural training loop (SPEC.md 99, B5).

All modes are pure functions of (batch, mode, rng) — deterministic given
the caller's seeded ``rng`` (G2). ``"none"`` is the exact legacy path: the
trainer never draws from the augmentation stream, so pre-v0.85 bit-exact
runs are untouched (the 18.7 / T2 pins stay green).

Modes (``config.AUGMENTS``):

  mixup     - x = lam*x_i + (1-lam)*x_j with lam ~ U(0.5, 1); the label is
              the dominant component's (softmax head) or the same convex
              blend (mse head).
  noise     - x + N(0, 0.01 * scale), scale = max(1, mean |x|).
  flip      - x * s with s ~ {+1, -1} per feature (last axis).
  time_warp - per-sample resampling of the time axis with linear
              interpolation (3-D (n, T, C): axis 1; 4-D (n, C, H, W):
              axis 2; 2-D (n, d): the feature axis).

The caller owns the rng: the trainer seeds it from (seed, 90000) — a
separate stream from the batch-sampler rng, so enabling augmentation
never perturbs the legacy sampling order.
"""
from __future__ import annotations

import numpy as np

from ..config import AUGMENTS, SpecError


def _mixup(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
           head: str) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    j = rng.integers(0, n, n)
    lam = rng.uniform(0.5, 1.0, n)
    shape = (n,) + (1,) * (x.ndim - 1)
    xb = lam.reshape(shape) * x + (1.0 - lam).reshape(shape) * x[j]
    if head == "mse":
        yx = np.asarray(y, dtype=np.float64)
        yb = lam * yx + (1.0 - lam) * yx[j]
    else:
        yv = np.asarray(y)
        yb = np.where(lam >= 0.5, yv, yv[j])
    return xb, yb


def _noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = max(1.0, float(np.abs(x).mean()))
    return x + rng.normal(0.0, 0.01 * scale, x.shape)


def _flip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
    signs = rng.integers(0, 2, size=shape).astype(np.float64)
    return x * (2.0 * signs - 1.0)


def _warp_axis(x: np.ndarray, axis: int, rng: np.random.Generator) -> np.ndarray:
    """Per-sample linear-interpolated resampling of `x` along `axis`
    (fully vectorized over the batch)."""
    T = x.shape[axis]
    if T < 2:
        return x
    n = x.shape[0]
    y = np.moveaxis(x, axis, 1)  # (n, T, ...)
    dst = np.sort(rng.uniform(0.0, float(T - 1), (n, T)))
    i0 = np.floor(dst).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (dst - i0)
    if y.ndim > 2:
        f = f[..., None]
    row = np.arange(n)[:, None]
    out = (1.0 - f) * y[row, i0] + f * y[row, i1]
    return np.moveaxis(out, 1, axis)


def _time_warp(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if x.ndim == 3:          # (n, T, C) sequences (conv1d / rnn)
        return _warp_axis(x, 1, rng)
    if x.ndim == 4:          # (n, C, H, W) grids (convnet): warp the rows
        return _warp_axis(x, 2, rng)
    if x.ndim == 2:          # (n, d) flat: the feature axis is the time axis
        return _warp_axis(x, 1, rng)
    raise SpecError(
        f"time_warp needs 2-D/3-D/4-D batches, got {x.ndim}-D "
        f"(SPEC.md 99 B5)")


def augment_batch(x: np.ndarray, y: np.ndarray, mode: str,
                  rng: np.random.Generator,
                  head: str = "softmax") -> tuple[np.ndarray, np.ndarray]:
    """One augmentation step: returns (x', y'). ``mode`` must be in
    ``config.AUGMENTS`` (``"none"`` returns the batch unchanged — but the
    trainer skips the call entirely in that case, so no rng draw happens)."""
    if mode not in AUGMENTS:
        raise SpecError(f"unknown augment mode {mode!r} (SPEC.md 99 B5)")
    if mode == "none":
        return x, y
    x = np.asarray(x, dtype=np.float64)
    if mode == "mixup":
        return _mixup(x, y, rng, head)
    if mode == "noise":
        return _noise(x, rng), y
    if mode == "flip":
        return _flip(x, rng), y
    if mode == "time_warp":
        return _time_warp(x, rng), y
    raise SpecError(f"unknown augment mode {mode!r} (SPEC.md 99 B5)")