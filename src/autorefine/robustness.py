"""FGSM adversarial robustness (SPEC.md 99, v0.85 C3).

`fgsm_perturb` / `robust_accuracy` give the acceptance gate an adversarial
view of a trained model: how many holdout predictions survive an
epsilon-ball input perturbation in the direction of steepest loss.

Design (deterministic, pure numpy — SPEC.md 3, no new dependency):
  * **Input gradients** — central differences of the per-sample loss,
    vectorized over the batch (2·D forward passes). Exact to O(eps_fd²)
    and model-agnostic (any `forward(x) -> (n, k)` contract).
  * **FGSM** — `x_adv = x + eps · sign(grad)`, the standard one-step attack.
  * **Robust accuracy** — the fraction of rows whose argmax is unchanged
    after the attack. A robustness gate (the env's `robust_eps` knob)
    requires a candidate to be at least as robust as the champion.

All functions are pure: same (model, X, y, eps) always yields the same
result (G2 — no RNG anywhere here).
"""
from __future__ import annotations

import numpy as np

__all__ = ["input_grads", "fgsm_perturb", "robust_accuracy"]

# finite-difference step for the input gradient (relative to the input
# scale — large enough to clear float noise, small enough to be linear)
_FD_REL = 1e-3


def _sample_loss(model, x: np.ndarray, y: np.ndarray, head: str) -> np.ndarray:
    """Per-row loss of `model.forward(x)` against `y` (softmax or mse)."""
    n = x.shape[0]
    out = model.forward(x)
    if head == "mse":
        target = np.asarray(y, dtype=np.float64).reshape(n, out.shape[1])
        return ((out - target) ** 2).sum(axis=1)
    z = out - out.max(axis=1, keepdims=True)
    logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    yv = np.asarray(y).reshape(n)
    return -logz[np.arange(n), yv]


def input_grads(model, x: np.ndarray, y: np.ndarray, head: str,
                rel_eps: float = _FD_REL) -> np.ndarray:
    """Central-difference gradient of the per-row loss w.r.t. the input.

    Returns (n, D) for 2-D input, else the same shape as `x` (flattened
    internally, reshaped back). Deterministic (forwards only)."""
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x2 = x.reshape(x.shape[0], -1)
    n, d = x2.shape
    scale = np.maximum(np.abs(x2).mean(axis=0, keepdims=True), 1e-6)
    h = rel_eps * scale  # (1, D) per-feature step
    grad = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        hp = np.zeros_like(h); hp[0, j] = h[0, j]
        hm = np.zeros_like(h); hm[0, j] = -h[0, j]
        lp = _sample_loss(model, x2 + hp, y, head)   # x + h along axis j
        lm = _sample_loss(model, x2 + hm, y, head)   # x - h along axis j
        grad[:, j] = (lp - lm) / (2.0 * h[0, j])
    return grad.reshape(orig_shape)


def fgsm_perturb(model, x: np.ndarray, y: np.ndarray, head: str,
                 eps: float) -> np.ndarray:
    """One-step FGSM: `x + eps · sign(input_grad)` (SPEC.md 99 C3)."""
    if eps <= 0.0:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    g = input_grads(model, x, y, head)
    return x + float(eps) * np.sign(g)


def robust_accuracy(model, X: np.ndarray, y: np.ndarray, eps: float,
                    head: str, max_rows: int = 128) -> dict:
    """Clean + FGSM-robust accuracy over (at most) the first `max_rows`
    rows — bounded so the gate stays cheap on large holdouts.

    Returns {"clean_acc", "robust_acc", "n", "eps"} (accuracies in [0, 1]);
    `eps == 0` returns clean == robust (the gate is off)."""
    n = min(int(X.shape[0]), max_rows)
    Xs = np.asarray(X[:n], dtype=np.float64)
    ys = np.asarray(y)[:n]
    clean = int((model.forward(Xs).argmax(axis=1) == ys).sum()) / n
    if eps <= 0.0:
        return {"clean_acc": float(clean), "robust_acc": float(clean),
                "n": n, "eps": 0.0}
    x_adv = fgsm_perturb(model, Xs, ys, head, eps)
    robust = int((model.forward(x_adv).argmax(axis=1) == ys).sum()) / n
    return {"clean_acc": float(clean), "robust_acc": float(robust),
            "n": n, "eps": float(eps)}
