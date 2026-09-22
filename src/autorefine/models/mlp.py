"""Small NumPy MLP with analytic backprop (cross-entropy head).

Deterministic given a seed; parameters are plain float64 arrays so checkpoints
are loadable with `np.load` alone (SPEC.md 9).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import ACTIVATIONS, SpecError


HEADS = ("softmax", "mse")


def _activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    raise SpecError(f"unknown activation {name!r}")


def _activation_grad(name: str, x: np.ndarray, out: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return 1.0 - out * out
    if name == "relu":
        return (x > 0).astype(np.float64)
    raise SpecError(f"unknown activation {name!r}")


# --- SPEC.md 70: hot-path helpers (bit-identical values, less per-call work) ---
_ARANGE_CACHE: dict[int, np.ndarray] = {}


def _row_idx(n: int) -> np.ndarray:
    """Cached `np.arange(n)`: batch size is stable within a training run, so
    re-creating it on every `loss_and_grads` call was pure overhead."""
    idx = _ARANGE_CACHE.get(n)
    if idx is None:
        idx = _ARANGE_CACHE[n] = np.arange(n)
    return idx


def _tanh_grad(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    return 1.0 - out * out


def _relu_act(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


# Bound function pairs per activation: hoists the name lookup out of the
# per-layer loop (identical results to `_activation` / `_activation_grad`).
_ACT_FNS: dict[str, tuple] = {
    "tanh": (np.tanh, _tanh_grad),
    "relu": (_relu_act, _relu_grad),
}


class MLP:
    """input (n, d_in) -> hidden layers -> (n, n_out) logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        n_out: int,
        activation: str,
        seed: int,
        head: str = "softmax",
        init_scale: float = 1.0,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise SpecError(f"unknown activation {activation!r}")
        if head not in HEADS:
            raise SpecError(f"unknown head {head!r}")
        if init_scale <= 0.0:
            raise SpecError(f"init_scale {init_scale!r} must be positive")
        rng = np.random.default_rng(seed)
        sizes = [in_dim, *hidden_sizes, n_out]
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Glorot uniform, scaled by init_scale (SPEC.md 19.1; 1.0 = plain)
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.layers.append((w, b))
        self.activation = activation
        self._act_fn, self._act_grad_fn = _ACT_FNS[activation]  # SPEC.md 70
        self.in_dim = in_dim
        self.n_out = n_out
        self.head = head
        self._is_mse = head == "mse"  # SPEC.md 74: dispatch hoisted to a flag
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- forward -----------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        a = x
        act = self._act_fn  # SPEC.md 70: dispatch hoisted out of the loop
        for i, (w, b) in enumerate(self.layers[:-1]):
            z = a @ w + b
            h = act(z)
            self._cache.append((a, z, h))
            a = h
        logits = a @ self.layers[-1][0] + self.layers[-1][1]
        self._cache.append((a, logits, logits))
        return logits

    # --- loss + analytic gradients -----------------------------------------
    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, label_smoothing: float = 0.0
    ) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
        """Returns (loss, [(Wg, bg), ...]).

        head "softmax": y is integer class labels (n,) — mean cross-entropy,
                        optionally with label smoothing epsilon (SPEC.md 17):
                        target = (1-eps) * onehot + eps / n_out.
        head "mse":     y is continuous targets (n,) or (n, n_out) — mean MSE
                        (SPEC.md 15: regression tasks such as SineRegressionV1).
        """
        n = x.shape[0]
        logits = self.forward(x)
        if self._is_mse:  # SPEC.md 74: flag instead of per-call string compare
            target = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
            diff = logits - target
            loss = float((diff * diff).mean())
            dz = 2.0 * diff / n  # d(mean MSE)/d(output)
        else:
            # numerically stable softmax cross-entropy
            z = logits - logits.max(axis=1, keepdims=True)
            logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            onehot = np.zeros_like(logz)
            onehot[_row_idx(n), y] = 1.0  # SPEC.md 70: cached arange
            eps = float(label_smoothing)
            # SPEC.md 70: with eps == 0.0 the smoothing expression is
            # bit-identical to `onehot` (x*1.0, x+0.0 are exact), so skip
            # the two extra array passes in the common case.
            target = onehot if eps == 0.0 else (1.0 - eps) * onehot + eps / self.n_out
            loss = float(-(target * logz).sum(axis=1).mean())
            # dL/dlogits for mean softmax CE with smoothed targets
            p = np.exp(logz)
            dz = p - target
            dz /= n

        grads: list[tuple[np.ndarray, np.ndarray]] = []
        d = dz
        for i in range(len(self.layers) - 1, -1, -1):
            a, _, _ = self._cache[i]
            w, _b = self.layers[i]
            gw = a.T @ d
            gb = d.sum(axis=0)
            if i > 0:
                # chain through the PREVIOUS layer's nonlinearity (its z/h)
                z_prev = self._cache[i - 1][1]
                h_prev = self._cache[i - 1][2]
                d = d @ w.T
                d *= self._act_grad_fn(z_prev, h_prev)  # SPEC.md 70 + 74
            grads.append((gw, gb))
        grads.reverse()
        return loss, grads

    # --- (de)serialization ---------------------------------------------------
    # Plain scalar/float arrays only: loadable with allow_pickle=False.
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        arrays["n_layers"] = np.array(len(self.layers))
        arrays["in_dim"] = np.array(self.in_dim)
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MLP":
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            layers = [
                (z[f"layer{i}_w"].astype(np.float64), z[f"layer{i}_b"].astype(np.float64))
                for i in range(n_layers)
            ]
            activation = str(z["activation"])
            in_dim = int(z["in_dim"])
            n_out = int(z["n_out"])
            head = str(z["head"]) if "head" in z.files else "softmax"
        m = cls.__new__(cls)
        m.layers = layers
        m.activation = activation
        m._act_fn, m._act_grad_fn = _ACT_FNS[activation]  # SPEC.md 70
        m.in_dim = in_dim
        m.n_out = n_out
        m.head = head  # default keeps pre-extension checkpoints loadable
        m._is_mse = head == "mse"  # SPEC.md 74 (keeps the 74 flag in sync)
        m._cache = []
        return m
