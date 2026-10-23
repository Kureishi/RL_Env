"""LateFusionMLP: two modality towers + a shared head (SPEC.md 99, v0.85 C1).

The "late" branch of the `fusion` spec field: an image-width slice of the
input feeds one tower, the remaining text-width slice feeds the other, and
a single head reads the concatenated tower outputs — the classic two-tower
late-fusion architecture, in the exact `.layers` + `loss_and_grads`
contract of `MLP` (SPEC.md 19.1) so the shared `_train_neural` loop,
the optimizer / schedule / clipping / early-stopping machinery, and the
npz save/load + warm-start surface all apply unchanged.

Layout (`.layers` order — tower1, tower2, head — is the canonical one the
80.3 `copy_weights` and 99 C2 `cross_modal_warm_start` slot-mapping read):

    tower1 : d_img -> h1 -> h2 -> ... -> hk      (every layer activated)
    tower2 : d_txt -> h1 -> h2 -> ... -> hk      (every layer activated)
    head   : (hk + hk) -> n_out                   (linear)

with `hk = architecture[-1]` and an empty architecture the towers are the
identity and the head is one linear layer on the full input.

Determinism (G2): Glorot init from a single seeded `default_rng` in the
fixed layer order (tower1, then tower2, then head); the forward is a plain
float64 matmul chain; the analytic gradients are verified against
central differences (the B1 AdapterMLP precedent).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import ACTIVATIONS, SpecError
from .mlp import _ACT_FNS, _row_idx, _sample_weights  # SPEC.md 70 helpers


class LateFusionMLP:
    """Input (n, d_img + d_txt) -> tower(img) + tower(txt) -> head -> logits."""

    def __init__(
        self,
        d_img: int,
        d_txt: int,
        hidden_sizes: Sequence[int],
        n_out: int,
        activation: str,
        seed: int,
        head: str = "softmax",
        init_scale: float = 1.0,
    ) -> None:
        if d_img < 1 or d_txt < 1:
            raise SpecError(
                f"LateFusionMLP needs d_img >= 1 and d_txt >= 1, got "
                f"d_img={d_img}, d_txt={d_txt} (SPEC.md 99)")
        if activation not in ACTIVATIONS:
            raise SpecError(f"unknown activation {activation!r}")
        if head not in ("softmax", "mse"):
            raise SpecError(f"unknown head {head!r}")
        if init_scale <= 0.0:
            raise SpecError(f"init_scale {init_scale!r} must be positive")
        self.d_img = int(d_img)
        self.d_txt = int(d_txt)
        self.in_dim = int(d_img) + int(d_txt)
        self.n_out = int(n_out)
        self.head = head
        self.activation = activation
        self._act_fn, self._act_grad_fn = _ACT_FNS[activation]  # SPEC.md 70
        self._is_mse = head == "mse"

        sizes = [int(h) for h in hidden_sizes]
        rng = np.random.default_rng(seed)

        def _layer(fan_in: int, fan_out: int):
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            return (rng.uniform(-limit, limit, (fan_in, fan_out)),
                    np.zeros(fan_out))

        # fixed draw order: tower1, tower2, head (G2)
        self._tower1: list[tuple[np.ndarray, np.ndarray]] = []
        prev = int(d_img)
        for h in sizes:
            self._tower1.append(_layer(prev, h))
            prev = h
        self._tower2: list[tuple[np.ndarray, np.ndarray]] = []
        prev = int(d_txt)
        for h in sizes:
            self._tower2.append(_layer(prev, h))
            prev = h
        h_last = int(sizes[-1]) if sizes else self.in_dim
        self._head = _layer(2 * h_last if sizes else self.in_dim, int(n_out))

        # the canonical `.layers` surface (SPEC.md 19.1 contract): the
        # optimizer loop, early stopping, save/load and weight transfer
        # all iterate this list
        self.layers = list(self._tower1) + list(self._tower2) + [self._head]
        self._cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self._h1: np.ndarray | None = None
        self._h2: np.ndarray | None = None
        self._concat: np.ndarray | None = None

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.in_dim:
            raise SpecError(
                f"LateFusionMLP expects (n, {self.in_dim}) input, got "
                f"{x.shape} (SPEC.md 99)")
        xi, xt = x[:, :self.d_img], x[:, self.d_img:]
        self._cache = []
        a = xi
        for w, b in self._tower1:
            z = a @ w + b
            h = self._act_fn(z)
            self._cache.append(("t1", a, z, h))
            a = h
        self._h1 = a
        a = xt
        for w, b in self._tower2:
            z = a @ w + b
            h = self._act_fn(z)
            self._cache.append(("t2", a, z, h))
            a = h
        self._h2 = a
        self._concat = np.concatenate([self._h1, self._h2], axis=1)
        w, b = self._head
        logits = self._concat @ w + b
        self._logits = logits
        return logits

    # --- loss + analytic gradients (the exact MLP head math, SPEC.md 70) ----
    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label_smoothing: float = 0.0,
        weights: np.ndarray | None = None,
    ) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
        """`(loss, grads)` with `grads` in `.layers` order (tower1, tower2,
        head) — the same head/CE/MSE/label-smoothing/sample-weight law as
        `MLP.loss_and_grads` (SPEC.md 17/74/84.1)."""
        n = x.shape[0]
        logits = self.forward(x)
        if self._is_mse:
            target = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
            diff = logits - target
            if weights is None:
                loss = float((diff * diff).mean())
                # elementwise mean over batch AND outputs -> n*n_out; for
                # n_out == 1 this is bit-identical to the MLP `2*diff/n`
                dz = 2.0 * diff / (n * self.n_out)
            else:
                w, W = _sample_weights(weights, n)
                loss = float((w * (diff * diff).sum(axis=1)).sum() / W)
                dz = 2.0 * diff * (w / W)[:, None]
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            onehot = np.zeros_like(logz)
            onehot[_row_idx(n), y] = 1.0  # SPEC.md 70: cached arange
            eps = float(label_smoothing)
            target = (onehot if eps == 0.0
                      else (1.0 - eps) * onehot + eps / self.n_out)
            if weights is None:
                loss = float(-(target * logz).sum(axis=1).mean())
                dz = (np.exp(logz) - target) / n
            else:
                w, W = _sample_weights(weights, n)
                loss = float((w * (-(target * logz)).sum(axis=1)).sum() / W)
                dz = (np.exp(logz) - target) * (w / W)[:, None]

        # head
        w_h, _b_h = self._head
        concat = self._concat
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        grads.append((concat.T @ dz, dz.sum(axis=0)))
        d_concat = dz @ w_h.T

        h_last = (self._h1.shape[1] if self._tower1
                  else self.d_img)
        d_h1 = d_concat[:, :h_last]
        d_h2 = d_concat[:, h_last:]
        # The tower loop invariant is `d = dL/dz_i` (PRE-activation gradient,
        # the same invariant MLP's backward relies on).  `d_h1` / `d_h2`
        # arrive as `dL/dh_final` — the POST-activation gradient — so first
        # push them through the last tower layer's nonlinearity.  A plain
        # MLP never needs this step because its final layer is a linear
        # head; a tower's final layer is activated, so skipping this is
        # exactly the "towers off by act'(z)" gradient bug (SPEC.md 99).
        if self._tower1:
            _a, z_last, h_last_act = self._t1_cache(len(self._tower1) - 1)
            d_h1 = d_h1 * self._act_grad_fn(z_last, h_last_act)
        if self._tower2:
            _a, z_last, h_last_act = self._t2_cache(len(self._tower2) - 1)
            d_h2 = d_h2 * self._act_grad_fn(z_last, h_last_act)
        for i in range(len(self._tower2) - 1, -1, -1):
            a, _z, _h = self._t2_cache(i)
            w, _b = self._tower2[i]
            grads.append((a.T @ d_h2, d_h2.sum(axis=0)))
            if i > 0:
                z_prev = self._t2_cache(i - 1)[1]
                h_prev = self._t2_cache(i - 1)[2]
                d_h2 = d_h2 @ w.T * self._act_grad_fn(z_prev, h_prev)
        for i in range(len(self._tower1) - 1, -1, -1):
            a, _z, _h = self._t1_cache(i)
            w, _b = self._tower1[i]
            grads.append((a.T @ d_h1, d_h1.sum(axis=0)))
            if i > 0:
                z_prev = self._t1_cache(i - 1)[1]
                h_prev = self._t1_cache(i - 1)[2]
                d_h1 = d_h1 @ w.T * self._act_grad_fn(z_prev, h_prev)
        grads.reverse()  # -> (.layers order: tower1, tower2, head)
        return loss, grads

    # --- cache helpers (tower-local indexing over the shared _cache) --------
    def _t1_cache(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tag, a, z, h = self._cache[i]
        if tag != "t1":
            raise RuntimeError("tower1 cache misaligned (SPEC.md 99)")
        return a, z, h

    def _t2_cache(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tag, a, z, h = self._cache[len(self._tower1) + i]
        if tag != "t2":
            raise RuntimeError("tower2 cache misaligned (SPEC.md 99)")
        return a, z, h

    # --- (de)serialization (the MLP npz surface + the split markers) --------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        arrays["n_layers"] = np.array(len(self.layers))
        arrays["d_img"] = np.array(self.d_img)
        arrays["d_txt"] = np.array(self.d_txt)
        arrays["n_tower1"] = np.array(len(self._tower1))
        arrays["n_tower2"] = np.array(len(self._tower2))
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "LateFusionMLP":
        with np.load(path, allow_pickle=False) as z:
            layers = [
                (z[f"layer{i}_w"].astype(np.float64),
                 z[f"layer{i}_b"].astype(np.float64))
                for i in range(int(z["n_layers"]))
            ]
            d_img = int(z["d_img"])
            d_txt = int(z["d_txt"])
            n_t1 = int(z["n_tower1"])
            n_t2 = int(z["n_tower2"])
            activation = str(z["activation"])
            head = str(z["head"])
            n_out = int(z["n_out"])
        m = cls.__new__(cls)
        m.d_img = d_img
        m.d_txt = d_txt
        m.in_dim = d_img + d_txt
        m.n_out = n_out
        m.head = head
        m.activation = activation
        m._act_fn, m._act_grad_fn = _ACT_FNS[activation]
        m._is_mse = head == "mse"
        m._tower1 = layers[:n_t1]
        m._tower2 = layers[n_t1:n_t1 + n_t2]
        m._head = layers[n_t1 + n_t2]
        m.layers = list(layers)
        m._cache = []
        m._h1 = None
        m._h2 = None
        m._concat = None
        return m
