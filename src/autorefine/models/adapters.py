"""AdapterMLP: a frozen MLP backbone + trainable rank-r adapters (A89.6).

The fine-tuning workhorse (SPEC.md 99, B1): the backbone is the ordinary
MLP (the SAME Glorot init as ``MLP`` for the same seed — so an adapter
model starts exactly where a plain one would) but its weights are FROZEN;
the only trainable parameters are low-rank adapter pairs:

    hidden block i (input d_i -> width h_i):   out = act(W_i x + b_i) + U_i (V_i x)
    head block (input h_last -> n_out):        logits = W_h x + b_h + Q (P x)

with ``V_i: (d_i, r)``, ``U_i: (r, h_i)``, ``P: (h_last, r)``,
``Q: (r, n_out)``. A depth-0 (linear) backbone has a single head block.
The trainable parameter count is the adapter count only (the
``config.spec_n_params`` adapter branch): for a (16, 8) backbone and
n_out 2, rank 4 trains 16*4 + 4*16 + 8*4 + 4*2 = 224 parameters instead of
the full 290.

The class exposes exactly the ``_train_neural`` contract (SPEC.md 19.1):

* ``.layers`` — the trainable (w, b)-style pairs ``[(V0, U0), ..., (P, Q)]``
  (the optimizer/early-stopping code treats the first element of each pair
  as the weight and the second as the bias — both are adapter matrices here,
  and both are trainable, so weight decay applies to both, as intended);
* ``.loss_and_grads(x, y, label_smoothing=, weights=)`` — (loss, grads) with
  the grads in the same order as ``.layers``;
* ``.forward(x)`` — the logits (the tasks' scoring contract);
* ``.save(path)`` / ``.load(path)`` — the checkpoint contract, tagged with
  the ``rank`` / ``ad{i}_V`` keys so ``finetune._detect_family`` routes
  these files to ``AdapterMLP.load`` (never to ``MLP.load``).

Deterministic given a seed (G2); the adapter init uses a SEPARATE seeded
stream (``seed + 1``) so the backbone stays bit-identical to ``MLP``'s.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import ACTIVATIONS, SpecError
from ._fourier import apply_scaled
from .mlp import HEADS, _ACT_FNS, _row_idx, _sample_weights


class AdapterMLP:
    """Frozen-backbone MLP + rank-r adapters (the B1 fine-tune mode)."""

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        n_out: int,
        activation: str,
        seed: int,
        head: str = "softmax",
        init_scale: float = 1.0,
        rank: int = 4,
        fourier_K: int = 0,
        fourier_scale: np.ndarray | None = None,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise SpecError(f"unknown activation {activation!r}")
        if head not in HEADS:
            raise SpecError(f"unknown head {head!r}")
        if init_scale <= 0.0:
            raise SpecError(f"init_scale {init_scale!r} must be positive")
        if int(rank) < 1:
            raise SpecError(f"adapter rank must be >= 1, got {rank!r}")
        self.rank = int(rank)
        # --- frozen backbone: the exact MLP init (same seed, same Glorot) ----
        rng = np.random.default_rng(seed)
        sizes = [in_dim, *hidden_sizes, n_out]
        self.backbone: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.backbone.append((w, b))
        # --- trainable adapters: a separate deterministic stream (G2) -------
        a_rng = np.random.default_rng(seed + 1)
        self.adapters: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            v = a_rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, self.rank))
            u = a_rng.normal(0.0, np.sqrt(1.0 / self.rank), (self.rank, fan_out))
            self.adapters.append((v, u))
        # --- the shared-loop interface (.layers == the trainable pairs) -----
        self.layers = self.adapters
        self.activation = activation
        self._act_grad_fn = _ACT_FNS[activation][1]
        self.in_dim = in_dim
        self.n_out = n_out
        self.head = head
        self._is_mse = head == "mse"
        self.fourier_K = int(fourier_K)
        self.fourier_scale = fourier_scale
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- forward -----------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        xx = np.asarray(x, dtype=np.float64)
        if self.fourier_K > 0 and self.fourier_scale is not None:
            xx = apply_scaled(xx, self.fourier_K, self.fourier_scale)
        act = _ACT_FNS[self.activation][0]
        a = xx
        for i in range(len(self.backbone) - 1):
            w, b = self.backbone[i]
            v, u = self.adapters[i]
            z = a @ w + b
            h = act(z)
            a_out = h + (a @ v) @ u
            # cache the block's INPUT (a) — loss_and_grads differentiates the
            # adapter path w.r.t. it; the head entry stores the head's input.
            self._cache.append((a, z, h))
            a = a_out
        w, b = self.backbone[-1]
        p, q = self.adapters[-1]
        logits = a @ w + b + (a @ p) @ q
        self._cache.append((a, logits, logits))
        return logits

    # --- loss + analytic gradients -----------------------------------------
    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label_smoothing: float = 0.0,
        weights: np.ndarray | None = None,
    ) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
        """(loss, [(gV0, gU0), ..., (gP, gQ)]) — the same heads, the same
        weighted-mean law as ``MLP.loss_and_grads`` (SPEC.md 17 / 84.1),
        with gradients only for the trainable adapters."""
        n = x.shape[0]
        logits = self.forward(x)
        if self._is_mse:
            target = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
            diff = logits - target
            if weights is None:
                loss = float((diff * diff).mean())
                dz = 2.0 * diff / n
            else:
                w, W = _sample_weights(weights, n)
                loss = float((w * (diff * diff).sum(axis=1)).sum() / W)
                dz = 2.0 * diff * (w / W)[:, None]
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            onehot = np.zeros_like(logz)
            onehot[_row_idx(n), y] = 1.0
            eps = float(label_smoothing)
            target = onehot if eps == 0.0 else (1.0 - eps) * onehot + eps / self.n_out
            if weights is None:
                loss = float(-(target * logz).sum(axis=1).mean())
                dz = (np.exp(logz) - target) / n
            else:
                w, W = _sample_weights(weights, n)
                loss = float((w * (-(target * logz)).sum(axis=1)).sum() / W)
                dz = (np.exp(logz) - target) * (w / W)[:, None]

        grads: list[tuple[np.ndarray, np.ndarray]] = []
        # head block: logits = a @ W + b + (a @ P) @ Q
        a = self._cache[-1][0]
        p, q = self.adapters[-1]
        hr = a @ p
        gq = hr.T @ dz
        gp = a.T @ (dz @ q.T)
        grads.append((gp, gq))
        # chain into the head block's input (main path + adapter path)
        w_last, _b_last = self.backbone[-1]
        d = dz @ w_last.T + dz @ q.T @ p.T
        # hidden blocks, back to front. The nonlinearity producing block i's
        # input is block i-1's activation (z_{i-1}); block 0's input is the
        # raw (fourier-expanded) feature row — no act' there (MLP pattern).
        for i in range(len(self.backbone) - 2, -1, -1):
            a_in, _z, _h = self._cache[i]
            w, _b = self.backbone[i]
            v, u = self.adapters[i]
            gu = (a_in @ v).T @ d
            gv = a_in.T @ (d @ u.T)
            grads.append((gv, gu))
            if i > 0:
                z_prev = self._cache[i - 1][1]
                h_prev = self._cache[i - 1][2]
                d = d @ w.T * self._act_grad_fn(z_prev, h_prev) \
                    + d @ u.T @ v.T
            else:
                d = d @ w.T + d @ u.T @ v.T
        grads.reverse()
        return loss, grads

    # --- (de)serialization ---------------------------------------------------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.backbone):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        for i, (v, u) in enumerate(self.adapters):
            arrays[f"ad{i}_V"] = v
            arrays[f"ad{i}_U"] = u
        arrays["n_blocks"] = np.array(len(self.backbone))
        arrays["rank"] = np.array(self.rank)
        arrays["in_dim"] = np.array(self.in_dim)
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        arrays["fourier_K"] = np.array(self.fourier_K)
        if self.fourier_scale is not None:
            arrays["fourier_scale"] = np.asarray(self.fourier_scale,
                                                 dtype=np.float64)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "AdapterMLP":
        with np.load(path, allow_pickle=False) as z:
            if "rank" not in z.files or "ad0_V" not in z.files:
                raise ValueError(
                    f"{path} is not an AdapterMLP checkpoint "
                    f"(missing the 'rank' / 'ad0_V' markers)")
            n_blocks = int(z["n_blocks"])
            rank = int(z["rank"])
            backbone = [
                (z[f"layer{i}_w"].astype(np.float64),
                 z[f"layer{i}_b"].astype(np.float64))
                for i in range(n_blocks)
            ]
            adapters = [
                (z[f"ad{i}_V"].astype(np.float64),
                 z[f"ad{i}_U"].astype(np.float64))
                for i in range(n_blocks)
            ]
            activation = str(z["activation"])
            in_dim = int(z["in_dim"])
            n_out = int(z["n_out"])
            head = str(z["head"]) if "head" in z.files else "softmax"
            fourier_K = int(z["fourier_K"]) if "fourier_K" in z.files else 0
            fourier_scale = (z["fourier_scale"].astype(np.float64)
                             if fourier_K > 0 and "fourier_scale" in z.files
                             else None)
        m = cls.__new__(cls)
        m.rank = rank
        m.backbone = backbone
        m.adapters = adapters
        m.layers = adapters
        m.activation = activation
        m._act_grad_fn = _ACT_FNS[activation][1]
        m.in_dim = in_dim
        m.n_out = n_out
        m.head = head
        m._is_mse = head == "mse"
        m.fourier_K = fourier_K
        m.fourier_scale = fourier_scale
        m._cache = []
        return m