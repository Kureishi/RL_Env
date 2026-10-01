"""Conv1D: a small NumPy temporal-conv family over (T, C) sequences (SPEC.md 83).

The native temporal model (83.2): one causal 1-D convolution over a
(T, C) input -- a sequence of ``C`` channels across ``T`` timesteps -- then
a tanh/relu, a stride-2 time pooling, a flatten, and a linear head
``(n, n_out)``. It is the temporal analogue of `models.convnet.ConvNet`
(spatial 3x3 -> temporal K-wide) and plugs into the shared neural training
loop (83.2.4) because it exposes the same `.layers` (list of `(w, b)`) +
`.loss_and_grads(x, y, label_smoothing)` interface as `MLP` / `ConvNet`.

Design (deterministic, pure numpy -- no new dependency, SPEC.md 3):
  * **Input** -- `X` is `(n, T, C)` (n sequences, T timesteps, C channels);
    `forward` also accepts flat `(n, T*C)` and reshapes (the 25.3 robustness
    rule). This is the sequence layout a `capabilities={"sequence"}` task
    yields from `make_dataset` (83.1).
  * **Conv** -- a causal valid 1-D conv with a `(F, C, K)` kernel (F output
    filters, C input channels, K the time width): `out[t, f] = sum_{c,k}
    w[f, c, k] * x[t+k, c] + b[f]`, `t in 0..T-K`. Im2col along the time axis
    (a `K`-wide gather), then one matmul per sample block.
  * **Pool** -- stride-2 time pooling (average over the true 1-D window),
    mirroring the convnet's 2x2 pool guard for short time axes.
  * **Head** -- flatten -> FC -> linear head, exactly the convnet tail.

`architecture` is unused (like knn/gp/gam, 25.2): the family's knobs are the
dedicated `conv1d_filters` (F) and `conv1d_kernel` (K) spec fields (83.2.3).
FC hidden 16 is fixed (off the spec surface, the `FC_HIDDEN` rule). Glorot x
`init_scale` init; checkpoints are plain arrays only (`allow_pickle=False`).
"""
from __future__ import annotations

import numpy as np

from ..config import SpecError

from .mlp import HEADS, _ACT_FNS, _row_idx, _sample_weights

# SPEC.md 83.2.3: FC hidden size is fixed (off the spec surface, 25.3 rule).
FC_HIDDEN = 16


def _time_pool(x: np.ndarray) -> np.ndarray:
    """Stride-2 average pool along the time axis of (n, T, F) -> (n, T//2, F).

    Mirrors the convnet's 2x2 time guard: a trailing odd step averages over a
    1-wide window (so a length-1 time axis passes through unchanged)."""
    n, t, f = x.shape
    out_t = max(1, t // 2)
    out = np.zeros((n, out_t, f), dtype=np.float64)
    for i in range(out_t):
        s = x[:, 2 * i:2 * i + 2, :]  # (n, w, f), w in {1, 2}
        out[:, i, :] = s.mean(axis=1)
    return out


def _time_pool_grad(dout: np.ndarray, pre_t: int, f: int) -> np.ndarray:
    """Backprop of `_time_pool`: scatter each output grad to its 1-D window,
    divided by the true window size so it matches the forward `.mean`."""
    n, out_t, _ = dout.shape
    out = np.zeros((n, pre_t, f), dtype=np.float64)
    for i in range(out_t):
        w = min(2, pre_t - 2 * i)
        # add the (n, F) output grad to each of the `w` pooled time-slices
        out[:, 2 * i:2 * i + w, :] += (dout[:, i, :] / w)[:, None, :]
    return out


def _conv1d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Causal valid 1-D conv of (n, T, C) with kernel (F, C, K) -> (n, T-K+1, F)."""
    n, t, c = x.shape
    f, _, k = w.shape
    out_t = t - k + 1
    if out_t < 1:
        raise SpecError(
            f"conv1d time {t} shorter than kernel {k} (need T >= K)")
    out = np.zeros((n, out_t, f), dtype=np.float64)
    for i in range(out_t):
        window = x[:, i:i + k, :]  # (n, k, c)
        # sum over (k, c): out[:, i, f] = sum_{k,c} w[f, c, k] * x[i+k, c]
        out[:, i, :] = np.einsum("nkc,fck->nf", window, w)
    out += b
    return out


def _conv1d_grad(x: np.ndarray, dout: np.ndarray, w: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop of `_conv1d_forward`. Returns (dw, db, dx)."""
    n, t, c = x.shape
    f, _, k = w.shape
    out_t = dout.shape[1]
    db = dout.sum(axis=(0, 1))  # (f,)
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    for i in range(out_t):
        window = x[:, i:i + k, :]  # (n, k, c)
        dout_i = dout[:, i, :]    # (n, f)
        # dw[f, c, j] += sum_n dout[n, i, f] * x[n, i+j, c]
        dw += np.einsum("nf,nkc->fck", dout_i, window)
        # dx[n, i+j, c] += sum_f dout[n, i, f] * w[f, c, j]
        dx[:, i:i + k, :] += np.einsum("nf,fck->nkc", dout_i, w)
    return dw, db, dx


class Conv1D:
    """Temporal conv over a (T, C) sequence; `forward(x) -> (n, n_out)` logits."""

    wants_sequence = True  # SPEC.md 83.1: a sequence-layout task family

    def __init__(
        self,
        timesteps: int,
        channels: int,
        filters: int,
        kernel: int,
        n_out: int,
        activation: str,
        seed: int,
        head: str = "softmax",
        init_scale: float = 1.0,
    ) -> None:
        if activation not in ("tanh", "relu"):
            raise SpecError(f"unknown activation {activation!r}")
        if head not in HEADS:
            raise SpecError(f"unknown head {head!r}")
        if init_scale <= 0.0:
            raise SpecError(f"init_scale {init_scale!r} must be positive")
        T, C, F, K = int(timesteps), int(channels), int(filters), int(kernel)
        if T < 2 or C < 1:
            raise SpecError(
                f"conv1d needs T >= 2 and C >= 1, got T={T}, C={C}")
        if K < 1 or K > T:
            raise SpecError(f"conv1d kernel {K} outside 1..T={T}")
        self.T, self.C, self.F, self.K = T, C, F, K
        self.n_out = int(n_out)
        self.activation = activation
        self._act_fn, self._act_grad_fn = _ACT_FNS[activation]
        self.head = head
        self.init_scale = float(init_scale)

        rng = np.random.default_rng(seed)

        def glorot(fan_in: int, fan_out: int, shape) -> np.ndarray:
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, shape)

        w_conv = glorot(C * K, F, (F, C, K))
        b_conv = np.zeros(F)
        # time dim after conv + pool (mirrors the convnet pooled-dim math)
        t1 = T - K + 1
        t1p = max(1, t1 // 2)
        flat = F * t1p
        wf = glorot(flat, FC_HIDDEN, (flat, FC_HIDDEN))
        bf = np.zeros(FC_HIDDEN)
        wh = glorot(FC_HIDDEN, n_out, (FC_HIDDEN, n_out))
        bh = np.zeros(n_out)
        # `.layers` order [conv, fc, head]; the shared loop consumes grads in
        # this same order (SPEC.md 83.2.4).
        self.layers: list[tuple[np.ndarray, np.ndarray]] = [
            (w_conv, b_conv), (wf, bf), (wh, bh),
        ]
        self._cache: dict = {}

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:  # flat (n, T*C) -> (n, T, C) (83.2.2 robustness)
            if x.shape[1] != self.T * self.C:
                raise SpecError(
                    f"conv1d flat input width {x.shape[1]} is not "
                    f"T*C={self.T * self.C} (T={self.T}, C={self.C})")
            x = x.reshape(x.shape[0], self.T, self.C)
        if x.ndim != 3 or x.shape[1:] != (self.T, self.C):
            raise SpecError(
                f"conv1d expects (n, T={self.T}, C={self.C}) (or flat "
                f"{self.T * self.C}), got {x.shape}")
        n = x.shape[0]

        w_conv, b_conv = self.layers[0]
        wf, bf = self.layers[1]
        wh, bh = self.layers[2]

        z1 = _conv1d_forward(x, w_conv, b_conv)  # (n, t1, F)
        act = self._act_fn
        a1 = act(z1)
        p1 = _time_pool(a1)  # (n, t1p, F)
        f = p1.reshape(n, self.F * p1.shape[1])
        zf = f @ wf + bf
        af = act(zf)
        logits = af @ wh + bh
        self._cache = {"x": x, "z1": z1, "a1": a1, "p1": p1,
                       "f": f, "zf": zf, "af": af}
        return logits

    # --- loss + analytic gradients -------------------------------------------
    def loss_and_grads(self, x, y, label_smoothing: float = 0.0,
                       weights: np.ndarray | None = None):
        """Returns (loss, [(Wg, bg), ...]) in `.layers` order (SPEC.md 83.2.4).

        `weights` (SPEC.md 84.1): optional per-example weights (weighted-mean
        loss/gradient); `None` is the exact legacy path (bit-identical)."""
        n = x.shape[0]
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.T, self.C)
        logits = self.forward(x)
        if self.head == "mse":
            target = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
            diff = logits - target
            if weights is None:
                loss = float((diff * diff).mean())
                dz = 2.0 * diff / n
            else:  # SPEC.md 84.1
                w, W = _sample_weights(weights, n)
                loss = float((w * (diff * diff).sum(axis=1)).sum() / W)
                dz = 2.0 * diff * (w / W)[:, None]
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            onehot = np.zeros_like(logz)
            onehot[_row_idx(n), y] = 1.0
            eps = float(label_smoothing)
            target = onehot if eps == 0.0 else \
                (1.0 - eps) * onehot + eps / self.n_out
            if weights is None:
                loss = float(-(target * logz).sum(axis=1).mean())
                dz = (np.exp(logz) - target) / n
            else:  # SPEC.md 84.1
                w, W = _sample_weights(weights, n)
                loss = float((w * (-(target * logz)).sum(axis=1)).sum() / W)
                dz = (np.exp(logz) - target) * (w / W)[:, None]

        c = self._cache
        act_grad = self._act_grad_fn
        # head
        gw_h = c["af"].T @ dz
        gb_h = dz.sum(axis=0)
        daf = dz @ self.layers[2][0].T
        daf *= act_grad(c["zf"], c["af"])
        # fc
        gw_f = c["f"].T @ daf
        gb_f = daf.sum(axis=0)
        dp1 = daf @ self.layers[1][0].T
        # unflatten back to the pool's (n, out_t, F) layout (the forward
        # flattened (n, out_t, F) -> (n, out_t*F) row-major)
        dp1 = dp1.reshape(n, c["p1"].shape[1], self.F)
        da1 = _time_pool_grad(dp1, c["a1"].shape[1], self.F)
        da1 *= act_grad(c["z1"], c["a1"])
        # conv
        dw_conv, db_conv, _dx = _conv1d_grad(c["x"], da1, self.layers[0][0])

        return loss, [(dw_conv, db_conv), (gw_f, gb_f), (gw_h, gb_h)]

    # --- (de)serialization: plain arrays only (allow_pickle=False) -----------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        arrays["n_layers"] = np.array(len(self.layers))
        arrays["T"] = np.array(self.T)
        arrays["C"] = np.array(self.C)
        arrays["F"] = np.array(self.F)
        arrays["K"] = np.array(self.K)
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        arrays["init_scale"] = np.array(self.init_scale)
        arrays["family"] = np.array("conv1d")
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Conv1D":
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            layers = [
                (z[f"layer{i}_w"].astype(np.float64),
                 z[f"layer{i}_b"].astype(np.float64))
                for i in range(n_layers)
            ]
            T, C, F, K = int(z["T"]), int(z["C"]), int(z["F"]), int(z["K"])
            n_out = int(z["n_out"])
            activation = str(z["activation"])
            head = str(z["head"]) if "head" in z.files else "softmax"
            init_scale = float(z["init_scale"]) if "init_scale" in z.files else 1.0
        m = cls.__new__(cls)
        m.layers = layers
        m.T, m.C, m.F, m.K = T, C, F, K
        m.n_out = n_out
        m.activation = activation
        m._act_fn, m._act_grad_fn = _ACT_FNS[activation]
        m.head = head
        m.init_scale = init_scale
        m._cache = {}
        return m
