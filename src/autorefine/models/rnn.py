"""Elman RNN: a small NumPy recurrent family over (T, C) sequences (SPEC.md 83).

The second native temporal model (83.3): a tanh Elman unit unrolled over the
time axis, reading out the **final** hidden state through a linear head. It
plugs into the shared neural training loop (83.3.5) because it exposes the
same `.layers` (list of `(w, b)`) + `.loss_and_grads(x, y, label_smoothing)`
interface as `MLP` / `ConvNet` / `Conv1D`.

Design (deterministic, pure numpy -- no new dependency, SPEC.md 3):
  * **Input** -- `X` is `(n, T, C)`; `forward` also accepts flat `(n, T*C)`
    (the 25.3 / 83.2.2 robustness rule). `T` and `C` are fixed at
    construction from the task's sequence layout.
  * **Cell** -- `h_t = act(W_in x_t + b_in + W_h h_{t-1} + b_h)`, `h_0 = 0`,
    `act` in {tanh, relu} (the spec's `activation`). Separate `b_in` / `b_h`
    biases keep each `.layers` slot independent (clean per-slot optimizer
    state, SPEC.md 83.3.5).
  * **Readout** -- `logits = W_out h_T + b_out` (final hidden state).
  * **Backprop** -- exact BPTT over the full unroll (short horizons --
    `rnn_steps` is the offered time axis, 83.3.3); no truncation, no RNG in
    the backward pass (G2).

`architecture` is unused (like conv1d, 83.2.3): the knob is the dedicated
`rnn_hidden` (H) spec field (83.3.3). Glorot x `init_scale` init; checkpoints
are plain arrays only (`allow_pickle=False`, SPEC.md 9).
"""
from __future__ import annotations

import numpy as np

from ..config import SpecError

from .mlp import HEADS, _ACT_FNS, _row_idx, _sample_weights


class RNN:
    """Elman RNN over a (T, C) sequence; `forward(x) -> (n, n_out)` logits."""

    wants_sequence = True  # SPEC.md 83.1: a sequence-layout task family

    def __init__(
        self,
        timesteps: int,
        channels: int,
        hidden: int,
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
        T, C, H = int(timesteps), int(channels), int(hidden)
        if T < 1 or C < 1 or H < 1:
            raise SpecError(
                f"rnn needs T, C, H >= 1, got T={T}, C={C}, H={H}")
        self.T, self.C, self.H = T, C, H
        self.n_out = int(n_out)
        self.activation = activation
        self._act_fn, self._act_grad_fn = _ACT_FNS[activation]
        self.head = head
        self.init_scale = float(init_scale)

        rng = np.random.default_rng(seed)

        def glorot(fan_in: int, fan_out: int, shape) -> np.ndarray:
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, shape)

        w_in = glorot(C, H, (C, H))       # input -> hidden
        b_in = np.zeros(H)                 # input bias
        w_h = glorot(H, H, (H, H))         # hidden -> hidden
        b_h = np.zeros(H)                  # recurrent bias
        w_out = glorot(H, n_out, (H, n_out))
        b_out = np.zeros(n_out)
        # `.layers` order [W_in, W_h, W_out]; the shared loop consumes grads in
        # this same order (SPEC.md 83.3.5). Each slot is an independent (w, b).
        self.layers: list[tuple[np.ndarray, np.ndarray]] = [
            (w_in, b_in), (w_h, b_h), (w_out, b_out),
        ]
        self._cache: list = []  # per-timestep (h_prev, x_t, h_t, z_t)

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:  # flat (n, T*C) -> (n, T, C) (83.2.2 robustness)
            if x.shape[1] != self.T * self.C:
                raise SpecError(
                    f"rnn flat input width {x.shape[1]} is not "
                    f"T*C={self.T * self.C} (T={self.T}, C={self.C})")
            x = x.reshape(x.shape[0], self.T, self.C)
        if x.ndim != 3 or x.shape[1:] != (self.T, self.C):
            raise SpecError(
                f"rnn expects (n, T={self.T}, C={self.C}) (or flat "
                f"{self.T * self.C}), got {x.shape}")
        n = x.shape[0]

        w_in, b_in = self.layers[0]
        w_h, b_h = self.layers[1]
        w_out, b_out = self.layers[2]

        act = self._act_fn
        h = np.zeros((n, self.H), dtype=np.float64)
        self._cache = []
        for t in range(self.T):
            h_prev = h
            xt = x[:, t, :]
            z = xt @ w_in + b_in + h_prev @ w_h + b_h
            h = act(z)
            self._cache.append((h_prev, xt, h, z))
        logits = h @ w_out + b_out
        return logits

    # --- loss + analytic gradients -------------------------------------------
    def loss_and_grads(self, x, y, label_smoothing: float = 0.0,
                       weights: np.ndarray | None = None):
        """Returns (loss, [(Wg, bg), ...]) in `.layers` order (SPEC.md 83.3.5).

        `weights` (SPEC.md 84.1): optional per-example weights (weighted-mean
        loss/gradient); `None` is the exact legacy path (bit-identical)."""
        n = x.shape[0]
        x = np.asarray(x, dtype=np.float64)
        logits = self.forward(x)  # the reshape + SpecError live in forward
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

        w_in, b_in = self.layers[0]
        w_h, b_h = self.layers[1]
        w_out, b_out = self.layers[2]
        act_grad = self._act_grad_fn

        # readout (final hidden state)
        h_final = self._cache[-1][2]
        gw_out = h_final.T @ dz
        gb_out = dz.sum(axis=0)
        dh = dz @ w_out.T  # dL/dh_T

        gw_in = np.zeros_like(w_in)
        gb_in = np.zeros_like(b_in)
        gw_h = np.zeros_like(w_h)
        gb_h = np.zeros_like(b_h)
        # BPTT: backprop through the unrolled cell (reverse time order)
        for t in range(self.T - 1, -1, -1):
            h_prev, xt, h_t, z_t = self._cache[t]
            dz_t = dh * act_grad(z_t, h_t)
            gw_in += xt.T @ dz_t
            gb_in += dz_t.sum(axis=0)
            gw_h += h_prev.T @ dz_t
            gb_h += dz_t.sum(axis=0)
            dh = dz_t @ w_h.T  # dL/dh_{t-1}

        return loss, [(gw_in, gb_in), (gw_h, gb_h), (gw_out, gb_out)]

    # --- (de)serialization: plain arrays only (allow_pickle=False) -----------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        arrays["n_layers"] = np.array(len(self.layers))
        arrays["T"] = np.array(self.T)
        arrays["C"] = np.array(self.C)
        arrays["H"] = np.array(self.H)
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        arrays["init_scale"] = np.array(self.init_scale)
        arrays["family"] = np.array("rnn")
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "RNN":
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            layers = [
                (z[f"layer{i}_w"].astype(np.float64),
                 z[f"layer{i}_b"].astype(np.float64))
                for i in range(n_layers)
            ]
            T, C, H = int(z["T"]), int(z["C"]), int(z["H"])
            n_out = int(z["n_out"])
            activation = str(z["activation"])
            head = str(z["head"]) if "head" in z.files else "softmax"
            init_scale = float(z["init_scale"]) if "init_scale" in z.files else 1.0
        m = cls.__new__(cls)
        m.layers = layers
        m.T, m.C, m.H = T, C, H
        m.n_out = n_out
        m.activation = activation
        m._act_fn, m._act_grad_fn = _ACT_FNS[activation]
        m.head = head
        m.init_scale = init_scale
        m._cache = []
        return m
