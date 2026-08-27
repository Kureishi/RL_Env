"""ConvNet: a small NumPy convnet over grid-structured features (SPEC.md 25.3).

The audio *temporal* model (log-mel spectrogram) and the image spatial model
share this family: conv 3x3 (C->c1) + act + 2x2 avg-pool -> conv 3x3 (c1->c2)
+ act + 2x2 avg-pool -> flatten -> FC(32) + act -> linear head (n, n_out).

`architecture = (c1, c2)`, each in CONV_FILTERS (SPEC.md 25.3); FC hidden 32
is fixed (deliberately off the spec surface). Im2col forward/backprop; Glorot
x `init_scale` init. Pooling applies only to dims >= 2 (guard for very short
time axes).

The model exposes the same `.layers` (list of `(w, b)`) +
`.loss_and_grads(x, y, label_smoothing)` interface as MLP, so the mlp
training loop (optimizer / LR schedule / grad-clip / early stopping / weight
decay / label smoothing / time cap) is shared verbatim (SPEC.md 25.3).
`wants_grid = True`; `forward` also accepts flat `(n, C*H*W)` input and
reshapes it (SPEC.md 25.3 robustness). Checkpoints are plain arrays only
(`allow_pickle=False`, SPEC.md 9).
"""
from __future__ import annotations

import numpy as np

from ..config import CONV_FILTERS, SpecError

from .mlp import HEADS, _activation, _activation_grad

# SPEC.md 25.3: FC hidden size is fixed (deliberately off the spec surface)
FC_HIDDEN = 32


def _im2col(x: np.ndarray, k: int = 3) -> tuple[np.ndarray, tuple[int, int]]:
    """Valid `k x k` im2col of (n, C, H, W) -> (n, C*k*k, (H-k+1)*(W-k+1))."""
    n, c, h, w = x.shape
    oh, ow = h - k + 1, w - k + 1
    col = np.empty((n, c, k, k, oh, ow), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            col[:, :, i, j, :, :] = x[:, :, i:i + oh, j:j + ow]
    return col.reshape(n, c * k * k, oh * ow), (oh, ow)


def _col2im(col: np.ndarray, oh: int, ow: int, n: int, c: int, h: int, w: int,
            k: int = 3) -> np.ndarray:
    """Inverse of `_im2col`: (n, C*k*k, oh*ow) -> (n, C, H, W) (accumulated)."""
    out = np.zeros((n, c, h, w), dtype=np.float64)
    patches = col.reshape(n, c, k, k, oh, ow)
    for i in range(k):
        for j in range(k):
            out[:, :, i:i + oh, j:j + ow] += patches[:, :, i, j, :, :]
    return out


def _avg_pool(x: np.ndarray) -> np.ndarray:
    """2x2 avg-pool per spatial dim; dims < 2 pass through (SPEC.md 25.3 guard)."""
    n, c, h, w = x.shape
    ph, pw = h // 2, w // 2
    if ph == 0:
        ph = 1
    if pw == 0:
        pw = 1
    out = np.zeros((n, c, ph, pw), dtype=np.float64)
    for i in range(ph):
        for j in range(pw):
            out[:, :, i, j] = x[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2].mean(axis=(2, 3))
    return out


class ConvNet:
    """Small convnet over a (C, H, W) grid; `forward(x) -> (n, n_out)` logits."""

    wants_grid = True  # SPEC.md 25.3: the task's score() routes grid rows to it

    def __init__(
        self,
        grid: tuple[int, int, int],
        c1: int,
        c2: int,
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
        C, H, W = (int(v) for v in grid)
        if C < 1 or H < 3 or W < 3:
            raise SpecError(
                f"convnet grid {(C, H, W)} too small for a valid 3x3 conv "
                "(need C >= 1, H, W >= 3)")
        c1, c2 = int(c1), int(c2)
        for c in (c1, c2):
            if c not in CONV_FILTERS:
                raise SpecError(f"convnet filter size {c} not in {CONV_FILTERS}")

        # spatial dims through the pipeline (valid 3x3 convs + 2x2 pools)
        def _pool_dim(d: int) -> int:
            return max(1, d // 2)
        h1, w1 = H - 2, W - 2
        h1p, w1p = _pool_dim(h1), _pool_dim(w1)
        if h1p < 3 or w1p < 3:
            raise SpecError(
                f"convnet grid {(C, H, W)} too small for the second conv layer "
                "(pooled dims must be >= 3)")
        h2, w2 = h1p - 2, w1p - 2
        h2p, w2p = _pool_dim(h2), _pool_dim(w2)
        flat_dim = c2 * h2p * w2p

        self.C, self.H, self.W = C, H, W
        self.c1, self.c2 = c1, c2
        self.n_out = int(n_out)
        self.activation = activation
        self.head = head
        self.init_scale = float(init_scale)
        self.flat_dim = flat_dim

        # Glorot x init_scale init (SPEC.md 25.3), deterministic given the seed
        rng = np.random.default_rng(seed)

        def glorot(fan_in: int, fan_out: int, shape) -> np.ndarray:
            limit = float(init_scale) * np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, shape)

        w1 = glorot(C * 9, c1, (c1, C, 3, 3))
        b1 = np.zeros(c1)
        w2 = glorot(c1 * 9, c2, (c2, c1, 3, 3))
        b2 = np.zeros(c2)
        # (fan_in, fan_out) orientation so `a @ w` matches the mlp convention
        wf = glorot(flat_dim, FC_HIDDEN, (flat_dim, FC_HIDDEN))
        bf = np.zeros(FC_HIDDEN)
        wh = glorot(FC_HIDDEN, n_out, (FC_HIDDEN, n_out))
        bh = np.zeros(n_out)
        # SPEC.md 25.3: `.layers` order [conv1, conv2, fc, head]; the mlp
        # training loop consumes gradients in this same order.
        self.layers: list[tuple[np.ndarray, np.ndarray]] = [
            (w1, b1), (w2, b2), (wf, bf), (wh, bh),
        ]
        self._cache: dict = {}

    # --- forward -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:  # flat (n, C*H*W) -> grid (SPEC.md 25.3 robustness)
            x = x.reshape(x.shape[0], self.C, self.H, self.W)
        elif x.ndim == 3:  # single grid sample (C, H, W)
            x = x[None, :]
        if x.ndim != 4 or x.shape[1:] != (self.C, self.H, self.W):
            raise SpecError(
                f"convnet expects grid {(self.C, self.H, self.W)} (or flat "
                f"{self.C * self.H * self.W}), got {x.shape}")
        n = x.shape[0]

        w1, b1 = self.layers[0]
        w2, b2 = self.layers[1]
        wf, bf = self.layers[2]
        wh, bh = self.layers[3]

        def _conv(inp, w, b):
            col, (oh, ow) = _im2col(inp, 3)  # (n, c_in*9, oh*ow)
            wmat = w.reshape(w.shape[0], -1).T  # (c_in*9, c_out)
            out = col.transpose(0, 2, 1) @ wmat + b  # (n, oh*ow, c_out)
            return out.transpose(0, 2, 1).reshape(n, w.shape[0], oh, ow), (oh, ow)

        z1, (oh1, ow1) = _conv(x, w1, b1)
        a1 = _activation(self.activation, z1)
        p1 = _avg_pool(a1)
        z2, (oh2, ow2) = _conv(p1, w2, b2)
        a2 = _activation(self.activation, z2)
        p2 = _avg_pool(a2)
        f = p2.reshape(n, self.flat_dim)
        zf = f @ wf + bf
        af = _activation(self.activation, zf)
        logits = af @ wh + bh
        self._cache = {
            "x": x, "z1": z1, "a1": a1, "p1": p1, "oh1": oh1, "ow1": ow1,
            "z2": z2, "a2": a2, "p2": p2, "oh2": oh2, "ow2": ow2,
            "f": f, "zf": zf, "af": af,
        }
        return logits

    # --- loss + analytic gradients -------------------------------------------
    def loss_and_grads(self, x, y, label_smoothing: float = 0.0):
        """Returns (loss, [(Wg, bg), ...]) in `.layers` order (SPEC.md 25.3)."""
        n = x.shape[0]
        logits = self.forward(x)
        if self.head == "mse":
            target = np.asarray(y, dtype=np.float64).reshape(n, self.n_out)
            diff = logits - target
            loss = float((diff * diff).mean())
            dz = 2.0 * diff / n
        else:
            z = logits - logits.max(axis=1, keepdims=True)
            logz = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            onehot = np.zeros_like(logz)
            onehot[np.arange(n), y] = 1.0
            eps = float(label_smoothing)
            target = (1.0 - eps) * onehot + eps / self.n_out
            loss = float(-(target * logz).sum(axis=1).mean())
            p = np.exp(logz)
            dz = (p - target) / n

        c = self._cache
        # head
        gw_h = c["af"].T @ dz
        gb_h = dz.sum(axis=0)
        daf = dz @ self.layers[3][0].T
        daf = daf * _activation_grad(self.activation, c["zf"], c["af"])
        # fc
        gw_f = c["f"].T @ daf
        gb_f = daf.sum(axis=0)
        dp2 = daf @ self.layers[2][0].T
        dp2 = dp2.reshape(n, self.c2, c["p2"].shape[2], c["p2"].shape[3])
        # pool2 backprop (2x2 avg: scatter each output's grad to its 2x2 window,
        # divided by the true block size so it matches the forward `.mean`)
        da2 = _avg_pool_grad(dp2, c["a2"].shape[2], c["a2"].shape[3])
        da2 = da2 * _activation_grad(self.activation, c["z2"], c["a2"])
        # conv2
        dw2, db2, da1 = _conv_grad(c["p1"], da2, self.layers[1][0], c["oh2"], c["ow2"])
        # pool1 backprop
        da1 = _avg_pool_grad(da1, c["a1"].shape[2], c["a1"].shape[3])
        da1 = da1 * _activation_grad(self.activation, c["z1"], c["a1"])
        # conv1
        dw1, db1, _dx = _conv_grad(c["x"], da1, self.layers[0][0], c["oh1"], c["ow1"])

        return loss, [(dw1, db1), (dw2, db2), (gw_f, gb_f), (gw_h, gb_h)]

    # --- (de)serialization: plain arrays only (allow_pickle=False) -----------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"layer{i}_w"] = w
            arrays[f"layer{i}_b"] = b
        arrays["n_layers"] = np.array(len(self.layers))
        arrays["C"] = np.array(self.C)
        arrays["H"] = np.array(self.H)
        arrays["W"] = np.array(self.W)
        arrays["c1"] = np.array(self.c1)
        arrays["c2"] = np.array(self.c2)
        arrays["n_out"] = np.array(self.n_out)
        arrays["activation"] = np.array(self.activation)
        arrays["head"] = np.array(self.head)
        arrays["init_scale"] = np.array(self.init_scale)
        arrays["flat_dim"] = np.array(self.flat_dim)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "ConvNet":
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            layers = [
                (z[f"layer{i}_w"].astype(np.float64), z[f"layer{i}_b"].astype(np.float64))
                for i in range(n_layers)
            ]
            C, H, W = int(z["C"]), int(z["H"]), int(z["W"])
            c1, c2 = int(z["c1"]), int(z["c2"])
            n_out = int(z["n_out"])
            activation = str(z["activation"])
            head = str(z["head"]) if "head" in z.files else "softmax"
            init_scale = float(z["init_scale"]) if "init_scale" in z.files else 1.0
            flat_dim = int(z["flat_dim"])
        m = cls.__new__(cls)
        m.layers = layers
        m.C, m.H, m.W = C, H, W
        m.c1, m.c2 = c1, c2
        m.n_out = n_out
        m.activation = activation
        m.head = head
        m.init_scale = init_scale
        m.flat_dim = flat_dim
        m._cache = {}
        return m


# --- conv backprop helpers --------------------------------------------------
def _conv_grad(inp: np.ndarray, dout: np.ndarray, w: np.ndarray,
               oh: int, ow: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop of a valid 3x3 conv. Returns (dw, db, dx)."""
    n, c_in, _, _ = inp.shape
    c_out = w.shape[0]
    col, _ = _im2col(inp, 3)  # (n, c_in*9, oh*ow)
    dout_mat = dout.transpose(0, 2, 3, 1).reshape(n, oh * ow, c_out)
    db = dout.sum(axis=(0, 2, 3))  # (c_out,)
    wmat = w.reshape(c_out, -1)  # (c_out, c_in*9)
    # dw[co, c9] = sum_{n,p} dout[n, p, co] * col[n, c9, p]
    dw_mat = (dout_mat.reshape(-1, c_out).T
              @ col.transpose(0, 2, 1).reshape(-1, c_in * 9))  # (c_out, c_in*9)
    dw = dw_mat.reshape(c_out, c_in, 3, 3)
    dcol = dout_mat @ wmat  # (n, oh*ow, c_in*9); wmat == W.T of the forward matmul
    dx = _col2im(dcol, oh, ow, n, c_in, inp.shape[2], inp.shape[3])
    return dw, db, dx


def _avg_pool_grad(dout: np.ndarray, pre_h: int, pre_w: int) -> np.ndarray:
    """Backprop of `_avg_pool`: scatter each output grad to its 2x2 window,
    divided by the true block size so it matches the forward `.mean` exactly."""
    n, c, oh, ow = dout.shape
    out = np.zeros((n, c, pre_h, pre_w), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            bh = min(2, pre_h - 2 * i)
            bw = min(2, pre_w - 2 * j)
            out[:, :, 2 * i:2 * i + bh, 2 * j:2 * j + bw] += \
                dout[:, :, i, j, None, None] / (bh * bw)
    return out
