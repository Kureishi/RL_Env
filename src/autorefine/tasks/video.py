"""VideoTask: a directory of labelled video clips as a fitting task
(SPEC.md 92.6, v0.78, A82).

The temporal-media modality: `imageio` (the optional `autorefine[video]`
extra, SPEC.md 92.6.1 — imported lazily, only when a clip is decoded, so
`import autorefine` stays imageio-free) reads the frames; each frame is a
deterministic grayscale block-average resize to `grid`×`grid` (the same
grayscale [0, 1] contract as ImageTask's `_decode`, 24.3, but
array-in / file-independent so the video path needs no second decoder).
The feature is a fixed 8·32·32 + 7 vector (92.6.2): the 8 evenly sampled
frames as 32×32 grayscale grids (8192 values) followed by the 7
inter-frame motion deltas. The labelled-directory protocol (24.2) is the
shared one — head inference, seed split, train-only standardization, the
§22.1 score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task
from .media import (_VIDEO_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)

_N_FRAMES = 8
_GRID = 32
MOTION_DIM = _N_FRAMES - 1  # 7 inter-frame motion deltas
VIDEO_FEATURE_DIM = _N_FRAMES * _GRID * _GRID + MOTION_DIM  # 8199 (92.6.2)

_VIDEO_HINT = (
    "decoding video needs imageio: pip install autorefine[video] "
    "(SPEC.md 92.6.1)"
)


def _to_gray(a: np.ndarray) -> np.ndarray:
    """One RGB(A) frame → float64 grayscale in [0, 1] (92.6.1).

    BT.601 weights for 3-channel frames (a 4th / alpha channel is
    ignored); a 2-D frame passes through. 8-bit (or wider) input is
    scaled to [0, 1] (float frames with max <= 1 pass through).
    Deterministic given the frame (G2).
    """
    a = np.asarray(a)
    if a.ndim == 2:
        g = a.astype(np.float64)
    else:
        a = a[..., :3]
        g = (0.299 * a[..., 0].astype(np.float64)
             + 0.587 * a[..., 1].astype(np.float64)
             + 0.114 * a[..., 2].astype(np.float64))
    if g.max() > 1.0:  # 8-bit (or wider) input -> [0, 1]
        g = g / 255.0
    return g


def _grid_resize(g: np.ndarray, grid: int) -> np.ndarray:
    """(H, W) grayscale → (grid, grid) by block averaging (92.6.1).

    A deterministic average-pool (no interpolation kernel for a
    platform to disagree with): output cell (i, j) is the mean of its
    input block (at least one pixel wide per axis). Same value range as
    the input. Deterministic given the frame (G2).
    """
    h, w = g.shape
    ys = np.round(np.linspace(0.0, h, grid + 1)).astype(np.int64)
    xs = np.round(np.linspace(0.0, w, grid + 1)).astype(np.int64)
    out = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        r0 = int(ys[i])
        r1 = max(int(ys[i + 1]), r0 + 1)
        for j in range(grid):
            c0 = int(xs[j])
            c1 = max(int(xs[j + 1]), c0 + 1)
            out[i, j] = float(g[r0:r1, c0:c1].mean())
    return out


def load_video_frames(p: Path, n_frames: int = _N_FRAMES,
                      grid: int = _GRID) -> np.ndarray:
    """SPEC.md 92.6.1 (v0.78): one video file → (n_frames, grid, grid)
    grayscale frames in [0, 1].

    `imageio` (the optional `autorefine[video]` extra) reads the clip;
    exactly `n_frames` are sampled evenly over the clip (a short clip
    repeats frames, so the feature dimension is constant for every
    clip, 92.6.2). A missing imageio is the install hint (92.6.1); an
    unreadable clip or a file with no frames is a fail-loud ValueError.
    Deterministic given the file bytes (G2).
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ValueError(_VIDEO_HINT) from exc
    try:
        frames = np.asarray(iio.imread(str(p)))
    except Exception as exc:
        raise ValueError(f"could not read video {p}: {exc}") from exc
    if frames.ndim == 3:  # a single frame -> (1, H, W, ...)
        frames = frames[None, ...]
    n = int(frames.shape[0]) if frames.ndim >= 4 else 0
    if n == 0:
        raise ValueError(f"{p}: no video frames found "
                         f"(shape {frames.shape}) (SPEC.md 92.6.1)")
    k = max(1, int(n_frames))
    idx = np.round(np.linspace(0.0, n - 1, k)).astype(np.int64)
    grids = [_grid_resize(_to_gray(frames[i]), grid) for i in idx]
    return np.array(grids, dtype=np.float64)


def video_features(grids: np.ndarray) -> np.ndarray:
    """SPEC.md 92.6.2 (v0.78): the raw (n·grid·grid + (n−1),) feature of
    one clip.

    The sampled grayscale frames flattened in frame order, followed by
    the n−1 inter-frame motion deltas — the mean |f[i+1] − f[i]| of the
    sampled grids (a coarse, resolution-free temporal signal). A
    one-frame clip yields the grid block only (n−1 = 0 deltas).
    Deterministic given the frames (G2).
    """
    grids = np.asarray(grids, dtype=np.float64)
    if grids.ndim != 3:
        raise ValueError(f"grids must be (n, grid, grid), got {grids.shape}")
    n = grids.shape[0]
    motion = np.array([float(np.abs(grids[i + 1] - grids[i]).mean())
                       for i in range(n - 1)], dtype=np.float64)
    return np.concatenate([grids.reshape(n, -1).ravel(), motion])


class VideoTask(Task):
    name = "video"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    feature_mean = None  # instance values (92.6)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 n_frames: int = _N_FRAMES, grid: int = _GRID) -> None:
        if path is None:
            raise ValueError(
                "VideoTask needs a directory of labelled video clips: use "
                "`autorefine fit --data DIR` or "
                "AutoRefineEnv(task='video', task_config={'path': DIR}) "
                "(SPEC.md 92.6)")
        self.seed = int(seed)
        self.path = str(path)
        self.n_frames = int(n_frames)
        self.grid = int(grid)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        items = collect_items(self.path, _VIDEO_EXTS, "video")
        x = np.array([video_features(load_video_frames(p, self.n_frames,
                                                       self.grid))
                      for p, _ in items], dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(x.shape[1])
        self.feature_names = ([f"frame {i}" for i in range(self.n_frames)]
                              + [f"motion delta {i}"
                                 for i in range(self.n_frames - 1)])
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"video-split")
        self._ho = ho
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean
        self.feature_std = std
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))

    # --- decoding -------------------------------------------------------------
    def item_features(self, path: str | Path) -> np.ndarray:
        """SPEC.md 92.6 (v0.78): decode one video clip to the raw
        (unstandardized) feature — the public wrapper `predict --item`
        calls."""
        return video_features(load_video_frames(Path(path), self.n_frames,
                                                self.grid))

    # --- protocol (SPEC.md 15, 24.2 — mirrors PointCloudTask) ----------------
    def _rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._x_ge, self._y_ge
        if s.startswith("train"):
            return self._x_tr, self._y_tr
        return self._x_ho, self._y_ho

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._rows_for(split)
        return x[: max(1, int(n))] if len(x) else x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states

    def make_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    def score(self, model, split: str, n: int) -> float:
        x, y = self._rows_for(split)
        if len(x) == 0:
            return 0.0
        n = min(int(n), len(x))
        pred = np.asarray(model.forward(x[:n]), dtype=np.float64)
        if self.head == "softmax":
            return float(100.0 * (pred.argmax(axis=1) == y[:n]).mean())
        pred = pred.reshape(-1)
        ss_res = float(((pred - y[:n]) ** 2).sum())
        ss_tot = float(((y[:n] - y[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    def holdout_errors(self, model, n: int = 200, n_max: int = 8) -> list[dict]:
        """SPEC.md 92.6 (v0.78): the misclassified holdout items (file
        entries; a video clip has no inline-renderable preview)."""
        if self.head != "softmax" or int(n_max) <= 0:
            return []
        x, y = self.holdout_rows(n)
        if len(x) == 0:
            return []
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        out: list[dict] = []
        for i in range(len(x)):
            if pred[i] == y[i]:
                continue
            p, _ = self._items[int(self._ho[i])]
            out.append({
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            })
            if len(out) >= int(n_max):
                break
        return out
