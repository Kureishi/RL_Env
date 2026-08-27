"""ImageTask: a directory of labelled images as a fitting task (SPEC.md 24.3).

Modality-as-task (G5): the features are a deterministic grid×grid grayscale
vector, and the head / splits / standardization / score protocol is the
§22.1 rule verbatim (shared with CsvTask via `media.py`, SPEC.md 24.2).
The improver loop, spec space, artifacts, `eval`, and dashboard then work
on it unchanged (SPEC.md 24).

Pillow is optional (SPEC.md 24.1, the §3/§21.1 pattern): it is imported
lazily, only when an image is decoded, so `import autorefine` stays
PIL-free; without it the constructor fails with the install hint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import _IMAGE_EXTS, collect_items, resolve_labels, split_indices

_PIL_HINT = (
    "decoding images needs Pillow: pip install autorefine[image] "
    "(SPEC.md 24.1)"
)


def _pil_resample():
    try:
        from PIL import Image
        return Image.Resampling.BILINEAR
    except AttributeError:  # Pillow < 9.1
        from PIL import Image
        return Image.BILINEAR


class ImageTask:
    name = "image"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    # per-data-dir attributes; __init__ overrides (defaults keep the class
    # usable for protocol introspection before a directory is given)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    # SPEC.md 25.3: the image modality is grid-structured (C=1, grid x grid)
    grid_capable = True
    feature_grid = None  # instance value: (1, grid, grid)

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 grid: int = 32) -> None:
        if path is None:
            raise ValueError(
                "ImageTask needs a directory of labelled images: use "
                "`autorefine fit --data DIR` or "
                "AutoRefineEnv(task='image', task_config={'path': DIR}) "
                "(SPEC.md 24.3)"
            )
        self.seed = int(seed)
        self.path = str(path)
        self.grid = int(grid)
        self.split_frac = float(split_frac)
        self.label_name = "class"  # subfolder name / index.csv label column

        items = collect_items(self.path, _IMAGE_EXTS, "image")
        x = np.array([self._decode(p, self.grid) for p, _ in items],
                     dtype=np.float64)

        # head / labels / splits / standardization — the §22.1 rule
        # (SPEC.md 24.2); raw labels resolve to §22.1 targets or string
        # class names (SPEC.md 24.2)
        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.state_dim = int(self.grid) ** 2
        self.feature_names = [f"{self.grid}x{self.grid} grayscale"]
        self.feature_grid = (1, self.grid, self.grid)  # SPEC.md 25.3
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"image-split")
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))  # SPEC.md 20.3 (items)

    # --- decoding -------------------------------------------------------------
    @staticmethod
    def _decode(p: Path, grid: int) -> np.ndarray:
        """Deterministic grayscale resize → (grid·grid,) in [0, 1]."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError(_PIL_HINT) from exc
        try:
            with Image.open(p) as im:
                im = im.convert("L").resize((grid, grid), _pil_resample())
                a = np.asarray(im, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"could not read image {p}: {exc}") from exc
        return (a / 255.0).ravel()

    # --- protocol (SPEC.md 15, 24.2 — mirrors CsvTask) ------------------------
    def _rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._x_ge, self._y_ge
        if s.startswith("train"):
            return self._x_tr, self._y_tr
        return self._x_ho, self._y_ho  # "holdout", "holdout-b3", ...

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._rows_for(split)
        return x[: max(1, int(n))] if len(x) else x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # already standardized in __init__

    def make_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Train split: what the improver is allowed to see (SPEC.md 15)."""
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    # --- grid protocol (SPEC.md 25.3) ---------------------------------------
    def grid_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Train split in grid layout (n, 1, grid, grid) — a pure reshape of
        the flat 1024-d rows, so the flat path stays bit-identical (25.3)."""
        x, y = self.make_dataset(n_points)
        return x.reshape(x.shape[0], *self.feature_grid), y

    def _grid_rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._rows_for(split)
        return x.reshape(x.shape[0], *self.feature_grid), y

    def score(self, model, split: str, n: int) -> float:
        # SPEC.md 25.3: a wants_grid model (convnet) gets grid-layout rows;
        # flat models get the flat rows (today's behavior, unchanged).
        if getattr(model, "wants_grid", False):
            x, y = self._grid_rows_for(split)
        else:
            x, y = self._rows_for(split)
        if len(x) == 0:
            return 0.0
        n = min(int(n), len(x))
        pred = np.asarray(model.forward(x[:n]), dtype=np.float64)
        if self.head == "softmax":
            acc = (pred.argmax(axis=1) == y[:n]).mean()
            return float(100.0 * acc)
        pred = pred.reshape(-1)
        ss_res = float(((pred - y[:n]) ** 2).sum())
        ss_tot = float(((y[:n] - y[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)
