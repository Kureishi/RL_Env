"""Fusion tasks: image + text rows for two-tower late fusion (SPEC.md 99, v0.85 C1).

`FusionTask` is the file-based modality task: a directory of
(image, text) pairs — either an `index.csv` with `image` / `text` /
`label` columns, or one subfolder per class holding one image and one
text file each. Each row is the concatenation of the image grid
vector (SPEC.md 24.3 decoding, 8x8 = 64-d by default) and the hashed
n-gram text vector (SPEC.md 45.2.1, 32-d by default), so the row is
`[image | text]` with the split point `fusion_split = 64`.

That split is the whole contract: the trainer's `fusion="late"` branch
feeds the first `fusion_split` dims to tower 1 and the rest to tower 2
(`LateFusionMLP`, SPEC.md 99); `fusion="early"` (the default) trains the
same row with the plain MLP, so the late/early comparison is the
architectural question the improver can answer.

`FusionV1` is the deterministic synthetic registry entry (3 classes,
class-specific 8x8 image pattern + class-specific text vector, seeded
noise) — the no-files counterpart so the late-fusion path is testable
and benchmarkable (the D3 `bench --suite multimodal` task set).

Head / splits / standardization / score are the §22.1 rule verbatim
(shared with ImageTask/TextTask via `media.py`, SPEC.md 24.2).
"""
from __future__ import annotations

import csv as _csv
import zlib
from pathlib import Path

import numpy as np

from .base import Task
from .images import ImageTask
from .media import (_IMAGE_EXTS, _TEXT_EXTS, class_label_str, resolve_labels,
                    split_indices)
from .text import ngram_features

IMG_GRID = 8      # 8x8 grid -> 64-d image part (the SPEC.md 24.3 decode)
TXT_DIM = 32      # 32-d hashed n-gram text part (SPEC.md 45.2.1)
FUSION_SPLIT = IMG_GRID ** 2  # 64: dims [0, 64) = image, [64, 96) = text

_FUSION_HINTS = {
    "image": ("image", "img", "pic", "photo"),
    "text": ("text", "txt", "sentence", "caption", "prompt"),
    "label": ("label", "target", "y", "class"),
}


def _decode_image(p: Path, grid: int = IMG_GRID) -> np.ndarray:
    """(grid*grid,) in [0, 1] — the SPEC.md 24.3 grayscale grid decode."""
    return ImageTask._decode(p, grid)


def _read_fusion_index(index: Path, d: Path
                       ) -> list[tuple[Path, Path, object]]:
    """(image, text, label) rows from an `index.csv`.

    Column resolution: named hints first (image/img, text/txt/sentence,
    label/target/y/class), else the positional order image, text, label.
    File cells are relative to `d` unless absolute; labels are raw
    (resolve_labels maps them, SPEC.md 24.2).
    """
    with open(index, newline="", encoding="utf-8-sig") as fh:
        reader = _csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{index} is empty") from None
        low = {h.lower(): i for i, h in enumerate(header)}

        def _col(kind: str, default: int) -> int:
            for hint in _FUSION_HINTS[kind]:
                if hint in low:
                    return low[hint]
            return default

        img_i, txt_i, lbl_i = _col("image", 0), _col("text", 1), _col("label", 2)
        seen = set()
        for i, name in ((img_i, "image"), (txt_i, "text"), (lbl_i, "label")):
            if i in seen:
                raise ValueError(
                    f"{index}: the {name} column must be distinct from the "
                    f"other columns (SPEC.md 99)")
            seen.add(i)
        items: list[tuple[Path, Path, object]] = []
        for raw in reader:
            if not raw or not any((c or "").strip() for c in raw):
                continue
            if max(img_i, txt_i, lbl_i) >= len(raw):
                continue
            img = Path(raw[img_i].strip())
            txt = Path(raw[txt_i].strip())
            if not img.is_absolute():
                img = d / img
            if not txt.is_absolute():
                txt = d / txt
            items.append((img, txt, raw[lbl_i].strip()))
    if not items:
        raise ValueError(f"{index} has no rows")
    return items


def collect_fusion_items(dir_path: str | Path) -> list[tuple[Path, Path, object]]:
    """(image, text, label) pairs for one fusion directory (SPEC.md 99).

    Layouts, in priority order:
      * `index.csv` with image / text / label columns (labels numeric or
        string class names — `resolve_labels` decides the head, 24.2);
      * one subfolder per class, each holding one image and one text file
        (the lexicographically first of each kind is used; deterministic).
    """
    d = Path(dir_path)
    if not d.is_dir():
        raise ValueError(
            f"no such fusion directory: {dir_path!r} — expected a directory "
            f"with an index.csv (image, text, label columns) or one "
            f"subfolder per class holding an image and a text file "
            f"(SPEC.md 99)"
        )
    index = d / "index.csv"
    if index.is_file():
        return _read_fusion_index(index, d)
    items: list[tuple[Path, Path, object]] = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sorted(f for f in sub.iterdir()
                      if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
        txts = sorted(f for f in sub.iterdir()
                      if f.is_file() and f.suffix.lower() in _TEXT_EXTS)
        if not imgs or not txts:
            continue
        items.append((imgs[0], txts[0], sub.name))
    if not items:
        raise ValueError(
            f"no (image, text) pairs under {dir_path!r} — add an "
            f"index.csv with image/text/label columns, or one subfolder "
            f"per class containing an image file and a text file "
            f"(SPEC.md 99)"
        )
    return items


class FusionTask(Task):
    """A directory of labelled (image, text) pairs (SPEC.md 99, v0.85 C1).

    Row = `[image grid (64-d) | n-gram text (32-d)]`; `fusion_split = 64`
    marks the tower boundary for `LateFusionMLP` (the trainer's
    `fusion="late"` branch). Everything else is the §22.1 protocol
    (24.2) — head inference, seed splits, train-only standardization,
    accuracy/R² scoring, holdout inspection.
    """
    name = "fusion"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = IMG_GRID ** 2 + TXT_DIM
    default_dataset_size = None  # instance value: len(train items)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    # SPEC.md 42.1.3: train-only standardization stats (instance values)
    feature_mean = None
    feature_std = None
    # SPEC.md 99 (v0.85 C1): the tower boundary — dims [0, fusion_split)
    # are the image side, [fusion_split, state_dim) the text side
    fusion_split = IMG_GRID ** 2

    def __init__(self, seed: int, path: str | Path | None = None,
                 split_frac: float = 0.2, img_grid: int = IMG_GRID,
                 txt_dim: int = TXT_DIM) -> None:
        if path is None:
            raise ValueError(
                "FusionTask needs a directory of (image, text) pairs: use "
                "`autorefine fit --data DIR --task fusion` or "
                "AutoRefineEnv(task='fusion', task_config={'path': DIR}) "
                "(SPEC.md 99)"
            )
        self.seed = int(seed)
        self.path = str(path)
        self.img_grid = int(img_grid)
        self.txt_dim = int(txt_dim)
        self.split_frac = float(split_frac)
        self.label_name = "class"
        self.fusion_split = int(img_grid) ** 2
        self.state_dim = self.fusion_split + int(txt_dim)

        items = collect_fusion_items(self.path)
        x = np.array([
            np.concatenate([_decode_image(p, self.img_grid),
                            ngram_features(t.read_text(encoding="utf-8",
                                                     errors="replace"),
                                           self.txt_dim)])
            for p, t, _ in items
        ], dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.feature_names = [f"{self.img_grid}x{self.img_grid} image",
                              f"{self.txt_dim}-d n-gram text"]
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"fusion-split")
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

    # --- protocol (SPEC.md 15, 24.2 — mirrors ImageTask) ----------------------
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

    def make_dataset(self, n_points: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Train split: what the improver is allowed to see (SPEC.md 15)."""
        n = len(self._x_tr) if n_points is None \
            else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    def item_features(self, image: str | Path, text: str | Path) -> np.ndarray:
        """SPEC.md 42.1.1-style single-item decode (unstandardized):
        [image grid | n-gram text] for one (image, text) pair."""
        return np.concatenate([
            _decode_image(Path(image), self.img_grid),
            ngram_features(Path(text).read_text(encoding="utf-8",
                                             errors="replace"), self.txt_dim),
        ])

    def score(self, model, split: str, n: int) -> float:
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

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 28.2 (C2): the holdout split `(x, y)`, clamped."""
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    def holdout_errors(self, model, n: int = 200, n_max: int = 8) -> list[dict]:
        """SPEC.md 28.3 (C3): misclassified holdout items; the `file`
        field names the image (the text file is the paired item)."""
        if self.head != "softmax" or int(n_max) <= 0:
            return []
        x, y = self.holdout_rows(n, model)
        if len(x) == 0:
            return []
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        out: list[dict] = []
        for i in range(len(x)):
            if pred[i] == y[i]:
                continue
            img, txt, _ = self._items[int(self._ho[i])]
            out.append({
                "file": img.name,
                "path": str(img),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            })
            if len(out) >= int(n_max):
                break
        return out


class FusionV1(Task):
    """v0.85 synthetic fusion task (SPEC.md 99, v0.85 C1): 3 classes,
    each with a distinct 8x8 image pattern (horizontal stripes / vertical
    stripes / checkerboard) and a distinct short text phrase, plus seeded
    per-sample noise. Rows are `[image 64-d | text 32-d]` with
    `fusion_split = 64` — the no-files counterpart of `FusionTask` so the
    late-fusion architecture is exercisable in benchmarks and tests.

    Deterministic given the seed (G2); score = 100 * accuracy, fresh
    seed-derived points per split (the ParityTask/SineTask pattern).
    """
    name = "fusion-v1"
    head = "softmax"
    n_outputs = 3
    max_steps = 1
    state_dim = IMG_GRID ** 2 + TXT_DIM
    default_dataset_size = 120  # SPEC.md 20.3 (points)
    metric = "accuracy"
    capabilities = frozenset()
    fusion_split = IMG_GRID ** 2

    # class -> (8x8 pattern (64-d), phrase) — fixed by construction
    # (deterministic): horizontal stripes / vertical stripes / checkerboard.
    _CLASSES = (
        ((np.arange(8) % 2)[:, None].repeat(8, axis=1).astype(np.float64).ravel(),
         "red circle small round"),
        ((np.arange(8) % 2)[None, :].repeat(8, axis=0).astype(np.float64).ravel(),
         "blue square large flat"),
        ((np.indices((8, 8)).sum(axis=0) % 2).astype(np.float64).ravel(),
         "green triangle tiny sharp"),
    )

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    @staticmethod
    def _split_rng(seed: int, split: str) -> np.random.Generator:
        seq = np.random.SeedSequence(
            [seed, zlib.crc32(b"fusion-v1"), zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    @classmethod
    def _points(cls, seed: int, split: str, n: int
                ) -> tuple[np.ndarray, np.ndarray]:
        """(rows (n, 96), labels (n,)) — balanced, seeded noise per sample."""
        rng = cls._split_rng(seed, split)
        x = np.empty((n, IMG_GRID ** 2 + TXT_DIM), dtype=np.float64)
        y = np.empty(n, dtype=np.int64)
        for i in range(n):
            c = i % 3  # deterministic balance
            base_img, phrase = cls._CLASSES[c]
            base_txt = ngram_features(phrase, TXT_DIM)
            x[i, :IMG_GRID ** 2] = base_img + rng.normal(0.0, 0.10,
                                                         IMG_GRID ** 2)
            # non-negative jitter on the text counts (scale in [0.5, 1.5])
            x[i, IMG_GRID ** 2:] = base_txt * rng.uniform(0.5, 1.5, TXT_DIM)
            y[i] = c
        return x, y

    # --- protocol (SPEC.md 15) -------------------------------------------------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._points(self.seed, split, n)
        return x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # O(1) features need no transform

    def make_dataset(self, n_points: int = 120) -> tuple[np.ndarray, np.ndarray]:
        """(rows (n, 96), labels (n,)) on the train split."""
        return self._points(self.seed, "train", n_points)

    def score(self, model, split: str, n: int) -> float:
        """100 * accuracy on fresh points of `split`; higher is better."""
        x, y = self._points(self.seed, split, n)
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        return float(100.0 * (pred == y).mean())
