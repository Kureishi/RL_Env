"""TextTask: a directory of labelled text files as a fitting task
(SPEC.md 45.2, v0.31).

The fourth modality with the same protocol (22.1/24.2) — mirroring
ImageTask (24.3): one subfolder per class (or an `index.csv`), the §22.1
head inference, the seed split, train-only standardization, and the §22.1
score. The feature decoding is a pure-NumPy/stdlib hashed word n-gram
count vector — zero new dependencies, deterministic given the file bytes
(G2; `zlib.crc32`, not the salted Python `hash()`).
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from .base import Task
from .media import (_TEXT_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def ngram_features(text: str, dim: int = 128) -> np.ndarray:
    """SPEC.md 45.2.1 (v0.31): a (dim,) float64 count vector of hashed
    word unigram + bigram n-grams.

    Lowercase, split on non-alphanumeric runs; each token (and each
    adjacent token pair) hashes with `zlib.crc32` of its UTF-8 bytes into
    bucket `h % dim` and is counted. Deterministic (G2) — `crc32`, not
    the per-process-salted `hash()` (45.2.1). Empty / no-token text →
    the zero vector.
    """
    d = int(dim)
    if d < 1:
        raise ValueError(f"dim must be >= 1, got {dim!r}")
    v = np.zeros(d, dtype=np.float64)
    tokens = _TOKEN_RE.findall((text or "").lower())
    for t in tokens:
        v[zlib.crc32(t.encode("utf-8")) % d] += 1.0
    for a, b in zip(tokens, tokens[1:]):
        v[zlib.crc32(f"{a} {b}".encode("utf-8")) % d] += 1.0
    return v


def _read_text(p: Path) -> str:
    """Deterministic text read (UTF-8, BOM-tolerant, replacement on
    malformed bytes — the file bytes are still the signal)."""
    return p.read_text(encoding="utf-8-sig", errors="replace")


class TextTask(Task):
    name = "text"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    # per-data-dir attributes; __init__ overrides (defaults keep the class
    # usable for protocol introspection before a directory is given)
    head = "softmax"
    n_outputs = 2
    state_dim = 1  # instance value: the n-gram dim (default 128)
    default_dataset_size = None  # instance value: len(train items)
    # SPEC.md 36.1 (v0.22, G1): declared metric + capabilities. Text is a
    # media modality (file/directory items); flat features — no "grid".
    metric = "accuracy"
    capabilities = frozenset({"media"})
    # SPEC.md 42.1.3 (v0.28): the train-only standardization stats —
    # instance values; None keeps class introspection usable.
    feature_mean = None
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 dim: int = 128) -> None:
        if path is None:
            raise ValueError(
                "TextTask needs a directory of labelled text files: use "
                "`autorefine fit --data DIR` or "
                "AutoRefineEnv(task='text', task_config={'path': DIR}) "
                "(SPEC.md 45.2)"
            )
        self.seed = int(seed)
        self.path = str(path)
        self.dim = int(dim)
        self.split_frac = float(split_frac)
        self.label_name = "class"  # subfolder name / index.csv label column

        items = collect_items(self.path, _TEXT_EXTS, "text")
        x = np.array([ngram_features(_read_text(p), self.dim)
                      for p, _ in items], dtype=np.float64)

        # head / labels / splits / standardization — the §22.1 rule
        # (SPEC.md 24.2); raw labels resolve to §22.1 targets or string
        # class names (SPEC.md 24.2)
        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(self.dim)
        self.feature_names = [f"hashed word n-grams ({self.dim})"]
        # SPEC.md 28.3 (C3): keep the items + holdout split indices so the
        # error gallery can map holdout row i -> items[ho[i]].
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"text-split")
        self._ho = ho
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean  # SPEC.md 42.1.3 (v0.28): expose the stats
        self.feature_std = std  # so `predict` standardizes new items like train
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))  # SPEC.md 20.3 (items)

    # --- decoding -------------------------------------------------------------
    def item_features(self, path: str | Path) -> np.ndarray:
        """SPEC.md 42.1.3 (v0.28): decode one text file to the raw
        (unstandardized) (dim,) feature — the public wrapper `predict
        --item` (42.1.1) and `autorefine.predict` call, so the CLI never
        reaches into task internals."""
        return ngram_features(_read_text(Path(path)), self.dim)

    # --- protocol (SPEC.md 15, 45.2 — mirrors ImageTask/CsvTask) ------------
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

    # --- holdout inspection (SPEC.md 28.2/28.3) ------------------------------
    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 28.2 (C2): the holdout split `(x, y)`, clamped like
        `score()`."""
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    def holdout_errors(self, model, n: int = 200,
                       n_max: int = 8) -> list[dict]:
        """SPEC.md 28.3 (C3): the holdout items the model misclassifies
        (argmax mismatch), in holdout order, <= `n_max` items, each
        `{"file", "path", "label", "predicted"}` — labels from the task's
        canonical `class_values`, not the raw item labels."""
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
            p, _ = self._items[int(self._ho[i])]  # holdout row i -> item
            out.append({
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            })
            if len(out) >= int(n_max):
                break
        return out
