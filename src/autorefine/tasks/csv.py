"""CsvTask: a user-supplied CSV file as a fitting task (SPEC.md 22.1).

The no-code path for "I have tabular data, which model fits?": the file is
loaded once through the minimal §15 protocol (make_dataset + score), with
deterministic, seed-derived train/holdout/gen splits (G2). The improver
loop, acceptance rule, artifacts, `eval`, and reporting all work on it
without further changes.

Rules (all deterministic, stdlib `csv` only — SPEC.md 22.1):
  * header + rows; columns with empty header names are dropped; all-empty
    rows are dropped;
  * the label column is the `label` argument if given (case-insensitive
    match), else the first of label/target/y/class, else the last column;
  * features are the numeric columns (every non-empty value finite-
    float-parseable) except the label, in file order; non-numeric
    non-label columns are ignored; a column with an empty value is
    non-numeric;
  * head inference: all-integer label values with 2–50 distinct classes →
    "softmax" (sorted unique values → classes 0…K−1); otherwise "mse";
  * splits: a seed-derived permutation partitions the rows into
    train / holdout / gen (default 80/10/10; `split_frac` is the non-train
    share, split evenly); features are standardized with train-only
    statistics (std 0 → 1);
  * score = 100·accuracy (softmax) or 100·R² clamped at 0 (mse).

Split strings are matched by prefix — `gen*` → gen, `train*` → train,
else holdout — so the §18.3 block-bootstrap names (`holdout-b3`, `gen-b2`)
work unchanged. The row set per split is fixed, so `n` clamps to its
length and every block scores the same rows (σ = 0; the §18.6 rule
reduces to a strict score comparison).
"""
from __future__ import annotations

import csv as _csv
import zlib
from pathlib import Path

import numpy as np

from .base import Task

_LABEL_HINTS = ("label", "target", "y", "class")
_MAX_CLASSES = 50  # integer labels with more distinct values → regression


class CsvTask(Task):
    name = "csv"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    # per-data-file attributes; __init__ overrides (defaults keep the class
    # usable for protocol introspection before a file is given)
    head = "mse"
    n_outputs = 1
    state_dim = 1
    default_dataset_size = None  # instance value: len(train rows)
    # SPEC.md 36.1 (v0.22, G1): declared metric + capabilities. Class-level
    # default matches the `head = "mse"` introspection default; __init__
    # sets the instance metric alongside `head` (accuracy vs r2 per data).
    metric = "r2"
    capabilities = frozenset()
    # SPEC.md 42.1.3 (v0.28): the train-only standardization stats —
    # instance values (the same mean/std __init__ already computes);
    # None keeps class introspection usable before a file is given.
    feature_mean = None
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2) -> None:
        if path is None:
            raise ValueError(
                "CsvTask needs a data path: use `autorefine fit --data FILE` "
                "or AutoRefineEnv(task='csv', task_config={'path': FILE})"
            )
        self.seed = int(seed)
        self.path = str(path)
        self.label_name: str | None = label
        self.split_frac = float(split_frac)

        names, rows = self._read_csv()
        self.label_name = self._pick_label(names, label)  # resolved column name
        label_name = self.label_name
        features = [c for c in self._numeric_cols(rows, names) if c != label_name]
        if not features:
            raise ValueError(f"CSV {self.path!r}: no numeric feature columns "
                             f"(columns: {names})")
        if len(rows) < 3:
            raise ValueError(f"CSV {self.path!r}: needs at least 3 rows, got {len(rows)}")

        x = np.array([[float(r[c]) for c in features] for r in rows], dtype=np.float64)
        y = np.array([float(r[label_name]) for r in rows], dtype=np.float64)

        # head inference (SPEC.md 22.1)
        classes = np.unique(y)
        integer_labels = bool(np.all(y == np.floor(y)))
        if integer_labels and 2 <= classes.size <= _MAX_CLASSES:
            self.head = "softmax"
            self.n_outputs = int(classes.size)
            self.class_values = [float(c) for c in classes]  # class i ↔ value
            self._y = np.array(
                [int(np.flatnonzero(classes == v)[0]) for v in y], dtype=np.int64
            )
            self.metric = "accuracy"  # SPEC.md 36.1.3 (score = 100*accuracy)
        else:
            self.head = "mse"
            self.n_outputs = 1
            self.class_values = None
            self._y = y
            self.metric = "r2"  # SPEC.md 36.1.3 (score = 100*R^2)
        self.state_dim = len(features)
        self.feature_names = features

        # seed-derived train/holdout/gen partition (G2)
        n = len(rows)
        n_train = int(n * (1.0 - self.split_frac))
        rest = n - n_train
        n_hold = rest // 2
        n_gen = rest - n_hold
        if n_train < 1 or n_hold < 1 or n_gen < 1:
            raise ValueError(f"CSV {self.path!r}: {n} rows are too few for "
                             f"train/holdout/gen splits with split_frac={self.split_frac}")
        seq = np.random.SeedSequence([self.seed, zlib.crc32(b"csv-split")])
        perm = np.random.default_rng(seq).permutation(n)
        tr, ho, ge = (perm[:n_train], perm[n_train:n_train + n_hold],
                      perm[n_train + n_hold:])

        # standardize with train-only statistics (no leakage)
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean  # SPEC.md 42.1.3 (v0.28): expose the stats
        self.feature_std = std  # so `predict` standardizes new rows like train

        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(n_train)  # SPEC.md 20.3 (points)

    # --- loading / inference ----------------------------------------------------
    def _read_csv(self) -> tuple[list[str], list[dict]]:
        """(column names, rows keyed by column name); names/rows are raw text."""
        with open(self.path, newline="", encoding="utf-8-sig") as fh:
            reader = _csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"CSV {self.path!r} is empty") from None
            cols = [i for i, h in enumerate(header) if h.strip() != ""]
            if not cols:
                raise ValueError(f"CSV {self.path!r} has no columns")
            name_at = {i: header[i].strip() for i in cols}
            rows: list[dict] = []
            for raw in reader:
                if not any((c or "").strip() for c in raw):
                    continue  # all-empty line
                rows.append({name_at[i]: (raw[i].strip() if i < len(raw) else "")
                             for i in cols})
        return [name_at[i] for i in cols], rows

    def _numeric_cols(self, rows: list[dict], names: list[str]) -> list[str]:
        out = []
        for c in names:
            vals = [r[c] for r in rows if r.get(c, "") != ""]
            if not vals:
                continue
            try:
                nums = [float(v) for v in vals]
            except ValueError:
                continue  # non-numeric column (ignored, SPEC.md 22.1)
            if all(np.isfinite(v) for v in nums):
                out.append(c)
        return out

    def _pick_label(self, names: list[str], label: str | None) -> str:
        lowered = {n.lower(): n for n in names}
        if label is not None:
            key = label.strip().lower()
            if key not in lowered:
                raise ValueError(f"CSV {self.path!r}: no label column {label!r} "
                                 f"(columns: {names})")
            return lowered[key]
        for hint in _LABEL_HINTS:
            if hint in lowered:
                return lowered[hint]
        return names[-1]  # last column

    # --- splits / protocol (SPEC.md 15, 22.1) ----------------------------------
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

    def _score_xy(self, model, x: np.ndarray, y: np.ndarray) -> float:
        """The head metric on given rows — softmax → 100·accuracy, mse →
        100·max(0, R²) — the exact §22.1 score contract, factored so
        `score` (a split) and `score_fold` (a subset) share one body.
        SPEC.md 44.1.2 (v0.30)."""
        if len(x) == 0:
            return 0.0
        pred = np.asarray(model.forward(x), dtype=np.float64)
        if self.head == "softmax":
            acc = (pred.argmax(axis=1) == y).mean()
            return float(100.0 * acc)
        pred = pred.reshape(-1)
        ss_res = float(((pred - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)

    def score(self, model, split: str, n: int) -> float:
        x, y = self._rows_for(split)
        n = min(int(n), len(x))
        return self._score_xy(model, x[:n], y[:n])

    def score_fold(self, model, split: str, fold_index: int, n: int) -> float:
        """SPEC.md 44.1.2 (v0.30): the `fold_index`-th deterministic
        random subset of the non-train pool (`holdout ∪ gen` rows, both
        already train-standardized) of size ≈ the holdout split —
        distinct per fold, seeded by (task seed, crc32(b"csv-kfold"),
        fold_index) (G2). The pool excludes the train split the model
        was trained on, so there is no leakage (44.1.4)."""
        if len(self._x_ho) == 0:
            return 0.0
        x_pool = np.vstack([self._x_ho, self._x_ge])
        y_pool = np.concatenate([self._y_ho, self._y_ge])
        size = min(len(self._x_ho), len(x_pool))
        seq = np.random.SeedSequence(
            [self.seed, zlib.crc32(b"csv-kfold"), int(fold_index)])
        idx = np.random.default_rng(seq).permutation(len(x_pool))[:size]
        x, y = x_pool[idx], y_pool[idx]
        n = min(int(n), len(x))
        return self._score_xy(model, x[:n], y[:n])

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 28.2 (C2): the holdout split `(x, y)`, clamped to its
        length — the rows `holdout_diagnostics` scores. CsvTask is flat-only,
        so `model` is accepted for a uniform protocol but ignored."""
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]
