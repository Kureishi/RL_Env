"""JsonlTask: a JSONL / nested-JSON log file as a fitting task
(SPEC.md 92.2, v0.78, A82).

The semi-structured-log modality: one file of records (JSON Lines, or a
JSON array), each record an object. The no-code rule mirrors CsvTask
(22.1) with the same protocol — head inference (24.2), seed split,
train-only standardization, the §22.1 score — but the records are
**flattened** first (92.2.1):

  * numeric leaf (int/float/bool) → one column, the dotted key path;
  * string leaf → a hashed bucket column ``s<B>`` (``zlib.crc32`` of the
    UTF-8 bytes % 8 buckets — the text.py 45.2.1 rule, G2);
  * list leaf → the mean of its numeric leaves (the list length when it
    has none);
  * null / dict leaves are skipped (dicts are recursed into).

The label is the ``label`` argument if given (top-level key first, then
flattened path), else the first of label/target/y/class (top-level, then
flattened). Missing feature values are 0.0 (the train-only standardization
then absorbs the shift, 22.1). String and numeric labels both resolve via
the 24.2 `resolve_labels` rule.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .base import Task
from .media import (resolve_labels, split_indices)

_JSONL_EXTS = (".jsonl", ".json")
_LABEL_HINTS = ("label", "target", "y", "class")
_N_STRING_BUCKETS = 8  # 92.2.1: the hashed string-bucket width


def flatten_record(obj: object) -> dict[str, float]:
    """SPEC.md 92.2.1 (v0.78): one JSON record → {column: value}.

    Deterministic (G2): numeric leaves → their dotted key path; strings →
    ``s<zlib.crc32(utf8) % 8>``; lists → the mean of their numeric leaves
    (length when there are none); nulls and bare dicts are skipped.
    """
    out: dict[str, float] = {}

    def _walk(value: object, prefix: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _walk(v, key)
            return
        if isinstance(value, (list, tuple)):
            nums = [float(v) for v in value
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and np.isfinite(float(v))]
            if nums:
                out[prefix] = float(sum(nums) / len(nums))
            else:
                out[prefix] = float(len(value))
            return
        if isinstance(value, bool):
            out[prefix] = 1.0 if value else 0.0
            return
        if isinstance(value, (int, float)):
            f = float(value)
            if np.isfinite(f):
                out[prefix] = f
            return
        if isinstance(value, str):
            key = f"s{zlib.crc32(value.encode('utf-8')) % _N_STRING_BUCKETS}"
            out[key] = out.get(key, 0.0) + 1.0
            return
        # None and anything else: no column (92.2.1)

    if isinstance(obj, dict):
        _walk(obj, "")
    return out


def _read_records(path: Path) -> list[dict]:
    """One JSONL / JSON-array file → a list of record dicts (92.2)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    stripped = text.lstrip()
    records: list[dict] = []
    if stripped.startswith("["):
        try:
            arr = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not a JSON array: {exc}") from exc
        if not isinstance(arr, list):
            raise ValueError(f"{path}: the JSON array must be a list of objects")
        for r in arr:
            if isinstance(r, dict):
                records.append(r)
    else:
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: not a JSON object: {exc}") from exc
            if isinstance(r, dict):
                records.append(r)
    if not records:
        raise ValueError(f"{path}: no JSON object records found "
                         "(expected JSONL lines or a JSON array of objects)")
    return records


def _pick_label(records: list[dict], label: str | None) -> tuple[str, list]:
    """(label key, raw label values) — the 92.2 rule (92.2.2)."""
    top_keys = {str(k) for r in records for k in r.keys()}
    flat_keys: set[str] = set()
    for r in records:
        flat_keys.update(flatten_record(r).keys())
    candidates = ([label] if label else []) + list(_LABEL_HINTS)
    for c in candidates:
        if c in top_keys:
            return c, [r.get(c) for r in records]
        if c in flat_keys:
            return c, [flatten_record(r).get(c) for r in records]
    raise ValueError(
        f"no label column: pass --label NAME (candidates tried: "
        f"{', '.join([c for c in candidates if c]) or 'none'}) — the label "
        f"is a top-level or dotted key of the JSON records (SPEC.md 92.2)")


class JsonlTask(Task):
    name = "jsonl"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train records)
    metric = "accuracy"
    capabilities = frozenset()
    feature_mean = None  # instance values (92.2)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2) -> None:
        if path is None:
            raise ValueError(
                "JsonlTask needs a data path: use `autorefine fit --data "
                "FILE.jsonl` or AutoRefineEnv(task='jsonl', "
                "task_config={'path': FILE}) (SPEC.md 92.2)")
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"no such JSONL file: {p!r} (SPEC.md 92.2)")
        if p.suffix.lower() not in _JSONL_EXTS:
            raise ValueError(
                f"{p}: expected a .jsonl or .json file, got {p.suffix!r} "
                f"(SPEC.md 92.2)")
        self.seed = int(seed)
        self.path = str(p)
        self.split_frac = float(split_frac)

        records = _read_records(p)
        key, raw_labels = _pick_label(records, label)
        self.label_name = key
        self._records = records  # 92.2.3: keep the rows for item decoding

        flat = [flatten_record(r) for r in records]
        columns = sorted({k for f in flat for k in f})  # fixed schema (G2)
        self.feature_names = list(columns)
        x = np.zeros((len(records), len(columns)), dtype=np.float64)
        for i, f in enumerate(flat):
            for j, c in enumerate(columns):
                x[i, j] = f.get(c, 0.0)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels(raw_labels)
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(len(columns))
        tr, ho, ge = split_indices(len(records), self.seed, self.split_frac,
                                   b"jsonl-split")
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
        """SPEC.md 92.2.3 (v0.78): one record file → the raw
        (unstandardized) feature (the first record of the file)."""
        records = _read_records(Path(path))
        flat = flatten_record(records[0])
        return np.array([flat.get(c, 0.0) for c in self.feature_names],
                        dtype=np.float64)

    def record_features(self, record: dict) -> np.ndarray:
        """SPEC.md 92.2.3 (v0.78): one in-memory JSON record → the raw
        feature over the task's fixed column schema."""
        flat = flatten_record(record)
        return np.array([flat.get(c, 0.0) for c in self.feature_names],
                        dtype=np.float64)

    # --- protocol (SPEC.md 15, 92.2 — mirrors CsvTask) -----------------------
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

    # no `holdout_errors`: records have no displayable media file (92.2.3)
