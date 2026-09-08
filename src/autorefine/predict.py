"""`predict` leaf (SPEC.md 42.1, v0.28) — score NEW rows with a finished
run's best model, using the task's own training-time preprocessing.

Pure stdlib + numpy (a core dependency), import-cycle-free (imports only
`numpy` and the `pathlib`/`csv`/`json`-free helpers it needs — no
autorefine modules — so the `cli` wires it, and tests can drive it with
fake tasks/models). The CLI (`cli._cmd_predict`) reconstructs the
(task, model) pair from the run dir and calls:

    raw  = row_to_features / csv_rows_to_features / media_item_features
    feat = standardize(task, raw)          # the task's train stats (42.1.3)
    out  = predict_features(task, model, feat)   # forward + decode (42.1.4)
"""
from __future__ import annotations

import csv as _csv
import math
from pathlib import Path

import numpy as np

__all__ = [
    "row_to_features",
    "standardize",
    "predict_features",
    "csv_rows_to_features",
    "media_item_features",
]


def _finite(v: object) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"row value {v!r} is not a finite number")
    return f


def row_to_features(row, state_dim, feature_names: list[str] | None = None) -> np.ndarray:
    """SPEC.md 42.1.1 (v0.28): one input row → a raw `(state_dim,)` feature
    vector (NOT yet standardized — that is `standardize`'s job, 42.1.3).

    - dict/object: keys match the task's `feature_names` case-insensitively
      (extra keys are ignored, a missing feature column is an error); a
      task without `feature_names` (e.g. `sine-v1`) accepts an object
      with exactly one key (its single unit-scaled input).
    - list/tuple/sequence: positional — exactly `state_dim` finite numbers.

    Deterministic; no side effects.
    """
    state_dim = int(state_dim)
    if state_dim < 1:
        raise ValueError(f"state_dim must be >= 1, got {state_dim}")
    if isinstance(row, dict):
        names = list(feature_names or [])
        if names:
            keys = {str(k).lower(): v for k, v in row.items()}
            out = []
            for n in names:
                if n.lower() not in keys:
                    raise ValueError(
                        f"row is missing feature column {n!r} "
                        f"(expected one of: {names})")
                out.append(_finite(keys[n.lower()]))
        else:
            if len(row) != 1:
                raise ValueError(
                    "task has no feature names: pass a one-key object "
                    f"(got {len(row)} keys) or a {state_dim}-element array")
            out = [_finite(next(iter(row.values())))]
    else:
        try:
            out = [_finite(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row values must be finite numbers: {exc}") from exc
    if len(out) != state_dim:
        raise ValueError(f"expected {state_dim} feature(s), got {len(out)}")
    return np.array(out, dtype=np.float64)


def standardize(task, raw: np.ndarray) -> np.ndarray:
    """SPEC.md 42.1.3 (v0.28): apply the task's own train-only
    standardization stats — `(raw − feature_mean) / feature_std` — exactly
    as the task standardized its train rows (the §22.1 rule); a task
    without the attributes (`sine-v1`) is the identity (already
    unit-scaled, §15). A shape mismatch between the stats and the row is
    an error (a wrong task/row pairing, caught here instead of a silent
    broadcast). Deterministic; no side effects.
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()
    mean = getattr(task, "feature_mean", None)
    std = getattr(task, "feature_std", None)
    if mean is None or std is None:
        return raw
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    if mean.shape != raw.shape or std.shape != raw.shape:
        raise ValueError(
            f"feature stats shape {mean.shape}/{std.shape} does not match "
            f"the row's shape {raw.shape} — is this row from this task's data?")
    return (raw - mean) / std


def predict_features(task, model, features: np.ndarray) -> dict:
    """SPEC.md 42.1.4 (v0.28): one (standardized) feature vector →
    `{"prediction": …, "probabilities": […]|null}`.

    - softmax head: the argmax class mapped back through
      `task.class_values` (0/1 values or string class names — the user's
      labels, not internal indices), plus `probabilities` = the softmax
      of the forward output (the tasks score on `argmax`, so the
      probabilities are a pure rendering of the same logits — 42.1.4);
    - mse head (or no head): the scalar output, `probabilities` is `None`.

    One forward pass, deterministic, no RNG.
    """
    x = np.atleast_2d(np.asarray(features, dtype=np.float64).ravel())
    out = np.asarray(model.forward(x), dtype=np.float64)
    if getattr(task, "head", None) == "softmax":
        logits = out.reshape(1, -1)[0]
        idx = int(logits.argmax())
        values = getattr(task, "class_values", None)
        pred = values[idx] if values else idx
        z = logits - logits.max()
        e = np.exp(z)
        probs = (e / e.sum()).tolist()
        return {"prediction": pred, "probabilities": [float(p) for p in probs]}
    return {"prediction": float(out.reshape(-1)[0]), "probabilities": None}


def _cell_is_number(s: str) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return True


def csv_rows_to_features(path, task) -> list[np.ndarray]:
    """SPEC.md 42.1.1 (v0.28): a CSV of NEW rows → a list of raw
    `(state_dim,)` feature vectors (unstandardized — the caller applies
    `standardize` per row, 42.1.3).

    Header auto-detect: if the first line has a non-numeric cell it is a
    header — cells match the task's `feature_names` case-insensitively
    (extra columns, such as a label column, are OK; a missing feature
    column is an error); a task without `feature_names` needs exactly
    `state_dim` columns. Otherwise the rows are positional and each must
    have exactly `state_dim` cells. Deterministic; stdlib `csv` only.
    """
    state_dim = int(getattr(task, "state_dim", 0))
    feature_names = list(getattr(task, "feature_names", None) or [])
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = _csv.reader(fh)
        rows = [[(c or "").strip() for c in raw] for raw in reader
                if any((c or "").strip() for c in raw)]
    if not rows:
        raise ValueError(f"CSV {p} has no rows")
    first = rows[0]
    if any(not _cell_is_number(c) for c in first):
        header = first
        data = rows[1:]
    else:
        header = None
        data = rows
    if not data:
        raise ValueError(f"CSV {p} has a header but no data rows")
    out: list[np.ndarray] = []
    for r in data:
        if header is not None:
            if feature_names:
                idx = {h.lower(): j for j, h in enumerate(header)}
                vec = []
                for n in feature_names:
                    j = idx.get(n.lower())
                    if j is None:
                        raise ValueError(
                            f"CSV {p}: header {header} is missing feature "
                            f"column {n!r}")
                    if j >= len(r) or r[j] == "":
                        raise ValueError(
                            f"CSV {p}: row {r} has no value for column {n!r}")
                    vec.append(_finite(r[j]))
            else:
                if len(header) != state_dim:
                    raise ValueError(
                        f"CSV {p}: task has no feature names — the header "
                        f"must have exactly {state_dim} column(s), got "
                        f"{header}")
                vec = [
                    _finite(r[j]) if j < len(r) and r[j] != ""
                    else (_raise_missing(p, n))
                    for j, n in enumerate(range(state_dim))
                ]
        else:
            if len(r) != state_dim:
                raise ValueError(
                    f"CSV {p}: expected {state_dim} cell(s) per row, "
                    f"got {len(r)}: {r}")
            vec = [_finite(c) for c in r]
        out.append(np.array(vec, dtype=np.float64))
    return out


def _raise_missing(p: Path, n: int) -> float:  # pragma: no cover — raises
    raise ValueError(f"CSV {p}: row is missing column {n}")


def media_item_features(task, path) -> np.ndarray:
    """SPEC.md 42.1.3 (v0.28): one media file → its raw (unstandardized)
    feature vector via the task's `item_features(path)` protocol method
    (`ImageTask`/`AudioTask` public wrappers, 42.1.3). A task without the
    method is an error — `predict --item` is media-tasks only."""
    fn = getattr(task, "item_features", None)
    if not callable(fn):
        raise ValueError(
            f"task {getattr(task, 'name', task)!r} has no `item_features` "
            "method — `predict --item` is for media tasks (image/audio, "
            "SPEC.md 42.1.3)")
    return np.asarray(fn(Path(path)), dtype=np.float64).ravel()
