"""Shared labelled-directory rules for the v0.10 media tasks (SPEC.md 24.2).

Both `ImageTask` and `AudioTask` load "one directory, one subfolder per
class (or an `index.csv` of file+label rows)" and then run the identical
§22.1 head-inference / seed-split / train-only-standardization protocol.
This module owns the directory + label rules so the two tasks share one
source of truth; the feature decoding (pixels / log-mel) lives in each
task's own file.

Rules (all deterministic, stdlib only — SPEC.md 24.2):
  * `index.csv` in the directory: column 1 = file path (relative to the
    directory, or absolute), column 2 = label; column names are any of
    path/file + label/target/y/class, else position;
  * otherwise one subfolder per class: `<dir>/<label>/<file>`;
  * head inference is the §22.1 rule verbatim (integer labels, 2–50
    distinct classes → softmax, else mse);
  * splits are the §22.1 seed permutation (80/10/10 by default).
"""
from __future__ import annotations

import csv as _csv
import zlib
from pathlib import Path

import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")   # SPEC.md 24.3
_AUDIO_EXTS = (".wav", ".mp3")                            # SPEC.md 24.4
_MAX_CLASSES = 50  # §22.1 rule: integer labels with more classes → regression
_INDEX_HINTS = ("label", "target", "y", "class")


def detect_modality(path: str | Path) -> str | None:
    """'image' | 'audio' | None for a directory (SPEC.md 24.5 auto-detect).

    Counts items by extension (subfolders only, plus `index.csv` which
    labels either modality — in that case the *files* decide).
    """
    d = Path(path)
    n_img = n_aud = 0
    if d.is_dir():
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in _IMAGE_EXTS:
                    n_img += 1
                elif ext in _AUDIO_EXTS:
                    n_aud += 1
    if n_img and not n_aud:
        return "image"
    if n_aud and not n_img:
        return "audio"
    if n_img and n_aud:
        return "mixed"  # caller must ask for an explicit --task
    return None


def collect_items(dir_path: str | Path, exts: tuple[str, ...],
                  what: str) -> list[tuple[Path, object]]:
    """(file, label) pairs for one labelled directory (SPEC.md 24.2).

    `exts` is the modality's accepted extensions; `what` is the human name
    used in error messages ("image" / "audio"). Labels are raw: the
    subfolder name (str) or the index.csv value (float) — `resolve_labels`
    maps them to model targets.
    """
    d = Path(dir_path)
    if not d.is_dir():
        raise ValueError(
            f"no such {what} directory: {dir_path!r} — expected a directory "
            f"with one subfolder per class, or an index.csv (SPEC.md 24.2)"
        )
    index = d / "index.csv"
    if index.is_file():
        return _read_index(index, d, what)
    items: list[tuple[Path, object]] = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                items.append((f, sub.name))
    if not items:
        raise ValueError(
            f"no {what} files under {dir_path!r} — put one subfolder per "
            f"class (e.g. {dir_path!r}/<class>/file) or add an index.csv "
            f"(SPEC.md 24.2); accepted extensions: "
            f"{', '.join(exts)}"
        )
    return items


def _read_index(index: Path, d: Path, what: str) -> list[tuple[Path, float]]:
    with open(index, newline="", encoding="utf-8-sig") as fh:
        reader = _csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{index} is empty") from None
        # column resolution: named hints first, else position (SPEC.md 24.2)
        low = {h.lower(): i for i, h in enumerate(header)}
        path_i = 0
        label_i = 1
        for hint in ("path", "file"):
            if hint in low:
                path_i = low[hint]
                break
        for hint in _INDEX_HINTS:
            if hint in low:
                label_i = low[hint]
                break
        items: list[tuple[Path, float]] = []
        for raw in reader:
            if not raw or not any((c or "").strip() for c in raw):
                continue
            if max(path_i, label_i) >= len(raw):
                continue
            p = Path(raw[path_i].strip())
            if not p.is_absolute():
                p = d / p
            try:
                label = float(raw[label_i].strip())
            except ValueError:
                raise ValueError(
                    f"{index}: label {raw[label_i]!r} is not numeric — "
                    f"use integer class ids (or one subfolder per class) "
                    f"(SPEC.md 24.2)"
                ) from None
            items.append((p, label))
    if not items:
        raise ValueError(f"{index} has no rows")
    return items


def infer_head(y: np.ndarray) -> tuple[str, int, list[float] | None, np.ndarray]:
    """The §22.1 head rule, verbatim (SPEC.md 24.2):
    (head, n_outputs, class_values, integer-class targets or raw)."""
    classes = np.unique(y)
    integer_labels = bool(np.all(y == np.floor(y)))
    if integer_labels and 2 <= classes.size <= _MAX_CLASSES:
        return ("softmax", int(classes.size),
                [float(c) for c in classes],
                np.array([int(np.flatnonzero(classes == v)[0]) for v in y],
                         dtype=np.int64))
    return ("mse", 1, None, y)


def resolve_labels(labels: list) -> tuple[str, int, list, np.ndarray]:
    """Raw item labels → (head, n_outputs, class_values, targets).

    Numeric labels (index.csv) take the §22.1 rule verbatim (SPEC.md 24.2);
    string labels (one subfolder per class) map to softmax classes over the
    sorted unique names, with the same 2–50 class bound.
    """
    try:
        y = np.array([float(v) for v in labels], dtype=np.float64)
    except (TypeError, ValueError):
        pass  # at least one label is not numeric → string class names
    else:
        return infer_head(y)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(
            f"need at least 2 classes, got {classes!r} — one subfolder per "
            f"class (SPEC.md 24.2)")
    if len(classes) > _MAX_CLASSES:
        raise ValueError(
            f"{len(classes)} string classes exceed the §22.1 limit of "
            f"{_MAX_CLASSES} (SPEC.md 24.2)")
    idx = {c: i for i, c in enumerate(classes)}
    return ("softmax", len(classes), list(classes),
            np.array([idx[v] for v in labels], dtype=np.int64))


def class_label_str(c: object) -> str:
    """Display string for a canonical `class_values` entry (SPEC.md 28.3).

    Numeric classes render without a trailing ".0" (1.0 -> "1"); string
    class names pass through. Deterministic, ASCII-safe for captions.
    """
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c)


def split_indices(n: int, seed: int, split_frac: float,
                  salt: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-derived train/holdout/gen partition — the §22.1 rule (G2)."""
    n_train = int(n * (1.0 - split_frac))
    rest = n - n_train
    n_hold = rest // 2
    n_gen = rest - n_hold
    if n_train < 1 or n_hold < 1 or n_gen < 1:
        raise ValueError(
            f"{n} items are too few for train/holdout/gen splits with "
            f"split_frac={split_frac}"
        )
    seq = np.random.SeedSequence([seed, zlib.crc32(salt)])
    perm = np.random.default_rng(seq).permutation(n)
    return (perm[:n_train], perm[n_train:n_train + n_hold],
            perm[n_train + n_hold:])
