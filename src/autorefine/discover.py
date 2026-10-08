"""SPEC.md 88.1 (v0.74, A1) — data discovery for a path-less ``go``.

The user's fewest words: ``cd`` into the folder that holds the data and
type ``autorefine go``. This leaf owns the one deterministic scan,
``discover_data(cwd)`` — CSV files in the directory first, then
directories holding a usable media layout (the 24.2 one-subfolder-per-
class or ``index.csv`` shapes, classified by ``tasks.detect_modality``)
— plus the small stdlib readers the ``found :`` line and the 88.8
``agreement :`` line share: ``summarize_csv`` and ``label_column``.

House rules (3 / 88 intro): stdlib only (plus the repo's own task
detector, which is import-cycle-free); pure and deterministic (G2) —
the ordering is (kind rank, mtime descending, name ascending), so two
scans of the same directory list the same candidates in the same
order; fail-loud where a guess would be silent (an empty directory
returns ``[]`` — the CLI says so, 88.1.2).
"""
from __future__ import annotations

import csv
from pathlib import Path

# 88.1.1: the candidate contract — one dict per usable item. ``task`` is
# the resolved task name (``csv`` for files; the detector's name for a
# media directory), ``reason`` the one phrase the ``found :`` line prints.
CANDIDATE_KEYS = ("path", "task", "reason")

# 88.1.1: the label-column vocabulary (case-insensitive), shared with the
# 88.8 ``agreement :`` line — the first hit wins, else the last column
# when there are at least two (a one-column file has no features to
# learn, so it is not a label table).
_LABEL_NAMES = ("label", "target", "y", "class")


def label_column(header: list[str]) -> int | None:
    """SPEC.md 88.1.1 (A1): the label column index of a CSV header — the
    first of ``label`` / ``target`` / ``y`` / ``class`` (case-
    insensitive), else the last column when there are at least two;
    ``None`` for an empty or single-column header (nothing to label).
    Deterministic (G2)."""
    names = [str(h).strip().lower() for h in header]
    if len(names) < 2:
        return None  # one column has no features to learn, so no label
    for i, name in enumerate(names):
        if name in _LABEL_NAMES:
            return i
    return len(names) - 1


def summarize_csv(path) -> tuple[int, int]:
    """SPEC.md 88.1.1 (A1): ``(n_rows, n_label_values)`` of a CSV via the
    stdlib reader (no numpy — a header-only file is ``(0, 0)`` and an
    unlabeled one keeps ``n_rows`` with ``0`` label values). The file is
    read at most once; a missing file is a fail-loud ``FileNotFoundError``
    (the caller decides how to say it)."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return (0, 0)
        col = label_column(header)
        n_rows = 0
        values: set = set()
        for row in reader:
            if not row:
                continue
            n_rows += 1
            if col is not None and col < len(row):
                values.add(row[col].strip())
    return (n_rows, len(values))


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:  # vanished between listdir and stat — sort last
        return float("-inf")


def discover_data(cwd) -> list[dict]:
    """SPEC.md 88.1 (A1): the deterministic candidate list of a directory.

    CSV files first (task ``csv``, kind rank 0), then directories whose
    media layout resolves to a single task (kind rank 1, the 24.5
    detector — ``mixed`` / empty are dropped); ordered by (kind rank,
    mtime descending, name ascending) so the newest usable item wins and
    ties break on the name (G2). An unreadable / missing directory is a
    fail-loud ``FileNotFoundError`` (88.1.2's rc 1 path)."""
    root = Path(cwd)
    if not root.is_dir():
        raise FileNotFoundError(f"no such directory: {root}")
    from .tasks import detect_modality  # local: keep the leaf import-light
    files: list[tuple[float, str]] = []
    dirs: list[tuple[float, str, str]] = []
    for child in root.iterdir():
        try:
            if child.is_file() and child.suffix.lower() == ".csv":
                files.append((_mtime(child), child.name))
            elif child.is_dir():
                mod = detect_modality(child)
                if mod in ("image", "audio", "text"):
                    dirs.append((_mtime(child), child.name, mod))
        except OSError:
            continue  # an item that disappears mid-scan is not a candidate
    out: list[dict] = []
    for mt, name in sorted(files, key=lambda t: (-t[0], t[1])):
        out.append({"path": str(root / name), "task": "csv",
                    "reason": "a CSV table"})
    for mt, name, mod in sorted(dirs, key=lambda t: (-t[0], t[1])):
        out.append({"path": str(root / name), "task": mod,
                    "reason": f"a folder of {mod} items"})
    return out


def pick_data(cwd) -> dict | None:
    """SPEC.md 88.1.2 (A1): the first candidate of ``discover_data`` (the
    newest usable item, 88.1.1), or ``None`` for an empty directory —
    the CLI's rc 1 path (88.1.2)."""
    cands = discover_data(cwd)
    return cands[0] if cands else None


__all__ = [
    "CANDIDATE_KEYS",
    "label_column",
    "summarize_csv",
    "discover_data",
    "pick_data",
]
