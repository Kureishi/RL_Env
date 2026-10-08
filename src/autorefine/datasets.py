"""SPEC.md 89.4 (v0.75, A4) + 89.7 (v0.75, B3): dataset snapshots and
recency-window helpers.

Pure stdlib over CSV files — no numpy, no side effects beyond the append to
``runs/datasets.jsonl`` (89.4.1) and the ``mkdir`` of the runs dir.
Deterministic (G2): the same file always snapshots to the same
``ds_id`` / hash.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .discover import label_column  # 88.1.1 (A1): the label column rule

#: 89.4.1: the snapshot registry lives in the runs dir (like registry.json)
DATASETS_FILENAME = "datasets.jsonl"


def hash_file(path) -> str:
    """89.4.1 (A4): the sha256 hex digest of a file's bytes (chunked —
    a missing file is a fail-loud ``FileNotFoundError``)."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_snapshot(runs_dir, snap: dict) -> None:
    """89.4.1 (A4): append one snapshot entry to ``runs/datasets.jsonl``
    (creating the runs dir; one compact JSON line, sorted keys)."""
    rd = Path(runs_dir)
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / DATASETS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, sort_keys=True) + "\n")


def snapshot_dataset(path, runs_dir) -> dict:
    """89.4.1 (A4): one deterministic snapshot entry for a CSV file —
    ``ds_id = "ds-" + sha256[:12]``, the full ``sha256``, ``path``,
    ``n_rows`` / ``n_cols``, the 88.1.1 ``label_column`` index plus its
    ``label_name``, ``class_balance`` (sorted value → count), the file
    ``mtime`` (epoch seconds), and ``created`` (GMT ISO). Appended via
    :func:`append_snapshot`; the entry is returned."""
    p = Path(path)
    sha = hash_file(p)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = [c.strip() for c in next(reader)]
        except StopIteration:
            header = []
        rows = [r for r in reader if any((c or "").strip() for c in r)]
    lbl = label_column(header)
    balance: dict[str, int] = {}
    if lbl is not None:
        for r in rows:
            if len(r) > lbl and (r[lbl] or "").strip():
                v = r[lbl].strip()
                balance[v] = balance.get(v, 0) + 1
    st = p.stat()
    snap = {
        "ds_id": "ds-" + sha[:12],
        "path": str(p),
        "sha256": sha,
        "n_rows": len(rows),
        "n_cols": len(header),
        "label_column": lbl,
        "label_name": (header[lbl] if lbl is not None and lbl < len(header)
                       else None),
        "class_balance": {k: balance[k] for k in sorted(balance)},
        "mtime": int(st.st_mtime),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_snapshot(runs_dir, snap)
    return snap


def load_snapshots(runs_dir) -> list[dict]:
    """89.4.2 (A4): the snapshot registry in file order; a missing file is
    an empty registry."""
    p = Path(runs_dir) / DATASETS_FILENAME
    if not p.is_file():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def find_snapshot(runs_dir, ds_id: str) -> dict | None:
    """89.4.3 (A4): resolve a snapshot by exact ``ds_id`` or a unique
    prefix (a prefix matching more than one entry — or none — is ``None``).
    """
    snaps = load_snapshots(runs_dir)
    for s in snaps:
        if s.get("ds_id") == ds_id:
            return s
    seen = set()
    hits = []
    for s in snaps:
        sid = str(s.get("ds_id", ""))
        if sid.startswith(ds_id) and sid not in seen:
            seen.add(sid)
            hits.append(s)
    return hits[0] if len(hits) == 1 else None


def latest_snapshot(runs_dir, path) -> dict | None:
    """89.4 (A4): the most recent snapshot of ``path`` (string compare,
    case-sensitive — the registry stores the caller's path), or ``None``."""
    want = str(path)
    hits = [s for s in load_snapshots(runs_dir) if str(s.get("path")) == want]
    return hits[-1] if hits else None


# --- 89.7 (v0.75, B3): recency windows --------------------------------------

_WINDOW_RE = re.compile(r"^(\d+(?:\.\d+)?)([dhms])$")
_UNITS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_window(text: str) -> float:
    """89.7.1 (B3): ``7d`` / ``24h`` / ``90m`` / ``30s`` → seconds. Any bad
    form (units, words, negatives, bare numbers) is a ``ValueError`` at
    parse time."""
    m = _WINDOW_RE.match(str(text).strip())
    if m is None:
        raise ValueError(
            f"bad window {text!r}: expected forms like 7d, 24h, 90m, 30s")
    return float(m.group(1)) * _UNITS[m.group(2)]


#: 89.7.1 (B3): the recognized date column names (case-insensitive)
DATE_COL_NAMES = ("date", "time", "timestamp", "ts", "day", "when", "created_at")


def find_date_col(header) -> int | None:
    """89.7.1 (B3): the index of the first header cell whose name is in
    :data:`DATE_COL_NAMES` (case-insensitive), else ``None``."""
    names = [str(h).strip().lower() for h in header]
    for i, n in enumerate(names):
        if n in DATE_COL_NAMES:
            return i
    return None


def parse_date(value) -> float | None:
    """89.7.1 (B3): one date cell → epoch seconds — ISO 8601 text (with or
    without a zone; naive is read as UTC) or an epoch-seconds number.
    Unparseable cells are ``None`` (dropped by the window cut)."""
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def window_rows(header, rows, window_s: float, date_col: int | None):
    """89.7.1 (B3): the recency slice — the rows whose date column is
    ``>= max_date − window_s`` (dates: 89.7.1 ISO 8601 or epoch seconds;
    unparseable cells never count as in-window). No date column, or no
    parseable dates at all, keeps every row (the caller decides what to say).
    Returns ``(kept_rows, total, lo_iso, hi_iso)`` — the bounds are
    ``None`` in the keep-all case."""
    rows = list(rows)
    if date_col is None or not rows:
        return rows, len(rows), None, None
    dates: list[float | None] = []
    for r in rows:
        dates.append(parse_date(r[date_col])
                     if len(r) > date_col and (r[date_col] or "").strip()
                     else None)
    valid = [d for d in dates if d is not None]
    if not valid:
        return rows, len(rows), None, None
    hi = max(valid)
    cut = hi - float(window_s)
    kept = [r for r, d in zip(rows, dates) if d is not None and d >= cut]
    return kept, len(rows), _iso(cut), _iso(hi)
