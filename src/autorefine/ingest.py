"""SPEC.md 89.1/89.2/89.3/89.6 (v0.75, A1/A2/A3/B2): the on-demand data
surfaces — the change poll, the drift gate, the champion/challenger
verdict, and the drop-folder merge.

Pure stdlib (no numpy): every function here is a deterministic decision or
a file read (G2). State files (``feed_state.json``,
``data_events.jsonl``, ``drop_state.json``) live where the caller says —
the runs dir for the feed, the drop dir itself for the ingest.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

# 89.1 (A1): the feed's per-data-file state + the scored-poll event log
FEED_STATE_FILENAME = "feed_state.json"
DATA_EVENTS_FILENAME = "data_events.jsonl"


def file_fingerprint(path) -> tuple[int, int] | None:
    """89.1.1 (A1): ``(mtime_ns, size)`` of a file — the change token the
    poll compares against; a missing / unreadable file is ``None``."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def poll(path, state) -> dict:
    """89.1.1 (A1): one deterministic change check. ``state`` is the
    previously persisted fingerprint (a ``[mtime_ns, size]`` list or
    ``None``); the FIRST poll (no state, or a changed fingerprint) always
    counts as changed. A vanished file reports ``changed: False`` with a
    ``None`` fingerprint (the caller decides how to say it)."""
    fp = file_fingerprint(path)
    if fp is None:
        return {"changed": False, "fingerprint": None}
    prev = state if isinstance(state, dict) else None
    prev_fp = prev.get("fingerprint") if prev is not None else None
    if prev_fp is None or list(prev_fp) != list(fp):
        return {"changed": True, "fingerprint": list(fp)}
    return {"changed": False, "fingerprint": list(fp)}


def load_state(runs_dir) -> dict:
    """89.1 (A1): the persisted feed state (``{path: {"fingerprint": [...]}}``);
    a missing / corrupt file is the empty state."""
    p = Path(runs_dir) / FEED_STATE_FILENAME
    if not p.is_file():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return obj if isinstance(obj, dict) else {}


def save_state(runs_dir, state) -> None:
    """89.1 (A1): persist the feed state (creating the runs dir; sorted
    keys for a stable file)."""
    rd = Path(runs_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / FEED_STATE_FILENAME).write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_event(runs_dir, event: dict) -> None:
    """89.1.3 (A1): append one scored-poll event to ``data_events.jsonl``
    (one compact JSON line, sorted keys)."""
    rd = Path(runs_dir)
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / DATA_EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def load_events(runs_dir) -> list[dict]:
    """89.1.3 / 89.8: the scored-poll events in file order (a missing file
    is an empty log)."""
    p = Path(runs_dir) / DATA_EVENTS_FILENAME
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


def drift_verdict(fresh: float, ref: float, bar: float = 2.0) -> tuple[str, float]:
    """89.2 (A2): the drift gate — the champion's score on the fresh rows
    (``fresh``) vs its reference score on its own data (``ref``): **alert**
    iff ``ref − fresh > bar``, else **stable**. Returns
    ``(status, margin)`` with ``margin = ref − fresh`` (a negative margin
    = the fresh data is EASIER). Pure, deterministic (G2)."""
    margin = round(float(ref) - float(fresh), 6)
    status = "alert" if margin > float(bar) else "stable"
    return status, margin


def refresh_verdict(champion: float, challenger: float,
                    margin: float = 0.0) -> tuple[str, str]:
    """89.3.2 (A3): **PROMOTE** iff ``challenger >= champion + margin``,
    else **KEEP** — with the one-line verdict text the ``refresh`` command
    prints (``<VERDICT> - challenger X.X vs champion Y.Y (margin M)``).
    Pure, deterministic (G2)."""
    ch, ca, m = float(champion), float(challenger), float(margin)
    if ca >= ch + m:
        return "PROMOTE", (
            f"PROMOTE - challenger {ca:.1f} vs champion {ch:.1f} "
            f"(margin {m:g})")
    return "KEEP", (
        f"KEEP - challenger {ca:.1f} vs champion {ch:.1f} (margin {m:g})")


# --- 89.6 (v0.75, B2): the drop-folder merge --------------------------------

def merge_tables(paths) -> tuple[list, list, list, list, int]:
    """89.6.1/89.6.2 (B2): concatenate CSV tables in the given order — the
    FIRST file's header wins; a later file whose header *set* differs is
    mismatched (skipped entirely, never a silent column shift);
    exact-duplicate rows (normalized cell tuple) are dropped. Returns
    ``(header, rows, merged_files, mismatched_files, dupes_skipped)``."""
    header: list[str] = []
    rows: list[list[str]] = []
    merged: list[str] = []
    mismatched: list[str] = []
    dupes = 0
    seen: set[tuple] = set()
    ref_set: set[str] | None = None
    for p in paths:
        p = str(p)
        with Path(p).open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            try:
                h = [c.strip() for c in next(reader)]
            except StopIteration:
                h = []
            file_rows = [r for r in reader if any((c or "").strip() for c in r)]
        hset = set(h)
        if ref_set is None:
            ref_set = hset
            header = h
        elif hset != ref_set:
            mismatched.append(p)
            continue
        for r in file_rows:
            key = tuple((c or "").strip() for c in r)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            rows.append(list(r))
        merged.append(p)
    return header, rows, merged, mismatched, dupes


def load_drop_state(drop_dir) -> dict:
    """89.6.1 (B2): the drop dir's ``drop_state.json``
    (``{"seen": {filename: sha256}}``); a missing / corrupt file is the
    fresh state."""
    p = Path(drop_dir) / "drop_state.json"
    if not p.is_file():
        return {"seen": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"seen": {}}
    seen = raw.get("seen") if isinstance(raw, dict) else None
    return {"seen": dict(seen) if isinstance(seen, dict) else {}}


def save_drop_state(drop_dir, state) -> None:
    """89.6.1 (B2): persist the drop state (creating the dir; sorted keys)."""
    d = Path(drop_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "drop_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def hash_csv(path) -> str:
    """89.6.1 (B2): the sha256 hex digest of a file's bytes (chunked)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
