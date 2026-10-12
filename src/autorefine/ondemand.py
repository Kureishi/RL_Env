"""SPEC.md 90 (v0.76, A80) — the on-demand UI surface: the same v0.75
ingestion machinery the CLI exposes (89) re-cut as pure, side-effect-free
cores the dashboard renders.

The dashboard (``dashboard_app``) is a thin renderer over these (23.1):
every function here is a deterministic decision or a read given its inputs
(G2). The only side effects are the challenger loop's own artifacts
(written under the caller's runs dir by ``DashboardRunner``, 23.1) and an
optional dataset snapshot.

Design notes (kept DRY with the CLI):
- champion reconstruction + the 42.1 scoring leaf live in ``cli`` (they are
  the one home, shared by ``feed``/``refresh``/``predict``); the cli imports
  many core modules and must not be imported *back* by them, so those helpers
  are imported lazily, inside the functions that need them (the same local-
  import pattern ``advanced`` uses for ``_task_from_summary``).
- the drift / refresh verdicts are ``ingest``'s (89.2/89.3) — reused, not
  re-implemented.
- the recency-window slice is ``datasets``' (89.7) — reused.
"""
from __future__ import annotations

import csv as _csv
import tempfile
from pathlib import Path

from .datasets import find_date_col, parse_window, window_rows  # 89.7 (B3)
from .ingest import drift_verdict, load_events, refresh_verdict  # 89.2/89.3
from .predict import (  # noqa: F401  (the 42.1 leaf, re-exported for tests)
    csv_rows_to_features,
    predict_features,
    standardize,
)
from .registry import load_registry  # 38.1

__all__ = [
    "latest_champion",
    "csv_preflight",
    "score_fresh",
    "quality_events",
    "refresh_model",
]


# --- 90.1 (A80) — the champion -------------------------------------------------

def latest_champion(runs_dir) -> Path | None:
    """90.1 (A80): the champion run — the registry's latest finished run
    (the 88.3.1 tie-break: timestamp, then run_id) that has a
    ``summary.json``. ``None`` when the registry is empty, unreadable, or
    the newest entry is not a finished run. Pure (a read only)."""
    rd = Path(runs_dir)
    try:
        entries = load_registry(rd)
    except Exception:
        entries = []
    if not entries:
        return None
    best = max(entries, key=lambda e: (str(e.get("timestamp") or ""),
                                      str(e.get("run_id") or "")))
    run_dir = rd / str(best.get("run_id", "?"))
    if not (run_dir / "summary.json").is_file():
        return None
    return run_dir


# --- 90.4 (A80) — a pure usability preflight -----------------------------------

def _read_cells(path) -> tuple[list[str], list[list[str]], bool]:
    """A CSV as stripped cells — the SAME header rule as the 42.1 leaf (the
    first row with a non-numeric cell is the header; blank rows skipped).
    Returns ``(header, data_rows, has_header)``."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        rows = [[(c or "").strip() for c in raw] for raw in _csv.reader(fh)
                if any((c or "").strip() for c in raw)]
    if not rows:
        return [], [], False
    first = rows[0]
    if any(_not_num(c) for c in first):
        return first, rows[1:], True
    return [], rows, False


def _not_num(s: str) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return True
    return False


def csv_preflight(csv_path) -> dict:
    """90.4 (A80): a pure usability check for a fresh CSV — ``n_rows``,
    ``n_cols``, the label column (the 88.1.1 rule), the ``class_balance``
    (value → count, sorted), and a ``usable`` flag. Unlike
    ``datasets.snapshot_dataset`` this writes nothing (a UI hover must not
    append a snapshot). Raises ``FileNotFoundError`` on a missing file."""
    from .discover import label_column  # 88.1.1 (A1): the label rule
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    header, rows, _has = _read_cells(csv_path)
    lbl = label_column(header)
    balance: dict[str, int] = {}
    if lbl is not None:
        for r in rows:
            if len(r) > lbl and (r[lbl] or "").strip():
                v = r[lbl].strip()
                balance[v] = balance.get(v, 0) + 1
    return {
        "path": str(p),
        "n_rows": len(rows),
        "n_cols": len(header),
        "header": list(header),
        "label_column": lbl,
        "label_name": (header[lbl]
                       if (lbl is not None and lbl < len(header)) else None),
        "class_balance": {k: balance[k] for k in sorted(balance)},
        "usable": (lbl is not None) and (len(rows) > 0),
    }


def _champion(champ_run) -> tuple[dict, object, object]:
    """90.2/90.3 (A80): reconstruct the champion run (summary, task, model)
    honoring the ondemand contract — a run dir that is not reconstructable
    (a foreign/malformed summary, a missing model file) is a ``ValueError``,
    never a raw ``KeyError``/``TypeError`` leaking out of the shared CLI
    helper (``_task_from_summary`` reads ``summary["seed"]`` directly). The
    app then shows a friendly error instead of the render crashing."""
    from .cli import _run_task_and_model  # local (cycle-free)
    try:
        return _run_task_and_model(champ_run)
    except (ValueError, FileNotFoundError):
        raise
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"champion run {Path(champ_run)!r} is not usable: {exc!r}") from exc


# --- 90.2 (A80) — score the champion on fresh rows -----------------------------

def score_fresh(run_dir, csv_path, bar: float = 2.0) -> dict:
    """90.2 (A80): the champion's accuracy (0-100) on a FRESH CSV, judged
    by the drift gate (89.2) against the champion's own reference score —
    its finished run's ``final_best_score``. Returns the JSON-safe dict
    ``{"fresh", "ref", "status", "margin", "n"}`` where ``status`` is
    ``"stable"`` / ``"alert"`` (89.2). Raises ``ValueError`` /
    ``FileNotFoundError`` on a broken run dir or an unusable CSV (the app
    shows a friendly error)."""
    from .cli import _score_csv_champion  # local (cycle-free)
    rd = Path(run_dir)
    summary, task, model = _champion(rd)
    ref = summary.get("final_best_score")
    if not isinstance(ref, (int, float)) or isinstance(ref, bool):
        raise ValueError(
            f"the champion run {rd} has no final_best_score to compare to")
    acc, n = _score_csv_champion(task, model, csv_path)
    status, margin = drift_verdict(acc, float(ref), bar=bar)
    return {"fresh": float(acc), "ref": float(ref), "status": status,
            "margin": float(margin), "n": int(n)}


# --- 90.5 (A80) — the quality-over-time event log ------------------------------

def quality_events(runs_dir) -> list[dict]:
    """90.5 (A80): the scored-poll events (``data_events.jsonl``, 89.1.3)
    in file order — the data ``plotting.svg_quality_over_time`` renders.
    A missing log is ``[]``."""
    return load_events(runs_dir)


# --- 90.3 (A80) — champion vs challenger on fresh data -------------------------

def _recency_train_path(csv_path, window_text: str | None):
    """89.7 (B3): restrict a CSV to its recency window for the challenger's
    training, writing the slice to the session temp dir (the dashboard
    reuses that dir for uploads). A CSV without a date column — or without
    parseable dates — keeps every row (the 89.7 keep-all rule). Returns
    ``(train_path, note)`` where ``note`` is ``None`` (no window asked) or a
    JSON-safe dict describing the cut."""
    if window_text is None or not str(window_text).strip():
        return Path(csv_path), None
    p = Path(csv_path)
    header, data, has_header = _read_cells(csv_path)
    if not data:
        raise ValueError(f"{csv_path} has no data rows to window")
    dcol = find_date_col(header) if has_header else None
    window_s = parse_window(str(window_text).strip())
    kept, total, lo, hi = window_rows(header, data, window_s, dcol)
    if lo is None and hi is None:
        # keep-all: no date column / no parseable dates — train on every row
        return p, {"window": str(window_text).strip(),
                   "kept": len(data), "total": total,
                   "note": "no date column found — trained on every row"}
    dest = Path(tempfile.gettempdir()) / "autorefine_dashboard"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"window_{p.name}"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        if has_header:
            w.writerow(header)
        for r in kept:
            w.writerow(r)
    return out, {"window": str(window_text).strip(), "kept": len(kept),
                 "total": total, "lo": lo, "hi": hi}


def refresh_model(csv_path, runs_dir, champion_run,
                  experiments: int = 8, target: float = 95.0, seed: int = 7,
                  margin: float = 0.0, window: str | None = None,
                  max_train_seconds: float = 30.0) -> dict:
    """90.3 (A80): champion/challenger on a FRESH CSV (89.3), UI-facing —
    the same decision the ``refresh`` command prints, returned as data:

    - champion: the named run's best model scored on the fresh rows (89.1);
    - challenger: a fresh ``DashboardRunner`` (23.1, bandit, v04) loop over
      the fresh rows (optionally restricted to a recency window, 89.7);
    - verdict: ``refresh_verdict`` (89.3.2) — **PROMOTE** iff the
      challenger is at/above the champion + margin, else **KEEP**.

    Returns the JSON-safe dict ``{"champion_score", "challenger_score",
    "n", "verdict", "line", "run_dir"}`` (+ ``"window"`` when a window was
    asked). The challenger loop writes its own artifacts under ``runs_dir``
    (23.1) and lands a registry entry. Raises ``ValueError`` /
    ``FileNotFoundError`` on a broken champion run or unusable CSV."""
    from .cli import _score_csv_champion  # local (cycle-free)
    from .dashboard import DashboardRunner  # 23.1 (lazy: the heavy leaf)
    champ_run = Path(champion_run)
    summary, task, model = _champion(champ_run)  # ValueError on a bad run
    data = Path(csv_path)
    if not data.is_file():
        raise FileNotFoundError(f"no such data file: {data}")

    train_path, window_note = _recency_train_path(data, window)
    # the champion is judged on the SAME rows the challenger trains on —
    # an apples-to-apples PROMOTE/KEEP comparison (89.3.1)
    champ_score, n = _score_csv_champion(task, model, train_path)

    runner = DashboardRunner(
        csv_path=str(train_path), target=float(target), policy="bandit",
        seed=int(seed), experiments=int(experiments),
        max_train_seconds=float(max_train_seconds), runs_dir=str(runs_dir),
        search_quality="v04")
    runner.start()
    while not runner.done:
        runner.next()
    res = runner.finish()
    challenger = float(res["final_best_score"])
    verdict, line = refresh_verdict(champ_score, challenger,
                                    margin=float(margin))
    out = {
        "champion_score": float(champ_score),
        "challenger_score": challenger,
        "n": int(n),
        "verdict": verdict,
        "line": line,
        "run_dir": (str(runner.env.run_dir) if runner.env is not None else "?"),
    }
    if window_note is not None:
        out["window"] = window_note
    return out
