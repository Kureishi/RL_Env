"""SPEC.md 55 (v0.41) — "make it feel live": the three live-view helpers.

The dashboard app (``dashboard_app.py``) is a thin renderer over these
pure functions (55.x); the run semantics live in ``dashboard.DashboardRunner``
unchanged. All of this is opt-in / additive, so the default synchronous
path and the A-pins stay byte-identical (G2). Pure, deterministic (G2).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .dashboard import eta_seconds, running_best_curve


def _finite_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def freshness_caption(n_updates, elapsed_s, done: bool = False) -> str:
    """55.1 (v0.41): the live freshness caption — the run's state
    (``live`` / ``done``), how many experiments are in, and how long ago
    the last tick landed (``just now`` under 5 s, else ``<n>s ago``; a
    non-finite ``elapsed_s`` reads ``?``). Pure, deterministic (G2)."""
    state = "done" if done else "live"
    n = int(n_updates) if _finite_num(n_updates) else 0
    if not _finite_num(elapsed_s):
        when = "?"
    elif float(elapsed_s) < 5:
        when = "just now"
    else:
        when = f"{float(elapsed_s):.0f}s ago"
    return f"{state} · {n} experiment(s) in · updated {when}"


def stream_tick_due(record) -> bool:
    """55.1 (v0.41): True when the worker thread is still alive and the
    run has not been drained — the app uses this to decide whether to
    ``st.rerun()`` and tick the page (the opt-in stream mode) or to finish
    the synchronous drain (the default path, byte-identical). Pure over
    the record dict; a dead / absent thread or a drained record reads as
    "not due"."""
    if not isinstance(record, dict) or record.get("drained"):
        return False
    th = record.get("thread")
    return th is not None and bool(th.is_alive())


def snapshot_inputs(rec, result) -> tuple:
    """55.2 (v0.41): the ``(info, stream, best_series, done)`` tuple that
    seeds ``status_snapshot`` — from the finished ``result`` (the drain's
    stored payload, 23.2) or, while a run is still live, derived from the
    worker's ``record`` messages (the ``info`` payload seeds the best curve;
    each ``update``'s ``best_score`` succeeds it). Pure over the two dicts
    (the app is a thin renderer, 23.1); ``(None, [], [], False)`` before a
    run (the widgets are gated on a run existing, 55.4)."""
    if result is not None:
        return (result.get("info"), result.get("stream"),
                result.get("best_series"), True)
    if rec is not None and isinstance(rec.get("messages"), list):
        info = None
        stream = []
        best = []
        for kind, payload in rec["messages"]:
            if kind == "info":
                info = payload
                best.append(round(payload["baseline_score"], 2))
            elif kind == "update":
                stream.append(payload)
                best.append(round(payload["best_score"], 2))
        return info, stream, best, False
    return None, [], [], False


def status_snapshot(info, stream, best_series, done: bool = False) -> str:
    """55.2 (v0.41): the mid-run status card — a self-contained text block
    (task, head, rows, baseline, target, the running best, the budget
    spent, the ETA, and the last 3 decisions) for a "how's it going?" chat
    message. Pure over the ``info`` dict (SPEC.md 23.1), the update
    ``stream`` (23.1), and the ``best_series`` (the baseline seeds it).
    ``done`` flips the header (live → final). Non-finite / missing fields
    degrade gracefully (omitted lines, never a crash)."""
    info = info if isinstance(info, dict) else {}
    lines = ["AutoRefine run — FINAL" if done else "AutoRefine run — LIVE"]
    label = info.get("label")
    head = info.get("head")
    lines.append(f"task: {label if label is not None else '?'}"
                 f" · head: {head if head is not None else '?'}")
    rows = info.get("rows")
    if isinstance(rows, dict) and rows:
        lines.append(f"rows: train {rows.get('train')} / holdout "
                     f"{rows.get('holdout')} / gen {rows.get('gen')}")
    base = info.get("baseline_score")
    if _finite_num(base):
        lines.append(f"baseline: {float(base):.2f}")
    tgt = info.get("target")
    if _finite_num(tgt):
        lines.append(f"target: {float(tgt):.1f}")
    bests = [float(v) for v in (best_series or []) if _finite_num(v)]
    if bests:
        lines.append(f"best so far: {max(bests):.2f}")
    # budget spent + ETA from the last update that carried experiments_left
    left = None
    for u in reversed(stream or []):
        if isinstance(u, dict) and _finite_num(u.get("experiments_left")):
            left = int(u["experiments_left"])
            break
    budget = info.get("budget_experiments")
    if left is not None and _finite_num(budget):
        spent = max(0, int(budget) - left)
        line = f"budget: {spent} of {int(budget)} experiments spent"
        if left > 0:
            line += f" · {left} left"
        lines.append(line)
        eta = eta_seconds(stream, left)
        if eta is not None:
            lines.append(f"eta: ~{eta:.1f}s remaining")
    # the last 3 decisions (scored or dup candidates, in stream order)
    decisions = []
    for u in reversed(stream or []):
        if not isinstance(u, dict):
            continue
        acc = "accepted" if u.get("accepted") else "rejected"
        cs = u.get("candidate_score")
        score = f"{float(cs):.2f}" if _finite_num(cs) else "dup"
        mut = ", ".join(f for f in (u.get("mutation") or []) if isinstance(f, str))
        idx = u.get("index")
        decisions.append(f"#{idx} {acc} ({score}): {mut or '-'}")
        if len(decisions) == 3:
            break
    lines.append("last 3 decisions:")
    if decisions:
        for d in decisions:
            lines.append(f"  - {d}")
    else:
        lines.append("  (none yet)")
    return "\n".join(lines)


def reference_curve(runs_dir, run_id) -> list[float]:
    """55.3 (v0.41): a past run's running-best score curve — the runs
    dir's ``experiments.jsonl`` run through ``running_best_curve``
    (SPEC.md 50.1.2). ``[]`` when the run is missing, unreadable, or has
    no scored rows (it then drops out of the overlay). Pure,
    deterministic (G2)."""
    try:
        ep = Path(runs_dir) / str(run_id) / "experiments.jsonl"
        if not ep.is_file():
            return []
        rows = [json.loads(line) for line in
                ep.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    return running_best_curve(rows)


def list_reference_runs(runs_dir) -> list[tuple[str, list[float]]]:
    """55.3 (v0.41): the runs dir's runs that carry a scored best curve,
    as ``[(run_id, best_series)]`` sorted by run_id — the options for the
    reference-overlay selectbox. ``[]`` when the dir is missing, not a
    directory, or has no scored run. Pure, deterministic (G2)."""
    try:
        root = Path(runs_dir)
        if not root.is_dir():
            return []
        out = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            curve = reference_curve(runs_dir, d.name)
            if curve:
                out.append((d.name, curve))
    except OSError:
        return []
    return out


__all__ = [
    "freshness_caption",
    "stream_tick_due",
    "snapshot_inputs",
    "status_snapshot",
    "reference_curve",
    "list_reference_runs",
]
