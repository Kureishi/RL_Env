"""SPEC.md 39.1 (v0.25, T3): watch mode / live progress.

Pure helpers that tail a run's `experiments.jsonl` and re-render the ASCII
charts (human mode) or emit compact JSON lines (`--tail`, for CI). A stdlib-
only leaf: it depends only on `plotting` for the chart frames and `memory`
for the kind registry. The polling loop itself lives in `cli._cmd_watch`
(39.1.1/39.1.4); everything here is a deterministic, side-effect-free
function of its inputs, so it is testable without sleeping.
"""
from __future__ import annotations

import json
from pathlib import Path

from .memory import KIND_BASELINE, KIND_EXPERIMENT  # 35.1 (C4): kind registry
from .plotting import ascii_pareto, ascii_score_curve


def read_new_entries(path: str | Path, offset: int = 0) -> tuple[list[dict], int]:
    """SPEC.md 39.1.2: tail `experiments.jsonl` from a byte offset.

    Only complete (``\\n``-terminated) lines are parsed; a partial trailing
    line (not yet flushed) is left for the next poll. A missing file returns
    ``([], 0)``. Re-reading from the same offset is idempotent.
    """
    p = Path(path)
    if not p.is_file():
        return [], 0
    data = p.read_bytes()
    if len(data) <= offset:
        return [], offset
    chunk = data[offset:]
    end = chunk.rfind(b"\n")
    if end == -1:  # no complete line yet — wait for the newline to land
        return [], offset
    complete = chunk[: end + 1]
    entries: list[dict] = []
    for line in complete.split(b"\n"):
        if not line.strip():
            continue
        entries.append(json.loads(line.decode("utf-8")))
    return entries, offset + end + 1


def run_finished(run_dir: str | Path) -> bool:
    """SPEC.md 39.1.2/39.1.4: the run is in its terminal state when
    `summary.json` exists (written at `_finish`)."""
    return (Path(run_dir) / "summary.json").is_file()


def _pareto_points(entries: list[dict]) -> list[dict]:
    """Scored frontier points from the log (same rule as `report --plot`)."""
    pts = []
    for e in entries or []:
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):
            continue
        score = e.get("holdout_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            pts.append({"score": score,
                        "train_seconds": e.get("train_seconds", 0.0)})
    return pts


def live_frame(entries: list[dict], pareto_points: list[dict] | None = None) -> str:
    """SPEC.md 39.1.2: the human frame — a live header plus the existing
    ASCII score curve + Pareto (SPEC 21.2). Deterministic string."""
    if pareto_points is None:
        pareto_points = _pareto_points(entries)
    # the best score so far = the max scored holdout score in the log
    scores = [e.get("holdout_score") for e in entries or []
              if e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)
              and isinstance(e.get("holdout_score"), (int, float))
              and not isinstance(e.get("holdout_score"), bool)]
    best = max(scores) if scores else None
    n_exp = sum(1 for e in entries or [] if e.get("kind") == KIND_EXPERIMENT)
    header = "watching: " + (f"{best:8.2f}" if best is not None else "  (no score yet)")
    header += f"   best score   |   experiments run: {n_exp}"
    parts = [header, "", ascii_score_curve(entries), "",
             "pareto frontier (score vs training seconds):",
             ascii_pareto(pareto_points)]
    return "\n".join(parts)


def tail_line(entry: dict, index: int) -> str:
    """SPEC.md 39.1.2: one compact machine line — a stable projection of the
    row (not the full spec / loss_history, which would bloat the stream)."""
    return json.dumps({
        "index": index,
        "kind": entry.get("kind"),
        "accepted": entry.get("accepted"),
        "holdout_score": entry.get("holdout_score"),
        "gen_gap": entry.get("gen_gap"),
        "train_seconds": entry.get("train_seconds"),
    }, sort_keys=True)


__all__ = [
    "read_new_entries",
    "run_finished",
    "live_frame",
    "tail_line",
]
