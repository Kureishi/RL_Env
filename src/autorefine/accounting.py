"""SPEC.md 39.2 (v0.25, T4): decision accounting for `report`.

A pure, stdlib-only "what happened" block derived *entirely* from
`experiments.jsonl` + `summary.json` — no new logged field, no change to
acceptance/scoring/summary (39.2.4). A leaf module: it reads the log kind
registry from `memory` and the §18.5 gen-gap tolerance from
`improver.meta_env` (single source of truth for `0.05`).

Only `cli._cmd_report` renders it (39.2.3); nothing else in the core reads
this module, and it has no side effects.
"""
from __future__ import annotations

import math

from .memory import (
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
)
from .improver.meta_env import GEN_GAP_TOL  # §18.5 tolerance (0.05); one home


def _num(value) -> float | None:
    """A finite number, else None (rows may lack a field on old runs)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


def _first_improvement(entries: list[dict]) -> dict:
    """SPEC.md 39.2.1: the first accepted *candidate* (kind=experiment) and
    the wall seconds from the baseline's `ts` to that row's `ts`."""
    baseline_ts = None
    for e in entries:
        if e.get("kind") == KIND_BASELINE:
            baseline_ts = _num(e.get("ts"))
            break
    improved = False
    index = None
    seconds = None
    ordinal = 0
    for e in entries:
        if e.get("kind") != KIND_EXPERIMENT:
            continue
        ordinal += 1
        if e.get("accepted") and not improved:
            improved = True
            index = ordinal
            ts = _num(e.get("ts"))
            if ts is not None and baseline_ts is not None:
                seconds = round(ts - baseline_ts, 6)
    return {
        "improved": improved,
        "experiment_index": index,
        "seconds": seconds,
    }


def _rejections(entries: list[dict]) -> dict:
    """SPEC.md 39.2.2: bucket the rejected *candidates* by the first failing
    gate (documented priority score → overfit → ci).

    The running best is reconstructed by scanning in order: the baseline
    seeds it, an accepted candidate raises it, and a curriculum step-up row
    re-pins it to its new baseline. Free-duplicate rejections (R3) are
    unspent and never logged, so the `duplicate` count is 0 by construction
    and the note says so (39.2.1)."""
    best_score: float | None = None
    accepted = 0
    buckets = {"score": 0, "overfit": 0, "ci": 0}
    for e in entries:
        kind = e.get("kind")
        if kind == KIND_BASELINE:
            best_score = _num(e.get("holdout_score"))
            if e.get("accepted"):  # 39.2.1: the baseline is in the scored set
                accepted += 1
            continue
        if kind == KIND_CURRICULUM:
            # a step-up re-baselines the champion (39.2.2); not a candidate
            rebase = _num(e.get("new_baseline_score"))
            if rebase is not None:
                best_score = rebase
            continue
        if kind != KIND_EXPERIMENT:
            continue  # screen / invalid_spec: not scored candidates here
        score = _num(e.get("holdout_score"))
        if e.get("accepted"):
            accepted += 1
            if score is not None:
                best_score = score
            continue
        # a rejected candidate: classify by the first failing gate
        if best_score is not None and score is not None and score <= best_score:
            buckets["score"] += 1
        else:
            gen_gap = _num(e.get("gen_gap"))
            if (score is not None and gen_gap is not None
                    and gen_gap > GEN_GAP_TOL * score):
                buckets["overfit"] += 1
            else:
                buckets["ci"] += 1
    rejected = sum(buckets.values())
    return {
        "accepted": accepted,
        "rejected": rejected,
        "score": buckets["score"],
        "overfit": buckets["overfit"],
        "ci": buckets["ci"],
        "duplicate": 0,  # free rejections are unspent + unlogged (39.2.1)
        "note": "free-duplicate rejections are unspent and not logged "
                "(they never appear in experiments.jsonl)",
    }


def _wall_time(summary: dict, entries: list[dict]) -> dict:
    """SPEC.md 39.2.1: baseline vs candidates vs the honest residual."""
    baseline = 0.0
    candidates = 0.0
    for e in entries:
        ts = _num(e.get("train_seconds"))
        if ts is None:
            continue
        if e.get("kind") == KIND_BASELINE:
            baseline += ts
        elif e.get("kind") == KIND_EXPERIMENT:
            candidates += ts
    total = _num(summary.get("wall_seconds")) or 0.0
    eval_overhead = round(total - baseline - candidates, 6)
    return {
        "baseline": round(baseline, 6),
        "candidates": round(candidates, 6),
        "eval_overhead": eval_overhead,
        "total": round(total, 6),
    }


def candidate_reason(accepted, candidate_score, gen_gap, best_before) -> str:
    """SPEC.md 51.3.1 (v0.37): the per-candidate gate verdict.

    ``"accepted"`` for an accepted candidate; for a rejection the first
    failing gate in the 39.2.2 priority — ``"score"`` (score <= the running
    best before this step), else ``"overfit"`` (gen_gap > 0.05 * score,
    the 18.5 tolerance), else ``"ci"``; an unscored rejection (no
    ``candidate_score``) is ``"dup"`` (the free duplicate/invalid
    rejections, 6 R3 / 25.4). Pure and deterministic (G2): the
    scored-rejection branching agrees with the ``_rejections`` buckets
    (39.2.2) row for row, so the app's reason column (51.3.2) and the
    report's accounting cannot drift."""
    if accepted:
        return "accepted"
    score = _num(candidate_score)
    if score is None:  # dup / invalid-spec: rejected without a score (R3)
        return "dup"
    if best_before is not None and score <= best_before:
        return "score"
    gap = _num(gen_gap)
    if gap is not None and gap > GEN_GAP_TOL * score:  # the 18.5 tolerance
        return "overfit"
    return "ci"


def account_run(summary: dict, entries: list[dict]) -> dict:
    """SPEC.md 39.2.3: the "what happened" block.

    Pure and deterministic (39.2.4): a function only of `summary.json` +
    `experiments.jsonl`. Returns the three sub-objects (rejections,
    time_to_first_improvement, wall_time) under stable keys.
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    summary = summary if isinstance(summary, dict) else {}
    return {
        "rejections": _rejections(entries),
        "time_to_first_improvement": _first_improvement(entries),
        "wall_time": _wall_time(summary, entries),
    }


__all__ = ["account_run", "candidate_reason"]
