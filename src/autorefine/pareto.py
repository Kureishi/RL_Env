"""Pareto memory: the score-vs-training-time frontier (SPEC.md 15).

Tracks every (score, train_seconds) point so the loop can reward *efficiency*,
not just score: a point is Pareto-optimal when no other point has both a
score >= and a train time <= (with at least one strict).
"""
from __future__ import annotations

from typing import Any


class ParetoFrontier:
    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []

    def add(self, score: float, seconds: float, spec_hash: str,
            cost: float | None = None) -> None:
        # SPEC.md 61.5.1 (A5, v0.47): an *optional* `cost` (per-episode /
        # per-prediction badness) rides along when supplied; the default
        # `cost=None` leaves the (score, seconds) contract byte-identical
        # (the A1–A50 pins never pass a cost, so this is additive).
        point = {"score": float(score), "seconds": float(seconds),
                 "spec_hash": spec_hash}
        if cost is not None:
            point["cost"] = float(cost)
        self.points.append(point)

    def cost_of(self, spec_hash: str) -> float | None:
        """SPEC.md 61.5.1 (A5): the `cost` recorded for a `spec_hash`, or
        ``None`` when the point is unknown or was added without a cost —
        the frontier's default ``(score, train)`` contract is untouched
        (the cost objective can be satisfied entirely in the gate, 61.5.2)."""
        for p in self.points:
            if p.get("spec_hash") == spec_hash:
                return None if "cost" not in p else float(p["cost"])
        return None

    def frontier(self) -> list[dict[str, Any]]:
        """Undominated points, sorted by training time (cheapest first)."""
        pts = self.points
        out = []
        for i, p in enumerate(pts):
            dominated = any(
                q["score"] >= p["score"]
                and q["seconds"] <= p["seconds"]
                and (q["score"] > p["score"] or q["seconds"] < p["seconds"])
                for j, q in enumerate(pts)
                if j != i
            )
            if not dominated:
                out.append(dict(p))
        return sorted(out, key=lambda p: (p["seconds"], -p["score"]))

    def best_score_at(self, seconds: float) -> float | None:
        """Best score achievable within a training-time budget."""
        cands = [p["score"] for p in self.points if p["seconds"] <= seconds]
        return max(cands) if cands else None

    def summary(self, time_budget: float = 1.0) -> dict[str, Any]:
        """JSON-serializable snapshot for summary.json / state."""
        frontier = self.frontier()
        best_score = max((p["score"] for p in self.points), default=None)
        within = self.best_score_at(time_budget)
        return {
            "pareto_frontier": [
                {"score": p["score"], "train_seconds": p["seconds"], "spec_hash": p["spec_hash"]}
                | ({"cost": p["cost"]} if "cost" in p else {})
                for p in frontier
            ],
            "best_score_at_1s": within,
            # efficiency: best score within 1s relative to the global best
            "efficiency_at_1s": (
                within / best_score if (within is not None and best_score) else None
            ),
        }
