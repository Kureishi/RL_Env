"""Objective-set acceptance (SPEC.md 37.2, v0.23 G4).

The §22.1 gate (`final >= target`) generalized to a small first-class
objective set — `score` / `train` / `model`. The default objective set
is exactly today's single score gate; the gate is evaluated after
`done` and never touches the loop (37.2.4).

Core module: stdlib + numpy (already a hard dependency), no new deps.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBJECTIVE_NAMES = ("score", "train", "model")
GATE_OPS = (">=", "<=")

_OPS = {
    ">=": lambda actual, thr: actual >= thr,
    "<=": lambda actual, thr: actual <= thr,
}


@dataclass(frozen=True)
class Objective:
    """One acceptance objective (SPEC.md 37.2.2): `name op threshold`,
    `name ∈ {score, train, model}`, `op ∈ {>=, <=}`."""

    name: str
    op: str
    threshold: float


def parse_objective(text: str) -> Objective:
    """The CLI form (SPEC.md 37.2.2): `score>=95`, `train<=30`,
    `model<=100000`. A bad name/op/threshold is a construction-time
    `ValueError`."""
    if not isinstance(text, str):
        raise ValueError(
            f"an objective must be a string like 'score>=95', "
            f"got {text!r} (SPEC.md 37.2.2)")
    m = re.fullmatch(
        r"\s*(\w+)\s*(>=|<=)\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*", text)
    if not m:
        raise ValueError(
            f"an objective must look like 'score>=95', got {text!r} "
            f"(SPEC.md 37.2.2)")
    name, op = m.group(1), m.group(2)
    if name not in OBJECTIVE_NAMES:
        raise ValueError(
            f"unknown objective name {name!r} — one of "
            f"{OBJECTIVE_NAMES} (SPEC.md 37.2.2)")
    threshold = float(m.group(3))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(
            f"objective threshold must be a finite non-negative number, "
            f"got {threshold!r} (SPEC.md 37.2.2)")
    return Objective(name=name, op=op, threshold=threshold)


def default_objectives(target: float) -> tuple[Objective, ...]:
    """The default objective set (SPEC.md 37.2.2) — exactly today's
    single score gate: `score >= target`."""
    return (Objective(name="score", op=">=", threshold=float(target)),)


def model_size(npz_path) -> int:
    """The `model` actual (SPEC.md 37.2.2): the total values stored in
    `best_model.npz` — uniform across families (metadata scalars
    included)."""
    with np.load(Path(npz_path)) as npz:
        return int(sum(int(npz[k].size) for k in npz.files))


def best_train_seconds(entries: list[dict], best_spec_hash: str) -> float | None:
    """The `train` actual (SPEC.md 37.2.2): the best candidate's train
    seconds — the memory entry whose `spec_hash` matches the best spec's
    fingerprint (the baseline row when no candidate improved)."""
    for e in entries:
        if e.get("spec_hash") == best_spec_hash \
                and isinstance(e.get("train_seconds"), (int, float)):
            return float(e["train_seconds"])
    return None


def evaluate(objectives, actuals: dict) -> dict:
    """Evaluate the objective set (SPEC.md 37.2.2): PASS iff ALL
    objectives pass (a missing actual fails its objective). Returns the
    JSON-safe breakdown `{pass, objectives[…]}`."""
    rows = []
    for o in objectives:
        actual = actuals.get(o.name)
        ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) \
            and _OPS[o.op](float(actual), float(o.threshold))
        rows.append({
            "name": o.name,
            "op": o.op,
            "threshold": float(o.threshold),
            "actual": None if actual is None else float(actual),
            "pass": bool(ok),
        })
    return {
        "pass": bool(rows) and all(r["pass"] for r in rows),
        "objectives": rows,
    }


def actuals_from_run(env, entries: list[dict]) -> dict:
    """The gate actuals for a finished env (SPEC.md 37.2.2): score from
    `env.best_score`, train from the best candidate's memory entry,
    model from the saved `best_model.npz`."""
    best_hash = env.best_spec.fingerprint() if env.best_spec else None
    return {
        "score": float(env.best_score),
        "train": best_train_seconds(entries, best_hash) if best_hash else None,
        "model": (int(model_size(env.run_dir / "best_model.npz"))
                  if env.run_dir is not None else None),
    }
