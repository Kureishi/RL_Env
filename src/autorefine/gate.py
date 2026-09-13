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

# SPEC.md 37.2 (v0.23, G4): the original score/train/model set — kept
# byte-identical (default_objectives / the A1–A50 pins).
_BASE_OBJECTIVE_NAMES = ("score", "train", "model")
# SPEC.md 61.1 (v0.47, A1): the constraint-aware names added on top —
# per_class_f1 / ece / cost / monotonic_in are the same `name op
# threshold` form (a lower bound on a goodness, or an upper bound on a
# badness). `default_objectives` still yields only the base score gate.
OBJECTIVE_NAMES = _BASE_OBJECTIVE_NAMES + (
    "per_class_f1", "ece", "cost", "monotonic_in")
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def compute_actuals(env, task, model, entries, extras=None) -> dict:
    """SPEC.md 61.1.2 (A1): the richer `actuals` producer for the
    constraint-aware gate. Starts from `actuals_from_run` (score / train /
    model) and, when `task` and `model` are given, adds the constraint
    actuals by reusing existing primitives (zero retraining):

    * `per_class_f1` — from the confusion matrix in
      `diagnostics.holdout_diagnostics` (`calibration.per_class_f1`, the
      worst non-empty class);
    * `ece` — over the model's holdout softmax probabilities
      (`calibration.ece`, 61.4);
    * `cost` — `task.cost(model, 'holdout', n)` when the task exposes it
      (61.5);
    * `monotonic_in` — `task.monotonic_probe(model)` when the task exposes
      it (61.1.3).

    `extras` (a dict) supplies a precomputed value for any name — a
    supplied finite number wins over recomputation (the app can pass an
    ECE it already computed). Any name with no actual stays `None` → its
    objective fails (the 37.2 rule), never a silent pass. `env` may be
    `None` (a bare task/model evaluation) — the score/train/model base is
    simply omitted then. Pure over already-produced data (G2).
    """
    from .calibration import ece as _ece
    from .calibration import per_class_f1 as _pcf1

    if env is None:
        result: dict = {}
    else:
        result = dict(actuals_from_run(env, entries or []))

    if task is not None and model is not None:
        # per_class_f1 — reuse the §28.2 confusion matrix (61.1.3)
        if getattr(task, "head", None) == "softmax":
            try:
                from .diagnostics import holdout_diagnostics
                diag = holdout_diagnostics(task, model, n=200)
            except Exception:  # a task that cannot be diagnosed contributes no actual
                diag = None
            if diag is not None:
                result["per_class_f1"] = _pcf1(diag["confusion"])
        # ece — over holdout softmax probabilities (61.4)
        if getattr(task, "head", None) == "softmax" \
                and callable(getattr(task, "holdout_rows", None)):
            x, y = task.holdout_rows(200, model)
            if len(x):
                logits = np.asarray(model.forward(x), dtype=np.float64)
                result["ece"] = _ece(_softmax(logits), np.asarray(y).ravel())
        # cost — task-level per-episode / per-prediction badness (61.5)
        if callable(getattr(task, "cost", None)):
            result["cost"] = float(task.cost(model, "holdout", 200))
        # monotonic_in — task-level finite-difference probe (61.1.3)
        if callable(getattr(task, "monotonic_probe", None)):
            result["monotonic_in"] = float(task.monotonic_probe(model))

    if extras:
        for name, val in extras.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                result[name] = float(val)
    return result


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
