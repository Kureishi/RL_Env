"""SPEC.md 49.2/49.3 (v0.35, advanced analysis): the app's advanced
panels over existing core — no new loop machinery.

- ``retrain_spec`` (49.2) — one controlled training pass of an edited
  spec against a finished run's reconstructed task: the run's own seed
  (49.2.2), the shared ``_task_from_summary`` reconstruction (22.1/28.4),
  ``train_from_task`` + ``evaluate_full``. The run dir is read-only.
- ``opposite_policy`` (49.3.1) — the policy A/B mapping: bandit ↔
  search; anything else (rl, unknown) is a ``ValueError`` (rl stays
  CLI-only by design, 23.1).

House rules: stdlib + numpy (a core dependency) only; no import cycles
(the heavy imports are function-local, mirroring the cli's local-loader
style); the module imports nothing from the app (23.1).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ModelSpec, SpecError

__all__ = ["retrain_spec", "opposite_policy"]


def opposite_policy(name) -> str:
    """SPEC.md 49.3.1 (v0.35): the A/B mapping — bandit ↔ search.

    Anything else (``rl``, an unknown name, a non-string) is a
    ``ValueError``: ``rl`` stays CLI-only by design (23.1), so the A/B
    re-run is the two policies the app can actually run (23.1).
    """
    if name == "bandit":
        return "search"
    if name == "search":
        return "bandit"
    raise ValueError(
        f"policy A/B covers 'bandit' and 'search' only, got {name!r} "
        f"('rl' stays CLI-only, SPEC.md 23.1/49.3.1)")


def retrain_spec(run_dir, spec, max_train_seconds: float | None = None) -> dict:
    """SPEC.md 49.2.1/49.2.2 (v0.35): re-score one spec against a run.

    Reconstructs the run's task from ``summary.json`` (the same
    ``_task_from_summary`` reconstruction ``eval``/``report`` use —
    22.1/28.4, including the curriculum-level fallback, 20.1), trains
    the given spec — a dict **or** a ``ModelSpec`` (a bad dict is a
    ``SpecError``, a loud typed error) — with the run's own seed (a
    deterministic re-score: same spec ⇒ same score, 49.2.2), evaluates
    with ``evaluate_full``, and returns the JSON-safe dict
    ``{"spec", "score", "gen_score", "gen_gap", "train_seconds",
    "final_loss", "time_capped"}``.

    The run directory is **read-only** — this writes nothing (49.2.1).
    An unknown task name (the reconstruction returns ``None``) is a
    ``ValueError`` naming the run dir.
    """
    # local imports: keep the module surface small and cycle-free
    # (the cli imports many core modules; this one must not be imported
    # back by it)
    from .cli import _task_from_summary
    from .evaluator import evaluate_full
    from .trainer import train_from_task

    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"no summary.json in {run_dir} — not a finished run "
                         f"(SPEC.md 49.2.1)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task = _task_from_summary(summary)
    if task is None:
        raise ValueError(
            f"cannot reconstruct the task of {run_dir} (unknown task name "
            f"or no curriculum level; SPEC.md 49.2.2)")
    if isinstance(spec, dict):
        try:
            spec = ModelSpec.from_dict(spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError(f"invalid spec: {exc} (SPEC.md 49.2.1)") from exc
    elif not isinstance(spec, ModelSpec):
        raise SpecError(
            f"spec must be a dict or a ModelSpec, got {type(spec).__name__} "
            f"(SPEC.md 49.2.1)")
    seed = int(summary["seed"])
    result = train_from_task(
        task, spec, seed,
        time_limit_seconds=None if max_train_seconds is None
        else float(max_train_seconds))
    ev = evaluate_full(task, result.model)
    return {
        "spec": spec.to_dict(),
        "score": float(ev["score"]),
        "gen_score": float(ev["gen_score"]),
        "gen_gap": float(ev["gen_gap"]),
        "train_seconds": float(result.train_seconds),
        "final_loss": float(result.final_loss),
        "time_capped": bool(result.time_capped),
    }
