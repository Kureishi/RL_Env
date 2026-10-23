"""Distillation: the smallest spec that keeps >= `keep` of the champion
score (SPEC.md 99, v0.85 B3).

Given a finished run's champion (spec + holdout score), enumerate a small
deterministic grid of *simpler* specs (fewer / smaller hidden layers,
capped train steps), train each on the same task + seed, and return the
smallest (by `spec_n_params`) whose holdout score keeps at least `keep`
(0.95) of the champion's. Pure derivation from the trainer — no new
machinery, no RNG beyond the seeded trainer (G2).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_SPEC, ModelSpec, SpecError, spec_n_params
from .trainer import train_from_task

__all__ = ["distill", "distill_from_run"]

# the simplification grid, smallest first (the search order is the
# determinism contract: same (task, champion, seed) => same answer)
_DISTILL_ARCHS: tuple[tuple[int, ...], ...] = ((), (8,), (16,), (32, 16))
_DISTILL_STEP_CAP = 400  # a distilled model rarely needs the champion's budget


def _candidate_specs(champion: ModelSpec) -> list[ModelSpec]:
    """Simpler specs, smallest expected size first; the champion itself
    is the last-resort candidate (distillation always has an answer)."""
    out: list[ModelSpec] = []
    seen: set[tuple] = set()
    for arch in _DISTILL_ARCHS:
        d = dict(champion.to_dict())
        d["architecture"] = list(arch)
        d["train_steps"] = min(int(champion.train_steps), _DISTILL_STEP_CAP)
        try:
            spec = ModelSpec.from_dict(d)
        except SpecError:
            continue  # e.g. a family where architecture is not the surface
        key = (tuple(spec.architecture), spec.train_steps)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    key = (tuple(champion.architecture), champion.train_steps)
    if key not in seen:
        out.append(dataclasses.replace(champion))
    return out


def distill(task, champion: ModelSpec, champion_score: float, seed: int,
            keep: float = 0.95, max_candidates: int = 6) -> dict:
    """Find the smallest spec keeping >= `keep` x champion holdout score.

    Returns {"spec", "score", "n_params", "champion": {...}, "kept",
    "reduction", "tried": [...]} (all JSON-safe)."""
    if not (0.0 < keep <= 1.0):
        raise ValueError(f"keep must be in (0, 1], got {keep!r}")
    threshold = keep * float(champion_score)

    def _n_params(spec: ModelSpec) -> int | None:
        return spec_n_params(spec, task.state_dim, task.n_outputs,
                             getattr(task, "feature_grid", None),
                             getattr(task, "fusion_split", None))

    champ_params = _n_params(champion)
    tried: list[dict] = []
    for spec in _candidate_specs(champion)[: max(int(max_candidates), 1)]:
        res = train_from_task(task, spec, seed=seed)
        score = float(task.score(res.model, "holdout", 200))
        n_params = _n_params(spec)
        tried.append({"spec": spec.to_dict(), "score": score,
                      "n_params": n_params})
        if score >= threshold:
            return {
                "spec": spec.to_dict(),
                "score": score,
                "n_params": n_params,
                "champion": {"spec": champion.to_dict(),
                             "score": float(champion_score),
                             "n_params": champ_params},
                "kept": score / float(champion_score) if champion_score else None,
                "reduction": (champ_params / n_params)
                if champ_params and n_params and n_params > 0 else None,
                "tried": tried,
            }
    raise RuntimeError(
        f"distillation failed: no candidate in the grid kept "
        f">{threshold:.4f} of the champion score {champion_score:.4f} "
        f"(SPEC.md 99)")


def distill_from_run(run_dir: str | Path, keep: float = 0.95) -> dict:
    """Distill a finished run's champion (SPEC.md 99 B3): reads
    summary.json + best_spec.json from `run_dir`, runs `distill` on the
    same task/seed, and writes `distill.json` back into the run dir."""
    from .tasks import TASKS  # local import: the registry is heavy

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    spec_dict = json.loads((run_dir / "best_spec.json").read_text())
    champion = ModelSpec.from_dict(spec_dict)
    task_name = summary["task"]
    if task_name not in TASKS:
        raise ValueError(f"run's task {task_name!r} is not registered")
    seed = int(summary.get("seed", 7))
    score = float(summary["final_best_score"])
    task = TASKS[task_name](seed=seed)
    out = distill(task, champion, score, seed=seed, keep=keep)
    (run_dir / "distill.json").write_text(json.dumps(out, indent=2))
    return out
