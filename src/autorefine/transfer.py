"""The search prior — carry a finished run's search knowledge to a similar
task (SPEC.md 86.1, v0.72).

A finished run leaves two artifacts that fully describe what its search
learned: ``summary.json`` (the task, the final best score, the experiment
count, and the canonical ``mutation_win_rate`` per-field credit — the same
computation the summary publishes, SPEC.md 26.1) and ``best_spec.json``
(the best spec). ``read_search_prior`` folds them into a frozen
``SearchPrior``; ``AutoRefineEnv(prior_run=...)`` then starts the new run
from the prior's best spec (retrained on the new task) and the bandit seeds
its field beliefs from the prior's credit (SPEC.md 86.2/86.3).

Only **finished** runs are transferable: an interrupted run has no
``summary.json``, and a missing or malformed artifact is a fail-loud
``ValueError`` (the 80.2.4 style), never a silent empty prior.

Core module: stdlib + numpy only (SPEC.md 3); deterministic (G2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchPrior:
    """One finished run's learned search knowledge (SPEC.md 86.1.1).

    ``trials`` / ``wins`` are per-field credit over the run's scored
    experiments (the summary's ``mutation_win_rate``); a field absent from
    them was never mutated in the source run (cold-start territory for the
    new run's bandit, 86.2.1).
    """
    task: str
    source_run: str
    best_spec: dict
    best_score: float
    trials: dict
    wins: dict
    n_experiments: int

    def to_dict(self) -> dict:
        """JSON-safe dict (86.1.1) — a stable, sorted-key round-trip (G2)."""
        return {
            "task": self.task,
            "source_run": self.source_run,
            "best_spec": dict(self.best_spec),
            "best_score": float(self.best_score),
            "trials": {k: int(self.trials[k]) for k in sorted(self.trials)},
            "wins": {k: float(self.wins[k]) for k in sorted(self.wins)},
            "n_experiments": int(self.n_experiments),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SearchPrior":
        """Reconstruct from `to_dict`'s shape (86.1.1); validate loudly."""
        if not isinstance(d, dict):
            raise ValueError(
                f"search prior must be a JSON object, got {type(d).__name__} "
                f"(SPEC.md 86.1.1)")
        for key in ("task", "source_run", "best_spec", "best_score",
                    "trials", "wins", "n_experiments"):
            if key not in d:
                raise ValueError(f"search prior is missing {key!r} "
                                 f"(SPEC.md 86.1.1)")
        return cls(
            task=str(d["task"]),
            source_run=str(d["source_run"]),
            best_spec=dict(d["best_spec"]),
            best_score=float(d["best_score"]),
            trials={str(k): int(v) for k, v in d["trials"].items()},
            wins={str(k): float(v) for k, v in d["wins"].items()},
            n_experiments=int(d["n_experiments"]),
        )

    def to_bandit_prior(self) -> dict:
        """The JSON-safe shape `BanditPolicy(prior=...)` consumes (86.1.1):
        ``{field: {"trials": int, "wins": float}}`` (86.2)."""
        return {
            field: {"trials": int(self.trials.get(field, 0)),
                    "wins": float(self.wins.get(field, 0.0))}
            for field in sorted(set(self.trials) | set(self.wins))
        }


def read_search_prior(run_dir: str | Path) -> SearchPrior:
    """Fold one finished run's artifacts into a ``SearchPrior`` (86.1.2).

    Reads ``summary.json`` (task, final best score, ``experiments_run``,
    ``mutation_win_rate``) and ``best_spec.json`` (the best spec). A
    missing or malformed artifact is a fail-loud ``ValueError`` — only
    finished, well-formed runs are transferable (86.1.2). Pure and
    deterministic (G2).
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    spec_path = run_dir / "best_spec.json"
    for path in (summary_path, spec_path):
        if not path.is_file():
            raise ValueError(
                f"not a finished run: {path} is missing — the search prior "
                f"transfers only from a finished run (SPEC.md 86.1.2)")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unreadable summary.json in {run_dir}: {exc} "
            f"(SPEC.md 86.1.2)") from exc
    try:
        best_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unreadable best_spec.json in {run_dir}: {exc} "
            f"(SPEC.md 86.1.2)") from exc
    if not isinstance(summary, dict):
        raise ValueError(
            f"summary.json in {run_dir} must be a JSON object "
            f"(SPEC.md 86.1.2)")
    for key in ("task", "final_best_score", "experiments_run"):
        if key not in summary:
            raise ValueError(
                f"summary.json in {run_dir} is missing {key!r} — not a "
                f"finished AutoRefine run (SPEC.md 86.1.2)")
    if not isinstance(best_spec, dict):
        raise ValueError(
            f"best_spec.json in {run_dir} must be a JSON object "
            f"(SPEC.md 86.1.2)")
    win_rate = summary.get("mutation_win_rate") or {}
    if not isinstance(win_rate, dict):
        raise ValueError(
            f"mutation_win_rate in {run_dir} must be a JSON object "
            f"(SPEC.md 86.1.2)")
    trials: dict[str, int] = {}
    wins: dict[str, float] = {}
    for field in sorted(win_rate):  # G2: fixed iteration order
        slot = win_rate[field]
        if not isinstance(slot, dict):
            raise ValueError(
                f"mutation_win_rate[{field!r}] in {run_dir} must be an "
                f"object (SPEC.md 86.1.2)")
        trials[str(field)] = int(slot.get("trials", 0))
        wins[str(field)] = float(slot.get("wins", 0.0))
    return SearchPrior(
        task=str(summary["task"]),
        source_run=run_dir.name,
        best_spec=dict(best_spec),
        best_score=float(summary["final_best_score"]),
        trials=trials,
        wins=wins,
        n_experiments=int(summary["experiments_run"]),
    )


def prior_spec(prior: SearchPrior):
    """The prior's best spec as a `ModelSpec` (86.1.3) — a malformed spec
    is a clean construction-time error, never a mid-run crash."""
    # local import: keeps transfer.py's import graph light (the advanced.py
    # local-loader pattern), and the package-init cycle-free
    from .config import ModelSpec
    return ModelSpec.from_dict(prior.best_spec)
