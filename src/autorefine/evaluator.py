"""Official scoring on the holdout / gen splits (SPEC.md 5.3, 15).

Task-agnostic: the OFFICIAL score is whatever the task defines on its splits
(`task.score(model, split, n)`). Interactive tasks (CartPoleV1, GridNavV1)
share the generic vectorized episode runner below; fitting tasks
(SineRegressionV1) implement `score` directly.
"""
from __future__ import annotations

import math

import numpy as np


def run_episodes(task, model, split: str, n: int) -> dict:
    """Run n seeded episodes of an interactive task; all stepping is
    vectorized across episodes. Returns per-episode stats:
    {"steps", "success", "mean_steps", "success_rate"}."""
    states = task.initial_conditions(split, n)
    n = len(states)
    alive = np.ones(n, dtype=bool)
    steps = np.zeros(n, dtype=np.int64)
    for _ in range(task.max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        logits = model.forward(task.prepare(states[idx]))
        acts = task.act(logits)
        nxt, term = task.step_vec(states[idx], acts)
        states[idx] = nxt
        steps[idx] += 1
        alive[idx] &= ~term
    success = task.terminal_success(states, steps)
    return {
        "steps": steps,
        "success": success,
        "mean_steps": float(steps.mean()),
        "success_rate": float(success.mean()),
    }


def evaluate_full(task, model, n: int = 200) -> dict:
    """Holdout score + generalization re-run (SPEC.md 5.3 gen_score)."""
    score = float(task.score(model, "holdout", n))
    gen_score = float(task.score(model, "gen", n))
    return {
        "score": score,
        "gen_score": gen_score,
        "gen_gap": score - gen_score,
    }


def score_with_ci(task, model, split: str = "holdout", n_blocks: int = 8,
                  block_size: int = 512) -> dict:
    """Block-bootstrap score confidence (SPEC.md 18.3).

    Scores `n_blocks` independent blocks of `block_size` fresh points each:
    block *i* is `task.score(model, f"{split}-b{i}", block_size)`. Split names
    are opaque seed-derived strings (G2), so this works for every task with
    zero task changes. Returns {"mean", "std", "blocks"} where std is the
    sample standard deviation over block scores (ddof=1; 0.0 for fewer than
    two blocks)."""
    n_blocks = int(n_blocks)
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    blocks = [float(task.score(model, f"{split}-b{i}", int(block_size)))
              for i in range(n_blocks)]
    mean = float(sum(blocks) / len(blocks))
    if len(blocks) >= 2:
        var = sum((b - mean) ** 2 for b in blocks) / (len(blocks) - 1)  # sample std
        std = float(math.sqrt(var))
    else:
        std = 0.0
    return {"mean": mean, "std": std, "blocks": blocks}


def kfold_score(task, model, split: str = "holdout", k: int = 5,
                n: int = 200) -> dict:
    """K-fold holdout scoring (SPEC.md 44.1, v0.30).

    Scores `k` distinct held-out subsets (the task's `score_fold` — a
    fresh seed-derived point set on generative tasks, a deterministic
    random subset of the non-train pool on CsvTask) and averages:
    returns {"score": mean, "std": sample std over the fold scores
    (ddof=1; 0.0 for K < 2), "folds": [...]} — the `score_with_ci`
    shape with `score` in place of `mean`. Scoring-only: the model is
    trained once, then scored K times (44.1.3)."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1 (SPEC.md 44.1)")
    folds = [float(task.score_fold(model, split, i, n)) for i in range(k)]
    score = float(sum(folds) / len(folds))
    if len(folds) >= 2:
        var = sum((f - score) ** 2 for f in folds) / (len(folds) - 1)  # sample std
        std = float(math.sqrt(var))
    else:
        std = 0.0
    return {"score": score, "std": std, "folds": folds}
