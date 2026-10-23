"""Bench suite: the small, deterministic multi-task leaderboard
(SPEC.md 99, v0.85 D3).

`bench_suite` runs the standard loop (the v0.4 search-quality preset, the
`SearchPolicy` improver, one fixed budget) over each task of a suite and
records baseline / best / family / wall time per task — the "which
modalities does the loop win, and by how much" view, with zero new
machinery: `AutoRefineEnv` + `SearchPolicy`, one call per task, and one
ASCII renderer.

The `multimodal` suite (the default) spans six built-in modalities:
flat Boolean (parity-v1), flat regression (sine-v1), spatial grid
(gridnav-v1), sequential control (cartpole-v1), 1-D sequence
(sequence-motif-v1), and image+text fusion (fusion-v1).

Determinism (G2): the per-task seed is a fixed derivation of the base
seed (the 89 pattern), the budget / preset / policy are fixed, and the
leaderboard rows are ordered by suite order — so the same
(tasks, seed, budget) reproduces the same leaderboard bit-for-bit.
Opt-in (99.2): this is a NEW command + module; the pinned legacy
streams are untouched.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

# the default suite — one task per modality (SPEC.md 99, A89.16)
SUITE_MULTIMODAL: tuple[str, ...] = (
    "parity-v1",           # flat Boolean (parity)
    "sine-v1",             # flat regression
    "gridnav-v1",          # spatial (grid)
    "cartpole-v1",         # sequential control
    "sequence-motif-v1",   # 1-D sequence (temporal)
    "fusion-v1",           # image + text (fusion)
)
SUITES: dict[str, tuple[str, ...]] = {"multimodal": SUITE_MULTIMODAL}


def bench_suite(tasks: tuple[str, ...] | None = None, seed: int = 7,
                budget: int = 6, runs_dir: str | Path = "runs",
                wall_seconds: float = 900.0, train_seconds: float = 30.0,
                verbose: bool = False) -> dict[str, Any]:
    """Run the standard loop over `tasks` (default: the `multimodal`
    suite) and return the leaderboard dict.

    Each task gets the SAME budget, preset, policy, and a stable derived
    seed (``seed * 1_000_003 + i``), so the comparison is apples-to-apples
    and the whole suite is reproducible (G2). Returns
    ``{"suite", "tasks", "seed", "budget", "rows", "ranked", "best",
    "total_wall_seconds"}`` (all JSON-safe)."""
    from .config import Budget
    from .improver.meta_env import AutoRefineEnv, search_quality_v04
    from .improver.policy import SearchPolicy
    from .tasks import TASKS

    task_list = tuple(tasks) if tasks else SUITE_MULTIMODAL
    if not task_list:
        raise ValueError("bench: tasks must be non-empty")
    for t in task_list:
        if t not in TASKS:
            raise ValueError(f"bench: task {t!r} is not registered")
    rows: list[dict[str, Any]] = []
    total_wall = 0.0
    for i, tname in enumerate(task_list):
        t_seed = int(seed) * 1_000_003 + i  # G2: stable, distinct
        env = AutoRefineEnv(
            task=tname, seed=t_seed,
            budget=Budget(int(budget), float(wall_seconds),
                          float(train_seconds)),
            runs_dir=Path(runs_dir) / f"bench-{tname}",
            policy="search", target=None,  # un-gated (like `run`)
            **search_quality_v04())
        state = env.reset()
        baseline = float(state["baseline_score"])
        policy = SearchPolicy(seed=t_seed)
        t0 = time.perf_counter()
        while not env.done:
            state, _r, _d, _i = env.step(policy.propose(state))
        wall = time.perf_counter() - t0
        total_wall += wall
        spec = env.best_spec
        summary = env.memory.load_summary() if env.memory else {}
        rows.append({
            "task": tname,
            "seed": t_seed,
            "budget": int(budget),
            "baseline": baseline,
            "best": float(env.best_score),
            "family": str(spec.model_family),
            "architecture": list(spec.architecture),
            "experiments_run": int(
                summary.get("experiments_run", env.bm.used_experiments)),
            "finished_reason": summary.get("finished_reason"),
            "wall_seconds": round(wall, 3),
            "run_dir": str(env.run_dir),
        })
        if verbose:
            print(f"bench {tname:20s} baseline {baseline:7.2f}  "
                  f"best {rows[-1]['best']:7.2f}  ({spec.model_family}, "
                  f"{wall:.1f}s)")
    ranked = sorted(rows, key=lambda r: (-r["best"], r["task"]))
    return {
        "suite": "multimodal" if tasks is None else "custom",
        "tasks": list(task_list),
        "seed": int(seed),
        "budget": int(budget),
        "rows": rows,
        "ranked": [r["task"] for r in ranked],
        "best": ranked[0] if ranked else None,
        "total_wall_seconds": round(total_wall, 3),
    }


def render_bench(out: dict[str, Any]) -> str:
    """The leaderboard as deterministic ASCII (the house console rule)."""
    lines = [
        f"bench suite : {out['suite']} ({len(out['tasks'])} tasks, "
        f"seed {out['seed']}, budget {out['budget']}/task)",
        "",
        f"{'rank':<5} {'task':<20} {'baseline':>9} {'best':>9} {'delta':>8}"
        f" {'family':<10} {'exp':>4} {'wall_s':>8}",
    ]
    for k, r in enumerate(out["rows"], 1):
        lines.append(
            f"{k:<5} {r['task']:<20} {r['baseline']:9.2f} {r['best']:9.2f}"
            f" {r['best'] - r['baseline']:+8.2f} {r['family']:<10}"
            f" {r['experiments_run']:>4} {r['wall_seconds']:8.2f}")
    best = out.get("best")
    if best is not None:
        lines.append("")
        lines.append(f"top     : {best['task']} ({best['family']}, arch "
                     f"{best['architecture']}, best {best['best']:.2f})")
    lines.append(f"total   : {out['total_wall_seconds']:.1f}s wall "
                 f"across {len(out['tasks'])} tasks")
    return "\n".join(lines)


__all__ = ["SUITE_MULTIMODAL", "SUITES", "bench_suite", "render_bench"]
