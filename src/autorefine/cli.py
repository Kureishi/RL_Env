"""Command-line interface (SPEC.md 11).

  python -m autorefine run    --task cartpole-v1 --seed 7 --experiments 30
  python -m autorefine report --run runs/<run_id>
  python -m autorefine eval   --run runs/<run_id>
  python -m autorefine doctor [--runs-dir runs]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from .config import Budget, ModelSpec
from .diagnostics import holdout_diagnostics
from .gate import (
    actuals_from_run,
    default_objectives,
    evaluate,
    parse_objective,
)
from .runconfig import (
    RUN_CONFIG_SCHEMA,
    RunConfig,
    format_recipe,
    runconfig_to_flags,
)
from .dashboard import diff_two_summaries, field_stats  # 42.2/42.3 (v0.28)
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.bandit import BanditPolicy
from .improver.curriculum import (  # 46.2 (v0.32): the three ladders
    CartPoleCurriculum,
    ParityCurriculum,
    SineCurriculum,
)
from .improver.policy import SearchPolicy
from .improver.rl_policy import MetaRLPolicy, train_policy
from .accounting import account_run  # 39.2 (T4): decision accounting
from .memory import KIND_BASELINE, KIND_EXPERIMENT, RunMemory
from .simulate import (  # 40 (v0.26) + 41 (v0.27): simulation
    estimate_wall,
    project_budget,
    projection_points,
    trace_lines,
    what_if,
)
from .registry import load_registry  # 40.1 (v0.26): wall-time estimate
from .watch import (  # 39.1 (T3): watch mode / live progress
    live_frame,
    read_new_entries,
    run_finished,
    tail_line,
)
from .plotting import (
    ascii_pareto,
    ascii_score_curve,
    html_report,
    svg_architecture,
    svg_confusion_matrix,
    svg_ladder_curve,
    svg_pareto,
    svg_per_class_bars,
    svg_score_curve,
)
from .predict import (  # 42.1 (v0.28): the predict leaf
    csv_rows_to_features,
    media_item_features,
    predict_features,
    row_to_features,
    standardize,
)
from .preflight import (  # 43.1 (v0.29): the data-health preflight leaf
    _SMOKE_WALL_SECONDS,
    data_health,
    format_health,
)
from .plugins import (
    POLICIES_GROUP,
    TASKS_GROUP,
    PluginError,
    discover,
    register_tasks,
)
from .tasks import (  # 46.2 (v0.32): the curriculum-level task fallbacks
    TASKS,
    CartPoleV1,
    ParityTask,
    SineRegressionV1,
)
from .models.mlp import MLP
from .evaluator import evaluate_full


def _drive(args: argparse.Namespace, env: AutoRefineEnv) -> None:
    """Shared improver driver for `run` and `fit` (SPEC.md 22.1).

    SPEC.md 47.3 (v0.33, `--quiet`): the loop's stdout (run dir / baseline /
    per-experiment / RL episode lines) is suppressed when `args.quiet`;
    the `=== summary ===` block is always kept (47.3.2)."""
    quiet = getattr(args, "quiet", False)  # 47.3 (run/fit only; safe elsewhere)
    if args.policy == "rl":
        # SPEC.md 15: meta-RL improver — policy trained on AutoRefineEnv itself
        policy = MetaRLPolicy(seed=args.seed)
        if not quiet:
            print(f"task    : {args.task}  (meta-RL, {args.rl_episodes} full-budget episodes)")
        rl_summary = train_policy(env, policy, args.rl_episodes,
                                  verbose=not quiet)  # 47.3.2
        if not quiet:
            print(f"rl episodes: {rl_summary['episodes']}, returns: "
                  f"{[round(r, 4) for r in rl_summary['episode_returns']]}")
        summary = env.memory.load_summary() if env.memory else {}
    else:
        state = env.reset()
        if not quiet:  # 47.3.2: the loop's stdout is the quiet surface
            print(f"run dir : {env.run_dir}")
            print(f"baseline: {state['baseline_score']:.2f}")

        # SPEC.md 17: the UCB field-bandit is a second reference policy
        policy = (SearchPolicy(seed=args.seed) if args.policy == "search"
                  else BanditPolicy(seed=args.seed))
        while not env.done:
            action = policy.propose(state)
            state, reward, done, info = env.step(action)
            if not quiet:  # 47.3.2: per-experiment line suppressed
                mark = "+" if info.get("accepted") else " "
                score = info.get("candidate_score")
                score_s = f"{score:8.2f}" if isinstance(score, (int, float)) else "  duplicate"
                print(f" {mark} exp {env.bm.used_experiments:3d}  score {score_s}  reward {reward:+.4f}")

        summary = env.memory.load_summary() if env.memory else {}
    print("\n=== summary ===")
    return _print_summary(args, env, summary)


def _cmd_run(args: argparse.Namespace) -> int:
    # SPEC.md 41.3 (v0.27, S5): the narrated demo — parity-v1, a tiny fixed
    # budget; the other `run` flags are ignored (41.3.1)
    if args.demo:
        return _cmd_demo(args)
    # SPEC.md 18.6: the v0.4 search-quality preset is the CLI default;
    # `--search-quality legacy` opts out (v0.3 behavior, exactly)
    quality = {} if args.search_quality == "legacy" else search_quality_v04()
    # SPEC.md 19.3: opt-in ensemble final evaluation over the top-2 frontier
    # models (off by default; v0.4 runs unchanged)
    # SPEC.md 20.1: opt-in curriculum (adaptive difficulty; parity for now)
    curriculum = None
    if args.curriculum:
        # SPEC.md 46.2 (v0.32): the ladder per task family (20.1 for parity,
        # 46.2.3 sine, 46.2.4 cartpole)
        if args.task == "parity-v1":
            curriculum = ParityCurriculum(seed=args.seed)
        elif args.task == "sine-v1":
            curriculum = SineCurriculum(seed=args.seed)
        elif args.task == "cartpole-v1":
            curriculum = CartPoleCurriculum(seed=args.seed)
        else:
            print("--curriculum supports parity-v1, sine-v1, and cartpole-v1 "
                  "(SPEC.md 46.2)", file=sys.stderr)
            return 1
    env = AutoRefineEnv(
        task=args.task, seed=args.seed, budget=_budget(args), runs_dir=args.runs_dir,
        ensemble_top_k=2 if args.ensemble_final else 0,
        curriculum=curriculum,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; None = off)
        screen_frac=args.screen_frac,  # v0.18 (SPEC.md 32.2; 1.0 = off)
        kfold=args.kfold,  # v0.30 (SPEC.md 44.1; 0 = off, legacy)
        # v0.23 (SPEC.md 37.1.3): driver metadata for the canonical recipe
        policy=args.policy,
        target=None,  # `run` is ungated (SPEC.md 37.2.3)
        rl_episodes=args.rl_episodes if args.policy == "rl" else None,
        **quality,
    )
    _drive(args, env)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """SPEC.md 41.3 (v0.27, S5): the demo loop — parity-v1 with the tiny
    fixed budget (3 experiments / 60 s wall / 10 s per train), the v0.4
    preset, the search policy, un-gated. Runs quietly, then prints the full
    narration with the 41.2 trace renderer (one renderer, two entry
    points). A demo run is a run: normal run dir, artifacts, registry
    entry; deterministic for a given seed (G2). rc 0 (41.3.3)."""
    env = AutoRefineEnv(
        task="parity-v1", seed=args.seed,
        budget=Budget(3, 60.0, 10.0),  # 41.3.1: the tiny fixed budget
        runs_dir=args.runs_dir,
        policy="search", target=None,  # un-gated (like `run`)
        **search_quality_v04())
    policy = SearchPolicy(seed=args.seed)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    summary = env.memory.load_summary() if env.memory else {}
    entries = env.memory.load_experiments()
    print("AutoRefine demo — one tiny, deterministic loop (SPEC.md 41.3)")
    print(f"task parity-v1   seed {args.seed}   budget 3 experiments / "
          f"60 s wall / 10 s per train   v0.4 preset   un-gated")
    print(f"run dir : {env.run_dir}")
    print()
    for line in trace_lines(entries):
        print(f"  {line}")
    print(f"\nbest spec: {json.dumps(env.best_spec.to_dict())}")
    print(json.dumps({k: summary.get(k) for k in (
        "baseline_score", "final_best_score", "improvement_factor",
        "experiments_run", "wall_seconds", "finished_reason")}, indent=2))
    print(f"artifacts: {env.run_dir}")
    return 0


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(
        max_experiments=args.experiments,
        max_wall_seconds=args.max_seconds,
        max_train_seconds=args.max_train_seconds,
    )


def _print_summary(args: argparse.Namespace, env: AutoRefineEnv,
                   summary: dict) -> None:
    print(json.dumps({k: summary.get(k) for k in (
        "baseline_score", "final_best_score", "improvement_factor",
        "experiments_run", "wall_seconds", "finished_reason",
    )}, indent=2))
    if summary.get("ensemble") and getattr(args, "ensemble_final", False):
        # SPEC.md 19.3 (only with --ensemble-final)
        ens = summary["ensemble"]
        print(f"ensemble  : top-{ens['top_k']} "
              f"members {ens['member_scores']} -> score {ens['ensemble_score']:.2f}")
    if summary.get("curriculum"):
        # SPEC.md 20.1 (only with --curriculum)
        cur = summary["curriculum"]
        print(f"curriculum: {len(cur['levels'])} step-up(s), final level "
              f"{cur['final_difficulty']} (ceiling {cur['final_ceiling']:.2f})")
        for row in cur["levels"]:
            print(f"  -> {row['difficulty']} (ceiling {row['ceiling']:.2f}): "
                  f"best {row['best_before_step']:.2f} -> baseline "
                  f"{row['new_baseline_score']:.2f}")
    print(f"best spec: {json.dumps(env.best_spec.to_dict())}")
    print(f"artifacts: {env.run_dir}")


def _resolve_fit_task(data: Path, want: str) -> str:
    """v0.10 (SPEC.md 24.5; v0.31 45.2.2): `--data` file → csv; directory
    → image/audio/text.

    `want` is the `--task` value ('auto' | 'csv' | 'image' | 'audio' |
    'text'); auto-detection counts modality extensions in the directory's
    subfolders (SPEC.md 45.2.1: one modality → its name, two or more →
    'mixed', none → None).
    """
    from .tasks import detect_modality
    if data.is_file():
        if want in ("auto", "csv"):
            return "csv"
        raise ValueError(f"--data {data} is a file: use --task csv (or auto)")
    if data.is_dir():
        if want == "auto":
            m = detect_modality(data)
            if m == "mixed":
                raise ValueError(
                    f"{data} holds items of more than one modality: pass "
                    f"--task image, --task audio, or --task text "
                    f"(SPEC.md 24.5)")
            if m is None:
                raise ValueError(
                    f"no image, audio, or text items under {data} — one "
                    f"subfolder per class, or an index.csv (SPEC.md 24.2); "
                    f"or pass a CSV file with --task csv")
            return m
        if want in ("image", "audio", "text"):
            return want
        raise ValueError(f"--task csv needs a CSV file, got directory {data}")
    raise ValueError(f"no such file or directory: {data}")


def _cmd_fit(args: argparse.Namespace) -> int:
    """`fit` (SPEC.md 22.1, v0.10 24.5, v0.23 37.1.4/37.2): data in, gated
    validated model out.

    = the `run` loop over the matching data task (task_config carries the
    path: csv file / image dir / audio dir), plus the acceptance gate. No
    `--gate` → the §22.1 score bar (PASS → rc 0, MISS → rc 2, byte-identical
    to pre-v0.23); `--gate "name op threshold"` (repeatable) → the objective
    set (37.2). `--from-run RUN_DIR` re-runs a finished run exactly (37.1.4)
    — mutually exclusive with `--data`. `--tasks A.csv,B.csv` (SPEC.md 46.3,
    v0.32) runs one shared budget over several CSV datasets — mutually
    exclusive with both single-file modes.
    """
    from .tasks import TASKS  # noqa: F401 (registry; the helpers use it)
    # SPEC.md 46.3 (v0.32): portfolio mode — first, because it is mutually
    # exclusive with the --data / --from-run checks below
    if getattr(args, "tasks", None) is not None:
        if args.data is not None or args.from_run is not None:
            print("--tasks is mutually exclusive with --data and --from-run "
                  "(SPEC.md 46.3)", file=sys.stderr)
            return 1
        if getattr(args, "dry_run", False):
            return _fit_dry_run_portfolio(args)
        return _fit_portfolio(args)
    if args.data is not None and args.from_run is not None:
        print("--data and --from-run are mutually exclusive (SPEC.md 37.1.4)",
              file=sys.stderr)
        return 1
    if getattr(args, "dry_run", False):
        return _fit_dry_run(args)  # 40.1 (v0.26): plan only, no training
    if args.from_run is not None:
        env, bar = _fit_from_run(args)
        if env is None:
            return 1
        metric = getattr(env.task, "metric", None) or "score"
    else:
        if args.data is None:
            print("one of --data or --from-run is required (SPEC.md 37.1.4)",
                  file=sys.stderr)
            return 1
        env, bar, metric = _fit_data(args)
    if env is None:
        return 1
    _drive(args, env)
    rc = _fit_gate(args, env, bar, metric)  # kept under --quiet (47.3.2)
    # C2/C3 (SPEC.md 28.2/28.3): after the gate line, per-class accuracy +
    # weakest-class hint (and the media error gallery)
    if not getattr(args, "quiet", False):  # 47.3.2: the diagnostics block
        _print_fit_diagnostics(env)
    return rc


def _portfolio_paths(args: argparse.Namespace) -> list[str] | int:
    """SPEC.md 46.3 (v0.32): parse + validate `--tasks` — the CSV file paths
    in order, or 1 (the CLI error already printed by the caller)."""
    paths = [p.strip() for p in str(args.tasks).split(",")]
    paths = [p for p in paths if p]
    if len(paths) < 2:
        print("--tasks needs at least 2 CSV files (SPEC.md 46.3)",
              file=sys.stderr)
        return 1
    return paths


def _fit_dry_run_portfolio(args: argparse.Namespace) -> int:
    """`fit --tasks A.csv,B.csv --dry-run` (SPEC.md 46.3, v0.32 S1×portfolio):
    the per-task dry-run plans — the same plan `fit --dry-run` prints for a
    single file, once per CSV in the given order, each with its sub-budget
    (experiments = ceil(total/N); the wall-time deadline is shared, so only
    the experiments count is split in the plan). rc 0 when every plan
    prints; rc 1 on a resolve/probe failure (the *same* errors `fit` would
    hit — e.g. a wrong label column — surfaced before any training)."""
    paths = _portfolio_paths(args)
    if isinstance(paths, int):
        return paths
    for p in paths:
        if not Path(p).is_file():
            print(f"portfolio: no such file: {p} (SPEC.md 46.3)",
                  file=sys.stderr)
            return 1
    sub_experiments = max(1, math.ceil(args.experiments / len(paths)))
    saved = (args.data, args.experiments)
    try:
        for k, p in enumerate(paths, start=1):
            print(f"\n=== portfolio task {k}/{len(paths)}: {p} ===")
            args.data = p  # the shared planner reads `args.data`
            args.experiments = sub_experiments
            rc = _fit_dry_run(args)
            if rc != 0:
                return rc
    finally:
        args.data, args.experiments = saved
    return 0


def _fit_portfolio(args: argparse.Namespace) -> int:
    """`fit --tasks A.csv,B.csv` (SPEC.md 46.3, v0.32): portfolio mode —
    one shared budget over several CSV datasets, run sequentially in the
    given order (v1: CSV files; cross-task policy transfer is a documented
    follow-up).

    Sub-budgets: each task gets experiments = ceil(total/N); the wall-time
    deadline is shared — the elapsed time is subtracted and clamped to a
    small positive floor (a task starting with no wall time left runs its
    baseline only; `Budget` requires max_wall_seconds > 0). Per-task gate
    lines as usual; rc 0 iff every task PASSes, else 2; a resolve failure
    is rc 1 (never a partial-portfolio score)."""
    from .tasks import CsvTask
    paths = _portfolio_paths(args)
    if isinstance(paths, int):
        return paths
    t0 = time.perf_counter()
    rcs: list[int] = []
    for k, p in enumerate(paths, start=1):
        f = Path(p)
        if not f.is_file():
            print(f"portfolio: no such file: {p} (SPEC.md 46.3)",
                  file=sys.stderr)
            return 1
        config: dict = {"path": str(f)}
        if args.label is not None:
            config["label"] = args.label
        if args.split_frac != 0.2:
            config["split_frac"] = args.split_frac
        if getattr(args, "temporal", False):  # v0.31 (SPEC.md 45.1.2)
            config["split_mode"] = "temporal"
        if getattr(args, "metric", "accuracy") != "accuracy":  # 46.1
            config["metric"] = args.metric
        try:
            probe = CsvTask(seed=args.seed, **config)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - t0
        sub = Budget(
            max(1, math.ceil(args.experiments / len(paths))),
            max(0.001, args.max_seconds - elapsed),
            args.max_train_seconds,
        )
        quality = {} if args.search_quality == "legacy" else search_quality_v04()
        args.task = "csv"  # _drive's print lines
        print(f"\n=== portfolio task {k}/{len(paths)}: {f} ===")
        env = AutoRefineEnv(
            task="csv", seed=args.seed, budget=sub, runs_dir=args.runs_dir,
            ensemble_top_k=2 if args.ensemble_final else 0,
            dataset_episodes=int(probe.default_dataset_size),
            task_config=config,
            stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1)
            screen_frac=args.screen_frac,  # v0.18 (SPEC.md 32.2)
            kfold=args.kfold,  # v0.30 (SPEC.md 44.1)
            policy=args.policy, target=args.target,
            rl_episodes=args.rl_episodes if args.policy == "rl" else None,
            **quality,
        )
        _drive(args, env)
        metric = getattr(probe, "metric", None) or "score"
        rcs.append(_fit_gate(args, env, args.target, metric))
        # 47.3.2: the diagnostics block is the quiet surface (kept header)
        if not getattr(args, "quiet", False):
            _print_fit_diagnostics(env)  # the fit output shape (28.2)
    return 0 if all(rc == 0 for rc in rcs) else 2


def _fit_dry_run(args: argparse.Namespace) -> int:
    """`fit --dry-run` (SPEC.md 40.1, v0.26 S1): resolve data -> task ->
    head/metric -> split and print the plan, exiting **before** the env is
    constructed — no run dir, no artifacts, no training.

    rc 0 when the plan prints; rc 1 on a resolve/probe failure (the *same*
    errors `fit` would hit — e.g. a wrong label column — surfaced before any
    training) or when combined with `--from-run` (40.1.3).
    """
    from .tasks import TASKS
    if args.from_run is not None:
        print("--dry-run and --from-run are mutually exclusive (SPEC.md 40.1.3)",
              file=sys.stderr)
        return 1
    if args.data is None:
        print("--dry-run requires --data (SPEC.md 40.1.1)", file=sys.stderr)
        return 1
    data = Path(args.data)
    try:
        task_name = _resolve_fit_task(
            data, getattr(args, "task", "auto") or "auto")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    config: dict = {"path": str(data)}
    if args.label is not None:
        config["label"] = args.label
    if args.split_frac != 0.2:
        config["split_frac"] = args.split_frac
    if getattr(args, "temporal", False):  # v0.31 (SPEC.md 45.1.2)
        config["split_mode"] = "temporal"
    if getattr(args, "metric", "accuracy") != "accuracy":  # 46.1 (v0.32)
        config["metric"] = args.metric
    # the *same* probe constructor `fit` uses — a bad label column raises here
    try:
        probe = TASKS[task_name](seed=args.seed, **config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    metric = getattr(probe, "metric", None) or "score"
    budget = _budget(args)
    dsz = getattr(probe, "default_dataset_size", None)
    print("dry-run: plan only — no training, no run dir, no artifacts "
          "(SPEC.md 40.1)")
    print(f"  data       : {data}")
    if args.label is not None:
        print(f"  label      : {args.label}")
    print(f"  split      : non-train share {args.split_frac:g} "
          f"(evenly into holdout + gen)")
    if getattr(args, "temporal", False):
        # v0.31 (SPEC.md 45.1): walk-forward splits — file order, csv tasks
        print("  split mode : temporal (walk-forward, file order; "
              "SPEC.md 45.1)")
    print(f"  recipe     : task {task_name} | seed {args.seed} | policy "
          f"{args.policy} | target {args.target:g}")
    print(f"  ensemble   : top-{2 if args.ensemble_final else 0}   "
          f"stall patience {args.stall_patience}   "
          f"screen frac {args.screen_frac:g}")
    if args.kfold > 0:  # v0.30 (SPEC.md 44.1): shown only when enabled
        print(f"  kfold      : K={args.kfold} distinct held-out subsets "
              f"per score (SPEC.md 44.1)")
    print(f"  task       : head {probe.head} | metric {metric} "
          f"| n_outputs {probe.n_outputs} | state_dim {probe.state_dim}")
    if dsz is not None:
        print(f"  dataset    : {dsz} points (the train split)")
    else:
        print("  dataset    : task default size")
    # SPEC.md 43.1 (v0.29): the data-health preflight — pure stats over the
    # probe already built above; warnings in the plan, never failures (43.1.3)
    # SPEC.md 46.1 (v0.32): the headroom note is an accuracy statement, so it
    # is suppressed when the declared metric is logloss (the class-balance
    # block still prints)
    _health_target = None if getattr(probe, "metric", None) == "logloss" \
        else args.target
    for line in format_health(data_health(probe, _health_target)):
        print(line)
    print(f"  budget     : experiments {budget.max_experiments} | "
          f"max-seconds {budget.max_wall_seconds:g} | "
          f"max-train-seconds {budget.max_train_seconds:g}")
    print(f"  catalog    : {_catalog_description(task_name, args.policy)}")
    est = estimate_wall(load_registry(args.runs_dir), task_name,
                        budget.max_experiments)
    if est["estimate"] is not None:
        print(f"  estimate   : ~{est['estimate']:g}s (median over "
              f"{est['runs']} past run(s) of this task in the registry, "
              f"SPEC.md 40.1.2)")
    else:
        print("  estimate   : (no past runs of this task in the registry)")
    return 0


def _catalog_description(task_name: str, policy: str) -> str:
    """SPEC.md 40.1.2 (v0.26): the policy's expected candidate space, one line.

    bandit → the offered families (25.5) + the union of their catalog
    actions; search → the legacy 14-field space; rl → the full mutation
    catalog.
    """
    from .improver import catalog
    if policy == "rl":
        return f"rl: {len(catalog.ACTIONS)}-action mutation catalog"
    if policy == "search":
        from .improver.actions import SEARCH_FIELDS
        return f"search: {len(SEARCH_FIELDS)}-field v1 spec space"
    fams = catalog.relevant_families(task_name)
    actions: set = set()
    for fam in fams:
        actions.update(catalog.relevant_actions(fam))
    return (f"bandit: {len(fams)} families ({', '.join(fams)}), "
            f"{len(actions)} catalog actions (union)")


def _fit_data(args: argparse.Namespace) -> tuple:
    """`fit --data` mode (SPEC.md 22.1/24.5): probe the data, build the env,
    and return `(env, bar, metric)` (or `(None, None, None)` on a bad
    path/task)."""
    from .tasks import TASKS
    data = Path(args.data)
    try:
        task_name = _resolve_fit_task(data, getattr(args, "task", "auto") or "auto")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None, None, None
    config: dict = {"path": str(data)}
    if args.label is not None:
        config["label"] = args.label
    if args.split_frac != 0.2:
        config["split_frac"] = args.split_frac
    if getattr(args, "temporal", False):  # v0.31 (SPEC.md 45.1.2)
        config["split_mode"] = "temporal"
    if getattr(args, "metric", "accuracy") != "accuracy":  # 46.1 (v0.32)
        config["metric"] = args.metric
    # deterministic probe for the train-split size (the env re-derives the
    # identical split from the same seed, SPEC.md 22.1)
    try:
        probe = TASKS[task_name](seed=args.seed, **config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None, None, None
    # SPEC.md 36.1.5 (v0.22, G1): the gate line names the task's declared
    # `metric` — read, not inferred
    metric = getattr(probe, "metric", None) or "score"
    quality = {} if args.search_quality == "legacy" else search_quality_v04()
    args.task = task_name  # _drive's print lines
    env = AutoRefineEnv(
        task=task_name, seed=args.seed, budget=_budget(args), runs_dir=args.runs_dir,
        ensemble_top_k=2 if args.ensemble_final else 0,
        dataset_episodes=int(probe.default_dataset_size),
        task_config=config,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; None = off)
        screen_frac=args.screen_frac,  # v0.18 (SPEC.md 32.2; 1.0 = off)
        kfold=args.kfold,  # v0.30 (SPEC.md 44.1; 0 = off, legacy)
        # v0.23 (SPEC.md 37.1.3): driver metadata for the canonical recipe
        policy=args.policy, target=args.target,
        rl_episodes=args.rl_episodes if args.policy == "rl" else None,
        **quality,
    )
    return env, args.target, metric


def _fit_from_run(args: argparse.Namespace) -> tuple:
    """`fit --from-run RUN_DIR` (SPEC.md 37.1.4): re-run the finished run
    exactly from its `run_config` (the recipe is the single source of
    truth) — the data path/label/split, seed, budget, policy (+rl_episodes),
    target, the search-quality knobs **verbatim** (not by preset), an
    **explicit** `dataset_size` (no re-probe), and ensemble/stall/screening.
    `--target` is ignored (the recipe's target is authoritative). Returns
    `(env, bar)` or `(None, None)` on error."""
    run_dir = Path(args.from_run)
    rc_path = run_dir / "run_config.json"
    if rc_path.is_file():
        raw = json.loads(rc_path.read_text(encoding="utf-8"))
    else:  # fallback: the additive summary.json key (34.2)
        sp = run_dir / "summary.json"
        if not sp.is_file():
            print(f"no run_config.json or summary.json in {run_dir} "
                  "(SPEC.md 37.1.4)", file=sys.stderr)
            return None, None
        raw = json.loads(sp.read_text(encoding="utf-8")).get("run_config")
    if not isinstance(raw, dict):
        print(f"{run_dir} has no run_config (a pre-v0.23 run) — re-run with "
              "--data instead (SPEC.md 37.1.4)", file=sys.stderr)
        return None, None
    try:
        config = RunConfig.from_dict(raw)
    except ValueError as exc:
        print(f"invalid run_config: {exc}", file=sys.stderr)
        return None, None
    if not config.task_config.get("path"):
        print("that run used a built-in task — use `autorefine run` instead "
              "(SPEC.md 37.1.4)", file=sys.stderr)
        return None, None
    # the recipe's target is authoritative; `--target` is ignored here
    bar = config.target if config.target is not None else 95.0
    # sync the CLI namespace the shared driver reads, so the re-run is
    # bit-identical (same policy seed, budget, knobs, explicit size — 37.1.5)
    args.seed = config.seed
    args.policy = config.policy or "bandit"
    args.rl_episodes = (config.rl_episodes
                        if config.rl_episodes is not None else args.rl_episodes)
    args.task = config.task
    args.ensemble_final = config.ensemble_top_k > 0
    env = AutoRefineEnv(
        task=config.task, seed=config.seed,
        budget=Budget(config.max_experiments, config.max_wall_seconds,
                      config.max_train_seconds),
        runs_dir=args.runs_dir,
        dataset_episodes=config.dataset_size,  # explicit, no re-probe (37.1.4)
        task_config=dict(config.task_config),
        ci_blocks=config.ci_blocks, z_accept=config.z_accept,
        efficiency_weight=config.efficiency_weight,
        gen_gap_penalty=config.gen_gap_penalty, block_size=config.block_size,
        ensemble_top_k=config.ensemble_top_k,
        stall_patience=config.stall_patience,  # v0.17 (SPEC.md 31.1)
        screen_frac=config.screen_frac,  # v0.18 (SPEC.md 32.2)
        kfold=config.kfold,  # v0.30 (SPEC.md 44.1): the recipe's kfold
        policy=config.policy, target=bar, rl_episodes=config.rl_episodes,
        # SPEC.md 38.2 (v0.24, T2): lineage — this run re-executes that one
        parent_run=run_dir.name,
    )
    return env, bar


def _fit_gate(args: argparse.Namespace, env: AutoRefineEnv, bar: float,
              metric: str) -> int:
    """The fit acceptance gate (SPEC.md 22.1 / 37.2). No `--gate` → the
    §22.1 score bar, byte-identical to pre-v0.23 (gate line + PASS/MISS text
    + rc 0/2); `--gate` (1+) → the objective set with per-objective rows
    (37.2.3). Returns the exit code."""
    final = env.best_score
    if not args.gate:
        # no --gate: the §22.1 gate, exactly
        print(f"\ngate    : final {final:.2f} on {metric} vs target {bar:.1f} "
              f"(SPEC.md 22.1; the loop maximizes the validated score, the bar is yours)")
        if final >= bar:
            print(f"PASS: final score {final:.2f} on {metric} >= {bar:.1f} "
                  "on held-out data (the loop never trained on these points; "
                  "gen_gap is the overfit guard)")
            return 0
        print(f"MISS: {final:.2f} < {bar:.1f} — the search space or budget is "
              "exhausted for this task")
        print("next: raise --experiments / --max-train-seconds, or extend the spec "
              "space / model families (README 'Extending')")
        return 2
    # --gate (SPEC.md 37.2): the objective set (score / train / model)
    objectives = tuple(parse_objective(g) for g in args.gate)
    actuals = actuals_from_run(env, env.memory.load_experiments())
    res = evaluate(objectives, actuals)
    n = len(res["objectives"])
    print(f"\ngate    : {n} objective(s) (SPEC.md 37.2)")
    for row in res["objectives"]:
        actual_s = (f"{row['actual']:.3g}" if row["actual"] is not None
                    else "n/a")
        print(f"  {row['name']:6s} {row['op']} {row['threshold']:.6g}   "
              f"actual {actual_s}   {'PASS' if row['pass'] else 'MISS'}")
    if res["pass"]:
        print(f"PASS: all {n} objective(s) met (SPEC.md 37.2)")
        return 0
    failed = ", ".join(r["name"] for r in res["objectives"] if not r["pass"])
    print(f"MISS: objective(s) not met: {failed} (SPEC.md 37.2)")
    return 2


def _cmd_variance(args: argparse.Namespace) -> int:
    """D1 (SPEC.md 29.1): the same budget under N seeds — "is the improvement
    real?" = the `fit` task resolution + `DashboardRunner.seed_sweep`.

    A variance report is a measurement, not a gate: per-seed `verdict` carries
    the §22.1 gate, but the command exits 0 (even with MISS seeds). Writes
    `seed_variance.svg` + `seed_curves.svg` (V1, SPEC.md 30.1) +
    `seed_sweep.json`; `--json` prints pure JSON (SPEC.md 21.2).
    """
    from .dashboard import DashboardRunner
    from .plotting import _quantile, svg_seed_curves, svg_seed_variance
    data = Path(args.data)
    try:
        _resolve_fit_task(data, getattr(args, "task", "auto") or "auto")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # v0.23 (SPEC.md 37.2): the per-seed verdicts carry the objective set —
    # no `--gate` → exactly the §22.1 score gate (byte-identical); `--gate`
    # (1+) → the score/train/model objective set. A variance report stays a
    # measurement (exit 0, SPEC.md 29.1).
    objectives = (tuple(parse_objective(g) for g in args.gate)
                  if args.gate else None)
    runner = DashboardRunner(
        csv_path=str(data), label=args.label, split_frac=args.split_frac,
        target=args.target, policy=args.policy, seed=args.seed,
        experiments=args.experiments, max_seconds=args.max_seconds,
        max_train_seconds=args.max_train_seconds, runs_dir=args.runs_dir,
        search_quality=args.search_quality, modality=args.task,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; per-seed)
        objectives=objectives,  # v0.23 (SPEC.md 37.2)
    )
    seeds = list(range(args.seed, args.seed + args.seeds))
    sweep = runner.seed_sweep(seeds, workers=args.workers)  # v0.17 (SPEC.md 31.2)
    rd = Path(args.runs_dir)
    rd.mkdir(parents=True, exist_ok=True)
    svg = rd / "seed_variance.svg"
    svg.write_text(svg_seed_variance(sweep, target=args.target), encoding="utf-8")
    curves = rd / "seed_curves.svg"  # V1 (SPEC.md 30.1): when the seeds diverge
    curves.write_text(svg_seed_curves(sweep, target=args.target), encoding="utf-8")
    js = rd / "seed_sweep.json"
    js.write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(sweep, indent=2, sort_keys=True))
        return 0
    print("seed  baseline   final  verdict")
    for s in sweep:
        print(f"{s['seed']:<5} {s['baseline']:>8.2f}  {s['final']:>7.2f}  {s['verdict']}")
    finals = [s["final"] for s in sweep]
    if finals:
        sf = sorted(finals)
        print(f"finals    : median {_quantile(sf, 0.5):.2f}  "
              f"min {min(finals):.2f}  max {max(finals):.2f}")
        beats = sum(1 for s in sweep if s["final"] > s["baseline"])
        print(f"baseline  : {beats}/{len(sweep)} seed(s) beat their own baseline")
    passing = sum(1 for s in sweep if s["pass"])
    print(f"target    : {passing}/{len(sweep)} seed(s) pass {args.target:.1f} "
          f"(a variance report is a measurement - the per-seed verdict carries the gate)")
    print(f"svg     : {svg}")
    print(f"svg     : {curves}")
    print(f"json    : {js}")
    return 0


def _cmd_policy_report(args: argparse.Namespace) -> int:
    """D2 (SPEC.md 29.2): train the meta-RL policy, capture the opt-in trace,
    and render the per-step action-probability + per-task-return views.

    RL stays CLI-only (SPEC.md 23.1): this writes `action_probabilities.svg`,
    `task_returns.svg`, `policy_trace_curve.svg` (v0.16, SPEC.md 30.6),
    `policy_trace.json`, `policy_returns.json`. Single (`--task`) or
    multi-task (`--multi --tasks ...`, SPEC.md 20.2).
    """
    from .improver.catalog import ACTIONS
    from .improver.rl_policy import MetaRLPolicy, train_multi_policy, train_policy
    from .plotting import (svg_action_probabilities, svg_policy_trace,
                           svg_task_returns)

    seed, episodes = args.seed, args.episodes
    trace: list[dict] = []
    if args.multi:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if not task_names:
            print("--multi requires --tasks TASK[,TASK,...]", file=sys.stderr)
            return 1
        for t in task_names:
            if t not in TASKS:
                print(f"unknown task {t!r} in --tasks", file=sys.stderr)
                return 1
        envs = [AutoRefineEnv(task=t, seed=seed, budget=_budget(args),
                              runs_dir=args.runs_dir) for t in task_names]
        policy = MetaRLPolicy(seed=seed, task_names=task_names)
        try:
            report = train_multi_policy(envs, policy, episodes_per_task=episodes,
                                        trace=trace)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        returns = report["episode_returns"]
    else:
        if args.task not in TASKS:
            print(f"unknown task {args.task!r}", file=sys.stderr)
            return 1
        env = AutoRefineEnv(task=args.task, seed=seed, budget=_budget(args),
                            runs_dir=args.runs_dir)
        policy = MetaRLPolicy(seed=seed)
        report = train_policy(env, policy, n_episodes=episodes, trace=trace)
        returns = {args.task: report["episode_returns"]}

    rd = Path(args.runs_dir)
    rd.mkdir(parents=True, exist_ok=True)
    ap = rd / "action_probabilities.svg"
    ap.write_text(svg_action_probabilities(trace, top_k=args.top_k,
                                           n_steps=args.n_steps), encoding="utf-8")
    tr = rd / "task_returns.svg"
    tr.write_text(svg_task_returns(returns), encoding="utf-8")
    pc = rd / "policy_trace_curve.svg"  # V6 (SPEC.md 30.6): r2g / baseline trace
    pc.write_text(svg_policy_trace(trace), encoding="utf-8")
    tj = rd / "policy_trace.json"
    tj.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    pj = rd / "policy_returns.json"
    pj.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for task, vals in returns.items():
        print(f"{task}: " + " ".join(f"{v:+.3f}" for v in vals))
    print(f"policy_updates: {report['policy_updates']}")
    print(f"trace steps  : {len(trace)}")
    if trace:
        last = trace[-1]
        order = sorted(range(len(last["probs"])), key=lambda i: (-last["probs"][i], i))
        top = [i for i in order[:args.top_k] if last["probs"][i] > 0.0]
        print("final step   : " + "  ".join(
            f"{ACTIONS[i][0]}={ACTIONS[i][1]} {last['probs'][i]:.2f}" for i in top))
    print(f"svg     : {ap}")
    print(f"svg     : {tr}")
    print(f"svg     : {pc}")
    print(f"json    : {tj}  {pj}")
    return 0


def _cmd_report_history(args: argparse.Namespace) -> int:
    """SPEC.md 38.3 (v0.24, T1): `report --history` — the run registry as a
    table (or pure JSON with `--json`). No registry / empty → rc 1 + hint
    (38.3.3)."""
    from .registry import format_table, load_registry
    runs_dir = Path(args.runs_dir)
    entries = load_registry(runs_dir)
    if not entries:
        print(f"no runs registered in {runs_dir} — finish at least one run "
              f"first (SPEC.md 38.3.3)", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0
    print(format_table(entries))
    print(f"\nregistry : {runs_dir / 'registry.json'} ({len(entries)} run(s))")
    return 0


def _print_accounting(acc: dict) -> None:
    """SPEC.md 39.2 (T4): render the \"what happened\" block in the human
    report (pure — a function of the `account_run` result, 39.2.3)."""
    rej = acc["rejections"]
    tti = acc["time_to_first_improvement"]
    wt = acc["wall_time"]
    total = rej["accepted"] + rej["rejected"]
    print("\n=== what happened ===")
    print(f"  candidates      : {total} total   accepted {rej['accepted']}   "
          f"rejected {rej['rejected']}")
    print(f"  rejected by gate: score {rej['score']}   overfit {rej['overfit']}   "
          f"ci {rej['ci']}")
    print(f"  duplicates      : {rej['duplicate']}   ({rej['note']})")
    if tti["improved"]:
        secs = tti["seconds"]
        secs_s = (f"{secs:+.1f}s after baseline" if secs is not None
                  else "(no timestamps in log)")
        print(f"  first improvement: candidate #{tti['experiment_index']}   "
              f"({secs_s})")
    else:
        print("  first improvement: none (never beat the baseline)")
    print(f"  wall time (s)   : baseline {wt['baseline']:.3f}   "
          f"candidates {wt['candidates']:.3f}   "
          f"eval+overhead {wt['eval_overhead']:.3f}   (total {wt['total']:.3f})")


def _report_what_if(args: argparse.Namespace, summary: dict,
                    entries: list) -> int:
    """`report --what-if` (SPEC.md 40.2, v0.26 S2): re-gate the run's logged
    history against a NEW objective set (40.2.4 block) and return the verdict
    rc — 0 PASS, 2 MISS, 1 on a bad/unsupported objective. Pure derivation
    from `experiments.jsonl`; nothing is retrained or written.
    """
    try:
        objectives = tuple(parse_objective(o) for o in args.what_if)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        wf = what_if(entries, objectives)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\nwhat-if   : re-gate the logged history against a NEW objective set "
          "(no retraining, SPEC.md 40.2)")
    for o in objectives:
        print(f"  bar       : {o.name} {o.op} {o.threshold:g}")
    actual_final = summary.get("final_best_score")
    actual_s = (f"{float(actual_final):.2f}" if actual_final is not None
                else "n/a")
    print(f"  pool      : {wf['pool']} scored candidate(s); {wf['passing']} pass "
          f"the bar")
    print(f"  actual final : {actual_s} (this run's best, for contrast)")
    if wf["final"] is not None:
        f = wf["final"]
        sh = f["spec_hash"]
        sh_s = (str(sh)[:8] if isinstance(sh, str) else str(sh))
        tr_s = f"{f['train']:g}s" if f["train"] is not None else "n/a"
        print(f"  counterfactual final: candidate #{f['cand']}, spec {sh_s}, "
              f"score {f['score']:.2f}, train {tr_s}")
    if wf["pass"]:
        print(f"  verdict   : PASS — {wf['passing']} logged candidate(s) meet "
              f"the bar (SPEC.md 40.2.5)")
        return 0
    print("  verdict   : MISS — no logged candidate meets the bar "
          "(SPEC.md 40.2.5)")
    return 2


def _report_project(args: argparse.Namespace, summary: dict,
                    run_dir: Path) -> None:
    """`report --project` (SPEC.md 41.1, v0.27 S3): project the budget to
    the target from the same-task history (41.1.2 points) — informational,
    rc 0 (the 41.1.1 guards handle the error paths above)."""
    reg = load_registry(run_dir.parent)  # 41.1.2.1: the runs-dir registry
    pts = projection_points(reg, summary, run_dir.name)
    e_cur = summary.get("experiments_run")
    if isinstance(e_cur, (int, float)) and not isinstance(e_cur, bool):
        e_cur = float(e_cur)
    else:
        e_cur = 0.0
    r = project_budget(pts, args.target, e_cur)
    print(f"\nprojection: will we reach target {args.target:g}? "
          "(saturating fit over same-task history, SPEC.md 41.1)")
    print(f"  points    : {r['points']} same-task run(s) "
          "(registry + this run, deduped by run id)")
    if not r["ok"]:
        print("  verdict   : INSUFFICIENT HISTORY — need >= 2 same-task runs "
              "with positive scores (SPEC.md 41.1.4)")
        return
    print(f"  curve     : score(e) = Vmax*e/(Km+e)   Vmax={r['vmax']:.2f}   "
          f"Km={r['km']:.2f}")
    if r["verdict"] == "ceiling":
        print(f"  verdict   : CEILING — the curve's asymptote ({r['vmax']:.2f}) "
              f"does not exceed the target ({args.target:g}); more "
              "experiments will not reach it (SPEC.md 41.1.4)")
    else:
        print(f"  verdict   : MORE — ~{r['more']} more experiment(s): the "
              f"curve reaches {args.target:g} at ~{r['e_target']:.1f} "
              f"experiments total (this run: {e_cur:g}) (SPEC.md 41.1.4)")


def _cmd_report(args: argparse.Namespace) -> int:
    # SPEC.md 40.2.1 (v0.26, S2): --what-if is a human counterfactual view,
    # mutually exclusive with the machine (--json) and history paths.
    if getattr(args, "what_if", None):
        if args.history:
            print("--what-if and --history are mutually exclusive "
                  "(SPEC.md 40.2.1)", file=sys.stderr)
            return 1
        if args.json:
            print("--what-if and --json are mutually exclusive "
                  "(SPEC.md 40.2.1)", file=sys.stderr)
            return 1
    # SPEC.md 44.2.2 (v0.30): --importance is a human view, mutually
    # exclusive with the machine (--json), the history path, the what-if
    # path, and the project/trace paths.
    if getattr(args, "importance", False):
        if (args.history or args.json or getattr(args, "what_if", None)
                or getattr(args, "project", False)
                or getattr(args, "trace", False)):
            print("--importance is mutually exclusive with "
                  "--json/--history/--what-if/--project/--trace "
                  "(SPEC.md 44.2.2)", file=sys.stderr)
            return 1
    # SPEC.md 41.1.1/41.2.1 (v0.27): --project / --trace are human views,
    # mutually exclusive with the machine (--json), the history path, and
    # the what-if path (A11 untouched).
    if getattr(args, "project", False) or getattr(args, "trace", False):
        if args.history:
            print("--project/--trace and --history are mutually exclusive "
                  "(SPEC.md 41.1.1/41.2.1)", file=sys.stderr)
            return 1
        if args.json:
            print("--project/--trace and --json are mutually exclusive "
                  "(SPEC.md 41.1.1/41.2.1)", file=sys.stderr)
            return 1
        if getattr(args, "what_if", None):
            print("--project/--trace and --what-if are mutually exclusive "
                  "(SPEC.md 41.1.1/41.2.1)", file=sys.stderr)
            return 1
    # SPEC.md 38.3.1: --history and --run are mutually exclusive; one is
    # required (the pre-v0.24 `report --run` path below is unchanged)
    if args.history:
        if args.run is not None:
            print("--run and --history are mutually exclusive (SPEC.md 38.3.1)",
                  file=sys.stderr)
            return 1
        return _cmd_report_history(args)
    if args.run is None:
        print("one of --run or --history is required (SPEC.md 38.3.1)",
              file=sys.stderr)
        return 1
    mem = RunMemory(Path(args.run))
    try:
        summary = mem.load_summary()
    except FileNotFoundError:
        print("no summary.json in run dir", file=sys.stderr)
        return 1
    entries = mem.load_experiments()
    run_dir = Path(args.run)
    extras = _report_extras(summary, run_dir)  # SPEC.md 28.2-28.4 (None-safe)
    if args.plot:  # SPEC.md 21.2: SVG files always land in the run dir
        pareto_pts = summary.get("pareto_frontier") or [
            {"score": e["holdout_score"], "train_seconds": e.get("train_seconds", 0.0)}
            for e in entries
            if e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)  # 35.1 (C4)
            and isinstance(e.get("holdout_score"), (int, float))
        ]
        (run_dir / "score_curve.svg").write_text(svg_score_curve(entries), encoding="utf-8")
        (run_dir / "pareto_frontier.svg").write_text(svg_pareto(pareto_pts), encoding="utf-8")
        if summary.get("curriculum"):  # SPEC.md 27.3: only runs with a ladder
            (run_dir / "ladder_curve.svg").write_text(
                svg_ladder_curve(entries, summary["curriculum"].get("levels")),
                encoding="utf-8")
        if extras["per_class_svg"]:  # SPEC.md 28.2 (C2)
            (run_dir / "per_class.svg").write_text(extras["per_class_svg"],
                                                   encoding="utf-8")
        if extras["confusion_svg"]:  # SPEC.md 28.2 (C2)
            (run_dir / "confusion.svg").write_text(extras["confusion_svg"],
                                                   encoding="utf-8")
        if extras["arch_svg"]:  # SPEC.md 28.4 (C4)
            (run_dir / "architecture.svg").write_text(extras["arch_svg"],
                                                      encoding="utf-8")
    if args.html:  # SPEC.md 22.2: one self-contained file in the run dir
        (run_dir / "report.html").write_text(
            html_report(summary, entries, diagnostics=extras["diagnostics"],
                        gallery=extras["gallery"], arch_svg=extras["arch_svg"]),
            encoding="utf-8")
    if args.json:  # SPEC.md 21.2: stdout is pure machine-readable JSON
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    diag = extras["diagnostics"]
    if diag:  # C2 (SPEC.md 28.2): per-class lines + weakest-class hint
        print(f"\nholdout diagnostics: {diag['correct']}/{diag['n']} correct")
        for i, lbl in enumerate(diag["class_labels"]):
            print(f"  class {str(lbl):<12s} {diag['per_class'][i]:5.1f}% "
                  f"({diag['confusion'][i][i]}/{diag['class_counts'][i]})")
        worst = min(range(len(diag["class_labels"])),
                    key=lambda i: diag["per_class"][i])
        if diag["per_class"][worst] < 100.0:
            print(f"  the model fails on class {diag['class_labels'][worst]}")
        if extras["gallery"]:  # C3 (SPEC.md 28.3): media error gallery, text
            print("holdout misclassifications:")
            for it in extras["gallery"]:
                print(f"  {it.get('file', '?')} - true {it.get('label')} -> "
                      f"predicted {it.get('predicted')}")
    for k in ("task", "seed", "finished_reason", "baseline_score",
              "final_best_score", "improvement_factor", "experiments_run", "wall_seconds"):
        print(f"{k:20s}: {summary.get(k)}")
    # SPEC.md 39.2 (T4): the "what happened" block — additive human output only
    # (--json / --plot / the summary file / the registry are byte-identical)
    _print_accounting(account_run(summary, entries))
    # SPEC.md 41.1 (v0.27, S3): budget projection — an informational human
    # view (rc 0); pure derivation, nothing is written.
    if getattr(args, "project", False):
        _report_project(args, summary, run_dir)
    # SPEC.md 41.2 (v0.27, S4): the line-per-experiment decision trace.
    if getattr(args, "trace", False):
        print("\n=== trace (one line per logged experiment) ===")
        for line in trace_lines(entries):
            print(f"  {line}")
    # SPEC.md 44.2 (v0.30, C): permutation feature importance — one extra
    # pass over the holdout rows; fitting tasks only (None -> n/a line).
    if getattr(args, "importance", False):
        rc = _cmd_importance(args, run_dir)
        if rc != 0:
            return rc
    # SPEC.md 40.2 (v0.26, S2): counterfactual re-gating — a human view that
    # returns its own verdict rc (0 PASS / 2 MISS / 1 bad objective) before
    # the rest of the report; the machine path above already returned.
    if getattr(args, "what_if", None):
        return _report_what_if(args, summary, entries)
    # SPEC.md 37.1.4: when the run carries the canonical recipe, render the
    # copy-pasteable command (fit for data tasks, run for built-in tasks)
    rc_cfg = summary.get("run_config")
    if rc_cfg:
        try:
            print("\nreproduce (copy-paste):")
            print(f"  {format_recipe(RunConfig.from_dict(rc_cfg))}")
        except ValueError:
            pass  # a pre-v0.23 / mutated run_config: skip the recipe line
    print("\nbest spec:")
    print(json.dumps(summary.get("best_spec"), indent=2))
    print(f"\n{'kind':10s} {'accepted':9s} {'score':>8s} {'gen_gap':>9s} {'mutation'}")
    for e in entries:
        print(
            f"{e.get('kind','?'):10s} {str(e.get('accepted')):9s} "
            f"{e.get('holdout_score', 0):8.2f} {e.get('gen_gap', 0):9.2f} "
            f"{','.join(e.get('mutation') or []) or '-'}"
        )
    if args.plot:  # SPEC.md 21.2: ASCII charts appended to the human report
        print("\nscore vs experiment:")
        print(ascii_score_curve(entries))
        print("\npareto frontier (score vs training seconds):")
        print(ascii_pareto(pareto_pts))
        print(f"svg     : {run_dir / 'score_curve.svg'}")
        print(f"svg     : {run_dir / 'pareto_frontier.svg'}")
        if summary.get("curriculum"):  # SPEC.md 27.3
            print(f"svg     : {run_dir / 'ladder_curve.svg'}")
        if extras["per_class_svg"]:  # SPEC.md 28.2
            print(f"svg     : {run_dir / 'per_class.svg'}")
        if extras["confusion_svg"]:  # SPEC.md 28.2
            print(f"svg     : {run_dir / 'confusion.svg'}")
        if extras["arch_svg"]:  # SPEC.md 28.4
            print(f"svg     : {run_dir / 'architecture.svg'}")
    if args.html:  # SPEC.md 22.2: point at the generated file
        print(f"html    : {run_dir / 'report.html'}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """SPEC.md 39.1 (T3): tail a run's `experiments.jsonl` and re-render the
    ASCII charts (human mode) or emit compact JSON lines (`--tail`, for CI).

    rc 0 when the run finishes; rc 130 on Ctrl-C; rc 1 when `--max-polls`
    is exhausted before the run finishes (a bounded wait, 39.1.4). The pure
    helpers live in `autorefine.watch`; this is only the thin poll loop.
    """
    import time
    run_dir = Path(args.run)
    exp_path = run_dir / "experiments.jsonl"
    entries: list[dict] = []
    offset = 0
    max_polls = (args.max_polls
                 if (args.max_polls is not None and args.max_polls > 0) else None)
    polls = 0
    try:
        while True:
            polls += 1
            if not run_dir.is_dir():
                # the run dir hasn't appeared yet (run starts elsewhere)
                if max_polls is not None and polls >= max_polls:
                    print(f"run dir not found (bounded wait): {run_dir}",
                          file=sys.stderr)
                    return 1
                time.sleep(args.interval)
                continue
            new_entries, offset = read_new_entries(exp_path, offset)
            entries.extend(new_entries)
            if args.tail:  # 39.1.3 machine mode: one compact line per new row
                base = len(entries) - len(new_entries)
                for j, e in enumerate(new_entries):
                    print(tail_line(e, base + j), flush=True)
            finished = run_finished(run_dir)
            if not args.tail and (new_entries or finished):  # 39.1.3 human mode
                if args.clear:  # 39.1.1 live refresh (off by default)
                    print("\x1b[2J\x1b[H", end="")
                print(live_frame(entries), flush=True)
            if finished:
                if not args.tail:
                    print("\n(run finished)", flush=True)
                return 0
            if max_polls is not None and polls >= max_polls:
                print("watch: still running after "
                      f"{max_polls} polls (use --max-polls 0 to wait "
                      "indefinitely)", file=sys.stderr)
                return 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


def _dashboard_command(runs_dir: str, port: int) -> list[str]:
    """The `streamlit run` command for the in-repo app (SPEC.md 23.4)."""
    app = Path(__file__).resolve().parent / "dashboard_app.py"
    return [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.headless", "true",
        f"--server.port={port}",
        "--", f"--runs-dir={runs_dir}",
    ]


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """`dashboard` (SPEC.md 23.4): launch the optional Streamlit app.

    The sole subprocess touch point (S1 exception, v0.9): it spawns the
    streamlit CLI on the fixed in-repo app file, never on user code.
    """
    try:
        import streamlit  # noqa: F401 — availability check only
    except ImportError:
        print("streamlit is not installed — the dashboard is an optional extra "
              "(SPEC.md 23.4)", file=sys.stderr)
        print("pip install autorefine[gui]     # or: pip install streamlit",
              file=sys.stderr)
        return 1
    cmd = _dashboard_command(args.runs_dir, args.port)
    return int(subprocess.call(cmd))


def _cmd_plugins_list(args: argparse.Namespace) -> int:
    for group in (TASKS_GROUP, POLICIES_GROUP):
        try:
            found = discover(group)
        except PluginError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if found:
            for name in sorted(found):
                obj = found[name]
                qual = getattr(obj, "__qualname__", obj.__name__)
                print(f"{group} : {name} = {obj.__module__}.{qual}")
        else:
            print(f"{group} : (none)")
    return 0


# SPEC.md 25.5: family loaders for `eval` (local imports keep the module
# surface small, mirroring the v0.5 boost loader)
def _load_tree(path: str):
    from .models.trees import TreeEnsemble
    return TreeEnsemble.load(path)


def _load_boost(path: str):
    from .models.trees import BoostingEnsemble
    return BoostingEnsemble.load(path)


def _load_knn(path: str):
    from .models.knn import KNN
    return KNN.load(path)


def _load_convnet(path: str):
    from .models.convnet import ConvNet
    return ConvNet.load(path)


def _task_from_summary(summary: dict):
    """The task a run trained (SPEC.md 22.1/28.4): `task`/`seed`/`task_config`
    reconstruction with the curriculum-level fallback (SPEC.md 20.1) —
    shared by `eval` and `report`. None when the name is unknown to both."""
    task_name = summary.get("task")
    seed = int(summary["seed"])
    cfg = summary.get("task_config") or {}  # e.g. csv: path/label
    if task_name in TASKS:
        return TASKS[task_name](seed=seed, **cfg) if cfg \
            else TASKS[task_name](seed=seed)
    if summary.get("curriculum"):  # SPEC.md 20.1/46.2: a curriculum level
        lvl = summary["curriculum"]["levels"][-1]
        if "n_bits" in lvl and "p_flip" in lvl:
            # the historical parity shape (20.1) — byte-identical rule
            return ParityTask(seed=seed, n_bits=lvl["n_bits"], p_flip=lvl["p_flip"])
        if "noise" in lvl and "freq_scale" in lvl and "amplitude" in lvl:
            # SPEC.md 46.2.3: the sine ladder level
            return SineRegressionV1(
                seed, noise=lvl["noise"], freq_scale=lvl["freq_scale"],
                amplitude=lvl["amplitude"])
        if "ic_scale" in lvl:
            # SPEC.md 46.2.4: the cartpole ladder level
            return CartPoleV1(seed, ic_scale=lvl["ic_scale"])
    return None


def _best_model_loaders() -> dict:
    """SPEC.md 25.5: family -> npz loader mapping (knn/convnet since v0.11)."""
    return {
        "mlp": lambda p: MLP.load(p),
        "tree": lambda p: _load_tree(p),
        "boost": lambda p: _load_boost(p),  # SPEC.md 19.2
        "knn": lambda p: _load_knn(p),      # SPEC.md 25.2
        "convnet": lambda p: _load_convnet(p),  # SPEC.md 25.3
    }


def _print_fit_diagnostics(env: AutoRefineEnv) -> None:
    """C2 (SPEC.md 28.2): the `fit` output after the gate line — per-class
    holdout accuracy lines + the weakest-class hint ("the model fails on
    class X"); for media tasks the C3 text error gallery (SPEC.md 28.3).
    Classification-only: episode/mse tasks print nothing (SPEC.md 28.5)."""
    if getattr(env.task, "head", None) != "softmax" or env.best_model is None:
        return
    diag = holdout_diagnostics(env.task, env.best_model)
    if not diag:
        return
    print(f"\nholdout diagnostics (SPEC.md 28.2): "
          f"{diag['correct']}/{diag['n']} correct")
    for i, lbl in enumerate(diag["class_labels"]):
        print(f"  class {str(lbl):<12s} {diag['per_class'][i]:5.1f}% "
              f"({diag['confusion'][i][i]}/{diag['class_counts'][i]})")
    worst = min(range(len(diag["class_labels"])),
                key=lambda i: diag["per_class"][i])
    if diag["per_class"][worst] < 100.0:
        print(f"  the model fails on class {diag['class_labels'][worst]} "
              f"({diag['per_class'][worst]:.1f}%)")
    hold_errors = getattr(env.task, "holdout_errors", None)
    if callable(hold_errors):  # C3 (SPEC.md 28.3), media tasks only
        errors = hold_errors(env.best_model)
        if errors:
            print("holdout misclassifications (SPEC.md 28.3):")
            for it in errors:
                print(f"  {it.get('file', '?')} - true {it.get('label')} -> "
                      f"predicted {it.get('predicted')}")


def _report_extras(summary: dict, run_dir: Path) -> dict:
    """C2/C3/C4 (SPEC.md 28.2-28.4): the learning views for `report` /
    `html`, when the task and best model are reconstructible from the run
    (same rule as `eval`); empty when they are not or when the task is not
    classification (SPEC.md 28.5) — the report still renders."""
    out = {"diagnostics": None, "gallery": None, "arch_svg": None,
           "per_class_svg": None, "confusion_svg": None}
    try:
        task = _task_from_summary(summary)
        if task is None:
            return out
        family = (summary.get("best_spec") or {}).get("model_family", "mlp")
        loaders = _best_model_loaders()
        if family not in loaders:
            return out
        model = loaders[family](str(run_dir / "best_model.npz"))
    except (FileNotFoundError, ValueError):
        return out
    diag = holdout_diagnostics(task, model)  # None for episode/mse (28.2)
    out["diagnostics"] = diag
    if diag:
        out["per_class_svg"] = svg_per_class_bars(diag)
        out["confusion_svg"] = svg_confusion_matrix(diag)
    hold_errors = getattr(task, "holdout_errors", None)
    if callable(hold_errors):  # C3 (SPEC.md 28.3), media tasks only
        g = hold_errors(model)
        out["gallery"] = g if g else None
    best_spec = summary.get("best_spec")
    if best_spec:  # C4 (SPEC.md 28.4)
        out["arch_svg"] = svg_architecture(
            best_spec, getattr(task, "state_dim", 1),
            getattr(task, "n_outputs", 1))
    return out


def _cmd_importance(args: argparse.Namespace, run_dir: Path) -> int:
    """`report --importance` (SPEC.md 44.2, v0.30): the winning model's
    per-column holdout score drop (permutation importance). Reconstructs
    the run through the 42.1 `_run_task_and_model` pattern (no re-
    implemented loading); the n/a case (44.2.3) prints a clear line and
    still returns rc 0; a broken run dir is rc 1."""
    from .importance import permutation_importance
    try:
        _summary, task, model = _run_task_and_model(run_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = permutation_importance(task, model,
                                    n_repeats=args.importance_repeats)
    if result is None:
        print("\nfeature importance: n/a for this run (episode task, "
              "no holdout rows, or non-flat features — SPEC.md 44.2.3)")
        return 0
    print(f"\nfeature importance (permutation, holdout n={result['n']}, "
          f"head {result['head']}, baseline {result['baseline']:.2f})")
    print("  importance = score drop when the column is shuffled "
          "(higher = the model uses it more)")
    for f in result["features"]:
        print(f"  {f['name']:<24s} {f['importance']:+9.3f}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    mem = RunMemory(Path(args.run))
    try:
        summary = mem.load_summary()
    except FileNotFoundError:
        print("no summary.json in run dir", file=sys.stderr)
        return 1
    task = _task_from_summary(summary)
    if task is None:
        print(f"unknown task {summary.get('task')!r} in summary", file=sys.stderr)
        return 1
    family = summary["best_spec"].get("model_family", "mlp")
    loaders = _best_model_loaders()
    if family not in loaders:
        print(f"unknown model_family {family!r} in summary", file=sys.stderr)
        return 1
    model = loaders[family](str(Path(args.run) / "best_model.npz"))
    spec = ModelSpec.from_dict(summary["best_spec"])
    ev = evaluate_full(task, model, n=args.episodes)
    print(f"spec      : {json.dumps(spec.to_dict())}")
    print(f"holdout   : {ev['score']:.2f}")
    print(f"gen score : {ev['gen_score']:.2f}")
    print(f"gen gap   : {ev['gen_gap']:.2f}")
    return 0


# --- v0.28 (SPEC.md 42): the "use the model" loop ----------------------------

def _run_task_and_model(run_dir: Path) -> tuple[dict, object, object]:
    """SPEC.md 42 (v0.28): reconstruct (summary, task, model) for a finished
    run — the `_report_extras` pattern (28.2), shared by `predict` and
    `explain`. Raises ValueError/FileNotFoundError on a broken run dir."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"no summary.json in run dir {str(run_dir)!r}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task = _task_from_summary(summary)
    if task is None:
        raise ValueError(f"unknown task {summary.get('task')!r} in summary")
    family = (summary.get("best_spec") or {}).get("model_family", "mlp")
    loaders = _best_model_loaders()
    if family not in loaders:
        raise ValueError(f"unknown model_family {family!r} in summary")
    model = loaders[family](str(run_dir / "best_model.npz"))
    return summary, task, model


def _pred_str(pred: object) -> str:
    """SPEC.md 42.1.4 (v0.28): the human rendering of a prediction —
    integer-valued floats without a trailing ".0" (the §28.3 label rule)."""
    if isinstance(pred, float) and pred.is_integer():
        return str(int(pred))
    return str(pred)


def _cmd_predict(args: argparse.Namespace) -> int:
    """`predict` (SPEC.md 42.1, v0.28): score NEW rows with the run's best
    model. Exactly one input mode (42.1.1); fitting tasks only (42.1.2);
    preprocessing = the task's own train stats (42.1.3); one forward pass
    per row (42.1.4)."""
    run_dir = Path(args.run)
    try:
        _summary, task, model = _run_task_and_model(run_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if int(getattr(task, "max_steps", 1)) > 1:  # 42.1.2: fitting tasks only
        print(f"task {getattr(task, 'name', '?')!r} is an episode task "
              "(max_steps > 1): predict supports fitting tasks (csv, sine-v1, "
              "image, audio) — use `autorefine eval` for episode tasks "
              "(SPEC.md 42.1.2)", file=sys.stderr)
        return 1
    if args.prob and getattr(task, "head", None) != "softmax":  # 46.1 (v0.32)
        print("predict: probabilities are only available for softmax-head "
              "models (SPEC.md 46.1)", file=sys.stderr)
        return 1
    modes = [m for m, on in (("row", args.row is not None),
                            ("csv", args.csv is not None),
                            ("stdin", args.stdin),
                            ("item", bool(args.item))) if on]
    if len(modes) != 1:
        print("exactly one input mode is required: --row, --csv, --stdin, "
              "or --item (SPEC.md 42.1.1)", file=sys.stderr)
        return 1
    try:
        if modes[0] == "row":
            parsed = json.loads(args.row)
            raws = [row_to_features(parsed, task.state_dim,
                                    getattr(task, "feature_names", None))]
        elif modes[0] == "csv":
            raws = csv_rows_to_features(args.csv, task)
        elif modes[0] == "stdin":
            raws = []
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                raws.append(row_to_features(parsed, task.state_dim,
                                            getattr(task, "feature_names", None)))
            if not raws:
                raise ValueError("no rows on stdin (SPEC.md 42.1.1)")
        else:  # "item" — media tasks only (42.1.3)
            raws = [media_item_features(task, p) for p in args.item]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"predict: {exc}", file=sys.stderr)
        return 1
    results = []
    for i, raw in enumerate(raws):
        try:
            feat = standardize(task, raw)
            r = predict_features(task, model, feat)
        except (ValueError, TypeError) as exc:
            print(f"predict: row {i}: {exc}", file=sys.stderr)
            return 1
        results.append({"row": i,
                        "prediction": r["prediction"],
                        "probabilities": r["probabilities"]})
    if args.json:
        print(json.dumps(results, sort_keys=True))
    else:
        for r in results:
            if args.prob:  # SPEC.md 46.1 (v0.32): the calibrated probabilities
                classes = list(task.class_values)
                for ci, cls in enumerate(classes):
                    print(f"{r['row']}\t{_pred_str(cls)}: "
                          f"p={r['probabilities'][ci]:.4f}")
            else:
                print(f"{r['row']}\t{_pred_str(r['prediction'])}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """`compare` (SPEC.md 42.2, v0.28): the app's Past-runs compare widget
    (38.4) in the terminal — `diff_two_summaries` (core `dashboard`,
    42.2.2) over two run dirs' summaries."""
    if len(args.runs) != 2:
        print("compare needs exactly two run dirs: --run A --run B "
              "(SPEC.md 42.2.1)", file=sys.stderr)
        return 1
    summaries = []
    for d in args.runs:
        p = Path(d) / "summary.json"
        if not p.is_file():
            print(f"no summary.json in run dir {d!r}", file=sys.stderr)
            return 1
        summaries.append(json.loads(p.read_text(encoding="utf-8")))
    diff = diff_two_summaries(summaries[0], summaries[1])
    if args.json:
        print(json.dumps(diff, indent=2, sort_keys=True))
        return 0
    print(f"run A : {args.runs[0]}")
    print(f"  score : {summaries[0].get('final_best_score')}")
    print(f"run B : {args.runs[1]}")
    print(f"  score : {summaries[1].get('final_best_score')}")
    print(f"delta   : {diff['score_delta']} (B - A)")
    if diff["spec_diff"]:
        print(f"best_spec diff ({diff['n_diff_fields']} field(s)):")
        for row in diff["spec_diff"]:
            print(f"  {row['field']:<24s} {row['a']} -> {row['b']}")
    else:
        print("best_spec: identical")
    return 0


def _explain_blocks(args: argparse.Namespace, run_dir: Path,
                    summary: dict, entries: list) -> dict:
    """SPEC.md 42.3.2 (v0.28): the five explain blocks — pure derivation
    from the run's artifacts (no writes)."""
    # result (42.3.2) — the run_config policy (37.1) is the header source
    result = {
        "task": summary.get("task"),
        "seed": summary.get("seed"),
        "policy": (summary.get("run_config") or {}).get("policy"),
        "baseline_score": summary.get("baseline_score"),
        "final_score": summary.get("final_best_score"),
        "improvement_factor": summary.get("improvement_factor"),
        "experiments_run": summary.get("experiments_run"),
        "finished_reason": summary.get("finished_reason"),
    }
    # tried (42.3.2) — 26.1 field_stats over the logged entries
    tried = field_stats(entries)
    # why (42.3.2) — the baseline champion + the accepted chain, log order
    why = []
    baseline = next((e for e in entries if e.get("kind") == KIND_BASELINE), None)
    if baseline is not None:
        why.append({"step": "baseline", "mutation": None,
                    "score": baseline.get("holdout_score")})
    for e in entries:
        if e.get("kind") == KIND_EXPERIMENT and e.get("accepted"):
            why.append({"step": "accepted",
                        "mutation": e.get("mutation"),
                        "score": e.get("holdout_score")})
    # weak (42.3.2) — 28.2 holdout diagnostics; n/a when they do not apply
    weak = {"diagnostics": None, "note": "n/a (not a classification task)"}
    try:
        _s, task, model = _run_task_and_model(run_dir)
        diag = holdout_diagnostics(task, model)  # None for episode/mse (28.5)
        if diag:
            worst = min(range(len(diag["class_labels"])),
                        key=lambda i: diag["per_class"][i])
            note = (f"the model fails on class {diag['class_labels'][worst]} "
                    f"({diag['per_class'][worst]:.1f}%)"
                    if diag["per_class"][worst] < 100.0
                    else "all classes at 100%")
            weak = {"diagnostics": diag, "note": note}
    except (ValueError, FileNotFoundError):
        pass  # keep the n/a note (42.3.3: informational, degrades)
    # next (42.3.2) — the 41.1 projection, same rule as `report --project`
    reg = load_registry(run_dir.parent)
    pts = projection_points(reg, summary, run_dir.name)
    e_cur = summary.get("experiments_run")
    e_cur = float(e_cur) if isinstance(e_cur, (int, float)) \
        and not isinstance(e_cur, bool) else 0.0
    proj = project_budget(pts, args.target, e_cur)
    nxt = {"target": args.target, "points": proj["points"],
           "verdict": proj["verdict"], "vmax": proj["vmax"],
           "km": proj["km"], "more": proj["more"]}
    return {"result": result, "tried": tried, "why": why,
            "weak": weak, "next": nxt}


def _cmd_explain(args: argparse.Namespace) -> int:
    """`explain` (SPEC.md 42.3, v0.28): the one-screen narrative —
    result / tried / why / weak / next (42.3.2). rc 0 (informational);
    rc 1 on a broken run dir (42.3.3)."""
    run_dir = Path(args.run)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        print(f"no summary.json in run dir {str(run_dir)!r}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = []
    exp_path = run_dir / "experiments.jsonl"
    if exp_path.is_file():
        entries = [json.loads(l) for l in
                   exp_path.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    blocks = _explain_blocks(args, run_dir, summary, entries)
    if args.json:
        print(json.dumps(blocks, sort_keys=True))
        return 0
    r = blocks["result"]
    print(f"=== explain {run_dir} (SPEC.md 42.3) ===")
    print("result")
    print(f"  task            : {r['task']}  (seed {r['seed']}, "
          f"policy {r['policy'] or '?'})")
    base = r["baseline_score"]
    fin = r["final_score"]
    b = f"{base:.2f}" if isinstance(base, (int, float)) else "?"
    f_ = f"{fin:.2f}" if isinstance(fin, (int, float)) else "?"
    fac = r["improvement_factor"]
    fac_s = f"{fac:.2f}x" if isinstance(fac, (int, float)) else "?"
    print(f"  baseline -> final: {b} -> {f_}   ({fac_s})")
    print(f"  experiments     : {r['experiments_run']}   "
          f"finished: {r['finished_reason']}")
    print("tried (per-field win rates)")
    if blocks["tried"]:
        for f in sorted(blocks["tried"]):
            s = blocks["tried"][f]
            print(f"  {f:<24s} {s['wins']:>3.0f}/{s['trials']:<3d} "
                  f"({100.0 * s['win_rate']:.0f}%)")
    else:
        print("  (no logged mutations)")
    print("why the best won (accepted chain)")
    for w in blocks["why"]:
        mut = ",".join(w["mutation"] or []) if w["mutation"] else "seed spec"
        s = w["score"]
        s_s = f"{s:.2f}" if isinstance(s, (int, float)) else "?"
        print(f"  [{w['step']}] {s_s}  {mut}")
    print("where it's weak (holdout diagnostics)")
    weak = blocks["weak"]
    diag = weak["diagnostics"]
    if diag is None:
        print(f"  {weak['note']}")
    else:
        print(f"  {diag['correct']}/{diag['n']} correct")
        for i, lbl in enumerate(diag["class_labels"]):
            print(f"  class {str(lbl):<12s} {diag['per_class'][i]:5.1f}% "
                  f"({diag['confusion'][i][i]}/{diag['class_counts'][i]})")
        print(f"  {weak['note']}")
    print("what's next (budget projection)")
    n = blocks["next"]
    if n["verdict"] == "insufficient":
        print(f"  insufficient history (target {n['target']:g}, "
              f"{n['points']} same-task point(s)) — need >= 2 (SPEC.md 41.1.4)")
    elif n["verdict"] == "ceiling":
        print(f"  CEILING — asymptote {n['vmax']:.2f} does not exceed the "
              f"target {n['target']:g} (SPEC.md 41.1.4)")
    else:
        print(f"  MORE — ~{n['more']} more experiment(s) to reach "
              f"{n['target']:g} (asymptote {n['vmax']:.2f}, SPEC.md 41.1.4)")
    return 0


# --- v0.29 (SPEC.md 43.2): `autorefine doctor` --------------------------------

def _smoke_train() -> tuple:
    """SPEC.md 43.2.2 (v0.29): the `run --demo` loop (41.3) at a smaller
    budget — parity-v1, 1 experiment / 20 s wall / 5 s per train, seed 7,
    search policy, un-gated — into a throwaway dir. Returns
    ("ok" | "warn" | "FAIL", line, measured wall or None)."""
    tmp = tempfile.mkdtemp(prefix="autorefine-doctor-")
    t0 = time.monotonic()
    try:
        env = AutoRefineEnv(
            task="parity-v1", seed=7,
            budget=Budget(1, 20.0, 5.0),  # 43.2.2: the tiny smoke budget
            runs_dir=tmp, policy="search", target=None,
            **search_quality_v04())
        policy = SearchPolicy(seed=7)
        state = env.reset()
        while not env.done:
            state, _r, _d, _i = env.step(policy.propose(state))
    except Exception as exc:  # a broken core is a FAIL, not a crash (43.2.2)
        return "FAIL", f"smoke train: error — {type(exc).__name__}: {exc}", None
    wall = time.monotonic() - t0
    if wall >= _SMOKE_WALL_SECONDS:
        return "warn", f"smoke train: parity-v1, 1 experiment, {wall:.2f} s " \
                       f"(>= {_SMOKE_WALL_SECONDS:g} s)", wall
    return "ok", f"smoke train: parity-v1, 1 experiment, {wall:.2f} s " \
                 f"(< {_SMOKE_WALL_SECONDS:g} s)", wall


def _probe_runs_dir(runs_dir: str) -> tuple:
    """SPEC.md 43.2.2: create the dir (parents included) and write + delete
    a probe file — a `fit` would die here, so any failure is `FAIL`."""
    p = Path(runs_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return "FAIL", f"runs dir: {runs_dir} not writable — {exc}", None
    return "ok", f"runs dir: {runs_dir} (writable)", None


def _cmd_doctor(args: argparse.Namespace) -> int:
    """`doctor` (SPEC.md 43.2, v0.29): the friendly first command —
    version / numpy / optional extras (via `find_spec`, never `import` —
    the A23 invariant), a micro parity smoke train (43.2.2), and
    `--runs-dir` writability. rc 0 when nothing is `FAIL` (warns — an
    absent extra, a slow smoke — do not fail); rc 1 on any `FAIL` (43.2.3).
    The smoke train's run dir is a `tempfile` dir, never the user's
    `--runs-dir` (doctor leaves no artifacts in the user's space)."""
    import autorefine as _af  # the 33.1 single version source
    counts = {"ok": 0, "warn": 0, "FAIL": 0}

    def emit(status: str, line: str) -> None:
        counts[status] += 1
        print(f"  {status:<4} {line}")

    print("autorefine doctor (SPEC.md 43.2)")
    emit("ok", f"autorefine {_af.__version__} "
               f"(python {sys.version.split()[0]})")
    try:
        import numpy as _np
        emit("ok", f"numpy {_np.__version__}")
    except Exception as exc:  # no core dependency available
        emit("FAIL", f"numpy — import failed: {exc}")
    # 43.2.2: the optional extras — probed, never imported (the A23
    # invariant): `find_spec` for presence, `importlib.metadata` for the
    # version (dist metadata — no module import, no sys.modules side effects)
    for mod, extra, pkg in (("PIL", "image", "Pillow"),
                            ("soundfile", "audio", "soundfile")):
        if importlib.util.find_spec(mod) is None:
            emit("warn", f"optional extra [{extra}] ({pkg}) not installed "
                         f"— pip install autorefine[{extra}]")
        else:
            try:
                version = importlib.metadata.version(pkg)
            except Exception:
                version = "?"
            emit("ok", f"optional extra [{extra}] ({pkg} {version})")
    status, line, _wall = _smoke_train()
    emit(status, line)
    status, line, _w = _probe_runs_dir(args.runs_dir)
    emit(status, line)
    print(f"result: {counts['ok']} ok, {counts['warn']} warn, "
          f"{counts['FAIL']} FAIL")
    return 1 if counts["FAIL"] else 0


# --- v0.33 (SPEC.md 47.4): `autorefine share` --------------------------------

def _share_report_html(run_dir: Path) -> str | None:
    """SPEC.md 47.4.2: the share bundle's `report.html` — ALWAYS regenerated
    in memory with the exact `report --html` call (22.2: `html_report` over
    the summary, the experiments, and the 28.2–28.4 extras) and never
    written into the run dir — so a share of an old run dir carries a
    current renderer's HTML. None when there is no summary (the caller
    already errors in that case; 47.4.4)."""
    mem = RunMemory(run_dir)
    try:
        summary = mem.load_summary()
    except FileNotFoundError:
        return None
    try:
        entries = mem.load_experiments()
    except FileNotFoundError:
        entries = []  # a summary-only dir: the report still renders (22.2)
    extras = _report_extras(summary, run_dir)  # None-safe (28.5)
    return html_report(summary, entries,
                       diagnostics=extras["diagnostics"],
                       gallery=extras["gallery"],
                       arch_svg=extras["arch_svg"])


def _cmd_share(args: argparse.Namespace) -> int:
    """`share` (SPEC.md 47.4, v0.33): bundle a finished run into one
    self-contained .zip — the "send this to a colleague" case, respecting
    the no-GUI-core stance (23.1: a file, not a server).

    Contents (47.4.2): `report.html` (always regenerated in memory, never
    written to the run dir), `summary.json`, `run_config.json` (when
    present, 37.1), `best_spec.json` (when present), and every flat
    `*.svg` in the run dir (21.2/28.2/30.1 artifacts).

    Deterministic (47.4.3, G2): the `ZipInfo` timestamp is fixed at
    1980-01-01 00:00:00, `ZIP_DEFLATED`, entries written in sorted name
    order — two shares of the same run dir are byte-identical.

    rc 0 with the entry listing (name + byte size) and the output path;
    rc 1 when the run dir is missing or has no `summary.json` (47.4.4).
    Stdlib `zipfile` only — no new dependency (SPEC.md 3).
    """
    run_dir = Path(args.run)
    if not (run_dir / "summary.json").is_file():
        print(f"no summary.json in run dir {str(run_dir)!r} — finish the run "
              f"first (SPEC.md 47.4.4)", file=sys.stderr)
        return 1
    out = (Path(args.out) if args.out is not None
           else run_dir.parent / f"{run_dir.name}-share.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, bytes] = {}
    html = _share_report_html(run_dir)  # 47.4.2: always regenerated
    if html is not None:
        payload["report.html"] = html.encode("utf-8")
    for name in ("summary.json", "run_config.json", "best_spec.json"):
        p = run_dir / name
        if p.is_file():
            payload[name] = p.read_bytes()
    for p in sorted(run_dir.glob("*.svg")):  # flat SVGs only (47.4.2)
        payload[p.name] = p.read_bytes()
    with zipfile.ZipFile(out, "w") as zf:
        for name in sorted(payload):  # 47.4.3: sorted entry order
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload[name])
    for name in sorted(payload):
        print(f"  {name:24s} {len(payload[name]):8d} bytes")
    print(f"share   : {out}")
    return 0


# --- v0.33 (SPEC.md 47.2): help polish ----------------------------------------

def _version_string() -> str:
    """SPEC.md 47.2.1: the `--version` line — the 33.1 single version
    source, imported inside the function (no module-level import cycle;
    the 43.2 `_cmd_doctor` pattern)."""
    import autorefine  # local: avoid the package-init cycle
    return f"autorefine {autorefine.__version__}"


# SPEC.md 47.2.3: the exit-code table the CLI already honors (0/1/2/130).
_TOP_EPILOG = """exit codes (SPEC.md 47.2.3):
  0    success (gate PASS where a gate applies)
  1    error (bad arguments, missing files, a resolve/probe failure)
  2    gate MISS (fit, variance, report --what-if)
  130  Ctrl-C while `watch` is polling
"""

# SPEC.md 47.2.2: the per-subcommand example blocks — the canonical
# invocations lifted from the README Quickstart, rendered verbatim by the
# RawDescriptionHelpFormatter (forwarded to every subparser).
_EP_RUN = """examples (README Quickstart):
  autorefine run --task cartpole-v1 --seed 7 --experiments 30
  autorefine run --task parity-v1 --policy bandit --seed 7 --experiments 6
  autorefine run --task parity-v1 --policy bandit --seed 7 --curriculum
  autorefine run --task sine-v1 --policy rl --rl-episodes 5
  autorefine run --demo
"""
_EP_REPORT = """examples (README Quickstart):
  autorefine report --run runs/<run_id>
  autorefine report --run runs/<run_id> --plot
  autorefine report --run runs/<run_id> --html
  autorefine report --run runs/<run_id> --json
  autorefine report --history
"""
_EP_WATCH = """examples:
  autorefine watch --run runs/<run_id>
  autorefine watch --run runs/<run_id> --tail
"""
_EP_FIT = """examples (README Quickstart):
  autorefine fit --data sales.csv --label churn --target 95.0
  autorefine fit --data examples/data/churn_sample.csv
  autorefine fit --data sales.csv --dry-run
  autorefine fit --from-run runs/<run_id>
  autorefine fit --tasks a.csv,b.csv
"""
_EP_DASHBOARD = """examples:
  autorefine dashboard            # needs pip install autorefine[gui]
"""
_EP_PLUGINS = """examples:
  autorefine plugins list
"""
_EP_EVAL = """examples:
  autorefine eval --run runs/<run_id>
"""
_EP_PREDICT = """examples:
  autorefine predict --run runs/<run_id> --row '{"x1": 0.5, "x2": -0.2}'
  autorefine predict --run runs/<run_id> --csv new_rows.csv --prob
"""
_EP_COMPARE = """examples:
  autorefine compare --run runs/<run_id_A> --run runs/<run_id_B>
"""
_EP_EXPLAIN = """examples:
  autorefine explain --run runs/<run_id>
  autorefine explain --run runs/<run_id> --json
"""
_EP_DOCTOR = """examples:
  autorefine doctor
"""
_EP_VARIANCE = """examples:
  autorefine variance --data sales.csv --seeds 5
  autorefine variance --data sales.csv --seeds 5 --json
"""
_EP_POLICY = """examples:
  autorefine policy-report --task sine-v1 --episodes 3
  autorefine policy-report --multi --tasks sine-v1,parity-v1,cartpole-v1
"""
_EP_SHARE = """examples:
  autorefine share --run runs/<run_id>
  autorefine share --run runs/<run_id> --out archive.zip
"""


def _add_config_flag(sp: argparse.ArgumentParser) -> None:
    """SPEC.md 47.1.2: the shared `--config` flag — every subcommand adds
    it exactly once (one place for the help text and the default)."""
    sp.add_argument("--config", default=None, metavar="FILE",
                    help="v0.33 (SPEC.md 47.1): read flag values from a JSON "
                         "file — a canonical run_config.json (schema "
                         "autorefine.run_config/1) or a plain JSON object of "
                         "kebab-case flag names; explicit CLI flags win over "
                         "the file, the file wins over the defaults")


def build_parser() -> argparse.ArgumentParser:
    """The full argparse tree (SPEC.md 33.2, C2): exposed separately so
    the KNOBS registry's CLI claims (which subcommand exposes which
    `--knob`) can be checked against the real parser by the A23 tests.
    Pure extraction from `main` — identical flags, no behavior change.

    SPEC.md 47.2 (v0.33): the top-level `--version` + the exit-code
    epilog, and the per-subcommand `examples:` epilog blocks (rendered
    verbatim — `RawDescriptionHelpFormatter` on the top parser, forwarded
    to every subparser)."""
    parser = argparse.ArgumentParser(
        prog="autorefine",
        description="Autonomous iterative model-improvement environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 47.2
        epilog=_TOP_EPILOG)  # 47.2.3: the exit-code table
    parser.add_argument("--version", action="version",
                        version=_version_string(),  # 47.2.1
                        help="print the version (autorefine <version>) and "
                             "exit (SPEC.md 47.2.1)")
    # 47.2.2: the epilog + RawDescriptionHelpFormatter are set on each
    # subparser below (CPython's add_subparsers does not forward them)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the autonomous improvement loop",
                           epilog=_EP_RUN,  # 47.2.2
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    p_run.add_argument("--task", default="cartpole-v1",
                       choices=sorted(TASKS), help=f"task ({', '.join(sorted(TASKS))})")
    p_run.add_argument("--policy", default="search",
                       choices=("search", "bandit", "rl"),
                       help="improver: v1 search (default), UCB field-bandit (v0.3),"
                            " or the meta-RL policy")
    p_run.add_argument("--rl-episodes", type=int, default=5,
                       help="meta-RL: number of full-budget episodes to train for")
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--experiments", type=int, default=30)
    p_run.add_argument("--max-seconds", type=float, default=900.0)
    p_run.add_argument("--max-train-seconds", type=float, default=30.0)
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--search-quality", default="v04", choices=("v04", "legacy"),
                       help="search-quality preset (SPEC.md 18): v0.4 CI/efficiency/"
                            "gen-gap acceptance (default) or the v0.3 legacy rule")
    p_run.add_argument("--ensemble-final", action="store_true",
                       help="v0.5 (SPEC.md 19.3): also report an ensemble final "
                            "score (average of the top-2 frontier models)")
    p_run.add_argument("--curriculum", action="store_true",
                       help="v0.6 (SPEC.md 20.1; v0.32 46.2): adaptive "
                            "difficulty — step the task up as the improver "
                            "saturates (parity-v1: p_flip then n_bits; "
                            "sine-v1: noise/frequency/amplitude; "
                            "cartpole-v1: initial-condition scale)")
    p_run.add_argument("--stall-patience", type=int, default=None,
                       help="v0.17 (SPEC.md 31.1): stop after K consecutive "
                            "non-improving experiments (finished_reason='stalled'); "
                            "default: off (pre-v0.17 behavior)")
    p_run.add_argument("--screen-frac", type=float, default=1.0,
                       help="v0.18 (SPEC.md 32.2): two-stage candidate "
                            "screening fraction — screen each candidate on the "
                            "first F·n rows and full-train only strict beats of "
                            "the (screened) baseline; 0 < F < 1 to enable, "
                            "default 1.0 = off (pre-v0.18 behavior)")
    p_run.add_argument("--kfold", type=int, default=0,
                       help="v0.30 (SPEC.md 44.1): score each model on K "
                            "distinct held-out subsets and average (k-fold "
                            "holdout scoring; 0 = off, the legacy single split)")
    p_run.add_argument("--demo", action="store_true",
                       help="v0.27 (SPEC.md 41.3): the narrated demo — one tiny, "
                            "deterministic parity-v1 loop (3 experiments, 60 s "
                            "wall) printed with the full decision trace; "
                            "--task/--experiments/etc. are ignored")
    p_run.add_argument("--quiet", action="store_true",
                       help="v0.33 (SPEC.md 47.3): suppress the per-experiment "
                            "loop output (keep the === summary === block); "
                            "--demo ignores it (already minimal, 41.3)")
    _add_config_flag(p_run)  # 47.1.2
    p_run.set_defaults(func=_cmd_run)

    p_rep = sub.add_parser(
        "report", help="print a run's summary and experiment log",
        epilog=_EP_REPORT,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_rep.add_argument("--run", default=None,
                       help="path to a run directory")
    # SPEC.md 38.3 (v0.24, T1): the run registry across finished runs
    p_rep.add_argument("--history", action="store_true",
                       help="v0.24 (SPEC.md 38.3): print the run registry "
                            "(one row per finished run) instead of a single "
                            "run's report; mutually exclusive with --run")
    p_rep.add_argument("--runs-dir", default="runs",
                       help="with --history: which runs dir's registry to read "
                            "(default runs)")
    p_rep.add_argument("--json", action="store_true",
                       help="v0.7 (SPEC.md 21.2): print summary.json to stdout "
                            "(machine-readable; with --plot the SVGs are still written)")
    p_rep.add_argument("--what-if", dest="what_if", action="append", default=[],
                       metavar="NAME OP THRESHOLD",
                       help="v0.26 (SPEC.md 40.2): re-gate the logged history "
                            "against a NEW objective set, e.g. 'score>=97', "
                            "'train<=30' (repeatable); 'model' is not supported; "
                            "mutually exclusive with --json/--history; rc 0 PASS / "
                            "rc 2 MISS")
    p_rep.add_argument("--project", action="store_true",
                       help="v0.27 (SPEC.md 41.1): project the budget — 'will we "
                            "reach the target?' — by fitting a saturating curve "
                            "to the same-task history (registry + this run); "
                            "informational, rc 0; mutually exclusive with "
                            "--json/--history/--what-if/--trace")
    p_rep.add_argument("--target", type=float, default=95.0,
                       help="with --project: the score target to project to "
                            "(default 95.0, like fit's gate)")
    p_rep.add_argument("--trace", action="store_true",
                       help="v0.27 (SPEC.md 41.2): terminal replay — one "
                            "decision line per experiments.jsonl entry "
                            "(candidate, mutation, score, accepted/rejected + "
                            "reason); mutually exclusive with --json/--history/"
                            "--what-if/--project")
    p_rep.add_argument("--importance", action="store_true",
                       help="v0.30 (SPEC.md 44.2): permutation feature "
                            "importance over the holdout — which columns the "
                            "winning model uses (one extra pass, zero training); "
                            "mutually exclusive with --json/--history/--what-if/"
                            "--project/--trace")
    p_rep.add_argument("--importance-repeats", type=int, default=5,
                       help="with --importance: shuffles per column "
                            "(default 5, SPEC.md 44.2)")
    p_rep.add_argument("--plot", action="store_true",
                       help="v0.7 (SPEC.md 21.2): ASCII charts on stdout + "
                            "score_curve.svg / pareto_frontier.svg in the run dir")
    p_rep.add_argument("--html", action="store_true",
                       help="v0.8 (SPEC.md 22.2): write a self-contained "
                            "report.html (embedded SVGs + tables) into the run dir")
    _add_config_flag(p_rep)  # 47.1.2
    p_rep.set_defaults(func=_cmd_report)

    p_watch = sub.add_parser(
        "watch", help="v0.25 (SPEC.md 39.1): tail a run's experiments.jsonl "
                      "live (re-render ASCII charts, or --tail JSON lines for CI)",
        epilog=_EP_WATCH,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_watch.add_argument("--run", required=True,
                         help="path to a run directory (may not exist yet)")
    p_watch.add_argument("--tail", action="store_true",
                         help="machine mode: emit one compact JSON line per "
                              "newly-logged row (for CI / external tools)")
    p_watch.add_argument("--interval", type=float, default=0.5,
                         help="poll period in seconds (default 0.5)")
    p_watch.add_argument("--max-polls", type=int, default=0,
                         help="safety cap on poll iterations (0 = unlimited; "
                              "a small N bounds a stuck wait and returns rc 1)")
    p_watch.add_argument("--clear", action="store_true",
                         help="human mode: ANSI clear-and-home before each "
                              "frame (live refresh); off by default")
    _add_config_flag(p_watch)  # 47.1.2
    p_watch.set_defaults(func=_cmd_watch)

    p_fit = sub.add_parser(
        "fit", help="v0.8 (SPEC.md 22.1) / v0.10 (24.5): fit a model on your data "
                    "(CSV file, or a directory of labelled images/audio) and "
                    "gate on a target",
        epilog=_EP_FIT,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_fit.add_argument("--data", default=None,
                       help="CSV file, or a directory of labelled images/audio "
                            "(v0.10, SPEC.md 24.5); or pass --from-run to re-run "
                            "a finished run (mutually exclusive, SPEC.md 37.1.4)")
    p_fit.add_argument("--from-run", dest="from_run", default=None,
                       help="v0.23 (SPEC.md 37.1.4): re-run a finished run exactly "
                            "from its run_config (mutually exclusive with --data)")
    p_fit.add_argument("--tasks", default=None,
                       metavar="CSV1,CSV2,...",
                       help="v0.32 (SPEC.md 46.3): portfolio mode — one shared "
                            "budget over several CSV files, run in the given "
                            "order (at least 2; mutually exclusive with --data "
                            "and --from-run; rc 0 iff every task passes its "
                            "gate)")
    p_fit.add_argument("--metric", default="accuracy",
                       choices=("accuracy", "logloss"),
                       help="v0.32 (SPEC.md 46.1): the CSV task metric — "
                            "accuracy (default, the §22.1 contract) or "
                            "logloss (softmax-head CSV files only; score = "
                            "100·(1 − mean NLL/ln 2), clamped at 0)")
    p_fit.add_argument("--dry-run", dest="dry_run", action="store_true",
                       help="v0.26 (SPEC.md 40.1): resolve data -> task -> head -> "
                            "split and print the plan + wall-time estimate, then exit "
                            "without training or artifacts (mutually exclusive with "
                            "--from-run)")
    p_fit.add_argument("--gate", action="append", default=[],
                       metavar="NAME OP THRESHOLD",
                       help="v0.23 (SPEC.md 37.2): an acceptance objective, e.g. "
                            "'score>=95', 'train<=30', 'model<=100000' "
                            "(repeatable); none → the §22.1 score gate")
    p_fit.add_argument("--task", default="auto",
                       choices=("auto", "csv", "image", "audio", "text"),
                       help="v0.10 (SPEC.md 24.5; v0.31 45.2.2): force the data "
                            "task; default auto (file -> csv, directory -> detected)")
    p_fit.add_argument("--label", default=None,
                       help="label column (default: label/target/y/class, else last column)")
    p_fit.add_argument("--split", dest="split_frac", type=float, default=0.2,
                       help="non-train share of the rows, split evenly into holdout/gen")
    p_fit.add_argument("--temporal", action="store_true", default=False,
                       help="v0.31 (SPEC.md 45.1): walk-forward split for CSV "
                            "data — train = first rows, holdout = next, gen = "
                            "last (file order; default: the random seed split)")
    p_fit.add_argument("--target", type=float, default=95.0,
                       help="your acceptance bar on final_best_score (score points)")
    p_fit.add_argument("--policy", default="bandit",
                       choices=("search", "bandit", "rl"),
                       help="improver: UCB field-bandit (default), v1 search, or meta-RL")
    p_fit.add_argument("--rl-episodes", type=int, default=5,
                       help="meta-RL: number of full-budget episodes to train for")
    p_fit.add_argument("--seed", type=int, default=7)
    p_fit.add_argument("--experiments", type=int, default=30)
    p_fit.add_argument("--max-seconds", type=float, default=900.0)
    p_fit.add_argument("--max-train-seconds", type=float, default=30.0)
    p_fit.add_argument("--runs-dir", default="runs")
    p_fit.add_argument("--search-quality", default="v04", choices=("v04", "legacy"),
                       help="search-quality preset (SPEC.md 18)")
    p_fit.add_argument("--ensemble-final", action="store_true",
                       help="v0.5 (SPEC.md 19.3): also report an ensemble final score")
    p_fit.add_argument("--stall-patience", type=int, default=None,
                       help="v0.17 (SPEC.md 31.1): stop after K consecutive "
                            "non-improving experiments (finished_reason='stalled'); "
                            "default: off (pre-v0.17 behavior)")
    p_fit.add_argument("--screen-frac", type=float, default=1.0,
                       help="v0.18 (SPEC.md 32.2): two-stage candidate "
                            "screening fraction (0 < F < 1 to enable; default "
                            "1.0 = off, pre-v0.18 behavior)")
    p_fit.add_argument("--kfold", type=int, default=0,
                       help="v0.30 (SPEC.md 44.1): score each model on K "
                            "distinct held-out subsets and average (k-fold "
                            "holdout scoring; 0 = off, the legacy single split)")
    p_fit.add_argument("--quiet", action="store_true",
                       help="v0.33 (SPEC.md 47.3): suppress the per-experiment "
                            "loop output and the per-class diagnostics (keep "
                            "the gate verdict + the === summary === block); "
                            "--dry-run ignores it (already minimal, 40.1)")
    _add_config_flag(p_fit)  # 47.1.2
    p_fit.set_defaults(func=_cmd_fit)

    p_dash = sub.add_parser(
        "dashboard", help="v0.9 (SPEC.md 23.4): launch the Streamlit app "
                          "(pip install autorefine[gui])",
        epilog=_EP_DASHBOARD,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_dash.add_argument("--port", type=int, default=8501,
                        help="streamlit server port (default 8501)")
    p_dash.add_argument("--runs-dir", default="runs",
                        help="where the dashboard's runs land (default runs)")
    _add_config_flag(p_dash)  # 47.1.2
    p_dash.set_defaults(func=_cmd_dashboard)

    p_plug = sub.add_parser("plugins", help="plugin entry points (SPEC.md 21.3)",
                            epilog=_EP_PLUGINS,  # 47.2.2
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    p_plug_sub = p_plug.add_subparsers(dest="plug_cmd", required=True)
    p_plug_sub.add_parser("list", help="report discovered autorefine.* entry points")
    _add_config_flag(p_plug)  # 47.1.2
    p_plug.set_defaults(func=_cmd_plugins_list)

    p_eval = sub.add_parser(
        "eval", help="re-score a run's best model on fresh splits",
        epilog=_EP_EVAL,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_eval.add_argument("--run", required=True, help="path to a run directory")
    p_eval.add_argument("--episodes", type=int, default=200)
    _add_config_flag(p_eval)  # 47.1.2
    p_eval.set_defaults(func=_cmd_eval)

    p_pred = sub.add_parser(
        "predict", help="v0.28 (SPEC.md 42.1): score NEW rows with a run's "
                        "best model (--row/--csv/--stdin/--item; fitting tasks)",
        epilog=_EP_PREDICT,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_pred.add_argument("--run", required=True, help="path to a run directory")
    p_pred.add_argument("--row", default=None,
                        help="one row as JSON: an object keyed by feature "
                             "name (case-insensitive) or an array of exactly "
                             "state_dim numbers")
    p_pred.add_argument("--csv", default=None,
                        help="a CSV of rows — a header line is matched by "
                             "name (extra columns OK), else positional with "
                             "exactly state_dim cells")
    p_pred.add_argument("--stdin", action="store_true",
                        help="rows as JSON lines on stdin (object or array "
                             "per line); zero lines is an error")
    p_pred.add_argument("--item", action="append", default=[],
                        metavar="FILE",
                        help="one media file (image/audio tasks); repeatable")
    p_pred.add_argument("--prob", action="store_true",
                        help="v0.32 (SPEC.md 46.1): also print the calibrated "
                             "per-class probabilities for every row (softmax-head "
                             "models only; one '<row>\t<class>: p=…' line per "
                             "class, in class order)")
    p_pred.add_argument("--json", action="store_true",
                        help="print the predictions as a pure JSON array "
                             "([{row, prediction, probabilities}, ...])")
    _add_config_flag(p_pred)  # 47.1.2
    p_pred.set_defaults(func=_cmd_predict)

    p_cmp = sub.add_parser(
        "compare", help="v0.28 (SPEC.md 42.2): the app's run-diff (38.4) in "
                        "the terminal — score delta + best_spec field diff",
        epilog=_EP_COMPARE,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_cmp.add_argument("--run", dest="runs", action="append", required=True,
                       metavar="DIR",
                       help="two run dirs, in order A then B (repeat --run)")
    p_cmp.add_argument("--json", action="store_true",
                       help="print the diff_two_summaries dict (pure JSON)")
    _add_config_flag(p_cmp)  # 47.1.2
    p_cmp.set_defaults(func=_cmd_compare)

    p_exp = sub.add_parser(
        "explain", help="v0.28 (SPEC.md 42.3): a one-screen narrative — "
                        "result / tried / why / weak / next",
        epilog=_EP_EXPLAIN,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_exp.add_argument("--run", required=True, help="path to a run directory")
    p_exp.add_argument("--target", type=float, default=95.0,
                       help="the score target for the next-block projection "
                            "(default 95.0, like fit's gate)")
    p_exp.add_argument("--json", action="store_true",
                       help="print {result, tried, why, weak, next} as pure JSON")
    _add_config_flag(p_exp)  # 47.1.2
    p_exp.set_defaults(func=_cmd_explain)

    p_doc = sub.add_parser(
        "doctor", help="v0.29 (SPEC.md 43.2): a friendly first command — "
                       "version / numpy / extras / smoke train / runs-dir",
        epilog=_EP_DOCTOR,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_doc.add_argument("--runs-dir", default="runs",
                       help="the runs dir to probe for writability (default runs)")
    _add_config_flag(p_doc)  # 47.1.2
    p_doc.set_defaults(func=_cmd_doctor)

    p_var = sub.add_parser(
        "variance", help="v0.15 (SPEC.md 29.1): the same budget under N seeds — "
                         "\"is the improvement real?\" (seed-variance box plot)",
        epilog=_EP_VARIANCE,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_var.add_argument("--data", required=True,
                       help="CSV file, or a directory of labelled images/audio/text")
    p_var.add_argument("--task", default="auto",
                       choices=("auto", "csv", "image", "audio", "text"),
                       help="force the data task; default auto (file -> csv, "
                            "directory -> detected; v0.31 45.2.2)")
    p_var.add_argument("--label", default=None,
                       help="label column (default: auto-detect)")
    p_var.add_argument("--split", dest="split_frac", type=float, default=0.2)
    p_var.add_argument("--target", type=float, default=95.0)
    p_var.add_argument("--seeds", type=int, required=True,
                       help="N distinct seeds (seed, seed+1, ... seed+N-1)")
    p_var.add_argument("--seed", type=int, default=7, help="base seed (default 7)")
    p_var.add_argument("--policy", default="bandit", choices=("search", "bandit"),
                       help="improver: UCB field-bandit (default) or v1 search")
    p_var.add_argument("--experiments", type=int, default=30)
    p_var.add_argument("--max-seconds", type=float, default=900.0)
    p_var.add_argument("--max-train-seconds", type=float, default=30.0)
    p_var.add_argument("--runs-dir", default="runs")
    p_var.add_argument("--search-quality", default="v04", choices=("v04", "legacy"),
                       help="search-quality preset (SPEC.md 18)")
    p_var.add_argument("--stall-patience", type=int, default=None,
                       help="v0.17 (SPEC.md 31.1): each sweep seed stops after K "
                            "consecutive non-improving experiments (per-seed "
                            "finished_reason='stalled'); default: off")
    p_var.add_argument("--workers", type=int, default=1,
                       help="v0.17 (SPEC.md 31.2): worker processes for the seed "
                            "sweep (default 1 = the exact serial path; per-seed "
                            "results bit-identical either way)")
    p_var.add_argument("--json", action="store_true",
                       help="print the sweep as pure JSON (SPEC.md 21.2)")
    p_var.add_argument("--gate", action="append", default=[],
                       metavar="NAME OP THRESHOLD",
                       help="v0.23 (SPEC.md 37.2): per-seed acceptance objective, "
                            "e.g. 'score>=95', 'train<=30', 'model<=100000' "
                            "(repeatable); none → the §22.1 score gate")
    _add_config_flag(p_var)  # 47.1.2
    p_var.set_defaults(func=_cmd_variance)

    p_pol = sub.add_parser(
        "policy-report", help="v0.15 (SPEC.md 29.2): train the meta-RL policy and "
                              "render the action-probability + task-return views",
        epilog=_EP_POLICY,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_pol.add_argument("--task", default="sine-v1", choices=sorted(TASKS),
                       help="task (single mode)")
    p_pol.add_argument("--seed", type=int, default=7)
    p_pol.add_argument("--episodes", type=int, default=3)
    p_pol.add_argument("--experiments", type=int, default=20)
    p_pol.add_argument("--max-seconds", type=float, default=900.0)
    p_pol.add_argument("--max-train-seconds", type=float, default=30.0)
    p_pol.add_argument("--runs-dir", default="runs")
    p_pol.add_argument("--multi", action="store_true",
                       help="multi-task (SPEC.md 20.2): one env per --tasks entry")
    p_pol.add_argument("--tasks", default="sine-v1,parity-v1,cartpole-v1",
                       help="comma-separated task names (with --multi)")
    p_pol.add_argument("--top-k", type=int, default=8, help="top-K actions per row")
    p_pol.add_argument("--n-steps", type=int, default=12, help="rows shown (last N steps)")
    _add_config_flag(p_pol)  # 47.1.2
    p_pol.set_defaults(func=_cmd_policy_report)

    p_share = sub.add_parser(
        "share", help="v0.33 (SPEC.md 47.4): bundle a finished run into one "
                      "deterministic, self-contained .zip (report.html + JSON "
                      "artifacts + SVGs)",
        epilog=_EP_SHARE,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_share.add_argument("--run", required=True, help="path to a run directory")
    p_share.add_argument("--out", default=None, metavar="FILE",
                         help="output .zip path (default: <run_dir>-share.zip "
                              "next to the run dir)")
    _add_config_flag(p_share)  # 47.1.2
    p_share.set_defaults(func=_cmd_share)

    return parser


def _config_value(action: argparse.Action, value, key: str):
    """SPEC.md 47.1.3: translate one JSON config value into the flag's value.

    `store_true`/`store_false` <- a JSON boolean; `append` flags (<nargs='*'>)
    <- a JSON list (a bare value is wrapped); typed flags <- the parser
    action's own `type`; `choices` are validated. Raises `ValueError` citing
    `--<key>` on any bad conversion (the caller prints it + exits 1)."""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise ValueError(f"--{key}: expected a JSON boolean, got {value!r}")
        return bool(value)
    if action.nargs == "*":  # append flags such as --gate / --item
        vals = value if isinstance(value, (list, tuple)) else [value]
        conv = action.type if action.type is not None else (lambda v: v)
        return [v if isinstance(v, str) else conv(str(v)) for v in vals]
    if action.type is not None:
        try:
            return action.type(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"--{key}: {value!r} is not a valid value "
                             f"({exc})") from exc
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"--{key}: {value!r} is not one of "
                         f"{sorted(action.choices)}")
    return value


def _apply_config(parser: argparse.ArgumentParser, args, argv: list[str]):
    """SPEC.md 47.1.1: apply the `--config` file to the parsed namespace.

    A file whose `schema` equals `RUN_CONFIG_SCHEMA` is a canonical
    run_config.json and is translated via `RunConfig.from_dict` +
    `runconfig_to_flags` (47.1.4); any other object is taken as plain
    kebab-case flag names. Precedence (47.1.5): explicit CLI flags > the
    config file > the parser defaults (an `--` token anywhere in `argv`
    wins, `--flag=value` included). Returns a non-zero exit code on any
    error (already printed to stderr), `None` on success."""
    path = args.config
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        print(f"error: --config {path}: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: --config {path}: not valid JSON ({exc})", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print(f"error: --config {path}: the top level must be a JSON object",
              file=sys.stderr)
        return 1
    if data.get("schema") == RUN_CONFIG_SCHEMA:  # 47.1.4: canonical recipe
        try:
            flags = runconfig_to_flags(RunConfig.from_dict(data))
        except ValueError as exc:
            print(f"error: --config {path}: {exc}", file=sys.stderr)
            return 1
    else:
        flags = data
    sub_action = next(a for a in parser._actions
                      if isinstance(a, argparse._SubParsersAction))
    sp = sub_action.choices[args.cmd]
    opts = {}
    for action in sp._actions:
        for opt in getattr(action, "option_strings", ()):
            if opt.startswith("--"):
                opts[opt[2:]] = action
    explicit = {tok[2:].split("=", 1)[0] for tok in argv
                if isinstance(tok, str) and tok.startswith("--")}
    for key, value in flags.items():
        action = opts.get(key)
        if action is None:  # 47.1.4: canonical configs pair with their command
            valid = ", ".join(f"--{k}" for k in sorted(opts))
            print(f"error: --config {path}: --{key} is not a flag of the "
                  f"'{args.cmd}' command (valid: {valid}); a canonical "
                  f"run_config.json pairs with the command that owns its "
                  f"flags (SPEC.md 47.1.4)", file=sys.stderr)
            return 1
        if key in explicit:
            continue  # 47.1.5: explicit CLI flags win
        try:
            setattr(args, action.dest, _config_value(action, value, key))
        except ValueError as exc:
            print(f"error: --config {path}: {exc}", file=sys.stderr)
            return 1
    return None


def main(argv: list[str] | None = None) -> int:
    # SPEC.md 21.3: external packages may contribute tasks via entry points;
    # register them before the parser is built so `--task` choices include them
    try:
        register_tasks()
    except PluginError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if argv is None:
        argv = list(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config", None) is not None:  # 47.1: optional config file
        rc = _apply_config(parser, args, list(argv))
        if rc is not None:
            return rc
    return int(args.func(args))
