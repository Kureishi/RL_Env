"""Command-line interface (SPEC.md 11).

  python -m autorefine run    --task cartpole-v1 --seed 7 --experiments 30
  python -m autorefine report --run runs/<run_id>
  python -m autorefine eval   --run runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import Budget, ModelSpec
from .diagnostics import holdout_diagnostics
from .gate import (
    actuals_from_run,
    default_objectives,
    evaluate,
    parse_objective,
)
from .runconfig import RunConfig, format_recipe
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.bandit import BanditPolicy
from .improver.curriculum import ParityCurriculum
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
from .plugins import (
    POLICIES_GROUP,
    TASKS_GROUP,
    PluginError,
    discover,
    register_tasks,
)
from .tasks import TASKS, ParityTask
from .models.mlp import MLP
from .evaluator import evaluate_full


def _drive(args: argparse.Namespace, env: AutoRefineEnv) -> None:
    """Shared improver driver for `run` and `fit` (SPEC.md 22.1)."""
    if args.policy == "rl":
        # SPEC.md 15: meta-RL improver — policy trained on AutoRefineEnv itself
        policy = MetaRLPolicy(seed=args.seed)
        print(f"task    : {args.task}  (meta-RL, {args.rl_episodes} full-budget episodes)")
        rl_summary = train_policy(env, policy, args.rl_episodes, verbose=True)
        print(f"rl episodes: {rl_summary['episodes']}, returns: "
              f"{[round(r, 4) for r in rl_summary['episode_returns']]}")
        summary = env.memory.load_summary() if env.memory else {}
    else:
        state = env.reset()
        print(f"run dir : {env.run_dir}")
        print(f"baseline: {state['baseline_score']:.2f}")

        # SPEC.md 17: the UCB field-bandit is a second reference policy
        policy = (SearchPolicy(seed=args.seed) if args.policy == "search"
                  else BanditPolicy(seed=args.seed))
        while not env.done:
            action = policy.propose(state)
            state, reward, done, info = env.step(action)
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
        if args.task != "parity-v1":
            print("--curriculum currently supports parity-v1 only "
                  "(SPEC.md 20.1)", file=sys.stderr)
            return 1
        curriculum = ParityCurriculum(seed=args.seed)
    env = AutoRefineEnv(
        task=args.task, seed=args.seed, budget=_budget(args), runs_dir=args.runs_dir,
        ensemble_top_k=2 if args.ensemble_final else 0,
        curriculum=curriculum,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; None = off)
        screen_frac=args.screen_frac,  # v0.18 (SPEC.md 32.2; 1.0 = off)
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
    """v0.10 (SPEC.md 24.5): `--data` file → csv; directory → image/audio.

    `want` is the `--task` value ('auto' | 'csv' | 'image' | 'audio');
    auto-detection counts modality extensions in the directory's subfolders.
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
                    f"{data} holds both image and audio items: pass "
                    f"--task image or --task audio (SPEC.md 24.5)")
            if m is None:
                raise ValueError(
                    f"no image or audio items under {data} — one subfolder "
                    f"per class, or an index.csv (SPEC.md 24.2); or pass a "
                    f"CSV file with --task csv")
            return m
        if want in ("image", "audio"):
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
    — mutually exclusive with `--data`.
    """
    from .tasks import TASKS  # noqa: F401 (registry; the helpers use it)
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
    rc = _fit_gate(args, env, bar, metric)
    # C2/C3 (SPEC.md 28.2/28.3): after the gate line, per-class accuracy +
    # weakest-class hint (and the media error gallery)
    _print_fit_diagnostics(env)
    return rc


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
    print(f"  recipe     : task {task_name} | seed {args.seed} | policy "
          f"{args.policy} | target {args.target:g}")
    print(f"  ensemble   : top-{2 if args.ensemble_final else 0}   "
          f"stall patience {args.stall_patience}   "
          f"screen frac {args.screen_frac:g}")
    print(f"  task       : head {probe.head} | metric {metric} "
          f"| n_outputs {probe.n_outputs} | state_dim {probe.state_dim}")
    if dsz is not None:
        print(f"  dataset    : {dsz} points (the train split)")
    else:
        print("  dataset    : task default size")
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
    if summary.get("curriculum"):  # SPEC.md 20.1: a curriculum level name
        lvl = summary["curriculum"]["levels"][-1]
        return ParityTask(seed=seed, n_bits=lvl["n_bits"], p_flip=lvl["p_flip"])
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


def build_parser() -> argparse.ArgumentParser:
    """The full argparse tree (SPEC.md 33.2, C2): exposed separately so
    the KNOBS registry's CLI claims (which subcommand exposes which
    `--knob`) can be checked against the real parser by the A23 tests.
    Pure extraction from `main` — identical flags, no behavior change."""
    parser = argparse.ArgumentParser(prog="autorefine",
                                     description="Autonomous iterative model-improvement environment")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the autonomous improvement loop")
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
                       help="v0.6 (SPEC.md 20.1): adaptive difficulty — step the "
                            "parity task up (p_flip, then n_bits) as the "
                            "improver saturates")
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
    p_run.add_argument("--demo", action="store_true",
                       help="v0.27 (SPEC.md 41.3): the narrated demo — one tiny, "
                            "deterministic parity-v1 loop (3 experiments, 60 s "
                            "wall) printed with the full decision trace; "
                            "--task/--experiments/etc. are ignored")
    p_run.set_defaults(func=_cmd_run)

    p_rep = sub.add_parser("report", help="print a run's summary and experiment log")
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
    p_rep.add_argument("--plot", action="store_true",
                       help="v0.7 (SPEC.md 21.2): ASCII charts on stdout + "
                            "score_curve.svg / pareto_frontier.svg in the run dir")
    p_rep.add_argument("--html", action="store_true",
                       help="v0.8 (SPEC.md 22.2): write a self-contained "
                            "report.html (embedded SVGs + tables) into the run dir")
    p_rep.set_defaults(func=_cmd_report)

    p_watch = sub.add_parser(
        "watch", help="v0.25 (SPEC.md 39.1): tail a run's experiments.jsonl "
                      "live (re-render ASCII charts, or --tail JSON lines for CI)")
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
    p_watch.set_defaults(func=_cmd_watch)

    p_fit = sub.add_parser(
        "fit", help="v0.8 (SPEC.md 22.1) / v0.10 (24.5): fit a model on your data "
                    "(CSV file, or a directory of labelled images/audio) and "
                    "gate on a target")
    p_fit.add_argument("--data", default=None,
                       help="CSV file, or a directory of labelled images/audio "
                            "(v0.10, SPEC.md 24.5); or pass --from-run to re-run "
                            "a finished run (mutually exclusive, SPEC.md 37.1.4)")
    p_fit.add_argument("--from-run", dest="from_run", default=None,
                       help="v0.23 (SPEC.md 37.1.4): re-run a finished run exactly "
                            "from its run_config (mutually exclusive with --data)")
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
    p_fit.add_argument("--task", default="auto", choices=("auto", "csv", "image", "audio"),
                       help="v0.10 (SPEC.md 24.5): force the data task; default auto "
                            "(file → csv, directory → detected)")
    p_fit.add_argument("--label", default=None,
                       help="label column (default: label/target/y/class, else last column)")
    p_fit.add_argument("--split", dest="split_frac", type=float, default=0.2,
                       help="non-train share of the rows, split evenly into holdout/gen")
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
    p_fit.set_defaults(func=_cmd_fit)

    p_dash = sub.add_parser(
        "dashboard", help="v0.9 (SPEC.md 23.4): launch the Streamlit app "
                          "(pip install autorefine[gui])")
    p_dash.add_argument("--port", type=int, default=8501,
                        help="streamlit server port (default 8501)")
    p_dash.add_argument("--runs-dir", default="runs",
                        help="where the dashboard's runs land (default runs)")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_plug = sub.add_parser("plugins", help="plugin entry points (SPEC.md 21.3)")
    p_plug_sub = p_plug.add_subparsers(dest="plug_cmd", required=True)
    p_plug_sub.add_parser("list", help="report discovered autorefine.* entry points")
    p_plug.set_defaults(func=_cmd_plugins_list)

    p_eval = sub.add_parser("eval", help="re-score a run's best model on fresh splits")
    p_eval.add_argument("--run", required=True, help="path to a run directory")
    p_eval.add_argument("--episodes", type=int, default=200)
    p_eval.set_defaults(func=_cmd_eval)

    p_var = sub.add_parser(
        "variance", help="v0.15 (SPEC.md 29.1): the same budget under N seeds — "
                         "\"is the improvement real?\" (seed-variance box plot)")
    p_var.add_argument("--data", required=True,
                       help="CSV file, or a directory of labelled images/audio")
    p_var.add_argument("--task", default="auto", choices=("auto", "csv", "image", "audio"),
                       help="force the data task; default auto (file -> csv, directory -> detected)")
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
    p_var.set_defaults(func=_cmd_variance)

    p_pol = sub.add_parser(
        "policy-report", help="v0.15 (SPEC.md 29.2): train the meta-RL policy and "
                              "render the action-probability + task-return views")
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
    p_pol.set_defaults(func=_cmd_policy_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    # SPEC.md 21.3: external packages may contribute tasks via entry points;
    # register them before the parser is built so `--task` choices include them
    try:
        register_tasks()
    except PluginError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
