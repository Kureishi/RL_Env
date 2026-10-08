"""Command-line interface (SPEC.md 11).

  python -m autorefine run    --task cartpole-v1 --seed 7 --experiments 30
  python -m autorefine report --run runs/<run_id>
  python -m autorefine eval   --run runs/<run_id>
  python -m autorefine doctor [--runs-dir runs]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # core dependency (used by the 62.3 domain ECE path)

from .config import Budget, ModelSpec
from .diagnostics import holdout_difficulty, holdout_diagnostics
from .gate import (
    actuals_from_run,
    default_objectives,
    evaluate,
    latency_line,
    measure_ms_per_row,
    parse_objective,
)
from .runconfig import (
    RUN_CONFIG_SCHEMA,
    RunConfig,
    format_recipe,
    runconfig_to_flags,
)
from .dashboard import diff_n_summaries, diff_two_summaries, field_stats  # 42.2/42.3 (v0.28) + 50.1.4 (v0.36)
from .sharing import share_payload, write_share_zip  # 47.4/50.2 (v0.36)
from .dossier import build_dossier  # 58.3 (v0.44): the run dossier
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.bandit import BanditPolicy
from .improver.curriculum import (  # 46.2 (v0.32): the three ladders
    CartPoleCurriculum,
    ParityCurriculum,
    SineCurriculum,
)
from .improver.policy import SearchPolicy
from .improver.rl_policy import (
    MetaRLPolicy,
    train_policy,
    save_policy,  # 67.1 (v0.53, A57): policy persistence
    load_policy,
)
from .steering import SteeringState, train_manual  # 59 (v0.45)
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
    html_cover,  # 56.2 (v0.42): the report cover block
    html_report,
    svg_architecture,
    svg_confusion_matrix,
    svg_ladder_curve,
    svg_pareto,
    svg_per_class_bars,
    svg_quality_over_time,  # 89.8 (v0.75, C1)
    svg_score_curve,
)
from .predict import (  # 42.1 (v0.28): the predict leaf
    csv_rows_to_features,
    media_item_features,
    predict_features,
    row_to_features,
    standardize,
)
from .provenance import (  # 56.1 (v0.42): the provenance certificate
    env_provenance,
    provenance_card,
    provenance_payload,
)
from .quickstart import (  # 65 (v0.51): the Quickstart (one source, 65.1)
    QUICKSTART_INTRO,
    quickstart_steps,
    render_quickstart,
)
from .audience import (  # 62 (v0.48, B1): the audience axis + verify
    AUDIENCES,
    build_view,
    data_fingerprint,
    html_view,
    render_verify,
    render_view,
    verify_run,
)
from .reporting import (  # 63 (v0.49, B2/B3/B4) + 64.1 (v0.50, B5)
    REPORT_FORMATS,
    benchmark_view,
    build_report_doc,
    decision_view,
    freshness_view,  # 89.10 (v0.75, C3)
    render_benchmark,
    render_benchmark_md,
    render_decision,
    render_freshness,  # 89.10 (v0.75, C3)
    render_report,
    render_report_pdf,
    render_user_guide,
    user_guide_view,
)
from .uncertainty import headline_uncertainty  # 64.2 (v0.50, B6): the ± read
from .goal import (  # 87.1/87.3 (v0.73, A1/A3) + 88.2-88.5 (v0.74, A2/B4/B5)
    ask_command,
    auto_target,
    experiments_for_time,
    extract_data_path,
    fair_bar_line,
    go_budget,
    measured_per_exp,
    resolve_goal,
)
from .vocab import plain_free  # 87.4 (v0.73, B1): the jargon guard
from .briefing import plain_verdict, so_what, why_this_model  # 87.5-87.6 (v0.73) + 88.9 (v0.74, D9)
from .discover import label_column, pick_data  # 88.1 (v0.74, A1): the path-less go
from .datasets import (  # 89.4/89.7 (v0.75, A4/B3): snapshots + windows
    DATE_COL_NAMES,
    find_date_col,
    find_snapshot,
    hash_file,
    load_snapshots,
    parse_window,
    snapshot_dataset,
    window_rows,
)
from .ingest import (  # 89.1/89.2/89.3/89.6 (v0.75, A1/A2/A3/B2)
    append_event,
    drift_verdict,
    hash_csv,
    load_drop_state,
    load_events,
    load_state,
    merge_tables,
    poll,
    refresh_verdict,
    save_drop_state,
    save_state,
)
from .modelcard import (  # 87.8 (v0.73, C2)
    model_card,
    render_model_card,
    render_model_card_md,
)
from .calibration import ece as _ece  # 61.4: the domain-view calibration metric
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
        # SPEC.md 67.4 (v0.53, A57): `run`-only persistence + exploration —
        # `--rl-load` resumes a saved policy (learn → persist → explore →
        # reuse); `--rl-epsilon`/`--rl-epsilon-decay` enable ε-greedy;
        # `--rl-save` writes the trained policy after training.
        if getattr(args, "rl_load", None):
            policy = load_policy(args.rl_load, seed=args.seed)
        else:
            policy = MetaRLPolicy(
                seed=args.seed,
                epsilon=getattr(args, "rl_epsilon", 0.0),
                epsilon_decay=getattr(args, "rl_epsilon_decay", 1.0))
        if not quiet:
            print(f"task    : {args.task}  (meta-RL, {args.rl_episodes} full-budget episodes"
                  + (f", ε={policy.epsilon:g}" if policy.epsilon > 0.0 else "")
                  + (
                      f" (loaded from {args.rl_load})"
                      if getattr(args, "rl_load", None) else ""))
        rl_summary = train_policy(env, policy, args.rl_episodes,
                                  verbose=not quiet)  # 47.3.2
        if not quiet:
            print(f"rl episodes: {rl_summary['episodes']}, returns: "
                  f"{[round(r, 4) for r in rl_summary['episode_returns']]}")
        if getattr(args, "rl_save", None):  # 67.4: persist the trained policy
            sp = save_policy(policy, args.rl_save)
            if not quiet:
                print(f"rl save   : policy written to {sp}")
        summary = env.memory.load_summary() if env.memory else {}
    else:
        state = env.reset()
        if not quiet:  # 47.3.2: the loop's stdout is the quiet surface
            print(f"run dir : {env.run_dir}")
            print(f"baseline: {state['baseline_score']:.2f}")

        # SPEC.md 17: the UCB field-bandit is a second reference policy
        # SPEC.md 59.2 (v0.45): steering pins drop their field from the
        # proposal pool (the env's force-set in step() is the backstop)
        excl = tuple(f for f, _v in env.steering.pins) if env.steering else ()
        # SPEC.md 86.4.2 (v0.72): the env's search prior seeds the bandit's
        # field beliefs (None = the exact legacy proposal stream, G2); the
        # search/RL policies own no per-field bookkeeping (86.4.2)
        prior = (env.prior.to_bandit_prior()
                 if getattr(env, "prior", None) is not None else None)
        policy = (SearchPolicy(seed=args.seed, exclude_fields=excl)
                  if args.policy == "search"
                  else BanditPolicy(seed=args.seed, exclude_fields=excl,
                                    prior=prior))
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


def _steering_value(text: str):
    """SPEC.md 59.4 (v0.45): one steering value from the CLI — a JSON token
    (a number, ``[2]``, ``[16, 8]``, a quoted string) parses to that value;
    a bare word (e.g. ``mlp``, ``cosine``) stays the raw string."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _steering_token(raw: str, verb: str) -> "tuple[str, object]":
    """SPEC.md 59.4 (v0.45): ``FIELD=VALUE`` → ``(field, value)``; a missing
    ``=`` or an empty field name is a `ValueError` naming the flag (loud)."""
    if "=" not in raw:
        raise ValueError(
            f"--{verb} {raw!r}: expected FIELD=VALUE (SPEC.md 59.2)")
    field, value = raw.split("=", 1)
    field = field.strip()
    if not field:
        raise ValueError(
            f"--{verb} {raw!r}: the field name is empty (SPEC.md 59.2)")
    return field, _steering_value(value)


def _steering_constrain_token(raw: str) -> "tuple[str, list]":
    """SPEC.md 59.4 (v0.45): ``FIELD=V1,V2`` → ``(field, [v1, v2])``; the set
    is comma-split into scalar values (architecture-tuple values are the
    app/Python API surface — the comma split cannot express them, 59.4)."""
    field, value = _steering_token(raw, "constrain")
    if isinstance(value, str):
        vs = [_steering_value(v) for v in value.split(",") if v.strip() != ""]
    else:  # a single JSON value (a number or a list)
        vs = [value]
    return field, vs


def _steering_from_args(args: argparse.Namespace):
    """SPEC.md 59.4 (v0.45): the `SteeringState` from the CLI's repeatable
    ``--pin`` / ``--bias`` / ``--constrain`` lists; `None` when all three are
    empty (the pre-v0.45 path, G2). A bad token is a `ValueError` (the
    caller prints it + exits 1)."""
    pins = list(getattr(args, "pin", None) or [])
    biases = list(getattr(args, "bias", None) or [])
    constrains = list(getattr(args, "constrain", None) or [])
    if not (pins or biases or constrains):
        return None
    state = SteeringState()
    for raw in pins:
        f, v = _steering_token(raw, "pin")
        state = state.pin(f, v)
    for raw in biases:
        f, v = _steering_token(raw, "bias")
        state = state.bias(f, v)
    for raw in constrains:
        f, vs = _steering_constrain_token(raw)
        state = state.constrain(f, vs)
    return state


def _cmd_run(args: argparse.Namespace) -> int:
    # SPEC.md 87.2 (v0.73, A2): the plain-language tour — the 41.3 tiny loop
    # narrated in three beats; like --demo, the other `run` flags are ignored
    if getattr(args, "tour", False):
        return _cmd_tour(args)
    # SPEC.md 41.3 (v0.27, S5): the narrated demo — parity-v1, a tiny fixed
    # budget; the other `run` flags are ignored (41.3.1)
    if args.demo:
        return _cmd_demo(args)
    if getattr(args, "prior_run", None) is not None \
            and getattr(args, "from_model", None) is not None:
        # SPEC.md 86.4.1 (v0.72): both define the baseline's starting point
        print("--prior-run and --from-model are mutually exclusive "
              "(SPEC.md 86.4.1)", file=sys.stderr)
        return 1
    # SPEC.md 59.2 (v0.45): the steering rules (None = off, pre-v0.45 path)
    try:
        steering = _steering_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
        steering=steering,  # SPEC.md 59.2 (v0.45)
        initial_model=args.from_model,  # v0.66 (SPEC.md 80; None = off)
        prior_run=args.prior_run,  # v0.72 (SPEC.md 86; None = off)
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


def _cmd_tour(args: argparse.Namespace) -> int:
    """SPEC.md 87.2 (v0.73, A2): `run --tour` — the tiny deterministic loop
    (the 41.3.1 env: parity-v1, 3 experiments / 60 s / 10 s, un-gated) told
    in three plain beats (the 87.4 vocabulary, asserted jargon-free) + the
    one-sentence verdict (87.6). Same env / budget / determinism as the
    demo (G2). rc 0 (41.3.3)."""
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
    base = summary.get("baseline_score")
    final = summary.get("final_best_score")
    n = summary.get("experiments_run", 3)
    base_s = (f"{float(base):.1f}"
              if isinstance(base, (int, float)) and not isinstance(base, bool)
              else "-")
    final_s = (f"{float(final):.1f}"
               if isinstance(final, (int, float)) and not isinstance(final, bool)
               else "-")
    # 87.2: the tour is ASCII-only (the 48.5 console rule)
    print("AutoRefine tour - one tiny loop, told in plain words")
    print(f"  1. Start simple: a starter model scores {base_s}.")
    print(f"  2. Try better: {n} tries, each a small change to the recipe.")
    print(f"  3. Pick the winner: the best try scores {final_s}.")
    print(f"verdict : {plain_verdict(summary, None)}")
    print(f"run dir : {env.run_dir}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """SPEC.md 87.1 (v0.73, A1): `autorefine ask "..."` — a goal in the
    user's words -> the plain plan (heard / note / plan) + the
    copy-pasteable `fit` command. No training, no run dir. rc 0 on a
    resolved plan; rc 1 on an unresolvable goal (a fail-loud
    `ValueError`, 87.1.3)."""
    try:
        resolved = resolve_goal(args.goal, data_path=args.data)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"heard  : {resolved['plain_summary']}")
    if resolved["data"] and not Path(resolved["data"]).exists():
        # 88.2.3 (v0.74, A2): the sentence carried a path that is not here
        # yet — the command stays a valid plan (rc 0)
        print("note   : I did not find that path on this machine yet - the "
              "command is ready for when it is")
    if resolved["unrecognized"]:
        print("note   : I did not pick out a specific ask in that sentence - "
              "the plan uses my defaults (a 95% bar, 30 experiments, "
              "auto task)")
    print(f"plan   : data {resolved['data'] or '<your data>'} | task "
          f"{resolved['task']} | target {resolved['target']:g} | "
          f"experiments {resolved['experiments']}")
    print(f"command: {' '.join(ask_command(resolved))}")
    return 0


def _go_probe(args: argparse.Namespace, data: Path, task_name: str):
    """SPEC.md 88.4 (v0.74, B4): the deterministic probe `fit` builds for
    the data — the *same* config `_fit_data` passes to the task
    constructor (one split, G2). ``None`` on a bad path/task (the caller
    degrades to a `note :` line; the fit path reports the error itself).
    """
    from .tasks import TASKS
    config: dict = {"path": str(data)}
    if args.label is not None:
        config["label"] = args.label
    if args.split_frac != 0.2:
        config["split_frac"] = args.split_frac
    if getattr(args, "temporal", False):
        config["split_mode"] = "temporal"
    if getattr(args, "metric", "accuracy") != "accuracy":
        config["metric"] = args.metric
    try:
        return TASKS[task_name](seed=args.seed, **config)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return None


def _latest_prior_run(runs_dir: str, task_name: str):
    """SPEC.md 88.6 (v0.74, C6): the most recent registry entry of the
    same task -> ``(run_dir, final_score | None)``; ``None`` when there
    is no such entry (or no readable registry). Newest by ``(timestamp,
    run_id)`` — the 88.3 tie-break (G2)."""
    try:
        entries = load_registry(runs_dir)
    except Exception:  # a vanished / unreadable registry is "no prior"
        return None
    same = [e for e in entries
            if isinstance(e, dict) and e.get("task") == task_name
            and e.get("run_id")]
    if not same:
        return None
    best = max(same, key=lambda e: (str(e.get("timestamp") or ""),
                                   str(e.get("run_id") or "")))
    score = best.get("final_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    return (str(Path(runs_dir) / str(best["run_id"])),
            None if score is None else float(score))


def _go_why(env: AutoRefineEnv, summary: dict) -> str:
    """SPEC.md 88.9 (v0.74, D9): the `why :` line — the starter spec vs
    the winning spec through `why_this_model` (the run's artifacts: the
    baseline entry, `best_spec.json` with the max-holdout entry as the
    fallback, and the summary's scores)."""
    entries = env.memory.load_experiments() if env.memory else []
    baseline_spec = next((e.get("spec") for e in entries
                          if e.get("kind") == KIND_BASELINE), None)  # 35.1 (C4)
    best_spec = None
    bp = Path(env.run_dir) / "best_spec.json"
    if bp.is_file():
        try:
            best_spec = json.loads(bp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            best_spec = None
    if best_spec is None:
        scored = [e for e in entries
                  if isinstance(e.get("holdout_score"), (int, float))]
        if scored:
            best_spec = max(scored, key=lambda e: e["holdout_score"]).get("spec")
    return why_this_model(baseline_spec, best_spec,
                          summary.get("baseline_score"),
                          summary.get("final_best_score"))


def _go_handoff(env: AutoRefineEnv, args: argparse.Namespace) -> None:
    """SPEC.md 88.7 (v0.74, C7): one command leaves the whole hand-off —
    `report.md` / `report.txt` (63.1 renderers over the 62 doc),
    `model_card.txt` / `model_card.md` (87.8), `report.pdf` when
    reportlab is present (63.1.3), and the deterministic share bundle
    (47.4) — each writer degrades to a `note :` line, never an error
    (88.7). Pure over the run's artifacts; the zip is byte-identical for
    a byte-identical run dir (G2)."""
    run_dir = Path(env.run_dir)
    summary = env.memory.load_summary() if env.memory else {}
    entries = env.memory.load_experiments() if env.memory else []
    written: list[str] = []
    doc = build_report_doc(summary, entries)
    for name, fmt in (("report.md", "md"), ("report.txt", "txt")):
        try:
            (run_dir / name).write_text(render_report(fmt, doc),
                                        encoding="utf-8")
            written.append(name)
        except Exception as exc:
            print(f"note : {name} skipped ({exc})")
    try:
        task_name, model_name = _user_guide_sources(summary, run_dir)
        hl = headline_uncertainty(
            summary.get("final_best_score"), _seed_spread_for(run_dir, summary))
        view = model_card(summary, entries, task=task_name, model=model_name,
                          n_examples=3, headline=hl)
        (run_dir / "model_card.txt").write_text(render_model_card(view),
                                                encoding="utf-8")
        written.append("model_card.txt")
        (run_dir / "model_card.md").write_text(render_model_card_md(view),
                                               encoding="utf-8")
        written.append("model_card.md")
    except Exception as exc:
        print(f"note : model card skipped ({exc})")
    try:
        render_report_pdf(doc, run_dir / "report.pdf")
        written.append("report.pdf")
    except ImportError:  # reportlab is optional (SPEC.md 3)
        print("note : report.pdf skipped (reportlab not installed)")
    except Exception as exc:
        print(f"note : report.pdf skipped ({exc})")
    zip_name = f"{run_dir.name}-share.zip"
    try:
        payload = share_payload(run_dir, _share_report_html(run_dir))
        write_share_zip(payload, run_dir.parent / zip_name)
        written.append(zip_name)
    except Exception as exc:
        print(f"note : {zip_name} skipped ({exc})")
    print(f"handoff : everything is in {run_dir}:")
    if written:
        print("  " + " \u00b7 ".join(written))


def _go_score_labels(path: Path) -> list[str] | None:
    """SPEC.md 88.8 (v0.74, D8): the score file's label-column values in
    row order (the 88.1.1 `label_column` read, stdlib csv, utf-8-sig);
    ``None`` when the file is unreadable or has no resolvable label
    column (no agreement line, 88.8)."""
    import csv as _csv
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = _csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return None
            col = label_column(header)
            if col is None:
                return None
            out: list[str] = []
            for row in reader:
                if not row:
                    continue
                out.append(row[col].strip() if col < len(row) else "")
            return out
    except OSError:
        return None


def _go_score(args: argparse.Namespace, env: AutoRefineEnv) -> bool:
    """SPEC.md 88.8 (v0.74, D8): score every row of `go --score CSV` with
    the run's best model — the 42.1 path (`csv_rows_to_features` +
    `standardize` + `predict_features`, the 42.1 per-row format) — plus,
    when the file carries a resolvable label column (88.1.1) and the task
    has `class_values`, the `agreement :` line (the fraction of rows
    whose prediction equals the file's label). A bad file is a fail-loud
    ``False`` (the run already finished; its artifacts stand, 88.8)."""
    try:
        _summary, task, model = _run_task_and_model(Path(env.run_dir))
    except (ValueError, FileNotFoundError) as exc:
        print(f"score: {exc}", file=sys.stderr)
        return False
    if int(getattr(task, "max_steps", 1)) > 1:  # 42.1.2: fitting tasks only
        print("score: episode tasks are not supported by the 42.1 scoring "
              "path (SPEC.md 42.1.2)", file=sys.stderr)
        return False
    try:
        raws = csv_rows_to_features(args.score, task)
    except (ValueError, TypeError, FileNotFoundError, OSError) as exc:
        print(f"score: {exc}", file=sys.stderr)
        return False
    if not raws:
        print(f"score: no rows in {args.score}", file=sys.stderr)
        return False
    preds: list = []
    for i, raw in enumerate(raws):
        try:
            feat = standardize(task, raw)
            r = predict_features(task, model, feat)
        except (ValueError, TypeError) as exc:
            print(f"score: row {i}: {exc}", file=sys.stderr)
            return False
        preds.append(r["prediction"])
    print(f"scored  : {len(preds)} rows with the best model "
          f"(task {getattr(task, 'name', '?')})")
    for i, p in enumerate(preds):
        print(f"{i}\t{_pred_str(p)}")
    labels = _go_score_labels(Path(args.score))
    if labels is not None and len(labels) == len(preds) \
            and getattr(task, "class_values", None):
        agree = 0
        for pred, lab in zip(preds, labels):
            try:
                agree += float(pred) == float(lab)
            except (TypeError, ValueError):
                agree += str(pred) == str(lab)
        print(f"agreement : {100.0 * agree / len(preds):.1f}% "
              f"({agree}/{len(preds)} rows agree with the label column)")
    return True


def _go_seeds(args: argparse.Namespace, env: AutoRefineEnv,
              task_name: str) -> list | None:
    """SPEC.md 88.10 (v0.74, D10): the additional seeds of `go --seeds N`
    — the 29.1 `seed_sweep` machinery (``seed+1 .. seed+N-1``) over the
    base run's core knobs; ``None`` on a bad setup (the base run stands,
    the caller degrades). A measurement (88.10.3): the gate and the exit
    code follow the base run."""
    from .dashboard import DashboardRunner
    try:
        runner = DashboardRunner(
            csv_path=str(args.data), label=args.label,
            split_frac=args.split_frac, target=args.target,
            policy=args.policy, seed=args.seed,
            experiments=args.experiments, max_seconds=args.max_seconds,
            max_train_seconds=args.max_train_seconds, runs_dir=args.runs_dir,
            search_quality=args.search_quality, modality=task_name,
            stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1)
            objectives=None)  # go's gate is the 22.1 score bar (no --gate)
        n = int(getattr(args, "seeds", 0))
        seeds = [int(args.seed) + i for i in range(1, n)]
        return list(runner.seed_sweep(seeds))
    except Exception as exc:
        print(f"seeds : {exc} (the base run stands)", file=sys.stderr)
        return None


def _cmd_go(args: argparse.Namespace) -> int:
    """SPEC.md 87.3 (v0.73, A3) + 88 (v0.74, A1..D10): `autorefine go` —
    data to verdict in one command, with the fewest possible inputs:

    - the data is found in the current folder when not named (88.1);
    - a previous same-task run is offered, and `--continue` takes it
      (88.6); `--auto-target` derives the bar from the data's class
      structure (88.4); `--time SECONDS` derives the budget from the
      measured per-experiment rate (88.5), else the data-sized default
      (87.3.2);
    - the loop reuses `_fit_data` / `_drive` / `_fit_gate` verbatim
      (byte-identical loop semantics);
    - after the gate: the plain verdict (87.3.3) with its seed band
      (88.10), the `why :` line (88.9), the one-line seed nudge (88.10.2),
      the full hand-off (88.7), and the optional second-file score (88.8).

    Exit codes are `fit`'s: 0 PASS / 2 MISS / 1 error (a bad `--score`
    file is rc 1 — the run already finished, its artifacts stand, 88.8).
    """
    # 89.12 (D2): --feeds is a different verb sharing the go core
    if getattr(args, "feeds", None) is not None:
        return _cmd_go_feeds(args)
    # 88.1 (A1): the data is found when it is not named
    if args.data is None:
        cand = pick_data(Path.cwd())
        if cand is None:
            print("go could not find a data file in the current folder - "
                  "pass --data PATH, or run this inside the folder holding "
                  "your data (SPEC.md 88.1.2)", file=sys.stderr)
            return 1
        args.data = cand["path"]
        print(f"found : {Path(cand['path']).name} ({cand['reason']}) - "
              f"using it (SPEC.md 88.1)")
    data = Path(args.data)

    # the task name, once, for the offers and derivations below (the env
    # re-resolves identically; a failure here only loses the hints)
    try:
        task_name = _resolve_fit_task(
            data, getattr(args, "task", "auto") or "auto")
    except ValueError:
        task_name = None

    # 88.6 (C6): the prior run is offered, and one flag takes it
    if getattr(args, "continue_", False):
        if task_name is None:
            print("continue : the data task did not resolve - starting "
                  "from scratch")
        else:
            hit = _latest_prior_run(args.runs_dir, task_name)
            if hit is None:
                print(f"continue : no earlier run of {task_name} in "
                      f"{args.runs_dir} - starting from scratch")
            else:
                prior_dir, score = hit
                args.prior_run = prior_dir
                s = f"{score:.1f}" if score is not None else "?"
                print(f"continue : starting from {Path(prior_dir).name} "
                      f"(final {s}) - its best spec seeds the search prior")
    elif task_name is not None:
        hit = _latest_prior_run(args.runs_dir, task_name)
        if hit is not None:
            prior_dir, score = hit
            s = f"{score:.1f}" if score is not None else "?"
            print(f"earlier  : {Path(prior_dir).name} scored {s} on "
                  f"{task_name} - pass --continue to start from its best "
                  f"spec instead of scratch")

    # 88.4 (B4): a fair bar when none was set
    if getattr(args, "auto_target", False):
        if task_name is None:
            print("target : auto - the data task did not resolve; keeping "
                  "the current bar")
        else:
            probe = _go_probe(args, data, task_name)
            if probe is None:
                print("target : auto - the data task did not resolve; "
                      "keeping the current bar")
            elif not getattr(probe, "class_values", None):
                print("target : auto - no class structure in this task; "
                      "keeping the current bar")
            else:
                try:
                    _x, y = probe.make_dataset()
                except Exception:
                    y = None
                counts: dict = {}
                n = 0
                if y is not None:
                    for v in y:
                        counts[str(v)] = counts.get(str(v), 0) + 1
                        n += 1
                if not n:
                    print("target : auto - no rows in the dataset; keeping "
                          "the current bar")
                else:
                    maj = max(counts.values()) / n
                    args.target = auto_target(maj)
                    print(f"target : auto - {fair_bar_line(maj, args.target)}")

    if getattr(args, "experiments", None) is None:
        # 87.3.2 (v0.73): the budget is a deterministic step function of
        # the train-split size — the *same* probe `fit` builds (one split,
        # G2); 88.5 (v0.74, B5): `--time` derives it from the measured
        # per-experiment rate instead
        try:
            task_name = _resolve_fit_task(
                data, getattr(args, "task", "auto") or "auto")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        probe = _go_probe(args, data, task_name)
        if probe is None:
            return 1
        if getattr(args, "time", None) is not None:
            per = measured_per_exp(load_registry(args.runs_dir), task_name)
            if per is not None:
                args.experiments = experiments_for_time(args.time, per)
                print(f"budget : {args.experiments} experiments (about "
                      f"{args.experiments * per:.0f}s at the measured "
                      f"{per:.1f}s/experiment on {task_name})")
            else:
                args.experiments = go_budget(
                    getattr(probe, "default_dataset_size", None))
                print(f"note : no measured per-experiment time for "
                      f"{task_name} yet - used the data-size default (the "
                      f"--time {args.time:g}s target was noted)")
        else:
            args.experiments = go_budget(
                getattr(probe, "default_dataset_size", None))
    env, bar, metric = _fit_data(args)
    if env is None:
        return 1
    _drive(args, env)
    rc = _fit_gate(args, env, bar, metric)
    # 89.11 (D1): the latency budget — a MISS (rc 2) when OVER it
    lrc = _latency_objective(args, env)
    if lrc is not None:
        return lrc
    if not getattr(args, "quiet", False):  # 47.3.2: the diagnostics block
        _print_fit_diagnostics(env)
    summary = env.memory.load_summary() if env.memory else {}

    # 88.10 (D10): the seed band — measured before the verdict so the
    # line can carry it (88.10.1); a measurement, not a gate (88.10.3)
    n_seeds = getattr(args, "seeds", None)
    band = ""
    if n_seeds is not None and int(n_seeds) >= 2 and task_name is not None:
        base_final = summary.get("final_best_score")
        finals: list[float] = []
        seed_items: list[str] = []
        if (isinstance(base_final, (int, float))
                and not isinstance(base_final, bool)):
            finals.append(float(base_final))
            seed_items.append(f"{args.seed} -> {float(base_final):.2f}")
        sweep = _go_seeds(args, env, task_name)
        if sweep is not None:
            for s in sweep:
                if isinstance(s.get("final"), (int, float)):
                    finals.append(float(s["final"]))
                seed_items.append(f"{int(s['seed'])} -> {s['final']:.2f}")
            if seed_items:
                print("seeds   : " + ", ".join(seed_items))
            if finals:
                mean = sum(finals) / len(finals)
                half = (max(finals) - min(finals)) / 2.0
                band = (f" - band {mean:g} +/- {half:g} "
                        f"across {len(finals)} seeds")
                print(f"spread  : {mean:.2f} +/- {half:.2f} "
                      f"(across {len(finals)} seeds)")
    print(f"verdict : {plain_verdict(summary, bar)}{band}")
    # 88.9 (D9): the verdict says why this model
    print(f"why     : {_go_why(env, summary)}")
    # 88.10.2: the one-line nudge when a single seed was run
    if n_seeds is None or int(n_seeds) < 2:
        print("note    : one seed was run - pass --seeds 3 for a spread "
              "estimate (how sure to be)")
    # 88.7 (C7): one command leaves the whole hand-off
    _go_handoff(env, args)
    # 88.8 (D8): the same command scores a second file
    if getattr(args, "score", None) is not None:
        if not _go_score(args, env):
            return 1
    return rc


# --- v0.75 (SPEC.md 89): time-based and on-demand data ingestion -----------

def _cell_is_num(s: str) -> bool:
    """89.1 (v0.75): the 42.1 header-detection cell test (finite number?)."""
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return True


def _read_csv_cells(path) -> tuple[list[str], list[list[str]], bool]:
    """89.1/89.6 (v0.75): a CSV as stripped cells — the SAME header rule as
    the 42.1 leaf (first row with a non-numeric cell = header; blank rows
    skipped), so this read's data rows align one-for-one with
    `csv_rows_to_features`. Returns ``(header, data_rows, has_header)``."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        rows = [[(c or "").strip() for c in raw] for raw in csv.reader(fh)
                if any((c or "").strip() for c in raw)]
    if not rows:
        return [], [], False
    first = rows[0]
    if any(not _cell_is_num(c) for c in first):
        return first, rows[1:], True
    return [], rows, False


def _score_csv_champion(task, model, csv_path) -> tuple[float, int]:
    """89.1.2 (v0.75, A1): the champion's accuracy (0-100) on the rows of a
    CSV, via the 42.1 leaf (`csv_rows_to_features` + `standardize` +
    `predict_features`) against the file's label column — the task's
    `label_name` when it names one, else the 88.1.1 rule (a named
    label/target/y/class, else the last column). Raises ``ValueError`` when
    the file or its label is unusable (the caller exits rc 1)."""
    header, rows, has_header = _read_csv_cells(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    idx = None
    names = [str(h).strip().lower() for h in header] if has_header else []
    lbl_name = getattr(task, "label_name", None)
    if has_header and lbl_name:
        want = str(lbl_name).strip().lower()
        if want in names:
            idx = names.index(want)
    if idx is None:
        idx = label_column(header) if has_header else None
    if idx is None:
        raise ValueError(
            f"{csv_path} has no label column (the 88.1.1 rule found none) "
            f"- the champion cannot be scored on it")
    raws = csv_rows_to_features(csv_path, task)
    if len(raws) != len(rows):
        raise ValueError(f"{csv_path}: {len(raws)} feature rows vs "
                         f"{len(rows)} label rows")
    correct = 0
    for raw, row in zip(raws, rows):
        x = standardize(task, raw)
        res = predict_features(task, model, x)
        lab = row[idx].strip() if len(row) > idx else ""
        if _pred_str(res["prediction"]) == lab:  # the 42.1.4 label rule
            correct += 1
    return 100.0 * correct / len(raws), len(raws)


def _resolve_champion_run(args, cmd: str) -> Path | None:
    """89.1/89.3 (v0.75): the champion run — `--champion RUN` (explicit
    wins) or the registry's latest (the 88.3 pattern: timestamp, then
    run_id). A failure prints one fail-loud line and returns ``None``
    (the caller exits rc 1, 89.1.4)."""
    if getattr(args, "champion", None) is not None:
        return Path(args.champion)
    if getattr(args, "latest", False):
        try:
            entries = load_registry(args.runs_dir)
        except Exception:
            entries = []
        if not entries:
            print(f"{cmd}: no finished runs in {args.runs_dir} - run "
                  f"autorefine fit or go first (SPEC.md 89.1.4)",
                  file=sys.stderr)
            return None
        best = max(entries, key=lambda e: (str(e.get("timestamp") or ""),
                                          str(e.get("run_id") or "")))
        run_dir = Path(args.runs_dir) / str(best.get("run_id", "?"))
        if not (run_dir / "summary.json").is_file():
            print(f"{cmd}: the champion run {run_dir} has no summary.json - "
                  f"finish it first (SPEC.md 89.1.4)", file=sys.stderr)
            return None
        return run_dir
    print(f"{cmd} needs --champion RUN or --latest (SPEC.md 89.1.4)",
          file=sys.stderr)
    return None


def _cmd_feed(args: argparse.Namespace) -> int:
    """SPEC.md 89.1/89.2 (v0.75, A1/A2): `autorefine feed --data P` — the
    data watcher. Each poll that detects new data snapshots the file
    (89.4), re-scores the champion on its rows with the 42.1 leaf, judges
    the drift (89.2), and appends one event to `data_events.jsonl` (89.1.3).
    Sleeps `--poll` seconds BETWEEN polls only. `--chart` renders the event
    log as the 89.8 SVG. rc 0 on success; rc 1 on a missing data file /
    champion run / label column (fail-loud, stderr)."""
    data = Path(args.data)
    if not data.is_file():
        print(f"feed: no such data file: {data} (SPEC.md 89.1.4)",
              file=sys.stderr)
        return 1
    run_dir = _resolve_champion_run(args, "feed")
    if run_dir is None:
        return 1
    try:
        summary, task, model = _run_task_and_model(run_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"feed: {exc} (SPEC.md 89.1.4)", file=sys.stderr)
        return 1
    ref = summary.get("final_best_score")
    if not isinstance(ref, (int, float)) or isinstance(ref, bool):
        print("feed: the champion run has no final_best_score "
              "(SPEC.md 89.2.1)", file=sys.stderr)
        return 1
    state = load_state(args.runs_dir)
    key = str(data)
    mine = state.get(key) if isinstance(state, dict) else None
    n_polls = max(1, int(args.max))
    for i in range(1, n_polls + 1):
        res = poll(data, mine)
        if not res["changed"]:
            print(f"feed : poll {i}: no change (skipped)")
        else:
            mine = {"fingerprint": res["fingerprint"]}
            state[key] = mine
            save_state(args.runs_dir, state)
            try:
                snap = snapshot_dataset(data, args.runs_dir)
                acc, n = _score_csv_champion(task, model, data)
            except (ValueError, OSError) as exc:
                print(f"feed : poll {i}: {exc} (SPEC.md 89.1.4)",
                      file=sys.stderr)
                return 1
            status, _margin = drift_verdict(acc, float(ref),
                                            bar=args.drift_bar)
            print(f"feed : poll {i}: {n} rows, champion {acc:.1f} "
                  f"(drift {status}) [{snap['ds_id']}]")
            if status == "alert":
                print(f"drift : fresh {acc:.1f} vs reference {float(ref):.1f} "
                      f"(bar {args.drift_bar:g}) - champion degraded on new "
                      f"data (SPEC.md 89.2)")
            now = datetime.now(timezone.utc)
            append_event(args.runs_dir, {
                "poll": i,
                "epoch": int(now.timestamp()),
                "iso": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rows": n,
                "champion_score": acc,
                "drift": status,
                "ds_id": snap["ds_id"],
            })
        if i < n_polls and args.poll > 0:
            time.sleep(args.poll)
    if getattr(args, "chart", None):
        svg = svg_quality_over_time(load_events(args.runs_dir))
        if svg:
            out = Path(args.chart)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(svg, encoding="utf-8")
            print(f"chart : wrote {out}")
        else:
            print("chart : no scored events to render (SPEC.md 89.8)")
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    """SPEC.md 89.3 (v0.75, A3): `autorefine refresh --data NEW.csv` —
    champion/challenger: the champion is the named run's (or the registry's
    latest) best model scored on NEW.csv (89.1); the challenger is a fresh
    22.1 loop (the 23.1 runner, bandit, `search_quality=v04`) over NEW.csv.
    `refresh_verdict` PROMOTEs at/above `--margin`, else KEEPs. rc 0 for
    either verdict (a measured decision, not a failure); rc 1 on a missing
    champion run / unusable data."""
    data = Path(args.data)
    if not data.is_file():
        print(f"refresh: no such data file: {data} (SPEC.md 89.3.3)",
              file=sys.stderr)
        return 1
    run_dir = _resolve_champion_run(args, "refresh")
    if run_dir is None:
        return 1
    try:
        summary, task, model = _run_task_and_model(run_dir)
        acc, n = _score_csv_champion(task, model, data)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"refresh: {exc} (SPEC.md 89.3.3)", file=sys.stderr)
        return 1
    print(f"champion : {run_dir.name} scores {acc:.1f} on the new data "
          f"({n} rows) (SPEC.md 89.3.1)")
    from .dashboard import DashboardRunner
    runner = DashboardRunner(
        csv_path=str(data), target=args.target, policy="bandit",
        seed=args.seed, experiments=args.experiments,
        runs_dir=args.runs_dir, search_quality="v04")
    runner.start()
    while not runner.done:
        runner.next()
    res = runner.finish()
    final = float(res["final_best_score"])
    challenger_run = (str(runner.env.run_dir) if runner.env is not None
                      else "?")
    print(f"challenger : final {final:.1f} after {args.experiments} "
          f"experiment(s) (run {challenger_run}) (SPEC.md 89.3.1)")
    verdict, line = refresh_verdict(acc, final, margin=args.margin)
    print(f"refresh : {line} (SPEC.md 89.3.2)")
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    """SPEC.md 89.4.2 (v0.75, A4): `autorefine datasets [--runs-dir D]` —
    list the snapshot registry (one line per snapshot: id, rows, cols,
    label, balance, created); an empty registry prints the empty message
    and stays rc 0."""
    snaps = load_snapshots(args.runs_dir)
    if not snaps:
        print("datasets : (none yet - feed, ingest, or fit --dataset "
              "snapshots land here; SPEC.md 89.4.2)")
        return 0
    for s in snaps:
        bal = (s.get("class_balance") or {})
        bal_s = ", ".join(f"{k}:{v}" for k, v in bal.items())
        lbl = s.get("label_column")
        lbl_n = s.get("label_name")
        lbl_s = (f"label {lbl_n} (col {lbl})" if lbl is not None and lbl_n
                 else (f"label col {lbl}" if lbl is not None else "no label"))
        print(f"{s.get('ds_id')}  rows {s.get('n_rows')}  cols "
              f"{s.get('n_cols')}  {lbl_s}  balance [{bal_s}]  "
              f"created {s.get('created')}")
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    """SPEC.md 89.5 (v0.75, B1): `autorefine schedule --data P` — N
    sequential 22.1 loops (the 23.1 runner); round k > 1 passes round
    k-1's run dir as its 86 search prior (`prior_run`), so the schedule
    COMPOUNDS. One line per round, a sleep of `--every` seconds between
    rounds only, and the closing best line. rc = the last round's gate rc
    (0 PASS / 2 MISS); a resolve failure is rc 1."""
    from .dashboard import DashboardRunner
    data = Path(args.data)
    if not data.exists():
        print(f"schedule: no such data path: {data} (SPEC.md 89.5.4)",
              file=sys.stderr)
        return 1
    rounds = int(args.rounds)
    if rounds < 1:
        print("schedule: --rounds must be >= 1 (SPEC.md 89.5.1)",
              file=sys.stderr)
        return 1
    try:
        _resolve_fit_task(data, getattr(args, "task", "auto") or "auto")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    prior: str | None = None
    best: float | None = None
    best_round = 0
    last_rc = 0
    for k in range(1, rounds + 1):
        runner = DashboardRunner(
            csv_path=str(data), target=args.target, policy="bandit",
            seed=args.seed, experiments=args.experiments,
            runs_dir=args.runs_dir, search_quality="v04",
            prior_run=prior)
        runner.start()
        while not runner.done:
            runner.next()
        res = runner.finish()
        final = float(res["final_best_score"])
        passed = final >= float(args.target)
        last_rc = 0 if passed else 2
        tag = "PASS" if passed else "MISS"
        print(f"schedule : round {k}/{rounds} done - final {final:.1f} "
              f"(target {args.target:g}, {tag}) (SPEC.md 89.5.2)")
        if best is None or final > best:
            best, best_round = final, k
        if runner.env is not None:
            prior = str(runner.env.run_dir)
        if k < rounds and args.every > 0:
            time.sleep(args.every)
    print(f"schedule : {rounds} rounds finished - best {best:.1f} "
          f"(round {best_round}) (SPEC.md 89.5.3)")
    return last_rc


def _cmd_ingest(args: argparse.Namespace) -> int:
    """SPEC.md 89.6 (v0.75, B2): `autorefine ingest DROP_DIR` — the
    drop-folder merge. The directory's `*.csv` (name order, the output
    file itself excluded) whose sha256 is not yet in the drop dir's
    `drop_state.json` are merged: the FIRST file's header wins, a later
    file with a different header set is skipped with a `note :` line (never
    a silent column shift), exact-duplicate rows are dropped. The result
    is written to `--out` (default `DROP_DIR/merged.csv`), snapshotted
    (89.4), and the state updated (a changed file re-arrives under its new
    sha). rc 0; rc 1 when there is nothing new (the "poll again later"
    signal for cron)."""
    drop = Path(args.drop_dir)
    if not drop.is_dir():
        print(f"ingest: no such directory: {drop} (SPEC.md 89.6.1)",
              file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else drop / "merged.csv"
    state = load_drop_state(drop)
    seen = state["seen"]
    csvs = sorted(drop.glob("*.csv"), key=lambda p: p.name)
    new: list[tuple[Path, str]] = []
    already = 0
    for p in csvs:
        if p == out or p == out.resolve():
            continue
        sha = hash_csv(p)
        if seen.get(p.name) == sha:
            already += 1
            continue
        new.append((p, sha))
    if not new:
        print(f"ingest : no new files in {drop} (SPEC.md 89.6.3)")
        return 1
    header, rows, merged, mismatched, dupes = merge_tables(
        [p for p, _ in new])
    for m in mismatched:
        print(f"note : {m} has a different header set - skipped "
              f"(never a silent column shift; SPEC.md 89.6.1)")
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    out.write_text(buf.getvalue(), encoding="utf-8")
    for p, sha in new:  # 89.6.1: seen = seen (merged OR mismatched)
        seen[p.name] = sha
    save_drop_state(drop, state)
    snap = snapshot_dataset(out, args.runs_dir)
    lbl = snap.get("label_column")
    lbl_name = (header[lbl] if lbl is not None and lbl < len(header)
                else "-")
    print(f"ingest : {len(new)} new file(s) in {drop} ({already} already "
          f"ingested) (SPEC.md 89.6.2)")
    print(f"ingest : merged {len(rows)} rows ({dupes} duplicate row(s) "
          f"skipped) -> {out} (SPEC.md 89.6.2)")
    print(f"ingest : snapshot {snap['ds_id']} ({snap['n_rows']} rows, "
          f"label '{lbl_name}') (SPEC.md 89.6.2)")
    return 0


def _fit_window_slice(args, window_s: float) -> str | None:
    """SPEC.md 89.7.2 (v0.75, B3): `fit --window` — pre-filter the CSV to
    the recency slice (89.7.1), write it to
    `<runs-dir>/windows/window-<sha12>.csv`, print the `window :` line,
    and return the slice path (the caller swaps `args.data`). ``None``
    (with a fail-loud line) when there is no date column, no parseable
    dates, or the window keeps every row."""
    path = Path(args.data)
    if not path.is_file():
        print(f"window : {path} is not a file (--window needs a CSV; "
              f"SPEC.md 89.7.2)", file=sys.stderr)
        return None
    header, rows, has_header = _read_csv_cells(path)
    col = find_date_col(header) if has_header else None
    if col is None and getattr(args, "window_col", None) is not None:
        names = [str(h).strip().lower() for h in header]
        want = str(args.window_col).strip().lower()
        if want in names:
            col = names.index(want)
    if col is None:
        print(f"window : no date column in {header} - expected one of "
              f"{list(DATE_COL_NAMES)} (or name it with --window-col; "
              f"SPEC.md 89.7.1)", file=sys.stderr)
        return None
    kept, total, lo, hi = window_rows(header, rows, window_s, col)
    if lo is None:
        print(f"window : no parseable dates in column {header[col]!r} - "
              f"expected ISO 8601 or epoch seconds (SPEC.md 89.7.1)",
              file=sys.stderr)
        return None
    if len(kept) == total:
        print(f"window : all {total} rows are inside the window ({lo} .. "
              f"{hi}) - nothing to cut (check the column or the window; "
              f"SPEC.md 89.7.2)", file=sys.stderr)
        return None
    buf = io.StringIO()
    w = csv.writer(buf)
    if has_header:
        w.writerow(header)
    w.writerows(kept)
    content = buf.getvalue()
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    wdir = Path(args.runs_dir) / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    out = wdir / f"window-{sha}.csv"
    out.write_text(content, encoding="utf-8")
    print(f"window : kept {len(kept)}/{total} rows ({lo} .. {hi}, col "
          f"'{header[col]}') (SPEC.md 89.7.2)")
    return str(out)


def _fit_dataset_verify(args) -> dict | None:
    """SPEC.md 89.4.3 (v0.75, A4): `fit --dataset` — resolve the snapshot
    (exact ds_id or a unique prefix), verify the file's sha256 still
    matches (a changed file is fail-loud; the caller exits rc 1), and
    return the entry (the caller swaps `args.data` and, after the gate,
    writes it to the run dir as `dataset.json`)."""
    snap = find_snapshot(args.runs_dir, args.dataset)
    if snap is None:
        print(f"dataset : no snapshot {args.dataset!r} in {args.runs_dir} "
              f"(autorefine datasets lists them; SPEC.md 89.4.3)",
              file=sys.stderr)
        return None
    if hash_file(snap["path"]) != snap["sha256"]:
        print(f"dataset : {snap['ds_id']} no longer matches its sha256 - "
              f"re-snapshot before fitting (SPEC.md 89.4.3)",
              file=sys.stderr)
        return None
    print(f"dataset : {snap['ds_id']} verified ({snap['n_rows']} rows) "
          f"(SPEC.md 89.4.3)")
    return snap


def _latency_objective(args, env) -> int | None:
    """SPEC.md 89.11 (v0.75, D1): `--latency-ms` — after the score gate,
    the median ms/row of the best model's forward over the task's dataset
    rows (<= 200) and the budget line; OVER budget makes the command MISS
    (rc 2). Returns the rc to apply (2 when over budget), ``None`` when
    the flag is unset, 1 when the measurement itself fails."""
    if getattr(args, "latency_ms", None) is None:
        return None
    try:
        x, _y = env.task.make_dataset()
        x = np.asarray(x)[:min(200, int(np.asarray(x).shape[0]))]
        if x.size == 0:
            raise ValueError("the task's dataset is empty")
        ms = measure_ms_per_row(env.best_model, x)
    except Exception as exc:  # a measurement failure is a plain error
        print(f"latency : could not measure ({exc}; SPEC.md 89.11.1)",
              file=sys.stderr)
        return 1
    ok, line = latency_line(ms, float(args.latency_ms))
    print(line)
    if not ok:
        return 2
    return None


def _cmd_go_feeds(args) -> int:
    """SPEC.md 89.12 (v0.75, D2): `go --feeds A.csv,B.csv` — the 46.3
    portfolio pattern at `go`'s level: one shared budget split evenly
    (`max(1, ceil(total/N))` per feed; `go`'s data-sized default when no
    total was given), the feeds run sequentially in the given order (each
    through the full 88 go core, quiet), and the closing
    `portfolio :` verdict. rc 0 iff every feed PASSes (2 if any MISS,
    1 on a resolve failure)."""
    if args.data is not None:
        print("--feeds is mutually exclusive with --data (SPEC.md 89.12.1)",
              file=sys.stderr)
        return 1
    feeds = [p.strip() for p in str(args.feeds).split(",") if p.strip()]
    if len(feeds) < 2:
        print("--feeds needs at least 2 CSV paths (SPEC.md 89.12.1)",
              file=sys.stderr)
        return 1
    total = args.experiments
    per = (max(1, math.ceil(total / len(feeds)))
           if total is not None else None)
    orig = (args.data, args.experiments, args.quiet, args.feeds)
    args.feeds = None  # the per-feed core must not re-dispatch to --feeds
    results: list[int] = []
    for f in feeds:
        args.data, args.experiments, args.quiet = f, per, True
        results.append(_cmd_go(args))
    args.data, args.experiments, args.quiet, args.feeds = orig
    n = len(results)
    ok = sum(1 for r in results if r == 0)
    print(f"portfolio : {n} feed(s), {ok}/{n} met the bar "
          f"(target {args.target:g}) (SPEC.md 89.12.2)")
    if all(r == 0 for r in results):
        return 0
    if any(r == 1 for r in results):
        return 1
    return 2


def _cmd_report_backtest(args) -> int:
    """SPEC.md 89.9 (v0.75, C2): `report --backtest --data FILE` — the
    expanding-window replay: window i = the first `ceil(n*i/K)` rows (the
    "if we had started with less data" axis), each a fresh 23.1 loop with
    the same seed (deterministic). One line per window; the slices land
    under `<runs-dir>/backtest/`. rc 0; rc 1 on an unusable file.
    Zero legacy surface touched."""
    from .dashboard import DashboardRunner
    data = Path(args.data)
    if not data.is_file():
        print(f"backtest : no such CSV file: {data} (SPEC.md 89.9.1)",
              file=sys.stderr)
        return 1
    header, rows, has_header = _read_csv_cells(data)
    n = len(rows)
    if n < 2:
        print(f"backtest : {data} has {n} row(s) - need at least 2 for "
              f"expanding windows (SPEC.md 89.9.1)", file=sys.stderr)
        return 1
    k = max(1, int(args.windows))
    budget = max(1, int(args.budget))
    bdir = Path(args.runs_dir) / "backtest"
    bdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, k + 1):
        cut = max(1, min(n, math.ceil(n * i / k)))
        buf = io.StringIO()
        w = csv.writer(buf)
        if has_header:
            w.writerow(header)
        w.writerows(rows[:cut])
        content = buf.getvalue()
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        slice_path = bdir / f"window_{i}-{sha}.csv"
        slice_path.write_text(content, encoding="utf-8")
        runner = DashboardRunner(
            csv_path=str(slice_path), target=args.target, policy="bandit",
            seed=args.seed, experiments=budget, runs_dir=args.runs_dir,
            search_quality="v04")
        runner.start()
        while not runner.done:
            runner.next()
        res = runner.finish()
        print(f"backtest : window {i}/{k} (rows 1-{cut}): baseline "
              f"{float(res['baseline_score']):.2f} -> final "
              f"{float(res['final_best_score']):.2f} (budget {budget}) "
              f"(SPEC.md 89.9.2)")
    return 0


def _cmd_card(args: argparse.Namespace) -> int:
    """SPEC.md 87.8 (v0.73, C2): `autorefine card --run DIR [--md]` — the
    model-card one-pager from a finished run's artifacts (summary +
    experiments.jsonl + best model, the 63.2 sources). `--md` renders the
    Markdown form. rc 0 on a readable run; rc 1 on a missing / broken run
    dir."""
    run_dir = Path(args.run)
    sp = run_dir / "summary.json"
    if not sp.is_file():
        print(f"no summary.json in {run_dir} (a finished run is needed)",
              file=sys.stderr)
        return 1
    summary = json.loads(sp.read_text(encoding="utf-8"))
    entries: list[dict] = []
    exp_path = run_dir / "experiments.jsonl"
    if exp_path.is_file():
        for line in exp_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    task, model = _user_guide_sources(summary, run_dir)
    hl = headline_uncertainty(
        summary.get("final_best_score"), _seed_spread_for(run_dir, summary))
    view = model_card(summary, entries, task=task, model=model,
                      n_examples=args.examples, headline=hl)
    print(render_model_card_md(view) if args.md else render_model_card(view))
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
    # 89.7/89.4 (v0.75, B3/A4): the recency window and the dataset
    # snapshot are single-file data modes — exclusive with the re-run and
    # portfolio modes below
    if getattr(args, "window", None) is not None or getattr(
            args, "dataset", None) is not None:
        if args.from_run is not None:
            print("--window/--dataset and --from-run are mutually "
                  "exclusive (SPEC.md 89.7/89.4)", file=sys.stderr)
            return 1
        if getattr(args, "tasks", None) is not None:
            print("--window/--dataset and --tasks are mutually "
                  "exclusive (SPEC.md 89.7/89.4)", file=sys.stderr)
            return 1
        if getattr(args, "dry_run", False):
            print("--window/--dataset and --dry-run are mutually "
                  "exclusive (SPEC.md 89.7/89.4)", file=sys.stderr)
            return 1
        if getattr(args, "window", None) is not None and getattr(
                args, "dataset", None) is not None:
            # the window cuts the --data file; a snapshot is already a file
            print("--window and --dataset are mutually exclusive "
                  "(SPEC.md 89.7/89.4)", file=sys.stderr)
            return 1
    # SPEC.md 46.3 (v0.32): portfolio mode — first, because it is mutually
    # exclusive with the --data / --from-run checks below
    if getattr(args, "tasks", None) is not None:
        if args.data is not None or args.from_run is not None:
            print("--tasks is mutually exclusive with --data and --from-run "
                  "(SPEC.md 46.3)", file=sys.stderr)
            return 1
        if getattr(args, "prior_run", None) is not None:
            # SPEC.md 86.4.1 (v0.72): one prior belongs to one task
            print("--prior-run and --tasks are mutually exclusive "
                  "(SPEC.md 86.4.1)", file=sys.stderr)
            return 1
        if getattr(args, "dry_run", False):
            return _fit_dry_run_portfolio(args)
        return _fit_portfolio(args)
    if getattr(args, "from_model", None) is not None:
        # SPEC.md 80.2.5 (v0.66): the uploaded model is a single-task seed —
        # a portfolio spreads one budget across tasks, a re-run carries its
        # own recipe (which may already name the model)
        if args.from_run is not None:
            print("--from-model and --from-run are mutually exclusive "
                  "(SPEC.md 80.2.5)", file=sys.stderr)
            return 1
        print("--from-model and --tasks are mutually exclusive "
              "(SPEC.md 80.2.5)", file=sys.stderr)
        return 1
    if args.data is not None and args.from_run is not None:
        print("--data and --from-run are mutually exclusive (SPEC.md 37.1.4)",
              file=sys.stderr)
        return 1
    if getattr(args, "prior_run", None) is not None:
        # SPEC.md 86.4.1 (v0.72): the search prior is a single-task seed —
        # a re-run executes the old recipe exactly (37.1.4), a portfolio
        # spans tasks (checked above); the uploaded model already claims
        # the baseline slot (80.2.5)
        if args.from_run is not None:
            print("--prior-run and --from-run are mutually exclusive "
                  "(SPEC.md 86.4.1)", file=sys.stderr)
            return 1
        if getattr(args, "from_model", None) is not None:
            print("--prior-run and --from-model are mutually exclusive "
                  "(SPEC.md 86.4.1)", file=sys.stderr)
            return 1
    if getattr(args, "dry_run", False):
        return _fit_dry_run(args)  # 40.1 (v0.26): plan only, no training
    snap = None  # 89.4.3 (A4): the fitted dataset entry (or the re-run's None)
    if args.from_run is not None:
        env, bar = _fit_from_run(args)
        if env is None:
            return 1
        metric = getattr(env.task, "metric", None) or "score"
    else:
        if args.data is None and getattr(args, "dataset", None) is None:
            print("one of --data, --dataset, or --from-run is required "
                  "(SPEC.md 37.1.4/89.4.3)", file=sys.stderr)
            return 1
        # 88.6 (v0.74, C6): --continue — the most recent same-task run is
        # the search prior (the 86 plumbing; an explicit --prior-run wins)
        if getattr(args, "continue_", False) and args.prior_run is None:
            try:
                _t = _resolve_fit_task(
                    Path(args.data), getattr(args, "task", "auto") or "auto")
            except ValueError:
                _t = None
            if _t is None:
                print("continue : the data task did not resolve - starting "
                      "from scratch")
            else:
                hit = _latest_prior_run(args.runs_dir, _t)
                if hit is None:
                    print(f"continue : no earlier run of {_t} in "
                          f"{args.runs_dir} - starting from scratch")
                else:
                    prior_dir, score = hit
                    args.prior_run = prior_dir
                    s = f"{score:.1f}" if score is not None else "?"
                    print(f"continue : starting from {Path(prior_dir).name} "
                          f"(final {s}) - its best spec seeds the search prior")
        # 89.7.2 (B3): --window pre-filters the CSV to the recency slice
        if getattr(args, "window", None) is not None:
            try:
                window_s = parse_window(args.window)
            except ValueError as exc:
                print(f"window : {exc} (SPEC.md 89.7.1)", file=sys.stderr)
                return 1
            slice_path = _fit_window_slice(args, window_s)
            if slice_path is None:
                return 1
            args.data = slice_path
        # 89.4.3 (A4): --dataset verifies the snapshot sha256 before fit
        snap = None
        if getattr(args, "dataset", None) is not None:
            snap = _fit_dataset_verify(args)
            if snap is None:
                return 1
            args.data = snap["path"]
        env, bar, metric = _fit_data(args)
    if env is None:
        return 1
    _drive(args, env)
    rc = _fit_gate(args, env, bar, metric)  # kept under --quiet (47.3.2)
    # 89.4.3 (A4): the fitted dataset lands in the run dir as dataset.json
    if snap is not None:
        (Path(env.run_dir) / "dataset.json").write_text(
            json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
    # 89.11 (D1): the latency budget — a MISS (rc 2) when OVER it
    lrc = _latency_objective(args, env)
    if lrc is not None:
        return lrc
    # C2/C3 (SPEC.md 28.2/28.3): after the gate line, per-class accuracy +
    # weakest-class hint (and the media error gallery)
    if not getattr(args, "quiet", False):  # 47.3.2: the diagnostics block
        _print_fit_diagnostics(env)
    return rc


def _cmd_manual(args: argparse.Namespace) -> int:
    """SPEC.md 59.3 (v0.45): manual/expert mode — set the exact spec over the
    field registry and train it once (validate + report, not discover).

    ``--data`` resolves a data task (csv/image/audio/text) via the same
    `_resolve_fit_task` the fit command uses (it wins over `--task`);
    otherwise `--task` is a built-in task. The ``--set FIELD=VALUE`` pairs
    (last one wins per field) are registry-validated (`manual_spec`,
    59.3.1) and trained once (`train_manual`, 59.3.2). The result is printed
    as JSON (the `retrain_spec` shape). rc 0 on success; rc 1 on a bad
    task/field/value (loud, stderr)."""
    task_name = args.task
    task_config: dict | None = None
    if getattr(args, "data", None) is not None:
        data = Path(args.data)
        try:
            task_name = _resolve_fit_task(data, "auto")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        task_config = {"path": str(data)}
        if args.label is not None:
            task_config["label"] = args.label
    values: dict = {}
    for raw in (args.set or []):
        if "=" not in raw:
            print(f"--set {raw!r}: expected FIELD=VALUE (SPEC.md 59.3.1)",
                  file=sys.stderr)
            return 1
        f, v = raw.split("=", 1)
        f = f.strip()
        if not f:
            print(f"--set {raw!r}: the field name is empty (SPEC.md 59.3.1)",
                  file=sys.stderr)
            return 1
        values[f] = _steering_value(v)
    try:
        out = train_manual(task_name, values, seed=args.seed,
                           task_config=task_config,
                           max_train_seconds=args.max_train)
    except ValueError as exc:  # unknown task / field / value (loud, 59.3)
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


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
    # SPEC.md 59.2 (v0.45): the steering rules (None = off, pre-v0.45 path)
    try:
        steering = _steering_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None, None, None
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
        steering=steering,  # SPEC.md 59.2 (v0.45)
        initial_model=args.from_model,  # v0.66 (SPEC.md 80; None = off)
        prior_run=args.prior_run,  # v0.72 (SPEC.md 86; None = off)
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
    # SPEC.md 59.4 (v0.45): the recipe's steering rules (empty = pre-v0.45)
    try:
        steering = (SteeringState.from_dict({
            "pins": config.pins, "biases": config.biases,
            "constraints": config.constraints})
            if (config.pins or config.biases or config.constraints) else None)
    except ValueError as exc:
        print(f"invalid steering in run_config: {exc}", file=sys.stderr)
        return None, None
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
        steering=steering,  # SPEC.md 59.4 (v0.45): the recipe's steering rules
        initial_model=config.initial_model,  # v0.66 (SPEC.md 80; None = off)
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


def _cmd_report_benchmark(args: argparse.Namespace) -> int:
    """SPEC.md 64.1 (v0.50, B5): `report --benchmark` — the benchmark /
    longitudinal report across the run registry: the per-task leaderboard
    (best score + spec fingerprint + gate), the same-spec-across-tasks
    generalization matrix, and the best-score trend across runs. The
    ``spec_map`` (run id -> best_spec) is loaded None-safe from each run
    dir's ``summary.json`` (64.1.5 — an unreadable dir degrades to an
    em-dash fingerprint, never a crash). ``--json`` prints the view model;
    otherwise the text rendering (``render_benchmark_md`` is the library /
    app form). Empty registry -> rc 1 + hint (the 38.3.3 rule)."""
    runs_dir = Path(args.runs_dir)
    entries = load_registry(runs_dir)
    if not entries:
        print(f"no runs registered in {runs_dir} — finish at least one run "
              f"first (SPEC.md 64.1.5)", file=sys.stderr)
        return 1
    spec_map: dict = {}
    for e in entries:  # bounded: one small summary.json per registered run
        rid = e.get("run_id")
        if not isinstance(rid, str) or not rid or rid in spec_map:
            continue
        try:
            s = json.loads(
                (runs_dir / rid / "summary.json").read_text(encoding="utf-8"))
            spec_map[rid] = s.get("best_spec")
        except (OSError, ValueError):
            continue  # 64.1.5: an unreadable / missing run dir degrades
    view = benchmark_view(entries, spec_map)
    if args.json:
        print(json.dumps(view, indent=2))
        return 0
    print(render_benchmark(view))
    return 0


def _seed_spread_for(run_dir: Path, summary: dict) -> list | None:
    """SPEC.md 64.2.2 (v0.50, B6): the same-task final-score spread the
    audience / guide views attach to the headline number — the 41.1
    machinery (``load_registry`` + ``projection_points``) reused one home
    for the three human views; degrades to ``None`` on a missing registry
    or any failure (the views then render the block as ``—``)."""
    try:
        reg = load_registry(run_dir.parent)
        pts = projection_points(reg, summary, run_dir.name)
        return [s for _e, s in pts] if pts else None
    except Exception:
        return None


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


def _cmd_report_audience(args, audience: str, summary: dict, entries, run_dir):
    """SPEC.md 62.1 (v0.48, B1): the exec / domain / regulator report.

    Re-skins the already-loaded summary + log (no training, no new capture).
    The three human-only machine paths (--json / --what-if / --project /
    --trace / --importance / --certificate) are mutually exclusive with an
    audience (rc 1); ``--html`` writes ``report_<audience>.html`` and
    ``--plot`` is a no-op (the audience view is a single text/HTML block)."""
    for flag, why in (
        ("json", "--json"), ("what_if", "--what-if"),
        ("project", "--project"), ("trace", "--trace"),
        ("importance", "--importance"), ("certificate", "--certificate"),
        ("fmt", "--format"), ("user", "--user"), ("decision", "--decision"),
        ("benchmark", "--benchmark"),  # 64.3.1 (v0.50)
    ):
        if getattr(args, flag, None):
            print(f"--audience {audience} is mutually exclusive with {why} "
                  f"(SPEC.md 62.1.3 / 63.4.1)", file=sys.stderr)
            return 1
    base_extras = _report_extras(summary, run_dir)  # diagnostics / gallery
    if audience == "domain":
        extras = dict(base_extras)
        ece, diff = _domain_calibration(summary, run_dir)
        if ece is not None:
            extras["ece"] = ece
        if diff is not None:
            extras["difficulty"] = diff
    else:
        extras = None
    provenance = trace = data = None
    if audience == "regulator":
        provenance, trace, data = _regulator_sources(summary, entries, run_dir)
    # 64.2.2 (B6): the ± read on the headline number — the same-task seed
    # spread (the 41.1 machinery), degrading to None (renders as —).
    seed_spread = _seed_spread_for(run_dir, summary)
    view = build_view(audience, summary, entries, diag=base_extras["diagnostics"],
                      extras=extras, provenance=provenance, trace=trace,
                      data=data, seed_spread=seed_spread)
    print(render_view(view, audience))
    # 89.10 (v0.75, C3): the freshness block — how old is the model relative
    # to the data (feed events when present, else the run's own timestamp)
    evs = load_events(run_dir.parent)
    if evs or summary.get("timestamp"):
        print(render_freshness(freshness_view(summary, evs)))
    if getattr(args, "html", False):
        out = run_dir / f"report_{audience}.html"
        out.write_text(html_view(audience, view), encoding="utf-8")
        print(f"html    : {out}")
    return 0


# --- 63 (v0.49, B2/B3/B4): formats + the end-user guide + the decision -------

# 63.4.1 (B2/B3/B4): the machine / other-human report paths each new view is
# mutually exclusive with — one shared tuple so the three guard blocks cannot
# drift (the 62.1.3 pattern, extended with the sibling human views).
_EXCLUSIVE_FLAGS = (
    ("json", "--json"), ("what_if", "--what-if"),
    ("project", "--project"), ("trace", "--trace"),
    ("importance", "--importance"), ("certificate", "--certificate"),
    ("html", "--html"),
    ("benchmark", "--benchmark"),  # 64.3.1 (v0.50): runs-dir-level view
)


def _exclusive_guard(name: str, args, spec_ref: str) -> int | None:
    """63.4.1 (v0.49): print + ``1`` when ``args`` sets any mutually
    exclusive flag for the human view ``name``; else ``None`` (proceed).
    A non-technical ``--audience`` is always exclusive too."""
    for flag, why in _EXCLUSIVE_FLAGS:
        if getattr(args, flag, None):
            print(f"{name} is mutually exclusive with {why} ({spec_ref})",
                  file=sys.stderr)
            return 1
    if getattr(args, "audience", "technical") != "technical":
        print(f"{name} is mutually exclusive with a non-technical "
              f"--audience ({spec_ref})", file=sys.stderr)
        return 1
    return None


def _cmd_report_format(args, fmt: str, summary: dict, entries, run_dir: Path):
    """SPEC.md 63.1 (v0.49, B2): the technical report in another format —
    ``md`` / ``txt`` to stdout (wikis / PRs / email / terminals), ``pdf``
    to ``report.pdf`` in the run dir (``reportlab`` is an optional extra;
    a clean rc 1 with the install hint when it is absent). One shared
    ``doc`` (``build_report_doc``) feeds all three renderers, so a
    re-render of the same doc is byte-identical (G2). ``--plot`` is a
    no-op (the document is self-contained)."""
    rc = _exclusive_guard(f"--format {fmt}", args, "SPEC.md 63.1.3")
    if rc is not None:
        return rc
    if getattr(args, "user", False) or getattr(args, "decision", False):
        print("--format is mutually exclusive with --user/--decision "
              "(SPEC.md 63.1.3)", file=sys.stderr)
        return 1
    doc = build_report_doc(summary, entries)
    if fmt == "pdf":
        out = run_dir / "report.pdf"
        try:
            render_report_pdf(doc, out)
        except ImportError as exc:  # reportlab is an optional extra
            print(str(exc), file=sys.stderr)
            return 1
        print(f"pdf     : {out}")
        return 0
    print(render_report(fmt, doc))
    return 0


def _user_guide_sources(summary: dict, run_dir: Path):
    """SPEC.md 63.2 (v0.49, B3): the ``(task, model)`` for the end-user
    guide — each degrades to ``None`` independently (the `_report_extras`
    rule), so a known task with a missing / unreadable best model still
    yields the "what it predicts" + "how to read" blocks."""
    task = None
    try:
        task = _task_from_summary(summary)
    except (ValueError, FileNotFoundError):
        task = None
    model = None
    if task is not None:
        family = (summary.get("best_spec") or {}).get("model_family", "mlp")
        loaders = _best_model_loaders()
        if family in loaders:
            try:
                model = loaders[family](str(run_dir / "best_model.npz"))
            except (ValueError, FileNotFoundError):
                model = None
    return task, model


def _cmd_report_user(args, summary: dict, entries, run_dir: Path) -> int:
    """SPEC.md 63.2 (v0.49, B3): the "what this model does" guide for the
    person who will *consume* the predictions — what it predicts, a few
    real input->output holdout examples, when to distrust it, known
    limitations, and how to read the number. One bounded holdout pass
    (28.2); episode / synthetic tasks degrade to an explanatory note
    (never invented rows). Informational, rc 0."""
    rc = _exclusive_guard("--user", args, "SPEC.md 63.2.3")
    if rc is not None:
        return rc
    if getattr(args, "fmt", None) is not None or getattr(args, "decision", False):
        print("--user is mutually exclusive with --format/--decision "
              "(SPEC.md 63.2.3)", file=sys.stderr)
        return 1
    task, model = _user_guide_sources(summary, run_dir)
    # 64.2.2 (B6): the ± read on the headline number (None-safe — a single
    # run has no spread to display; the A53 pin keeps its output intact).
    hl = headline_uncertainty(
        summary.get("final_best_score"), _seed_spread_for(run_dir, summary))
    view = user_guide_view(summary, entries, task=task, model=model,
                           headline=hl)
    print(render_user_guide(view))
    return 0


def _cmd_report_decision(args, summary: dict, entries,
                         run_dir: Path) -> int:
    """SPEC.md 63.3 (v0.49, B4): the go/no-go decision artifact — verdict
    (the 62.2.2 rule, one home) + target margin, the confidence read
    (same-task seed variance + the logged CI / overfit rejections), the
    top-3 failure modes, and one concrete next step (the 41.1 projection
    with ``--target`` feeds it: "run ~N more" / "relax or expand" /
    "collect history"). The ``seed_spread`` is the registry's same-task
    final scores (the 41.1.2 points). Informational, rc 0."""
    rc = _exclusive_guard("--decision", args, "SPEC.md 63.3.3")
    if rc is not None:
        return rc
    if getattr(args, "fmt", None) is not None or getattr(args, "user", False):
        print("--decision is mutually exclusive with --format/--user "
              "(SPEC.md 63.3.3)", file=sys.stderr)
        return 1
    proj = None
    pts: list = []
    try:  # the 41.1 machinery — degrade to None on a missing registry
        reg = load_registry(run_dir.parent)
        pts = projection_points(reg, summary, run_dir.name)
        e_cur = summary.get("experiments_run")
        if not (isinstance(e_cur, (int, float))
                and not isinstance(e_cur, bool)):
            e_cur = 0.0
        proj = project_budget(pts, args.target, e_cur)
    except Exception:
        proj = None
    seed_spread = [s for _e, s in pts] if pts else None
    diag = _report_extras(summary, run_dir)["diagnostics"]  # None-safe (28.5)
    view = decision_view(summary, entries, proj=proj,
                         seed_spread=seed_spread, diag=diag)
    print(render_decision(view))
    return 0


def _domain_calibration(summary: dict, run_dir):
    """SPEC.md 62.3 (A52): the domain calibration / difficulty extras — the
    holdout ECE (the 61.4 ``ece`` over the model's softmax) and the 58.1
    difficulty ranking, for a softmax fitting task with a reconstructible
    best model. Degrades to ``(None, None)`` on any failure (None-safe, like
    ``_report_extras``)."""
    try:
        task = _task_from_summary(summary)
        family = (summary.get("best_spec") or {}).get("model_family", "mlp")
        loaders = _best_model_loaders()
        if task is None or family not in loaders \
                or getattr(task, "head", None) != "softmax":
            return None, None
        model = loaders[family](str(run_dir / "best_model.npz"))
        x, y = task.holdout_rows(200, model)
        if len(x) == 0:
            return None, None
        logits = np.asarray(model.forward(x), dtype=np.float64)
        probs = _softmax(logits)
        labels = np.asarray(y, dtype=np.int64).ravel()
        e = float(_ece(probs, labels))
        diff = holdout_difficulty(task, model)
        return e, diff
    except (FileNotFoundError, ValueError, Exception):
        return None, None


def _regulator_sources(summary: dict, entries, run_dir):
    """SPEC.md 62.4 (A52): the regulator chain-of-custody sources — the
    56.1 provenance payload, the 41.2 decision trace, and the 62.4 data
    fingerprint. The ``target`` follows the ``_cmd_report --certificate``
    rule (the summary, else the canonical run_config)."""
    rc_cfg = None
    rc_path = run_dir / "run_config.json"
    if rc_path.is_file():
        try:
            rc_cfg = json.loads(rc_path.read_text(encoding="utf-8"))
        except Exception:
            rc_cfg = None
    _target = summary.get("target")
    if _target is None and isinstance(summary.get("run_config"), dict):
        _target = summary["run_config"].get("target")
    provenance = provenance_payload(
        env_provenance(), seed=summary.get("seed"),
        task=summary.get("task"), target=_target, run_config=rc_cfg)
    trace = trace_lines(entries)
    data = data_fingerprint(summary.get("task_config"))
    return provenance, trace, data


def _softmax(logits) -> "np.ndarray":
    """A stable softmax (the 61.4 ECE needs probabilities; we compute it
    inline rather than importing ``calibration``'s private helper)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cmd_verify(args: argparse.Namespace) -> int:
    """SPEC.md 62.5 (A52): ``autorefine verify --run DIR`` — independently
    re-derive every reported number from the run's artifacts and assert
    them. rc 0 = all checks pass; rc 2 = one or more checks FAIL (a number
    the log does not support); rc 1 = a missing / invalid run dir."""
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"verify: no such run dir: {run_dir}", file=sys.stderr)
        return 1
    result = verify_run(run_dir)
    print(render_verify(result, run_dir))
    if result.get("error"):
        return 1
    return 0 if result["passed"] else 2


def _cmd_report_nonlinearity(args, summary: dict, entries,
                             run_dir: Path) -> int:
    """SPEC.md 82.5 (v0.68): the ``report --nonlinearity`` human view —
    the 82.2.1 profile printed as a labeled block (verdict + gap, the
    linear best, the overall best, the per-family table, the
    spectral-expansion line, the steer hint). ``rc 0`` on success; the
    default report (no flag) stays byte-identical (the 63.5 pin
    anchor)."""
    from .research import nonlinearity_profile
    prof = nonlinearity_profile(entries)
    print(f"\nnonlinearity profile ({prof['n_scored']} scored "
          f"candidates, {prof['n_families']} families)")
    if prof["n_scored"] == 0:
        print("  no scored candidates to profile")
        return 0
    gap = prof["gap"]
    gap_s = f"{gap:.1f}" if gap is not None else "n/a (no linear candidate)"
    print(f"  verdict      : {prof['verdict']} (gap {gap_s})")
    lin = prof["linear"]
    if lin["best"] is not None:
        print(f"  linear best  : {lin['best']:.1f}  "
              f"(depth-0 mlp, {lin['n']} candidate"
              f"{'s' if lin['n'] != 1 else ''})")
    else:
        print("  linear best  : n/a (the run never proposed a depth-0 mlp)")
    ov = prof["overall"]
    print(f"  overall best : {ov['best']:.1f}  (family: {ov['family']})")
    print("  families     :")
    for f in prof["families"]:
        delta = f["delta"]
        d_s = f"{delta:+.1f} over linear" if delta is not None \
            else "delta n/a"
        print(f"    {f['name']:<8s} best {f['best']:6.1f}  "
              f"({f['n']} candidate{'s' if f['n'] != 1 else ''}, {d_s})")
    fourier = prof["fourier"]
    if fourier["on_mean"] is not None and fourier["off_mean"] is not None:
        print(f"  spectral     : fourier_features>0 on "
              f"{fourier['on_mean']:.1f} vs off "
              f"{fourier['off_mean']:.1f} "
              f"(delta {fourier['delta']:+.1f})")
    else:
        print("  spectral     : the spectral expansion (fourier_features) "
              "was not tried in this run")
    print(f"  steer        : {prof['steer']}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    # 89.9 (v0.75, C2): --backtest replays the loop over expanding windows
    # of the given CSV — exclusive with every run-level view below
    if getattr(args, "backtest", False):
        for flag, why in (
                ("run", "--run"), ("history", "--history"),
                ("benchmark", "--benchmark"), ("json", "--json"),
                ("what_if", "--what-if"), ("project", "--project"),
                ("trace", "--trace"), ("importance", "--importance"),
                ("certificate", "--certificate"), ("fmt", "--format"),
                ("user", "--user"), ("decision", "--decision"),
                ("nonlinearity", "--nonlinearity")):
            if getattr(args, flag, None) not in (None, False, []):
                print(f"--backtest and {why} are mutually exclusive "
                      f"(SPEC.md 89.9)", file=sys.stderr)
                return 1
        if getattr(args, "audience", "technical") not in (None, "technical"):
            print("--backtest and --audience are mutually exclusive "
                  "(SPEC.md 89.9)", file=sys.stderr)
            return 1
        return _cmd_report_backtest(args)
    # SPEC.md 64.1 (v0.50, B5): --benchmark is a runs-dir-level view (like
    # --history): mutually exclusive with --run, --history, and every
    # single-run human view (64.3.1 — the guard table is symmetric); it
    # shares --runs-dir (and --json) with --history.
    if getattr(args, "benchmark", False):
        if args.run is not None:
            print("--run and --benchmark are mutually exclusive "
                  "(SPEC.md 64.1.5)", file=sys.stderr)
            return 1
        if args.history:
            print("--history and --benchmark are mutually exclusive "
                  "(SPEC.md 64.1.5)", file=sys.stderr)
            return 1
        for flag, why in (
            ("what_if", "--what-if"), ("project", "--project"),
            ("trace", "--trace"), ("importance", "--importance"),
            ("certificate", "--certificate"), ("fmt", "--format"),
            ("user", "--user"), ("decision", "--decision"),
        ):
            if getattr(args, flag, None):
                print(f"--benchmark is mutually exclusive with {why} "
                      f"(SPEC.md 64.1.5)", file=sys.stderr)
                return 1
        if getattr(args, "audience", "technical") != "technical":
            print("--benchmark is mutually exclusive with a non-technical "
                  "--audience (SPEC.md 64.1.5)", file=sys.stderr)
            return 1
        return _cmd_report_benchmark(args)
    # SPEC.md 40.2.1 (v0.26, S2): --what-if is a human counterfactual view,
    # mutually exclusive with the machine (--json) and history paths.
    if getattr(args, "what_if", None):
        if args.history:
            print("--what-if and --history are mutually exclusive "
                  "(SPEC.md 40.2.1)", file=sys.stderr)
            return 1
        if getattr(args, "benchmark", False):
            print("--what-if and --benchmark are mutually exclusive "
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
                or getattr(args, "trace", False)
                or getattr(args, "benchmark", False)):
            print("--importance is mutually exclusive with "
                  "--json/--history/--what-if/--project/--trace/--benchmark "
                  "(SPEC.md 44.2.2)", file=sys.stderr)
            return 1
    # SPEC.md 56.1.1 (v0.42): --certificate is a human view — the
    # provenance card on stdout + the cover in report.html (with --html) —
    # mutually exclusive with the machine path (--json) and --history (it
    # needs a run dir). The default path (no flag) is byte-identical (56.6).
    if getattr(args, "certificate", False):
        if args.json:
            print("--certificate and --json are mutually exclusive "
                  "(SPEC.md 56.1.1)", file=sys.stderr)
            return 1
        if args.history:
            print("--certificate and --history are mutually exclusive "
                  "(SPEC.md 56.1.1)", file=sys.stderr)
            return 1
        if getattr(args, "benchmark", False):
            print("--certificate and --benchmark are mutually exclusive "
                  "(SPEC.md 56.1.1)", file=sys.stderr)
            return 1
    # SPEC.md 41.1.1/41.2.1 (v0.27): --project / --trace are human views,
    # mutually exclusive with the machine (--json), the history path, and
    # the what-if path (A11 untouched).
    if getattr(args, "project", False) or getattr(args, "trace", False):
        if args.history:
            print("--project/--trace and --history are mutually exclusive "
                  "(SPEC.md 41.1.1/41.2.1)", file=sys.stderr)
            return 1
        if getattr(args, "benchmark", False):
            print("--project/--trace and --benchmark are mutually exclusive "
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
    # SPEC.md 82.5 (v0.68): --nonlinearity is a human view (the 63.2/
    # 63.4 --user/--decision pattern) — mutually exclusive with --history;
    # the default report (no flag) stays byte-identical (the 63.5 pin
    # anchor). Guarded before the --history dispatch so the pair is a
    # loud error, not a silent history report.
    if getattr(args, "nonlinearity", False) and args.history:
        print("--nonlinearity and --history are mutually exclusive "
              "(SPEC.md 82.5)", file=sys.stderr)
        return 1
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
    # SPEC.md 62.1 (v0.48, B1): the audience axis — exec / domain / regulator
    # re-skin the same logged data; `technical` (the default) falls through to
    # the byte-identical path below (62.1.4 / the A52 pin anchor).
    audience = getattr(args, "audience", "technical")
    if audience in ("exec", "domain", "regulator"):
        return _cmd_report_audience(args, audience, summary, entries, run_dir)
    # SPEC.md 63 (v0.49, B2/B3/B4): three more human renderings of the same
    # logged data — a format (md/txt/pdf), the end-user guide, and the
    # decision artifact. Each is opt-in; the default path (no new flag)
    # stays byte-identical (63.5, the A53 pin anchor).
    if getattr(args, "fmt", None) is not None:
        return _cmd_report_format(args, args.fmt, summary, entries, run_dir)
    if getattr(args, "user", False):
        return _cmd_report_user(args, summary, entries, run_dir)
    if getattr(args, "decision", False):
        return _cmd_report_decision(args, summary, entries, run_dir)
    # SPEC.md 82.5 (v0.68): the nonlinearity profile — a human view over
    # the same logged entries; dispatches before the default (byte-
    # identical) report path.
    if getattr(args, "nonlinearity", False):
        return _cmd_report_nonlinearity(args, summary, entries, run_dir)
    # SPEC.md 56.1.1 (v0.42): the provenance certificate — the text card on
    # stdout + (with --html) the cover block in report.html. The payload is
    # pure over the run's identity + the env facts (56.1); `target` comes
    # from the summary or the canonical run_config (37.1); the run-config
    # hash is the 56.1 canonical hash of `run_config.json`.
    cert_cover = None
    cert_payload = None
    if getattr(args, "certificate", False):
        rc_cfg = None
        rc_path = run_dir / "run_config.json"
        if rc_path.is_file():
            try:
                rc_cfg = json.loads(rc_path.read_text(encoding="utf-8"))
            except Exception:
                rc_cfg = None
        _rcd = summary.get("run_config")
        _target = summary.get("target")
        if _target is None and isinstance(_rcd, dict):
            _target = _rcd.get("target")
        cert_payload = provenance_payload(
            env_provenance(), seed=summary.get("seed"),
            task=summary.get("task"), target=_target, run_config=rc_cfg)
        cert_cover = html_cover(summary, cert_payload)
        print(provenance_card(cert_payload))
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
        # 56.2 (v0.42): the brand masthead + provenance cover — present only
        # with --certificate (cert_cover is None otherwise; byte-identical)
        (run_dir / "report.html").write_text(
            html_report(summary, entries, diagnostics=extras["diagnostics"],
                        gallery=extras["gallery"], arch_svg=extras["arch_svg"],
                        cover=cert_cover),
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


def _cmd_quickstart(args: argparse.Namespace) -> int:
    """SPEC.md 65.3 (v0.51): print the Quickstart — the same steps the
    README and the dashboard's first screen carry (one source, 65.1).
    ``--json`` prints the step model (``{intro, steps}``) instead of the
    text rendering. Zero training, zero writes; rc 0 (65.3.2)."""
    if getattr(args, "json", False):
        print(json.dumps({"intro": QUICKSTART_INTRO,
                          "steps": quickstart_steps()}, indent=2))
        return 0
    print(render_quickstart())
    return 0


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


def _load_gp(path: str):
    from .models.gp import GP
    return GP.load(path)


def _load_gam(path: str):
    from .models.gam import GAM
    return GAM.load(path)


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
        "gp": lambda p: _load_gp(p),        # SPEC.md 75 (v0.61)
        "gam": lambda p: _load_gam(p),      # SPEC.md 75 (v0.61)
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
    """`predict` (SPEC.md 42.1, v0.28; 88.3, v0.74): score NEW rows with
    the run's best model. The run is named (`--run DIR`, explicit wins)
    or the most recent finished run from the registry (`--latest`,
    88.3.1); neither is a fail-loud rc 1 (88.3.2). Exactly one input mode
    (42.1.1); fitting tasks only (42.1.2); preprocessing = the task's own
    train stats (42.1.3); one forward pass per row (42.1.4). The scoring
    path is 42.1's, byte-identical."""
    if args.run:
        run_dir = Path(args.run)
    elif getattr(args, "latest", False):
        # 88.3 (v0.74, A3): the registry's most recent run (timestamp,
        # then run_id — the 88.3.1 tie-break)
        try:
            entries = load_registry(args.runs_dir)
        except Exception:
            entries = []
        if not entries:
            print(f"predict: no finished runs in {args.runs_dir} - run "
                  f"autorefine go or fit first (SPEC.md 88.3.2)",
                  file=sys.stderr)
            return 1
        best = max(entries, key=lambda e: (str(e.get("timestamp") or ""),
                                          str(e.get("run_id") or "")))
        run_dir = Path(args.runs_dir) / str(best.get("run_id", "?"))
        if not (run_dir / "summary.json").is_file():
            print(f"predict: the latest run {run_dir} has no summary.json - "
                  f"finish it first (SPEC.md 88.3.2)", file=sys.stderr)
            return 1
        score = best.get("final_score")
        score_s = (f"{float(score):g}"
                   if isinstance(score, (int, float)) and not isinstance(score, bool)
                   else "?")
        print(f"using   : latest run {best.get('run_id')} (task "
              f"{best.get('task')}, final {score_s}) (SPEC.md 88.3)")
    else:
        print("predict needs --run DIR or --latest (SPEC.md 88.3)",
              file=sys.stderr)
        return 1
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
    """`compare` (SPEC.md 42.2, v0.28; 50.1.4, v0.36): the app's
    Past-runs compare widget (38.4) in the terminal — two runs via
    `diff_two_summaries` (core `dashboard`, 42.2.2), three runs via
    `diff_n_summaries` (50.1.1). The two-run output is byte-identical
    to 42.2.3 (the A32 pin); one or ≥ 4 run dirs is rc 1."""
    if len(args.runs) not in (2, 3):
        print("compare needs two or three run dirs: --run A --run B [--run C] "
              "(SPEC.md 42.2.1, 50.1.4)", file=sys.stderr)
        return 1
    summaries = []
    for d in args.runs:
        p = Path(d) / "summary.json"
        if not p.is_file():
            print(f"no summary.json in run dir {d!r}", file=sys.stderr)
            return 1
        summaries.append(json.loads(p.read_text(encoding="utf-8")))
    if len(args.runs) == 2:
        # --- the pinned 42.2.3 surface (A32): byte-identical, unchanged ---
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
    # --- three runs (50.1.4): the diff_n_summaries gate rows + spec table ---
    tagged = [dict(s) for s in summaries]  # 50.1.4: never mutate the loaded dicts
    for i, s in enumerate(tagged):
        s["run_id"] = Path(args.runs[i]).name
    diff = diff_n_summaries(tagged)
    if args.json:
        print(json.dumps(diff, indent=2, sort_keys=True))
        return 0
    for i, row in enumerate(diff["runs"]):
        gate = {True: "PASS", False: "MISS"}.get(row["met_target"], "—")
        print(f"run {i + 1} : {args.runs[i]}")
        print(f"  score : {row['final_score']} | target : {row['target']} "
              f"| gate : {gate}")
        print(f"  exps  : {row['experiments_run']} | wall   : "
              f"{row['wall_seconds']}s | policy : {row['policy']}")
    if diff["spec_fields"]:
        print(f"best_spec diff ({len(diff['spec_fields'])} field(s)):")
        for row in diff["spec_fields"]:
            vals = " | ".join("—" if v is None else str(v) for v in row["values"])
            print(f"  {row['field']:<24s} {vals}")
    else:
        print("best_spec: identical")
    if diff["best_run_id"] is not None:
        print(f"best    : {diff['best_run_id']} "
              f"(span {diff['score_span']})")
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

    v0.36 (50.2.1): the payload/zip machinery moved to the `sharing`
    core (the app's "Build share bundle" panel, 50.2.3, shares it) —
    this command's contract is byte-identical (47.4.3).
    """
    run_dir = Path(args.run)
    if not (run_dir / "summary.json").is_file():
        print(f"no summary.json in run dir {str(run_dir)!r} — finish the run "
              f"first (SPEC.md 47.4.4)", file=sys.stderr)
        return 1
    out = (Path(args.out) if args.out is not None
           else run_dir.parent / f"{run_dir.name}-share.zip")
    payload = share_payload(run_dir, _share_report_html(run_dir))  # 47.4.2/50.2.1
    write_share_zip(payload, out)
    for name in sorted(payload):
        print(f"  {name:24s} {len(payload[name]):8d} bytes")
    print(f"share   : {out}")
    return 0


def _cmd_dossier(args: argparse.Namespace) -> int:
    """`dossier` (SPEC.md 58.3, v0.44): the run dossier — one scrollable,
    self-contained HTML file (inline ``<style>``, no external assets,
    22.2) combining the whole story of a finished run:

    1. **Summary** — the headline numbers (the report's table, 22.1).
    2. **Recipe** — the canonical `run_config.json` (37.1) as the
       copy-pasteable `format_recipe` line (33.2/47.1).
    3. **Provenance** — the `svg_provenance` certificate (56.1).
    4. **Frontier** — the 2-objective `svg_pareto` (25.5) plus the
       3-objective `svg_frontier3` (58.2) score × time × size view.
    5. **Lineage** — the spec-lineage graph (57.2).
    6. **Sensitivity** — the per-field response surfaces (57.2).
    7. **Error analysis** — the per-class bars + confusion matrix (28.2),
       the per-input difficulty ranking (58.1), and the media error
       gallery (28.3) when the task supports it.

    Pure rendering of artifacts already on disk (zero new training, G2):
    a re-render of the same run dir is byte-identical (no timestamps in
    the output). The task/model geometry (state_dim, n_out, the conv
    grid) comes from the reconstructed task — the
    `_run_task_and_model` contract (42).

    rc 0 with the written path; rc 1 when the run dir is missing or has
    no `summary.json` (the `_cmd_share` contract, 47.4.4).
    """
    run_dir = Path(args.run)
    try:
        _summary, task, model = _run_task_and_model(run_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"dossier: {exc} (SPEC.md 58.3/42)", file=sys.stderr)
        return 1
    diag = holdout_diagnostics(task, model)
    gallery: list | None = None
    hold_errors = getattr(task, "holdout_errors", None)
    if callable(hold_errors):  # 28.3: media tasks only
        g = hold_errors(model)
        gallery = g if g else None  # no misclassifications -> no gallery
    out = (Path(args.out) if args.out is not None
           else run_dir / "dossier.html")
    html_text = build_dossier(
        run_dir,
        state_dim=task.state_dim,
        n_out=task.n_outputs,
        grid=getattr(task, "feature_grid", None),
        difficulty=holdout_difficulty(task, model),
        gallery=gallery,
        per_class_svg=svg_per_class_bars(diag) if diag else None,
        confusion_svg=svg_confusion_matrix(diag) if diag else None,
    )
    out.write_text(html_text, encoding="utf-8")
    print(f"dossier : {out}")
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
  autorefine report --benchmark   # v0.50 (SPEC.md 64.1): the longitudinal view
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
_EP_ASK = """examples:
  autorefine ask "reach 96% on this churn table, quick"
  autorefine ask "at least 90 percent on my photos" --data examples/data
"""
_EP_GO = """examples:
  autorefine go --data examples/data/churn_sample.csv
  autorefine go --data sales.csv --target 93 --experiments 20
"""
_EP_CARD = """examples:
  autorefine card --run runs/<run_id>
  autorefine card --run runs/<run_id> --md
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
  autorefine compare --run A --run B --run C   # v0.36 (SPEC.md 50.1.4)
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
_EP_DOSSIER = """examples:
  autorefine dossier --run runs/<run_id>
  autorefine dossier --run runs/<run_id> --out dossier.html
"""
_EP_QUICKSTART = """examples (SPEC.md 65.3):
  autorefine quickstart
  autorefine quickstart --json
"""

_EP_MANUAL = """examples (manual/expert mode, SPEC.md 59.3):
  autorefine manual --task parity-v1 \\
      --set architecture=[16,8] --set model_family=mlp --set train_steps=400
  autorefine manual --data sales.csv --label churn \\
      --set model_family=mlp --set architecture=[32,16]
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
    p_run.add_argument("--rl-save", default=None, metavar="PATH",
                       help="v0.53 (SPEC.md 67.4): save the trained policy "
                            "(learn → persist → reuse)")
    p_run.add_argument("--rl-load", default=None, metavar="PATH",
                       help="v0.53 (SPEC.md 67.4): resume a policy saved with "
                            "--rl-save before training")
    p_run.add_argument("--rl-epsilon", type=float, default=0.0,
                       help="v0.53 (SPEC.md 67.2): ε-greedy exploration in "
                            "[0, 1] (0 = pure softmax, the default)")
    p_run.add_argument("--rl-epsilon-decay", type=float, default=1.0,
                       help="v0.53 (SPEC.md 67.2): per-episode ε decay "
                            "factor in (0, 1] (1.0 = no decay)")
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
    p_run.add_argument("--from-model", default=None, metavar="NPZ",
                       help="v0.66 (SPEC.md 80): start from an uploaded "
                            "trained model (*.npz) — scored as the baseline, "
                            "its weights warm-start compatible candidates; "
                            "default: none (a fresh baseline)")
    p_run.add_argument("--prior-run", default=None, metavar="RUN_DIR",
                       help="v0.72 (SPEC.md 86): start from a finished run's "
                           "search knowledge — its best spec becomes the "
                           "baseline (retrained on this task) and the bandit "
                           "starts informed; default: none (a fresh run)")
    p_run.add_argument("--demo", action="store_true",
                       help="v0.27 (SPEC.md 41.3): the narrated demo — one tiny, "
                            "deterministic parity-v1 loop (3 experiments, 60 s "
                            "wall) printed with the full decision trace; "
                            "--task/--experiments/etc. are ignored")
    p_run.add_argument("--tour", action="store_true",
                       help="v0.73 (SPEC.md 87.2): the plain-language tour — "
                            "the same tiny parity-v1 loop told in three plain "
                            "beats + a one-sentence verdict (no jargon); "
                            "--task/--experiments/etc. are ignored")
    p_run.add_argument("--quiet", action="store_true",
                       help="v0.33 (SPEC.md 47.3): suppress the per-experiment "
                            "loop output (keep the === summary === block); "
                            "--demo ignores it (already minimal, 41.3)")
    # SPEC.md 59.2 (v0.45): the human-in-the-loop steering verbs
    p_run.add_argument("--pin", action="append", default=[],
                       metavar="FIELD=VALUE",
                       help="v0.45 (SPEC.md 59.2): freeze a spec field to a value "
                            "(every candidate + the baseline); e.g. "
                            "--pin architecture=[16,8] (repeatable)")
    p_run.add_argument("--bias", action="append", default=[],
                       metavar="FIELD=VALUE",
                       help="v0.45 (SPEC.md 59.2): redirect a mutation of a spec "
                            "field to a value; e.g. --bias learning_rate=1e-3 "
                            "(repeatable)")
    p_run.add_argument("--constrain", action="append", default=[],
                       metavar="FIELD=V1,V2",
                       help="v0.45 (SPEC.md 59.2): reject candidates outside an "
                            "allowed value set (invalid_spec); e.g. "
                            "--constrain train_steps=200,400 (repeatable)")
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
    p_rep.add_argument("--certificate", action="store_true",
                       help="v0.42 (SPEC.md 56.1): the provenance certificate "
                            "— the text card on stdout (config hash, version, "
                            "seed, dep pins, git sha) + the cover block in "
                            "report.html (with --html); mutually exclusive "
                            "with --json/--history")
    p_rep.add_argument("--plot", action="store_true",
                       help="v0.7 (SPEC.md 21.2): ASCII charts on stdout + "
                            "score_curve.svg / pareto_frontier.svg in the run dir")
    p_rep.add_argument("--html", action="store_true",
                       help="v0.8 (SPEC.md 22.2): write a self-contained "
                            "report.html (embedded SVGs + tables) into the run dir")
    p_rep.add_argument("--audience",
                       choices=AUDIENCES, default="technical",
                       help="v0.48 (SPEC.md 62.1): the reader the report is "
                            "written for — exec (go/no-go, one screen), domain "
                            "(per-class / calibration / where-it-fails), "
                            "technical (the default, today's report, "
                            "byte-identical), or regulator (chain-of-custody + "
                            "decision trace); mutually exclusive with "
                            "--json/--what-if/--project/--trace/--importance/"
                            "--certificate; --html writes report_<audience>.html")
    p_rep.add_argument("--format", dest="fmt", default=None,
                       choices=REPORT_FORMATS,
                       help="v0.49 (SPEC.md 63.1): render the technical "
                            "report in another format — 'md' (Markdown) and "
                            "'txt' (plain text) to stdout, 'pdf' to "
                            "report.pdf in the run dir (needs the optional "
                            "reportlab extra); mutually exclusive with "
                            "--json/--audience/--what-if/--project/--trace/"
                            "--importance/--certificate/--html; --plot is a "
                            "no-op")
    p_rep.add_argument("--user", action="store_true",
                       help="v0.49 (SPEC.md 63.2): the end-user guide — what "
                            "this model does, real input->output holdout "
                            "examples, when to distrust it, and how to read "
                            "the number (for the person who consumes the "
                            "predictions, not the one who trained it); "
                            "mutually exclusive with the other human views "
                            "and --json; informational, rc 0")
    p_rep.add_argument("--decision", action="store_true",
                       help="v0.49 (SPEC.md 63.3): the go/no-go decision "
                            "artifact — verdict + margin, confidence (seed "
                            "variance + the logged CI/overfit rejections), "
                            "top-3 failure modes, and one concrete next step "
                            "(--target feeds the 41.1 projection); mutually "
                            "exclusive with the other human views and --json; "
                            "informational, rc 0")
    p_rep.add_argument("--nonlinearity", action="store_true",
                       help="v0.68 (SPEC.md 82.5): the nonlinearity profile "
                            "— how much of the task's difficulty is "
                            "nonlinear (the depth-0 linear best vs the "
                            "overall best), the per-family deltas, and the "
                            "measured spectral-expansion effect; a human "
                            "view, mutually exclusive with --history; "
                            "informational, rc 0")
    p_rep.add_argument("--benchmark", action="store_true",
                       help="v0.50 (SPEC.md 64.1): the benchmark / longitudinal "
                            "report across the run registry — the per-task "
                            "leaderboard (best score + spec fingerprint + gate), "
                            "the same-spec-across-tasks generalization matrix, "
                            "and the best-score trend across runs; uses "
                            "--runs-dir (like --history); --json for the "
                            "machine form; mutually exclusive with --run, "
                            "--history, and the single-run views")
    # v0.75 (SPEC.md 89.9, C2): the expanding-window backtest replay
    p_rep.add_argument("--backtest", action="store_true",
                       help="v0.75 (SPEC.md 89.9): replay the loop over "
                            "expanding windows of --data (window i = the "
                            "first ceil(n*i/--windows) rows; the 'if we had "
                            "started with less data' axis); one line per "
                            "window; mutually exclusive with the run-level "
                            "views")
    p_rep.add_argument("--data", default=None, metavar="CSV",
                       help="v0.75 (SPEC.md 89.9): the CSV for --backtest")
    p_rep.add_argument("--seed", type=int, default=7,
                       help="v0.75 (SPEC.md 89.9): the backtest loops' seed "
                            "(default 7 - deterministic)")
    p_rep.add_argument("--windows", type=int, default=3,
                       help="v0.75 (SPEC.md 89.9): K backtest windows "
                            "(default 3)")
    p_rep.add_argument("--budget", type=int, default=3,
                       help="v0.75 (SPEC.md 89.9): each backtest window's "
                            "experiment budget (default 3)")
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
    p_fit.add_argument("--rl-epsilon", type=float, default=0.1,
                       help="v0.59 (SPEC.md 73.3): ε-greedy exploration in "
                            "[0, 1] for the meta-RL improver (default 0.1 — "
                            "SPEC.md 73.9: ε=0 deadlocks the loop — the greedy "
                            "policy re-proposes its initial action, the "
                            "duplicate-stall guard ends the episode, and the "
                            "policy never learns; pass --rl-epsilon 0 to "
                            "restore the pre-v0.59 pure-softmax path)")
    p_fit.add_argument("--rl-epsilon-decay", type=float, default=1.0,
                       help="v0.59 (SPEC.md 73.3): per-episode ε decay "
                            "factor in (0, 1] (1.0 = no decay)")
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
    p_fit.add_argument("--from-model", default=None, metavar="NPZ",
                       help="v0.66 (SPEC.md 80): start from an uploaded "
                            "trained model (*.npz) — scored as the baseline, "
                            "its weights warm-start compatible candidates; "
                            "default: none (a fresh baseline)")
    p_fit.add_argument("--prior-run", default=None, metavar="RUN_DIR",
                       help="v0.72 (SPEC.md 86): start from a finished run's "
                           "search knowledge — its best spec becomes the "
                           "baseline (retrained on this task) and the bandit "
                           "starts informed; default: none (a fresh run)")
    p_fit.add_argument("--continue", dest="continue_", action="store_true",
                       default=False,
                       help="v0.74 (SPEC.md 88.6): start from the most recent "
                            "same-task run in the registry — one-flag alias "
                            "for --prior-run over the registry (an explicit "
                            "--prior-run wins; --data mode only)")
    p_fit.add_argument("--quiet", action="store_true",
                       help="v0.33 (SPEC.md 47.3): suppress the per-experiment "
                            "loop output and the per-class diagnostics (keep "
                            "the gate verdict + the === summary === block); "
                            "--dry-run ignores it (already minimal, 40.1)")
    # SPEC.md 59.2 (v0.45): the human-in-the-loop steering verbs
    p_fit.add_argument("--pin", action="append", default=[],
                       metavar="FIELD=VALUE",
                       help="v0.45 (SPEC.md 59.2): freeze a spec field to a value "
                            "(every candidate + the baseline); e.g. "
                            "--pin architecture=[16,8] (repeatable)")
    p_fit.add_argument("--bias", action="append", default=[],
                       metavar="FIELD=VALUE",
                       help="v0.45 (SPEC.md 59.2): redirect a mutation of a spec "
                            "field to a value; e.g. --bias learning_rate=1e-3 "
                            "(repeatable)")
    p_fit.add_argument("--constrain", action="append", default=[],
                       metavar="FIELD=V1,V2",
                       help="v0.45 (SPEC.md 59.2): reject candidates outside an "
                            "allowed value set (invalid_spec); e.g. "
                            "--constrain train_steps=200,400 (repeatable)")
    # v0.75 (SPEC.md 89): time-based and on-demand data ingestion
    p_fit.add_argument("--window", default=None, metavar="7d",
                       help="v0.75 (SPEC.md 89.7): train on the recency slice "
                            "only - keep rows whose date column is within "
                            "7d/24h/90m/30s of the file's newest date "
                            "(CSV with a date column; --data mode only)")
    p_fit.add_argument("--window-col", default=None,
                       help="v0.75 (SPEC.md 89.7): name the date column "
                            "explicitly (default: auto-detect the usual "
                            "date names, then any parseable first match)")
    p_fit.add_argument("--dataset", default=None, metavar="ID",
                       help="v0.75 (SPEC.md 89.4): fit a snapshot by its "
                            "dataset id (or unique prefix) instead of --data "
                            "- the sha256 is verified before the loop and the "
                            "entry lands in the run dir as dataset.json")
    p_fit.add_argument("--latency-ms", type=float, default=None,
                       metavar="MS",
                       help="v0.75 (SPEC.md 89.11): also gate on the best "
                            "model's measured median ms/row - OVER the budget "
                            "is a MISS (rc 2), like the score bar")
    _add_config_flag(p_fit)  # 47.1.2
    p_fit.set_defaults(func=_cmd_fit)

    # v0.73 (SPEC.md 87.1, A1): the plain-language goal entry
    p_ask = sub.add_parser(
        "ask", help="v0.73 (SPEC.md 87.1): say what you want in your own "
                   "words — get back a plain plan + the fit command to copy "
                   "(no training)",
        epilog=_EP_ASK,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_ask.add_argument("goal",
                       help='the goal in a sentence, e.g. "reach 96% on '
                            'this churn table, quick"')
    p_ask.add_argument("--data", default=None,
                       help="optional data path to fold into the plan "
                            "(file or directory)")
    _add_config_flag(p_ask)  # 47.1.2
    p_ask.set_defaults(func=_cmd_ask)

    # v0.73 (SPEC.md 87.3, A3): `fit` with a data-sized default budget
    p_go = sub.add_parser(
        "go", help="v0.73 (SPEC.md 87.3): fit your data with a budget sized "
                   "to the data (or --experiments to override) and a plain-"
                   "language verdict after the gate",
        epilog=_EP_GO,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_go.add_argument("--data", default=None,
                      help="CSV file, or a directory of labelled images/audio "
                           "(optional: found in the current folder when "
                           "omitted, SPEC.md 88.1)")
    p_go.add_argument("--label", default=None,
                      help="label column (default: label/target/y/class, else "
                           "last column)")
    p_go.add_argument("--task", default="auto",
                      choices=("auto", "csv", "image", "audio", "text"),
                      help="force the data task; default auto (file -> csv, "
                           "directory -> detected)")
    p_go.add_argument("--target", type=float, default=95.0,
                      help="your acceptance bar on final_best_score "
                           "(default 95.0)")
    p_go.add_argument("--auto-target", dest="auto_target", action="store_true",
                      default=False,
                      help="v0.74 (SPEC.md 88.4): derive the bar from the "
                           "data's own class structure — 5 points above the "
                           "majority-class ceiling, held between the 95 "
                           "default and the 99.5 cap (classification tasks "
                           "only; otherwise the current bar stands)")
    p_go.add_argument("--experiments", type=int, default=None,
                      help="override the data-sized budget (default: derived "
                           "from the train-split size — small tables need "
                           "fewer experiments)")
    p_go.add_argument("--time", type=float, default=None, metavar="SECONDS",
                      help="v0.74 (SPEC.md 88.5): spend about this many "
                           "seconds, budget derived from this task's "
                           "measured per-experiment rate (only while "
                           "--experiments is absent; an explicit "
                           "--experiments wins)")
    p_go.add_argument("--continue", dest="continue_", action="store_true",
                      default=False,
                      help="v0.74 (SPEC.md 88.6): start from the most recent "
                           "same-task run in the registry — its best spec "
                           "seeds the search prior instead of scratch")
    p_go.add_argument("--score", default=None, metavar="CSV",
                      help="v0.74 (SPEC.md 88.8): after the gate, also score "
                           "the rows of this CSV with the best model (one "
                           "prediction per row, + an agreement line when the "
                           "file carries a label column)")
    p_go.add_argument("--seeds", type=int, default=None, metavar="N",
                      help="v0.74 (SPEC.md 88.10): N >= 2 — also run seeds "
                           "seed+1..seed+N-1 with the same budget and report "
                           "the spread band (a measurement; the gate and "
                           "exit code follow the base run)")
    p_go.add_argument("--seed", type=int, default=7)
    p_go.add_argument("--policy", default="bandit",
                      choices=("search", "bandit", "rl"),
                      help="improver: UCB field-bandit (default), v1 search, "
                           "or meta-RL")
    p_go.add_argument("--runs-dir", default="runs")
    p_go.add_argument("--quiet", action="store_true",
                      help="suppress the per-experiment loop output and the "
                           "per-class diagnostics (keep the gate verdict, "
                           "the summary block, and the verdict line)")
    p_go.add_argument("--latency-ms", type=float, default=None,
                      metavar="MS",
                      help="v0.75 (SPEC.md 89.11): also gate on the best "
                           "model's measured median ms/row - OVER the budget "
                           "is a MISS (rc 2), like the score bar")
    p_go.add_argument("--feeds", default=None, metavar="A.csv,B.csv",
                      help="v0.75 (SPEC.md 89.12): a portfolio of feed CSVs - "
                           "one shared budget split evenly across them, run "
                           "sequentially; rc 0 iff every feed PASSes "
                           "(mutually exclusive with --data)")
    # 87.3.4: the shared driver / gate / budget read these — the fit
    # defaults, materialized once (the `go` surface keeps its own flag set)
    p_go.set_defaults(
        split_frac=0.2, temporal=False, metric="accuracy", rl_episodes=5,
        search_quality="v04", ensemble_final=False, stall_patience=None,
        screen_frac=1.0, kfold=0, from_model=None, prior_run=None,
        gate=[], max_seconds=900.0, max_train_seconds=30.0,
        tasks=None, dry_run=False, from_run=None, dataset=None,
        window=None, window_col=None)
    _add_config_flag(p_go)  # 47.1.2
    p_go.set_defaults(func=_cmd_go)

    # v0.75 (SPEC.md 89): time-based and on-demand data ingestion
    p_feed = sub.add_parser(
        "feed", help="v0.75 (SPEC.md 89.1): watch a data file - each poll that "
                     "detects new data snapshots it, re-scores the champion on "
                     "it, and judges the drift (the on-call loop)")
    p_feed.add_argument("--data", required=True, help="the CSV file to watch")
    p_feed.add_argument("--runs-dir", default="runs")
    p_feed.add_argument("--max", type=int, default=1, metavar="N",
                        help="number of polls (default 1 - one watch cycle; "
                             "the state survives between calls)")
    p_feed.add_argument("--poll", type=float, default=0.5,
                        help="seconds to sleep BETWEEN polls (default 0.5; "
                             "no sleep after the last poll)")
    p_feed.add_argument("--champion", default=None, metavar="RUN",
                        help="the champion run dir (default with --latest: "
                             "the registry's latest run)")
    p_feed.add_argument("--latest", action="store_true",
                        help="the champion is the registry's latest finished "
                             "run (timestamp, then run id)")
    p_feed.add_argument("--drift-bar", type=float, default=2.0, metavar="PTS",
                        help="drift alert when the champion's fresh score "
                             "falls more than this below its reference "
                             "(default 2.0)")
    p_feed.add_argument("--chart", default=None, metavar="OUT.svg",
                        help="render the scored-event log as the quality-over-"
                        "time SVG to OUT.svg (the 89.8 view)")
    p_feed.add_argument("--label", default=None, help="label column hint")
    _add_config_flag(p_feed)  # 47.1.2
    p_feed.set_defaults(func=_cmd_feed)

    p_refresh = sub.add_parser(
        "refresh", help="v0.75 (SPEC.md 89.3): champion/challenger - score the "
                        "champion's model on NEW data, run a fresh challenger "
                        "loop on it, and decide PROMOTE / KEEP")
    p_refresh.add_argument("--data", required=True,
                           help="the NEW data CSV")
    p_refresh.add_argument("--champion", default=None, metavar="RUN",
                           help="the champion run dir (default with --latest: "
                                "the registry's latest run)")
    p_refresh.add_argument("--latest", action="store_true",
                           help="the champion is the registry's latest "
                                "finished run")
    p_refresh.add_argument("--experiments", type=int, default=5,
                           help="the challenger loop's budget (default 5)")
    p_refresh.add_argument("--target", type=float, default=95.0,
                           help="the challenger loop's gate bar (default 95.0)")
    p_refresh.add_argument("--seed", type=int, default=7)
    p_refresh.add_argument("--margin", type=float, default=0.0,
                           help="PROMOTE when the challenger beats the "
                                "champion by at least this (default 0.0 - "
                                "a tie PROMOTEs)")
    p_refresh.add_argument("--runs-dir", default="runs")
    _add_config_flag(p_refresh)  # 47.1.2
    p_refresh.set_defaults(func=_cmd_refresh)

    p_datasets = sub.add_parser(
        "datasets", help="v0.75 (SPEC.md 89.4): list the dataset snapshot "
                         "registry (id, rows, cols, label, balance, created)")
    p_datasets.add_argument("--runs-dir", default="runs")
    _add_config_flag(p_datasets)  # 47.1.2
    p_datasets.set_defaults(func=_cmd_datasets)

    p_sched = sub.add_parser(
        "schedule", help="v0.75 (SPEC.md 89.5): N sequential training rounds "
                         "that compound - each round starts from the previous "
                         "round's search knowledge")
    p_sched.add_argument("--data", required=True,
                         help="the CSV file (or a labelled media directory)")
    p_sched.add_argument("--rounds", type=int, default=3,
                         help="number of sequential rounds (default 3)")
    p_sched.add_argument("--every", type=float, default=0.0,
                         help="seconds to sleep BETWEEN rounds (default 0; "
                              "no sleep after the last round)")
    p_sched.add_argument("--experiments", type=int, default=3,
                         help="each round's budget (default 3)")
    p_sched.add_argument("--target", type=float, default=95.0,
                         help="each round's gate bar (default 95.0)")
    p_sched.add_argument("--seed", type=int, default=7)
    p_sched.add_argument("--runs-dir", default="runs")
    p_sched.add_argument("--label", default=None, help="label column hint")
    _add_config_flag(p_sched)  # 47.1.2
    p_sched.set_defaults(func=_cmd_schedule)

    p_ingest = sub.add_parser(
        "ingest", help="v0.75 (SPEC.md 89.6): merge the new CSVs dropped in a "
                       "folder (deduped rows, header-checked), snapshot the "
                       "result; rc 1 when nothing is new (poll again later)")
    p_ingest.add_argument("drop_dir", help="the drop folder holding *.csv")
    p_ingest.add_argument("--out", default=None, metavar="CSV",
                          help="the merged output (default <drop_dir>/"
                               "merged.csv; excluded from the merge)")
    p_ingest.add_argument("--runs-dir", default="runs")
    _add_config_flag(p_ingest)  # 47.1.2
    p_ingest.set_defaults(func=_cmd_ingest)

    # v0.73 (SPEC.md 87.8, C2): the model-card one-pager
    p_card = sub.add_parser(
        "card", help="v0.73 (SPEC.md 87.8): print the model-card one-pager "
                     "for a finished run (what it predicts, real examples, "
                     "when to distrust it, how to read the number)",
        epilog=_EP_CARD,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_card.add_argument("--run", required=True, help="path to a run directory")
    p_card.add_argument("--md", action="store_true",
                        help="render the Markdown form (wikis / PRs / Slack)")
    p_card.add_argument("--examples", type=int, default=3,
                        help="how many real holdout examples to show "
                             "(default 3)")
    _add_config_flag(p_card)  # 47.1.2
    p_card.set_defaults(func=_cmd_card)

    # SPEC.md 59.3 (v0.45): manual/expert mode — train one exact spec
    p_man = sub.add_parser(
        "manual", help="v0.45 (SPEC.md 59.3): manual/expert mode — set the exact "
                       "spec over the field registry and train it once "
                       "(validate + report, not discover)",
        epilog=_EP_MANUAL,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_man.add_argument("--task", default="cartpole-v1",
                       choices=sorted(TASKS),
                       help="built-in task (default cartpole-v1); or pass --data "
                            "for a data task (SPEC.md 59.3)")
    p_man.add_argument("--data", default=None,
                       help="a CSV file / media directory → a data task "
                            "(csv/image/audio/text), like fit --data (overrides "
                            "--task when both are given)")
    p_man.add_argument("--label", default=None,
                       help="label column (data tasks; default: auto-detect)")
    p_man.add_argument("--set", action="append", default=[],
                       metavar="FIELD=VALUE",
                       help="a spec field override (repeatable); the value is "
                            "registry-validated — e.g. --set architecture=[16,8] "
                            "--set model_family=mlp (SPEC.md 59.3.1)")
    p_man.add_argument("--seed", type=int, default=7)
    p_man.add_argument("--max-train", type=float, default=None,
                       help="per-train wall-time cap in seconds (default: none)")
    _add_config_flag(p_man)  # 47.1.2
    p_man.set_defaults(func=_cmd_manual)

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

    p_qs = sub.add_parser(
        "quickstart",
        help="v0.51 (SPEC.md 65.3): print the Quickstart — the same steps "
             "the README and the dashboard's first screen carry",
        epilog=_EP_QUICKSTART,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_qs.add_argument("--json", action="store_true",
                      help="print {intro, steps} as pure JSON")
    _add_config_flag(p_qs)  # 47.1.2
    p_qs.set_defaults(func=_cmd_quickstart)

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
    p_pred.add_argument("--run", default=None,
                        help="path to a run directory (or --latest for the "
                             "most recent finished run)")
    p_pred.add_argument("--latest", action="store_true",
                        help="v0.74 (SPEC.md 88.3): score with the most "
                             "recent finished run from the registry (an "
                             "explicit --run wins when both are given)")
    p_pred.add_argument("--runs-dir", default="runs",
                        help="the runs dir the registry lives in "
                             "(default runs; for --latest)")
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
        "compare", help="v0.28 (SPEC.md 42.2; 50.1.4 v0.36): the app's "
                        "run-diff (38.4) in the terminal — 2-run score delta + "
                        "best_spec diff, or 3-run gate rows + spec table",
        epilog=_EP_COMPARE,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_cmp.add_argument("--run", dest="runs", action="append", required=True,
                       metavar="DIR",
                       help="two or three run dirs, in order (repeat --run)")
    p_cmp.add_argument("--json", action="store_true",
                       help="print the diff dict (diff_two_summaries for two "
                            "runs, diff_n_summaries for three — pure JSON)")
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

    p_dossier = sub.add_parser(
        "dossier", help="v0.44 (SPEC.md 58.3): render the run dossier — one "
                        "self-contained HTML file combining the whole story "
                        "of a finished run (recipe, provenance, frontier, "
                        "lineage, sensitivity, error analysis)",
        epilog=_EP_DOSSIER,  # 47.2.2
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_dossier.add_argument("--run", required=True, help="path to a run directory")
    p_dossier.add_argument("--out", default=None, metavar="FILE",
                           help="output .html path (default: <run_dir>/dossier.html)")
    _add_config_flag(p_dossier)  # 47.1.2
    p_dossier.set_defaults(func=_cmd_dossier)

    # SPEC.md 62.5 (v0.48, B1): the one genuinely new bit — independently
    # re-derive every reported number from the run's artifacts and assert them.
    p_ver = sub.add_parser(
        "verify", help="v0.48 (SPEC.md 62.5): independently re-derive every "
                       "reported number from a run's artifacts (summary.json + "
                       "experiments.jsonl) and assert them — the auditor / "
                       "chain-of-custody check; rc 0 PASS / rc 2 FAIL / rc 1 "
                       "bad run dir",
        epilog="examples:\n"
               "  autorefine verify --run runs/<run_id>",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_ver.add_argument("--run", required=True, help="path to a run directory")
    _add_config_flag(p_ver)  # 47.1.2
    p_ver.set_defaults(func=_cmd_verify)

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
