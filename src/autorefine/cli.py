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
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.bandit import BanditPolicy
from .improver.curriculum import ParityCurriculum
from .improver.policy import SearchPolicy
from .improver.rl_policy import MetaRLPolicy, train_policy
from .memory import RunMemory
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
        **quality,
    )
    _drive(args, env)
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
    """`fit` (SPEC.md 22.1, v0.10 24.5): data in, gated validated model out.

    = the `run` loop over the matching data task (task_config carries the
    path: csv file / image dir / audio dir), plus the §21.5 target gate:
    PASS → exit 0, MISS → exit 2.
    """
    from .tasks import TASKS
    data = Path(args.data)
    try:
        task_name = _resolve_fit_task(data, getattr(args, "task", "auto") or "auto")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
        return 1
    quality = {} if args.search_quality == "legacy" else search_quality_v04()
    args.task = task_name  # _drive's print lines
    env = AutoRefineEnv(
        task=task_name, seed=args.seed, budget=_budget(args), runs_dir=args.runs_dir,
        ensemble_top_k=2 if args.ensemble_final else 0,
        dataset_episodes=int(probe.default_dataset_size),
        task_config=config,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; None = off)
        screen_frac=args.screen_frac,  # v0.18 (SPEC.md 32.2; 1.0 = off)
        **quality,
    )
    _drive(args, env)
    final = env.best_score
    print(f"\ngate    : final {final:.2f} vs target {args.target:.1f} "
          f"(SPEC.md 22.1; the loop maximizes the validated score, the bar is yours)")
    if final >= args.target:
        print(f"PASS: final score {final:.2f} >= {args.target:.1f} on held-out data "
              "(the loop never trained on these points; gen_gap is the overfit guard)")
        rc = 0
    else:
        print(f"MISS: {final:.2f} < {args.target:.1f} — the search space or budget is "
              "exhausted for this task")
        print("next: raise --experiments / --max-train-seconds, or extend the spec "
              "space / model families (README 'Extending')")
        rc = 2  # the user's acceptance step, machine-readable for pipelines
    # C2/C3 (SPEC.md 28.2/28.3): after the gate line, per-class accuracy +
    # weakest-class hint (and the media error gallery)
    _print_fit_diagnostics(env)
    return rc


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
    runner = DashboardRunner(
        csv_path=str(data), label=args.label, split_frac=args.split_frac,
        target=args.target, policy=args.policy, seed=args.seed,
        experiments=args.experiments, max_seconds=args.max_seconds,
        max_train_seconds=args.max_train_seconds, runs_dir=args.runs_dir,
        search_quality=args.search_quality, modality=args.task,
        stall_patience=args.stall_patience,  # v0.17 (SPEC.md 31.1; per-seed)
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


def _cmd_report(args: argparse.Namespace) -> int:
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
            if e.get("kind") in ("baseline", "experiment")
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


def main(argv: list[str] | None = None) -> int:
    # SPEC.md 21.3: external packages may contribute tasks via entry points;
    # register them before the parser is built so `--task` choices include them
    try:
        register_tasks()
    except PluginError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
    p_run.set_defaults(func=_cmd_run)

    p_rep = sub.add_parser("report", help="print a run's summary and experiment log")
    p_rep.add_argument("--run", required=True, help="path to a run directory")
    p_rep.add_argument("--json", action="store_true",
                       help="v0.7 (SPEC.md 21.2): print summary.json to stdout "
                            "(machine-readable; with --plot the SVGs are still written)")
    p_rep.add_argument("--plot", action="store_true",
                       help="v0.7 (SPEC.md 21.2): ASCII charts on stdout + "
                            "score_curve.svg / pareto_frontier.svg in the run dir")
    p_rep.add_argument("--html", action="store_true",
                       help="v0.8 (SPEC.md 22.2): write a self-contained "
                            "report.html (embedded SVGs + tables) into the run dir")
    p_rep.set_defaults(func=_cmd_report)

    p_fit = sub.add_parser(
        "fit", help="v0.8 (SPEC.md 22.1) / v0.10 (24.5): fit a model on your data "
                    "(CSV file, or a directory of labelled images/audio) and "
                    "gate on a target")
    p_fit.add_argument("--data", required=True,
                       help="CSV file, or a directory of labelled images/audio "
                            "(v0.10, SPEC.md 24.5)")
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

    args = parser.parse_args(argv)
    return int(args.func(args))
