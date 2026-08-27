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
    svg_pareto,
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
        **quality,
    )
    _drive(args, env)
    final = env.best_score
    print(f"\ngate    : final {final:.2f} vs target {args.target:.1f} "
          f"(SPEC.md 22.1; the loop maximizes the validated score, the bar is yours)")
    if final >= args.target:
        print(f"PASS: final score {final:.2f} >= {args.target:.1f} on held-out data "
              "(the loop never trained on these points; gen_gap is the overfit guard)")
        return 0
    print(f"MISS: {final:.2f} < {args.target:.1f} — the search space or budget is "
          "exhausted for this task")
    print("next: raise --experiments / --max-train-seconds, or extend the spec "
          "space / model families (README 'Extending')")
    return 2  # the user's acceptance step, machine-readable for pipelines


def _cmd_report(args: argparse.Namespace) -> int:
    mem = RunMemory(Path(args.run))
    try:
        summary = mem.load_summary()
    except FileNotFoundError:
        print("no summary.json in run dir", file=sys.stderr)
        return 1
    entries = mem.load_experiments()
    run_dir = Path(args.run)
    if args.plot:  # SPEC.md 21.2: SVG files always land in the run dir
        pareto_pts = summary.get("pareto_frontier") or [
            {"score": e["holdout_score"], "train_seconds": e.get("train_seconds", 0.0)}
            for e in entries
            if e.get("kind") in ("baseline", "experiment")
            and isinstance(e.get("holdout_score"), (int, float))
        ]
        (run_dir / "score_curve.svg").write_text(svg_score_curve(entries), encoding="utf-8")
        (run_dir / "pareto_frontier.svg").write_text(svg_pareto(pareto_pts), encoding="utf-8")
    if args.html:  # SPEC.md 22.2: one self-contained file in the run dir
        (run_dir / "report.html").write_text(html_report(summary, entries),
                                             encoding="utf-8")
    if args.json:  # SPEC.md 21.2: stdout is pure machine-readable JSON
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
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


def _cmd_eval(args: argparse.Namespace) -> int:
    mem = RunMemory(Path(args.run))
    try:
        summary = mem.load_summary()
    except FileNotFoundError:
        print("no summary.json in run dir", file=sys.stderr)
        return 1
    task_name = summary["task"]
    seed = int(summary["seed"])
    cfg = summary.get("task_config") or {}  # SPEC.md 22.1 (e.g. csv: path/label)
    if task_name in TASKS:
        task = TASKS[task_name](seed=seed, **cfg) if cfg else TASKS[task_name](seed=seed)
    elif summary.get("curriculum"):  # SPEC.md 20.1: a curriculum level name
        lvl = summary["curriculum"]["levels"][-1]
        task = ParityTask(seed=seed, n_bits=lvl["n_bits"], p_flip=lvl["p_flip"])
    else:
        print(f"unknown task {task_name!r} in summary", file=sys.stderr)
        return 1
    family = summary["best_spec"].get("model_family", "mlp")
    # SPEC.md 25.5: the family -> loader mapping (knn/convnet since v0.11)
    loaders = {
        "mlp": lambda p: MLP.load(p),
        "tree": lambda p: _load_tree(p),
        "boost": lambda p: _load_boost(p),  # SPEC.md 19.2
        "knn": lambda p: _load_knn(p),      # SPEC.md 25.2
        "convnet": lambda p: _load_convnet(p),  # SPEC.md 25.3
    }
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

    args = parser.parse_args(argv)
    return int(args.func(args))
