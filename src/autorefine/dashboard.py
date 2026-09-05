"""DashboardRunner: the streamlit-free core of the v0.9 visual dashboard
(SPEC.md 23.1).

The app (dashboard_app.py) is a thin rendering layer over this class; the
run semantics live here so they stay testable without streamlit and so
same-seed runs reproduce the CLI `fit` outcome with the same flags (G2).
No streamlit import (SPEC.md 23.1, §3 optional-dependency pattern).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import Budget
from .diagnostics import holdout_diagnostics
from .improver.bandit import BanditPolicy
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.policy import SearchPolicy
from .memory import KIND_BASELINE, KIND_CURRICULUM, KIND_EXPERIMENT  # 35.1 (C4)
from .plotting import (
    html_report,
    svg_architecture,
    svg_confusion_matrix,
    svg_decision_boundary,  # V3 view (SPEC.md 30.3)
    svg_family_bars,  # V5 view (SPEC.md 30.5)
    svg_field_value_matrix,  # V2 view (SPEC.md 30.2)
    svg_ladder_curve,
    svg_loss_curves,
    svg_mutation_timeline,
    svg_pareto,
    svg_per_class_bars,
    svg_score_curve,
    svg_score_gap_scatter,
    svg_score_strip,
)
from .tasks import TASKS, detect_modality
from .tasks.media import class_label_str  # V3 class labels (SPEC.md 30.3)

_RL_HINT = (
    "policy 'rl' does not map to a live per-experiment stream (it trains over "
    "several full-budget episodes); use `autorefine fit --policy rl` "
    "(SPEC.md 23.1)"
)


class DashboardRunner:
    """CSV in → live per-experiment updates → gated verdict + artifacts.

    = the `fit` loop (SPEC.md 22.1) decomposed into `start()` / `next()` /
    `finish()` so a UI can render one update at a time (SPEC.md 23.1).
    """

    def __init__(self, csv_path, label: str | None = None, split_frac: float = 0.2,
                 target: float = 95.0, policy: str = "bandit", seed: int = 7,
                 experiments: int = 30, max_seconds: float = 900.0,
                 max_train_seconds: float = 30.0, runs_dir: str = "runs",
                 search_quality: str = "v04", modality: str | None = None,
                 stall_patience: int | None = None) -> None:
        if policy not in ("bandit", "search"):
            raise ValueError(f"unsupported policy {policy!r}: {_RL_HINT}")
        if search_quality not in ("v04", "legacy"):
            raise ValueError(f"search_quality must be 'v04' or 'legacy', "
                             f"got {search_quality!r}")
        if modality is not None and modality not in ("auto", "csv", "image", "audio"):
            raise ValueError(f"modality must be one of auto/csv/image/audio, "
                             f"got {modality!r} (SPEC.md 24.5)")
        # `csv_path` is the data path (v0.10: a CSV file or a media directory)
        self.csv_path = csv_path
        self.modality = modality or "auto"  # SPEC.md 24.5
        self.label = label or None
        self.split_frac = float(split_frac)
        self.target = float(target)
        self.policy_name = policy
        self.seed = int(seed)
        self.experiments = int(experiments)
        self.max_seconds = float(max_seconds)
        self.max_train_seconds = float(max_train_seconds)
        self.runs_dir = str(runs_dir)
        self.search_quality = search_quality
        self.stall_patience = stall_patience  # v0.17 (SPEC.md 31.1; None = off)
        self.env: AutoRefineEnv | None = None
        self.policy: SearchPolicy | BanditPolicy | None = None
        self._state: dict | None = None
        self._phase = "idle"  # idle -> running -> done
        self._updates: list[dict] = []

    @property
    def done(self) -> bool:
        """True once the loop has hit `done` and `finish()` is valid."""
        return self._phase == "done"

    # --- lifecycle ----------------------------------------------------------
    def _resolve_task(self, data: Path) -> str:
        """Data path → task name (v0.10, SPEC.md 24.5): file → csv;
        directory → detected or forced via `modality`."""
        mod = self.modality
        if data.is_file():
            if mod in ("auto", "csv"):
                return "csv"
            raise ValueError(f"{data} is a file: modality must be 'csv' or 'auto'")
        if data.is_dir():
            if mod in ("image", "audio"):
                return mod
            m = detect_modality(data)
            if m == "mixed":
                raise ValueError(
                    f"{data} holds both image and audio items: set "
                    f"modality='image' or 'audio' (SPEC.md 24.5)")
            if m is None:
                raise ValueError(
                    f"no image or audio items under {data} — one subfolder "
                    f"per class, or an index.csv (SPEC.md 24.2)")
            return m
        raise ValueError(f"no such CSV file: {data}")

    def start(self) -> dict:
        """Probe the data, build the env, train+score the baseline (SPEC.md 23.1).

        v0.10 (SPEC.md 24.5): the data is a CSV file (csv task) or a
        directory of labelled images/audio (image/audio tasks).
        """
        if self._phase != "idle":
            raise RuntimeError(f"start() again in phase {self._phase!r}")
        data = Path(self.csv_path)
        if not data.exists():
            raise ValueError(f"no such CSV file: {data}")
        task_name = self._resolve_task(data)
        config: dict = {"path": str(data)}
        if self.label is not None:
            config["label"] = self.label
        if self.split_frac != 0.2:
            config["split_frac"] = self.split_frac

        # deterministic probe: the env re-derives the identical split from
        # the same seed (SPEC.md 22.1)
        probe = TASKS[task_name](seed=self.seed, **config)
        self.env = AutoRefineEnv(
            task=task_name, seed=self.seed,
            budget=Budget(self.experiments, self.max_seconds, self.max_train_seconds),
            runs_dir=self.runs_dir,
            dataset_episodes=int(probe.default_dataset_size),
            task_config=config,
            # the CLI default (SPEC.md 18.6); 'legacy' = the deterministic
            # v0.3 rule (strict score comparison, no wall-clock eff term)
            **({} if self.search_quality == "legacy" else search_quality_v04()),
            # v0.17 (SPEC.md 31.1): plateau early stop (None = off)
            stall_patience=self.stall_patience,
        )
        self.policy = (SearchPolicy(seed=self.seed) if self.policy_name == "search"
                       else BanditPolicy(seed=self.seed))
        self._state = self.env.reset()
        self._phase = "running"
        return {
            "label": probe.label_name,
            "head": probe.head,
            "class_values": probe.class_values,
            "features": probe.feature_names,
            "rows": {
                "train": int(len(probe._x_tr)),
                "holdout": int(len(probe._x_ho)),
                "gen": int(len(probe._x_ge)),
            },
            "baseline_score": self._state["baseline_score"],
            "target": self.target,
            "policy": self.policy_name,
            "seed": self.seed,
            "budget_experiments": self.experiments,
            "run_dir": str(self.env.run_dir),
        }

    def next(self) -> dict:
        """One improver step → one JSON-safe update dict (SPEC.md 23.1).

        The update also carries the decision views (SPEC.md 26): `spec_diff`
        (26.2), `field_stats` (26.1), `field_value_stats` (30.2), and `ucb`
        (26.4, bandit only) — pure functions of the update stream so far
        (G2).
        """
        if self._phase == "idle":
            raise RuntimeError("call start() before next()")
        if self._phase == "done":
            raise RuntimeError("the run is done; call finish()")
        action = self.policy.propose(self._state)
        # D2 (SPEC.md 26.2): the best spec *before* this step, and the
        # proposed spec, for the old → new chips
        prev_best = self.env.best_spec.to_dict() if self.env.best_spec else {}
        action_dict = action if isinstance(action, dict) else action.to_dict()
        self._state, reward, done, info = self.env.step(action)
        mutation = list((self._state.get("last") or {}).get("fields") or [])
        update = {
            "index": len(self._updates) + 1,
            "accepted": bool(info.get("accepted", False)),
            "reason": info.get("reason"),
            "candidate_score": info.get("candidate_score"),
            "gen_gap": info.get("gen_gap"),
            "train_seconds": info.get("train_seconds"),
            "experiments_left": info.get("experiments_left",
                                         self.env.bm.experiments_left),
            "mutation": mutation,
            "spec_diff": spec_diff(prev_best, action_dict, mutation),
            # C1 (SPEC.md 28.1): the candidate's bounded loss history, for the
            # live learning-curve view (empty for dup/invalid rejections)
            "loss_history": info.get("loss_history") or [],
            "best_score": self.env.best_score,
            "reward": reward,
            "done": bool(done),
        }
        self._updates.append(update)
        # decision views over the stream so far (SPEC.md 26.5; pure, G2)
        update["field_stats"] = field_stats(self._updates)
        update["field_value_stats"] = field_value_stats(self._updates)  # V2 (SPEC.md 30.2)
        if self.policy_name == "bandit":
            trace = ucb_trace(self._updates, alpha=self.policy.alpha)
            update["ucb"] = {f: series[-1] for f, series in trace.items()}
        else:
            update["ucb"] = None  # UCB is a bandit concept (SPEC.md 26.4)
        if done:
            self._phase = "done"
        return update

    def run_all(self, on_update=None) -> dict:
        """Drive the loop to `done`, calling `on_update(update)` per step."""
        while self._phase == "running":
            update = self.next()
            if on_update is not None:
                on_update(update)
        return self.finish()

    # --- D1 seed-variance (SPEC.md 29.1) ------------------------------------
    def _sweep_params(self) -> dict:
        """The plain-value `DashboardRunner` constructor kwargs (picklable,
        SPEC.md 31.2) — everything a sweep seed carries except its seed."""
        return dict(
            csv_path=self.csv_path, label=self.label, split_frac=self.split_frac,
            target=self.target, policy=self.policy_name,
            experiments=self.experiments, max_seconds=self.max_seconds,
            max_train_seconds=self.max_train_seconds, runs_dir=self.runs_dir,
            search_quality=self.search_quality, modality=self.modality,
            stall_patience=self.stall_patience,  # v0.17 (SPEC.md 31.1)
        )

    def _clone(self, seed: int) -> "DashboardRunner":
        """D1 (SPEC.md 29.1): a fresh runner from `self`'s parameters (same
        flags, a new seed) so a sweep run is fully isolated — this never
        disturbs the caller's own runner or the §18.7 pin."""
        return DashboardRunner(seed=int(seed), **self._sweep_params())

    def seed_sweep(self, seeds, on_update=None, workers: int = 1) -> list:
        """D1 (SPEC.md 29.1): the same flags re-run under N seeds — the
        "is the improvement real?" question (SPEC.md 29).

        For each seed it builds a fresh `DashboardRunner` (`_clone`), drives
        the exact §23.1 loop `start()` → `next()`…→ `finish()` (`on_update`
        forwarded per step), and appends one JSON-safe dict
        `{seed, baseline, final, curve, target, pass, verdict,
        experiments_run, run_dir}` — `curve` (SPEC.md 30.1) is
        `[baseline, best after step 1, …]`, one point per `next()` call
        (free duplicate rejections are steps too, R3). Empty `seeds` → `[]`.
        The caller's own runner is untouched.

        v0.17 (SPEC.md 31.2): `workers > 1` runs the independent seeds on
        a `ProcessPoolExecutor` — each seed is a self-contained
deterministic run, so per-seed results are bit-identical to the serial
        path (except the timestamped `run_dir`); `workers == 1` (default)
        keeps the exact pre-v0.17 serial loop. `on_update` is a callback
        and cannot run in worker processes (it raises with `workers > 1`).
        """
        seeds = [int(s) for s in (seeds or [])]
        if not seeds:
            return []
        if not isinstance(workers, int) or isinstance(workers, bool) \
                or workers < 1:
            raise ValueError(f"workers must be an int >= 1, got {workers!r} "
                             f"(SPEC.md 31.2)")
        if workers > 1:
            if on_update is not None:
                raise ValueError(
                    "on_update is a callback and cannot run in worker "
                    "processes — pass workers=1 for the serial loop "
                    "(SPEC.md 31.2)")
            from concurrent.futures import ProcessPoolExecutor  # stdlib
            params = self._sweep_params()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # pool.map preserves input order (SPEC.md 31.2)
                return list(pool.map(_sweep_seed_worker,
                                     [(params, s) for s in seeds]))
        # workers == 1 (default): the exact pre-v0.17 serial loop (31.2)
        out: list[dict] = []
        for seed in seeds:
            out.append(_drive_sweep_runner(self._clone(seed), on_update))
        return out

    # --- results ------------------------------------------------------------
    def finish(self) -> dict:
        """Verdict (§22.1 gate) + summary + plots + artifact payloads."""
        if self._phase != "done":
            raise RuntimeError("the run is not finished yet (loop to done first)")
        mem = self.env.memory
        run_dir = self.env.run_dir
        summary = mem.load_summary()
        entries = mem.load_experiments()
        final = float(self.env.best_score)
        pass_ = final >= self.target  # the §22.1 gate, exactly
        pareto_pts = summary.get("pareto_frontier") or [
            {"score": e["holdout_score"], "train_seconds": e.get("train_seconds", 0.0)}
            for e in entries
            if e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)  # 35.1 (C4)
            and isinstance(e.get("holdout_score"), (int, float))
        ]
        # learning views (SPEC.md 28), computed live from memory (no reloads)
        diag = holdout_diagnostics(self.env.task, self.env.best_model)
        gallery: list | None = None
        hold_errors = getattr(self.env.task, "holdout_errors", None)
        if callable(hold_errors) and self.env.best_model is not None:
            g = hold_errors(self.env.best_model)
            gallery = g if g else None  # no misclassifications -> no gallery
        arch_svg = (svg_architecture(self.env.best_spec.to_dict(),
                                     self.env.task.state_dim,
                                     self.env.task.n_outputs)
                    if self.env.best_spec else None)
        fvs = field_value_stats(self._updates)  # V2 (SPEC.md 30.2)
        fams = family_stats(entries)  # V5 (SPEC.md 30.5): over the scored entries
        boundary = decision_boundary(self.env.task, self.env.best_model)  # V3
        return {
            "verdict": "PASS" if pass_ else "MISS",
            "target": self.target,
            "final_best_score": final,
            "baseline_score": float(self.env.baseline_score),
            "improvement_factor": summary.get("improvement_factor"),
            "experiments_run": summary.get("experiments_run"),
            "wall_seconds": summary.get("wall_seconds"),
            "finished_reason": summary.get("finished_reason"),
            "summary": summary,
            "updates": list(self._updates),
            "best_spec": (self.env.best_spec.to_dict()
                          if self.env.best_spec else None),
            # SPEC.md 21.2 SVGs + 22.2 report, generated in-memory
            "svg_score": svg_score_curve(entries),
            "svg_pareto": svg_pareto(pareto_pts),
            # decision views, final state (SPEC.md 26.5)
            "field_stats": field_stats(self._updates),
            # V2 (SPEC.md 30.2): field x value win matrix + its stats
            "field_value_stats": fvs,
            "field_value_svg": svg_field_value_matrix(fvs),
            # V5 (SPEC.md 30.5): model-family score bars + its stats
            "family_stats": fams,
            "family_bars_svg": svg_family_bars(fams),
            "ucb_trace": (ucb_trace(self._updates, alpha=self.policy.alpha)
                          if self.policy_name == "bandit" else None),
            "timeline_svg": svg_mutation_timeline(self._updates),
            # acceptance-gate views (SPEC.md 27): G1 + G2 always (derived
            # from the update stream); G3 only when the run has a ladder
            # (None for non-curriculum runs, SPEC.md 27.3)
            "strip_svg": svg_score_strip(self._updates),
            "scatter_svg": svg_score_gap_scatter(self._updates),
            "ladder_svg": (svg_ladder_curve(
                entries, summary.get("curriculum", {}).get("levels"))
                if summary.get("curriculum") else None),
            # learning views (SPEC.md 28): C1 curves, C2 diagnostics +
            # per-class/confusion SVGs, C3 error gallery, C4 architecture SVG
            "loss_curves": self._loss_curves(entries),
            "diagnostics": diag,
            "per_class_svg": svg_per_class_bars(diag) if diag else None,
            "confusion_svg": svg_confusion_matrix(diag) if diag else None,
            # V3 (SPEC.md 30.3): decision boundary on the 2-D feature plane
            # (None for 1-feature / mse-head / episode tasks)
            "boundary": boundary,
            "boundary_svg": svg_decision_boundary(boundary) if boundary else None,
            "error_gallery": gallery,
            "arch_svg": arch_svg,
            "report_html": html_report(summary, entries, diagnostics=diag,
                                       gallery=gallery, arch_svg=arch_svg),
            "artifacts": {
                "best_spec.json": (run_dir / "best_spec.json").read_bytes(),
                "best_model.npz": (run_dir / "best_model.npz").read_bytes(),
                "experiments.jsonl": (run_dir / "experiments.jsonl").read_bytes(),
            },
            "run_dir": str(run_dir),
        }

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _loss_curves(entries) -> dict[str, list]:
        """C1 (SPEC.md 28.1): `{label: loss_history}` for every logged row
        that has one — baseline, experiment 1..n, curriculum 1..n — in log
        order; rows without a history (dup/invalid rejections) are skipped.
        Deterministic (G2); JSON-safe (the history is already JSON-safe)."""
        counters: dict[str, int] = {}
        curves: dict[str, list] = {}
        for e in entries or []:
            kind = e.get("kind")
            hist = e.get("loss_history")
            if kind not in (KIND_BASELINE, KIND_EXPERIMENT, KIND_CURRICULUM) \
                    or not hist:  # SPEC.md 35.1 (C4)
                continue
            counters[kind] = counters.get(kind, 0) + 1
            label = (KIND_BASELINE if kind == KIND_BASELINE
                     else f"{kind} {counters[kind]}")
            curves[label] = hist
        return curves

    def updates_json(self) -> str:
        """The full update stream as JSON (machine-readable live feed)."""
        return json.dumps(self._updates, indent=2, sort_keys=True)


# --- v0.17 sweep worker (SPEC.md 31.2) ----------------------------------------
# Module-level plain functions: picklable, so the parallel seed sweep can run
# one full isolated run per worker process (Windows `spawn` re-imports the
# package; bound methods and lambdas would not be picklable — SPEC.md 31.2).

def _drive_sweep_runner(runner: "DashboardRunner", on_update=None) -> dict:
    """Drive one sweep seed — `start()` → `next()`… → `finish()` — and
    return its JSON-safe dict (SPEC.md 29.1).

    The exact serial body of the pre-v0.17 `seed_sweep` loop (SPEC.md 31.2):
    the serial path uses it per seed with `on_update` forwarded, the
    parallel worker uses it without one (31.2). `curve[0] == baseline`,
    `curve[-1] == final` (SPEC.md 30.1).
    """
    info = runner.start()
    curve = [float(info["baseline_score"])]  # V1 (SPEC.md 30.1)
    while not runner.done:
        update = runner.next()
        curve.append(float(update["best_score"]))
        if on_update is not None:
            on_update(update)
    res = runner.finish()
    final = float(res["final_best_score"])
    target = float(res["target"])
    return {
        "seed": runner.seed,
        "baseline": float(res["baseline_score"]),
        "final": final,
        "curve": curve,
        "target": target,
        "pass": bool(final >= target),  # the §22.1 gate, exactly
        "verdict": res["verdict"],
        "experiments_run": res["experiments_run"],
        "run_dir": str(res["run_dir"]),
    }


def _sweep_seed_worker(job: tuple[dict, int]) -> dict:
    """v0.17 (SPEC.md 31.2): one independent sweep seed on a worker process.

    Builds the same `_clone(seed)` runner from the picklable constructor
    kwargs (`_sweep_params`) and drives the same loop — the seeds share no
    state, so the per-seed result is bit-identical to the serial path
    (except the timestamped `run_dir`). No `on_update` here: it is a
    callback and `seed_sweep` raises with `workers > 1` (31.2).
    """
    params, seed = job
    return _drive_sweep_runner(DashboardRunner(seed=int(seed), **params), None)


# --- decision views (SPEC.md 26) ---------------------------------------------
# Pure functions of the runner's update stream (SPEC.md 23.1 rows): the app
# renders them; the runner attaches them to each update (SPEC.md 26.5).
# G2: same stream → bit-identical output. No streamlit (SPEC.md 23.1).

def field_stats(updates) -> dict[str, dict]:
    """D1 (SPEC.md 26.1): per-field {trials, wins, win_rate}, crediting each
    update's `mutation` fields with its `accepted` outcome — the exact
    (field, accepted) credit pairs the bandit consumes (SPEC.md 17)."""
    acc: dict[str, list] = {}
    for u in updates or []:
        for f in (u.get("mutation") or []):
            if not isinstance(f, str):
                continue
            s = acc.setdefault(f, [0, 0.0])
            s[0] += 1
            if u.get("accepted"):
                s[1] += 1.0
    return {f: {"trials": acc[f][0], "wins": acc[f][1],
                "win_rate": acc[f][1] / acc[f][0]}
            for f in sorted(acc)}


def _fv_key(value) -> str:
    """V2 (SPEC.md 30.2): a spec value → its matrix cell key — `None` →
    `"-"`, list/tuple values joined with `,` (e.g. layers `[8, 16]` →
    `"8,16"`), else `str(v)`."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def field_value_stats(updates) -> dict[str, dict[str, dict]]:
    """V2 (SPEC.md 30.2): per `(field, value)` {trials, wins, win_rate} —
    the v0.12 per-field rollup (`field_stats`, SPEC.md 26.1) refined to
    *which value won* (e.g. `layers` 8,16 vs 16,8). For each update and each
    entry of its `spec_diff` (SPEC.md 26.2: {field, old, new}), the new value
    is keyed via `_fv_key` and the cell credited with the update's
    `accepted` outcome. Fields and values are returned sorted (values
    numeric-first, matching `_fv_sort_key` in plotting.py). Pure over the
    update stream (G2): the per-field trial/wins sums equal `field_stats`'
    whenever `spec_diff` is present.
    """
    acc: dict[str, dict[str, list]] = {}
    for u in updates or []:
        diff = u.get("spec_diff")
        if not isinstance(diff, list):
            continue
        for d in diff:
            if not isinstance(d, dict):
                continue
            f = d.get("field")
            if not isinstance(f, str):
                continue
            cell = acc.setdefault(f, {}).setdefault(_fv_key(d.get("new")), [0, 0.0])
            cell[0] += 1
            if u.get("accepted"):
                cell[1] += 1.0

    def _sort(k: str) -> tuple:
        try:
            return (0.0, float(k), "")
        except (TypeError, ValueError):
            return (1.0, 0.0, k)

    return {
        f: {v: {"trials": acc[f][v][0], "wins": acc[f][v][1],
                "win_rate": acc[f][v][1] / acc[f][v][0]}
            for v in sorted(acc[f], key=_sort)}
        for f in sorted(acc)
    }


def family_stats(entries) -> list[dict]:
    """V5 (SPEC.md 30.5): best holdout score per model family — over the
    scored entries (kind baseline/experiment, finite `holdout_score`),
    grouped by the entry's `spec.model_family` (`"unknown"` when absent or
    not a string): `{family, best_score, best_seconds (the cost of the
    best member — ties keep the first), trials, wins (accepted count)}`,
    sorted by `(-best_score, family)`. Pure (G2); the renderer
    (`svg_family_bars`) reads exactly these keys."""
    groups: dict[str, list] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):  # 35.1 (C4)
            continue
        s = e.get("holdout_score")
        if isinstance(s, bool) or not isinstance(s, (int, float)) \
                or not math.isfinite(float(s)):
            continue
        spec = e.get("spec")
        fam = spec.get("model_family") if isinstance(spec, dict) else None
        if not isinstance(fam, str) or not fam:
            fam = "unknown"
        groups.setdefault(fam, []).append(e)
    out = []
    for fam in groups:
        members = groups[fam]
        # max() is stable: the first best member wins ties, so
        # `best_seconds` is that member's cost (SPEC.md 30.5)
        best = max(members, key=lambda m: float(m["holdout_score"]))
        out.append({
            "family": fam,
            "best_score": float(best["holdout_score"]),
            "best_seconds": float(best.get("train_seconds", 0.0) or 0.0),
            "trials": len(members),
            "wins": float(sum(1 for m in members if m.get("accepted"))),
        })
    out.sort(key=lambda f: (-f["best_score"], f["family"]))
    return out


def spec_diff(prev_best, action, fields) -> list[dict]:
    """D2 (SPEC.md 26.2): old → new per mutated field; a value absent from
    that spec is None (rendered as "—"). `prev_best`/`action` are spec dicts."""
    prev_best = prev_best if isinstance(prev_best, dict) else {}
    action = action if isinstance(action, dict) else {}
    return [{"field": f, "old": prev_best.get(f), "new": action.get(f)}
            for f in (fields or []) if isinstance(f, str)]


def ucb_trace(updates, alpha: float = 1.0) -> dict[str, list]:
    """D4 (SPEC.md 26.4): the bandit's UCB (SPEC.md 17) after crediting each
    update: wins[f]/t + sqrt(alpha * ln(max(1, total)) / t); fields not yet
    credited are None (a chart gap). Pure in (updates, alpha)."""
    names = sorted({f for u in (updates or [])
                    for f in (u.get("mutation") or []) if isinstance(f, str)})
    trials = {f: 0 for f in names}
    wins = {f: 0.0 for f in names}
    trace: dict[str, list] = {f: [] for f in names}
    for u in updates or []:
        for f in (u.get("mutation") or []):
            if f not in trials:
                continue
            trials[f] += 1
            if u.get("accepted"):
                wins[f] += 1.0
        total = max(1, sum(trials.values()))
        for f in names:
            t = trials[f]
            trace[f].append(
                None if t == 0
                else wins[f] / t + math.sqrt(alpha * math.log(total) / t))
    return trace


def decision_boundary(task, model, n_grid: int = 24, n_points: int = 200) -> dict | None:
    """V3 (SPEC.md 30.3): the decision-boundary data for a 2-feature
    classification task — where does the model fail on a 2-D feature plane?
    The flat-task complement of the media error gallery (SPEC.md 28.3).

    Applies only when `task.head == "softmax"`, `task.state_dim == 2`,
    `task.class_values` is non-empty, `task.holdout_rows` (SPEC.md 28.2) is
    callable, and `model is not None` — otherwise `None` (one feature, mse
    head, episode tasks). One forward pass over the holdout (clamped by
    `n_points`, the same `n` `holdout_diagnostics` uses) and one over an
    `n_grid x n_grid` grid spanning the holdout's per-feature ranges
    (padded 5%; a degenerate range → ±1); no RNG (G2). Returns the
    JSON-safe dict of SPEC.md 30.3: `x0`/`x1` (feature names, or
    `x0`/`x1`), `x0_range`/`x1_range`, `n_grid`, `grid_preds` (flat,
    row-major: index `i*n + j` is cell (i, j), class index), `points`
    (`{x, y, true, pred}` per holdout row), `classes` (labels via
    `class_label_str`, SPEC.md 28.3), `n`, and `correct` = #{true == pred}.
    """
    if getattr(task, "head", None) != "softmax":
        return None
    if int(getattr(task, "state_dim", -1)) != 2:
        return None
    values = getattr(task, "class_values", None)
    if not values:
        return None
    if not callable(getattr(task, "holdout_rows", None)):
        return None
    if model is None:
        return None

    n = max(1, int(n_grid))
    x, y = task.holdout_rows(int(n_points), model)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] == 0:
        return None
    k = len(values)
    y_true = np.clip(np.asarray(y, dtype=np.int64).ravel()[: x.shape[0]],
                     0, k - 1)
    pred = np.clip(np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1),
                   0, k - 1)

    def _padded(idx: int) -> list[float]:
        lo, hi = float(x[:, idx].min()), float(x[:, idx].max())
        span = hi - lo
        if span > 0.0:  # padded 5% on both sides (SPEC.md 30.3)
            return [lo - 0.05 * span, hi + 0.05 * span]
        return [lo - 1.0, hi + 1.0]  # degenerate range → ±1 (SPEC.md 30.3)

    r0, r1 = _padded(0), _padded(1)

    def _nodes(lo: float, hi: float) -> list[float]:
        if n == 1:
            return [(lo + hi) / 2.0]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    g0, g1 = _nodes(*r0), _nodes(*r1)
    grid = np.array([[v0, v1] for v0 in g0 for v1 in g1], dtype=np.float64)
    grid_pred = np.clip(
        np.asarray(model.forward(grid), dtype=np.float64).argmax(axis=1),
        0, k - 1)

    names = getattr(task, "feature_names", None)
    if isinstance(names, (list, tuple)) and len(names) >= 2:
        x0name, x1name = str(names[0]), str(names[1])
    else:
        x0name, x1name = "x0", "x1"

    m = int(x.shape[0])
    return {
        "x0": x0name,
        "x1": x1name,
        "x0_range": r0,
        "x1_range": r1,
        "n_grid": n,
        "grid_preds": [int(p) for p in grid_pred],  # row-major i*n + j
        "points": [
            {"x": float(x[i, 0]), "y": float(x[i, 1]),
             "true": int(y_true[i]), "pred": int(pred[i])}
            for i in range(m)
        ],
        "classes": [class_label_str(c) for c in values],
        "n": m,
        "correct": int((y_true == pred).sum()),
    }
