"""DashboardRunner: the streamlit-free core of the v0.9 visual dashboard
(SPEC.md 23.1).

The app (dashboard_app.py) is a thin rendering layer over this class; the
run semantics live here so they stay testable without streamlit and so
same-seed runs reproduce the CLI `fit` outcome with the same flags (G2).
No streamlit import (SPEC.md 23.1, §3 optional-dependency pattern).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Budget
from .improver.bandit import BanditPolicy
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.policy import SearchPolicy
from .plotting import html_report, svg_pareto, svg_score_curve
from .tasks import CsvTask

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
                 search_quality: str = "v04") -> None:
        if policy not in ("bandit", "search"):
            raise ValueError(f"unsupported policy {policy!r}: {_RL_HINT}")
        if search_quality not in ("v04", "legacy"):
            raise ValueError(f"search_quality must be 'v04' or 'legacy', "
                             f"got {search_quality!r}")
        self.csv_path = csv_path
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
    def start(self) -> dict:
        """Probe the CSV, build the env, train+score the baseline (SPEC.md 23.1)."""
        if self._phase != "idle":
            raise RuntimeError(f"start() again in phase {self._phase!r}")
        data = Path(self.csv_path)
        if not data.is_file():
            raise ValueError(f"no such CSV file: {data}")
        config: dict = {"path": str(data)}
        if self.label is not None:
            config["label"] = self.label
        if self.split_frac != 0.2:
            config["split_frac"] = self.split_frac

        # deterministic probe: the env re-derives the identical split from
        # the same seed (SPEC.md 22.1)
        probe = CsvTask(seed=self.seed, **config)
        self.env = AutoRefineEnv(
            task="csv", seed=self.seed,
            budget=Budget(self.experiments, self.max_seconds, self.max_train_seconds),
            runs_dir=self.runs_dir,
            dataset_episodes=int(probe.default_dataset_size),
            task_config=config,
            # the CLI default (SPEC.md 18.6); 'legacy' = the deterministic
            # v0.3 rule (strict score comparison, no wall-clock eff term)
            **({} if self.search_quality == "legacy" else search_quality_v04())
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
        """One improver step → one JSON-safe update dict (SPEC.md 23.1)."""
        if self._phase == "idle":
            raise RuntimeError("call start() before next()")
        if self._phase == "done":
            raise RuntimeError("the run is done; call finish()")
        action = self.policy.propose(self._state)
        self._state, reward, done, info = self.env.step(action)
        update = {
            "index": len(self._updates) + 1,
            "accepted": bool(info.get("accepted", False)),
            "reason": info.get("reason"),
            "candidate_score": info.get("candidate_score"),
            "gen_gap": info.get("gen_gap"),
            "train_seconds": info.get("train_seconds"),
            "experiments_left": info.get("experiments_left",
                                         self.env.bm.experiments_left),
            "mutation": list((self._state.get("last") or {}).get("fields") or []),
            "best_score": self.env.best_score,
            "reward": reward,
            "done": bool(done),
        }
        self._updates.append(update)
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
            if e.get("kind") in ("baseline", "experiment")
            and isinstance(e.get("holdout_score"), (int, float))
        ]
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
            "report_html": html_report(summary, entries),
            "artifacts": {
                "best_spec.json": (run_dir / "best_spec.json").read_bytes(),
                "best_model.npz": (run_dir / "best_model.npz").read_bytes(),
                "experiments.jsonl": (run_dir / "experiments.jsonl").read_bytes(),
            },
            "run_dir": str(run_dir),
        }

    # --- helpers ------------------------------------------------------------
    def updates_json(self) -> str:
        """The full update stream as JSON (machine-readable live feed)."""
        return json.dumps(self._updates, indent=2, sort_keys=True)
