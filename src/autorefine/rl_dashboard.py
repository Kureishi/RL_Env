"""RL runners for the dashboard (SPEC.md 68.1, v0.54, A58) — the streamlit-
free core of the in-app RL loop.

The app's worker protocol (SPEC.md 51.2.3) is policy-agnostic: it only ever
calls ``start()`` → ``next()``…→ ``finish()`` and renders the posted
messages. ``DashboardRunner`` (dashboard.py) is one implementation of that
contract for the bandit/search loop; this module adds the RL side:

  * ``RLRunner`` — one data source, N full-budget episodes of
    ``propose → step → observe`` (exactly ``train_policy``'s loop body,
    SPEC.md 68.1.2), yielded one step at a time so the app can render it
    live; keep-best swaps the policy to the best-episode checkpoint
    (SPEC.md 67.3.1) before serializing (68.1.3).
  * ``RLMultiRunner`` — multi-task transfer (SPEC.md 20.2): one shared
    policy over several built-in tasks, round-robin episodes — the exact
    ``train_multi_policy`` order (68.1.4).
  * ``describe_policy`` — the JSON-safe preview dict for an uploaded /
    trained policy (SPEC.md 68.2.2).

No streamlit import (the SPEC.md 23.1 pattern): run semantics live here,
the app (dashboard_app.py) is a thin renderer over these classes.
"""
from __future__ import annotations

from pathlib import Path

from .config import Budget
from .diagnostics import holdout_difficulty, holdout_diagnostics
from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.rl_policy import (
    MetaRLPolicy,
    load_policy,
    policy_from_bytes,
    policy_to_bytes,
)
from .dashboard import (  # the shared pure helpers (one home, SPEC.md 23.1)
    DashboardRunner,
    family_stats,
    field_stats,
    field_value_stats,
    spec_diff,
)
from .plotting import (
    html_report,
    svg_architecture,
    svg_confusion_matrix,
    svg_mutation_timeline,
    svg_pareto,
    svg_per_class_bars,
    svg_score_curve,
    svg_score_gap_scatter,
    svg_score_strip,
    svg_time_strip,
)
from .tasks import TASKS, detect_modality

# SPEC.md 68.1.4 (v0.54, A58): the multi-task runner's task set — the
# built-in synthetic tasks (the user-data path needs a CSV, which the
# single RLRunner covers).
BUILTIN_TASKS = ("parity-v1", "sine-v1", "cartpole-v1")

_VALID_MODALITIES = ("auto", "csv", "image", "audio", "text")


def describe_policy(policy: MetaRLPolicy) -> dict:
    """SPEC.md 68.2.2 (v0.54, A58): the policy preview — a JSON-safe dict
    of the loadable state (what an upload *is*, before a run resumes it).
    Pure over the policy (G2)."""
    return {
        "n_actions": int(policy.n_actions),
        "state_dim": int(policy.state_dim),
        "task_names": list(policy.task_names) if policy.task_names else None,
        "n_updates": int(policy.n_updates),
        "epsilon": float(policy.epsilon),
        "epsilon_decay": float(policy.epsilon_decay),
        "lr": float(policy.lr),
        "baseline_decay": float(policy.baseline_decay),
        "has_bias": policy.B is not None,
    }


def _info_rows(task) -> dict:
    """The (train, holdout, gen) row counts for the info block (68.1.1).

    The CSV task pre-splits and exposes `_x_tr` / `_x_ho` / `_x_ge`; the
    built-in synthetic tasks (the multi path) carry only the dataset size
    and split lazily inside the env. Fall back to the dataset size so
    `start()` never crashes on a built-in task — the rows field is
    display-only and the multi info block is not drained (68.1.4)."""
    if all(hasattr(task, k) for k in ("_x_tr", "_x_ho", "_x_ge")):
        return {"train": int(len(task._x_tr)),
                "holdout": int(len(task._x_ho)),
                "gen": int(len(task._x_ge))}
    n = int(task.default_dataset_size)
    return {"train": n, "holdout": 0, "gen": 0}


class RLRunner:
    """RL loop in the dashboard's runner contract (SPEC.md 68.1.1).

    = ``train_policy`` (SPEC.md 15/67) decomposed into ``start()`` /
    ``next()`` / ``finish()`` — the same message shapes ``DashboardRunner``
    posts (68.1.1), so the app's worker/drain renders an RL run with no
    new plumbing (68.3.2).
    """

    def __init__(self, csv_path, label: str | None = None, split_frac: float = 0.2,
                 target: float = 95.0, seed: int = 7, experiments: int = 30,
                 max_seconds: float = 900.0, max_train_seconds: float = 30.0,
                 runs_dir: str = "runs", search_quality: str = "v04",
                 modality: str | None = None, stall_patience: int | None = None,
                 episodes: int = 5, epsilon: float = 0.0,
                 epsilon_decay: float = 1.0, keep_best: bool = False,
                 initial_policy: bytes | str | Path | None = None) -> None:
        # 68.5: the construction guards — a bad budget knob is a ValueError,
        # not a silent zero-episode run. ε bounds are validated by
        # MetaRLPolicy (67.2.1) at start(); the same values round-trip here.
        if (not isinstance(episodes, int) or isinstance(episodes, bool)
                or episodes < 1):
            raise ValueError(f"episodes must be an int >= 1, got {episodes!r}")
        if search_quality not in ("v04", "legacy"):
            raise ValueError(f"search_quality must be 'v04' or 'legacy', "
                             f"got {search_quality!r}")
        if modality is not None and modality not in _VALID_MODALITIES:
            raise ValueError(f"modality must be one of auto/csv/image/"
                             f"audio/text, got {modality!r}")
        self.csv_path = csv_path
        self.modality = modality or "auto"
        self.label = label or None
        self.split_frac = float(split_frac)
        self.target = float(target)
        self.policy_name = "rl"  # the worker/drain gate UCB on "bandit"
        self.seed = int(seed)
        self.experiments = int(experiments)
        self.max_seconds = float(max_seconds)
        self.max_train_seconds = float(max_train_seconds)
        self.runs_dir = str(runs_dir)
        self.search_quality = search_quality
        self.stall_patience = stall_patience
        # SPEC.md 68.1 (v0.54, A58): the RL knobs (the 67.x features)
        self.episodes = episodes
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.keep_best = bool(keep_best)
        self.initial_policy = initial_policy
        self.env: AutoRefineEnv | None = None
        self.policy: MetaRLPolicy | None = None
        self._state: dict | None = None
        self._phase = "idle"  # idle -> running -> done
        self._updates: list[dict] = []
        self._trace: list[dict] = []  # D2 records (the live policy views)
        self._returns: list[float] = []
        self._episodes_done = 0
        self._best_snap: dict | None = None  # 67.3.1 strictly-greater
        # 68.1.2 (v0.54, A58): one reset per episode — True once start() (or a
        # previous step's boundary reset) has begun the current episode; the
        # next() step that ends an episode clears it so the following step
        # resets exactly once, the train_policy loop body (a per-step reset
        # would hand every step a fresh budget: episode 2+ never ends)
        self._episode_started = False
        self._resumed = False
        # the app's stop request (51.2.2) — the env's stop_check reads it
        self._stop_requested = False

    @property
    def done(self) -> bool:
        return self._phase == "done"

    def request_stop(self) -> None:
        """51.2.2 (same as DashboardRunner): honored by the env between
        experiments; the in-flight episode's gradient is flushed by the
        env's done signal, so the partial run stays honest (68.1.1)."""
        self._stop_requested = True

    def _stop_check(self) -> bool:
        return self._stop_requested

    # --- task resolution (the DashboardRunner rule, SPEC.md 24.5) -----------
    def _resolve_task(self, data: Path) -> str:
        mod = self.modality
        if data.is_file():
            if mod in ("auto", "csv"):
                return "csv"
            raise ValueError(f"{data} is a file: modality must be 'csv' "
                             f"or 'auto'")
        if data.is_dir():
            if mod in ("image", "audio", "text"):
                return mod
            m = detect_modality(data)
            if m == "mixed":
                raise ValueError(
                    f"{data} holds items of more than one modality: set "
                    f"modality='image', 'audio', or 'text' ")
            if m is None:
                raise ValueError(
                    f"no image, audio, or text items under {data} — one "
                    f"subfolder per class, or an index.csv ")
            return m
        raise ValueError(f"no such data file: {data}")

    # --- lifecycle ----------------------------------------------------------
    def start(self) -> dict:
        """Probe the data, build the env + policy, train+score the baseline
        of episode 1 (the 23.1 split; the 68.1.1 info keys)."""
        if self._phase != "idle":
            raise RuntimeError(f"start() again in phase {self._phase!r}")
        data = Path(self.csv_path)
        if not data.exists():
            raise ValueError(f"no such data file: {data}")
        task_name = self._resolve_task(data)
        config: dict = {"path": str(data)}
        if self.label is not None:
            config["label"] = self.label
        if self.split_frac != 0.2:
            config["split_frac"] = self.split_frac

        probe = TASKS[task_name](seed=self.seed, **config)
        self.env = AutoRefineEnv(
            task=task_name, seed=self.seed,
            budget=Budget(self.experiments, self.max_seconds,
                          self.max_train_seconds),
            runs_dir=self.runs_dir,
            dataset_episodes=int(probe.default_dataset_size),
            task_config=config,
            **({} if self.search_quality == "legacy"
               else search_quality_v04()),
            stall_patience=self.stall_patience,
            stop_check=self._stop_check,  # 51.2.2: the app's stop hook
            # v0.23 (37.1.3): driver metadata for the canonical recipe —
            # `run`-style: the policy is the RL loop with its episode count
            policy=self.policy_name, target=self.target,
            rl_episodes=self.episodes,
        )
        # 68.2.2 (v0.54, A58): resume from a saved policy (bytes in memory
        # — the app's upload — or a path, the CLI's file) or build a fresh
        # one with the ε schedule (67.2)
        if self.initial_policy is not None:
            if isinstance(self.initial_policy, (bytes, bytearray)):
                self.policy = policy_from_bytes(
                    bytes(self.initial_policy), seed=self.seed)
            else:
                self.policy = load_policy(self.initial_policy, seed=self.seed)
            self._resumed = True
        else:
            self.policy = MetaRLPolicy(
                self.seed, epsilon=self.epsilon,
                epsilon_decay=self.epsilon_decay)
        self._state = self.env.reset()
        self._phase = "running"
        self._episode_started = True  # start() owns episode 1's reset
        info = {
            "label": probe.label_name,
            "head": probe.head,
            "class_values": probe.class_values,
            "features": probe.feature_names,
            "rows": _info_rows(probe),
            "baseline_score": self._state["baseline_score"],
            "baseline_train_seconds": self._baseline_train_seconds(
                self.env.memory.load_experiments()),
            "target": self.target,
            "policy": self.policy_name,
            "seed": self.seed,
            "budget_experiments": self.experiments,
            "run_dir": str(self.env.run_dir),
            # 68.1.1: the RL knobs (the live caption + the result block)
            "episodes": self.episodes,
            "epsilon": self.policy.epsilon,
            "epsilon_decay": self.policy.epsilon_decay,
            "n_updates": int(self.policy.n_updates),
            "resumed": self._resumed,
        }
        return info

    def next(self) -> dict:
        """One improver step → one JSON-safe update (the 68.1.1 shape: the
        DashboardRunner keys + the RL extras). The loop body is exactly
        ``train_policy``'s (68.1.2): a fresh ``env.reset()`` per episode,
        then ``propose → step → observe``.
        """
        if self._phase == "idle":
            raise RuntimeError("call start() before next()")
        if self._phase == "done":
            raise RuntimeError("the run is done; call finish()")
        # episode boundary (R5: reset starts a fresh episode + run dir) —
        # exactly once per episode, before its first step (68.1.2: the
        # train_policy loop body; a per-step reset would hand every step a
        # fresh budget and episode 2+ would never end)
        if not self._episode_started:
            self._state = self.env.reset()
            self._episode_started = True
        state = self._state
        ep_no = self._episodes_done + 1
        action = self.policy.propose(state)
        prev_best = self.env.best_spec.to_dict() if self.env.best_spec else {}
        action_dict = action if isinstance(action, dict) else action.to_dict()
        self._state, reward, done, info = self.env.step(action)
        # SPEC.md 73.2 (v0.59, A63): free no-op steps are invisible to the
        # update; the D2 trace below still records them.
        self.policy.observe(reward, done,
                            no_op=info.get("candidate_score") is None)
        # D2 (SPEC.md 29.2): the live policy views' per-step record
        a, p = self.policy.last_proposal
        self._trace.append({
            "task": state.get("task"),
            "action": int(a),
            "probs": [float(v) for v in p],
            "reward": float(reward),
        })
        episode_done = bool(done)
        if episode_done:
            self._episodes_done += 1
            self._returns.append(self.policy.last_episode_return)
            # 67.3.1: the strictly-greater best-episode snapshot (keep-best)
            r = self.policy.last_episode_return
            if (self._best_snap is None
                    or r > self._best_snap["return"]):
                self._best_snap = {
                    "episode": ep_no - 1,
                    "return": float(r),
                    "w": self.policy.w.copy(),
                    "b": self.policy.b.copy(),
                    "B": (self.policy.B.copy()
                          if self.policy.B is not None else None),
                }
            if self._episodes_done >= self.episodes:
                self._phase = "done"
            self._episode_started = False  # the next step starts a new episode
        # a Stop-honored step (51.2.2) ends the whole run honestly (68.1.1)
        if info.get("reason") == "stopped":
            self._phase = "done"
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
            "loss_history": info.get("loss_history") or [],
            "effective_score": info.get("effective_score"),
            "se": info.get("se"),
            "best_score": self.env.best_score,
            "reward": reward,
            # the worker's break signal — True only on the LAST step of the
            # LAST episode (the app's worker loop, 51.2.3)
            "done": self._phase == "done",
            # 68.1.1: the RL extras (the drain's episode-boundary line)
            "episode": ep_no,
            "epsilon": float(self.policy.epsilon),
            "episode_return": (self.policy.last_episode_return
                               if episode_done else None),
            "episode_done": episode_done,
            "task": state.get("task"),
            "ucb": None,  # UCB is a bandit concept (SPEC.md 26.4)
        }
        self._updates.append(update)
        # the same decision views DashboardRunner attaches (SPEC.md 26.5)
        update["field_stats"] = field_stats(self._updates)
        update["field_value_stats"] = field_value_stats(self._updates)
        return update

    def run_all(self, on_update=None) -> dict:
        """Drive the loop to done, calling ``on_update(update)`` per step."""
        while self._phase == "running":
            update = self.next()
            if on_update is not None:
                on_update(update)
        return self.finish()

    # --- results ------------------------------------------------------------
    def finish(self) -> dict:
        """Verdict + artifacts (the DashboardRunner shared keys) + the
        ``rl`` block (68.1.3) — the app's result view renders from it."""
        if self._phase != "done":
            raise RuntimeError("the run is not finished yet (loop to done first)")
        env = self.env
        mem = env.memory
        run_dir = env.run_dir
        summary = mem.load_summary()
        entries = mem.load_experiments()
        final = float(env.best_score)
        pass_ = final >= self.target  # the 22.1 gate, exactly
        pareto_pts = summary.get("pareto_frontier") or [
            {"score": e["holdout_score"],
             "train_seconds": e.get("train_seconds", 0.0)}
            for e in entries
            if e.get("kind") in ("baseline", "experiment")
            and isinstance(e.get("holdout_score"), (int, float))
        ]
        # 68.1.3 (v0.54, A58): keep-best — the downloaded policy IS the
        # best-episode checkpoint (67.3.1's strictly-greater snapshot);
        # keep_best=False (the CLI parity default) keeps the final weights.
        if self.keep_best and self._best_snap is not None:
            self.policy.w = self._best_snap["w"].copy()
            self.policy.b = self._best_snap["b"].copy()
            if self._best_snap["B"] is not None:
                self.policy.B = self._best_snap["B"].copy()
        policy_bytes = policy_to_bytes(self.policy)
        # the same learning views DashboardRunner.finish() computes
        diag = holdout_diagnostics(env.task, env.best_model)
        diff = (holdout_difficulty(env.task, env.best_model)
                if env.best_model is not None else None)
        gallery: list | None = None
        hold_errors = getattr(env.task, "holdout_errors", None)
        if callable(hold_errors) and env.best_model is not None:
            g = hold_errors(env.best_model)
            gallery = g if g else None
        arch_svg = (svg_architecture(env.best_spec.to_dict(),
                                     env.task.state_dim,
                                     env.task.n_outputs)
                    if env.best_spec else None)
        baseline_ts = self._baseline_train_seconds(entries)
        return {
            "verdict": "PASS" if pass_ else "MISS",
            "target": self.target,
            "final_best_score": final,
            "baseline_score": float(env.baseline_score),
            "improvement_factor": summary.get("improvement_factor"),
            "experiments_run": summary.get("experiments_run"),
            "wall_seconds": summary.get("wall_seconds"),
            "finished_reason": summary.get("finished_reason"),
            "summary": summary,
            "updates": list(self._updates),
            "best_spec": (env.best_spec.to_dict() if env.best_spec else None),
            "svg_score": svg_score_curve(entries),
            "svg_pareto": svg_pareto(pareto_pts),
            "field_stats": field_stats(self._updates),
            "field_value_stats": field_value_stats(self._updates),
            "family_stats": family_stats(entries),
            "timeline_svg": svg_mutation_timeline(self._updates),
            "strip_svg": svg_score_strip(self._updates),
            "scatter_svg": svg_score_gap_scatter(self._updates),
            "baseline_train_seconds": baseline_ts,
            "time_strip_svg": svg_time_strip(self._updates, baseline_ts),
            "loss_curves": DashboardRunner._loss_curves(entries),
            "diagnostics": diag,
            "per_class_svg": svg_per_class_bars(diag) if diag else None,
            "confusion_svg": svg_confusion_matrix(diag) if diag else None,
            "difficulty": diff,
            "state_dim": env.task.state_dim,
            "n_out": env.task.n_outputs,
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
            # SPEC.md 68.1.3 (v0.54, A58): the RL block — the app's result
            # section (68.3.3) renders from exactly these keys
            "rl": {
                "episodes": self.episodes,
                "episode_returns": list(self._returns),
                "best": ({"episode": self._best_snap["episode"],
                          "return": self._best_snap["return"]}
                         if self._best_snap is not None else None),
                "n_updates": int(self.policy.n_updates),
                "epsilon_final": float(self.policy.epsilon),
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "trace": list(self._trace),
                "returns": {env.task_name: list(self._returns)},
                "task_names": [env.task_name],
                "policy_bytes": policy_bytes,
                "keep_best": bool(self.keep_best),
                "resumed": self._resumed,
            },
        }

    @staticmethod
    def _baseline_train_seconds(entries) -> float | None:
        return DashboardRunner._baseline_train_seconds(entries)


class RLMultiRunner:
    """Multi-task transfer in the app (SPEC.md 68.1.4) — one shared policy
    over several built-in tasks, round-robin episodes (the exact
    ``train_multi_policy`` order), driven through the same
    ``start`` / ``next`` / ``done`` / ``finish`` interface so the Compare
    tab can run it live (68.3.4).
    """

    def __init__(self, tasks, episodes_per_task: int = 1, seed: int = 7,
                 experiments: int = 30, max_seconds: float = 900.0,
                 max_train_seconds: float = 30.0, runs_dir: str = "runs",
                 search_quality: str = "v04",
                 stall_patience: int | None = None,
                 epsilon: float = 0.0, epsilon_decay: float = 1.0,
                 initial_policy: bytes | str | Path | None = None,
                 keep_best: bool = False) -> None:
        tasks = [str(t) for t in (tasks or [])]
        if not tasks:
            raise ValueError("tasks must be non-empty")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"tasks must be unique, got {tasks!r}")
        unknown = [t for t in tasks if t not in BUILTIN_TASKS]
        if unknown:
            raise ValueError(
                f"unknown task(s) {unknown!r}: the multi-task runner "
                f"supports the built-ins {list(BUILTIN_TASKS)}")
        if (not isinstance(episodes_per_task, int)
                or isinstance(episodes_per_task, bool)
                or episodes_per_task < 1):
            raise ValueError("episodes_per_task must be an int >= 1, "
                             f"got {episodes_per_task!r}")
        if search_quality not in ("v04", "legacy"):
            raise ValueError(f"search_quality must be 'v04' or 'legacy', "
                             f"got {search_quality!r}")
        self.tasks = tasks
        self.episodes_per_task = episodes_per_task
        self.total_episodes = episodes_per_task * len(tasks)
        self.seed = int(seed)
        self.experiments = int(experiments)
        self.max_seconds = float(max_seconds)
        self.max_train_seconds = float(max_train_seconds)
        self.runs_dir = str(runs_dir)
        self.search_quality = search_quality
        self.stall_patience = stall_patience
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.initial_policy = initial_policy
        self.keep_best = bool(keep_best)  # unused for multi (67.3.2) — kept
        # for the shared interface; the best snapshot stays None there
        self.policy_name = "rl"
        self.envs: list[AutoRefineEnv] = []
        self.policy: MetaRLPolicy | None = None
        self._active: AutoRefineEnv | None = None
        self._state: dict | None = None
        self._phase = "idle"
        self._updates: list[dict] = []
        self._trace: list[dict] = []
        self._returns: dict[str, list[float]] = {t: [] for t in tasks}
        self._episodes_done = 0
        # 68.1.2 (v0.54, A58): one reset per episode (the train_policy /
        # train_multi_policy loop body) — see RLRunner._episode_started
        self._episode_started = False
        self._resumed = False
        self._stop_requested = False

    @property
    def done(self) -> bool:
        return self._phase == "done"

    @property
    def env(self) -> AutoRefineEnv:
        """The drain's ``runner.env.*`` reads (z_accept, task dims) — the
        most recently active env (the first before any step)."""
        return self._active if self._active is not None else self.envs[0]

    def request_stop(self) -> None:
        self._stop_requested = True

    def _stop_check(self) -> bool:
        return self._stop_requested

    def _active_env(self) -> AutoRefineEnv:
        # 68.1.4: round-robin — episode k takes task (k mod n), the exact
        # train_multi_policy order (20.2)
        return self.envs[self._episodes_done % len(self.envs)]

    def start(self) -> dict:
        """Build the per-task envs + the shared policy, and reset the first
        task (episode 0) for the info block (68.1.1)."""
        if self._phase != "idle":
            raise RuntimeError(f"start() again in phase {self._phase!r}")
        quality = ({} if self.search_quality == "legacy"
                   else search_quality_v04())
        self.envs = []
        for name in self.tasks:
            probe = TASKS[name](seed=self.seed)
            self.envs.append(AutoRefineEnv(
                task=name, seed=self.seed,
                budget=Budget(self.experiments, self.max_seconds,
                              self.max_train_seconds),
                runs_dir=self.runs_dir,
                dataset_episodes=int(probe.default_dataset_size),
                **quality,
                stall_patience=self.stall_patience,
                stop_check=self._stop_check,
                policy=self.policy_name, target=None,
                rl_episodes=self.total_episodes,
            ))
        # 68.2.2: resume (must carry the same task_names — 67.1.2) or fresh
        if self.initial_policy is not None:
            if isinstance(self.initial_policy, (bytes, bytearray)):
                self.policy = policy_from_bytes(
                    bytes(self.initial_policy), seed=self.seed)
            else:
                self.policy = load_policy(self.initial_policy, seed=self.seed)
            self._resumed = True
        else:
            self.policy = MetaRLPolicy(
                self.seed, task_names=list(self.tasks),
                epsilon=self.epsilon, epsilon_decay=self.epsilon_decay)
        if self.policy.B is None:
            raise ValueError("the multi-task policy needs task_names "
                             "(a loaded policy must carry them)")
        if self.policy.task_names != self.tasks:
            raise ValueError(
                f"the loaded policy's tasks {self.policy.task_names} "
                f"do not match {self.tasks}")
        first = self.envs[0]
        # the env's own task instance (the multi path is built-ins only, so
        # it carries no task_config — a fresh probe would just re-split the
        # same dataset). The built-in tasks expose only the Task protocol
        # (head, default_dataset_size, ...) — no `class_values` /
        # `feature_names` (those belong to the user-data tasks) — and the
        # info block is display-only (the `_info_rows` dataset-size
        # fallback, 68.1.1), so the absent fields read None instead of
        # crashing start() (68.1.4: the Compare tab drives this live).
        task0 = first.task
        self._active = first
        self._state = first.reset()
        self._phase = "running"
        self._episode_started = True  # start() owns episode 1's reset
        return {
            "label": "multi: " + ", ".join(self.tasks),
            "head": getattr(task0, "head", None),
            "class_values": getattr(task0, "class_values", None),
            "features": getattr(task0, "feature_names", None),
            "rows": _info_rows(task0),
            "baseline_score": self._state["baseline_score"],
            "baseline_train_seconds": self._baseline_train_seconds(
                first.memory.load_experiments()),
            "target": 100.0,  # multi is un-gated (like `run`); verdict n/a
            "policy": self.policy_name,
            "seed": self.seed,
            "budget_experiments": self.experiments,
            "run_dir": str(first.run_dir),
            "episodes": self.total_episodes,
            "episodes_per_task": self.episodes_per_task,
            "tasks": list(self.tasks),
            "epsilon": self.policy.epsilon,
            "epsilon_decay": self.policy.epsilon_decay,
            "n_updates": int(self.policy.n_updates),
            "resumed": self._resumed,
        }

    def next(self) -> dict:
        """One step of the current (task, episode) — the same update shape
        as ``RLRunner.next`` with the update's ``task`` set (68.1.4)."""
        if self._phase == "idle":
            raise RuntimeError("call start() before next()")
        if self._phase == "done":
            raise RuntimeError("the run is done; call finish()")
        # episode boundary — exactly once per episode, before its first
        # step (68.1.2: the train_multi_policy loop body)
        if not self._episode_started:
            self._active = self._active_env()
            self._state = self._active.reset()
            self._episode_started = True
        task = self.tasks[self._episodes_done % len(self.tasks)]
        env = self._active
        state = self._state
        ep_no = self._episodes_done + 1
        action = self.policy.propose(state)
        prev_best = env.best_spec.to_dict() if env.best_spec else {}
        action_dict = action if isinstance(action, dict) else action.to_dict()
        self._state, reward, done, info = env.step(action)
        # SPEC.md 73.2 (v0.59, A63): no-op exclusion (per-task, 73.2.4)
        self.policy.observe(reward, done,
                            no_op=info.get("candidate_score") is None)
        a, p = self.policy.last_proposal
        self._trace.append({
            "task": task,
            "action": int(a),
            "probs": [float(v) for v in p],
            "reward": float(reward),
        })
        episode_done = bool(done)
        if episode_done:
            self._episodes_done += 1
            self._returns[task].append(self.policy.last_episode_return)
            if self._episodes_done >= self.total_episodes:
                self._phase = "done"
            self._episode_started = False  # the next step starts a new episode
        if info.get("reason") == "stopped":
            self._phase = "done"
        mutation = list((self._state.get("last") or {}).get("fields") or [])
        update = {
            "index": len(self._updates) + 1,
            "accepted": bool(info.get("accepted", False)),
            "reason": info.get("reason"),
            "candidate_score": info.get("candidate_score"),
            "gen_gap": info.get("gen_gap"),
            "train_seconds": info.get("train_seconds"),
            "experiments_left": info.get("experiments_left",
                                         env.bm.experiments_left),
            "mutation": mutation,
            "spec_diff": spec_diff(prev_best, action_dict, mutation),
            "loss_history": info.get("loss_history") or [],
            "effective_score": info.get("effective_score"),
            "se": info.get("se"),
            "best_score": env.best_score,
            "reward": reward,
            "done": self._phase == "done",
            "episode": ep_no,
            "epsilon": float(self.policy.epsilon),
            "episode_return": (self.policy.last_episode_return
                               if episode_done else None),
            "episode_done": episode_done,
            "task": task,
            "ucb": None,
        }
        self._updates.append(update)
        update["field_stats"] = field_stats(self._updates)
        update["field_value_stats"] = field_value_stats(self._updates)
        return update

    def run_all(self, on_update=None) -> dict:
        while self._phase == "running":
            update = self.next()
            if on_update is not None:
                on_update(update)
        return self.finish()

    def finish(self) -> dict:
        """The shared result keys (over the last active env) + the ``rl``
        block with the per-task returns (68.1.4). ``best`` stays ``None``
        for multi (67.3.2)."""
        if self._phase != "done":
            raise RuntimeError("the run is not finished yet (loop to done first)")
        env = self._active if self._active is not None else self.envs[0]
        mem = env.memory
        run_dir = env.run_dir
        summary = mem.load_summary()
        entries = mem.load_experiments()
        final = float(env.best_score)
        diag = holdout_diagnostics(env.task, env.best_model)
        baseline_ts = self._baseline_train_seconds(entries)
        return {
            "verdict": "PASS" if final >= 100.0 else "MISS",
            "target": 100.0,
            "final_best_score": final,
            "baseline_score": float(env.baseline_score),
            "improvement_factor": summary.get("improvement_factor"),
            "experiments_run": summary.get("experiments_run"),
            "wall_seconds": summary.get("wall_seconds"),
            "finished_reason": summary.get("finished_reason"),
            "summary": summary,
            "updates": list(self._updates),
            "best_spec": (env.best_spec.to_dict() if env.best_spec else None),
            "svg_score": svg_score_curve(entries),
            "svg_pareto": svg_pareto([
                {"score": e["holdout_score"],
                 "train_seconds": e.get("train_seconds", 0.0)}
                for e in entries
                if e.get("kind") in ("baseline", "experiment")
                and isinstance(e.get("holdout_score"), (int, float))
            ]),
            "field_stats": field_stats(self._updates),
            "loss_curves": DashboardRunner._loss_curves(entries),
            "diagnostics": diag,
            "per_class_svg": svg_per_class_bars(diag) if diag else None,
            "confusion_svg": svg_confusion_matrix(diag) if diag else None,
            "state_dim": env.task.state_dim,
            "n_out": env.task.n_outputs,
            "report_html": html_report(summary, entries, diagnostics=diag),
            "artifacts": {
                "best_spec.json": (run_dir / "best_spec.json").read_bytes(),
                "best_model.npz": (run_dir / "best_model.npz").read_bytes(),
                "experiments.jsonl": (run_dir / "experiments.jsonl").read_bytes(),
            },
            "run_dir": str(run_dir),
            "rl": {
                "episodes": self.total_episodes,
                "episodes_per_task": self.episodes_per_task,
                "episode_returns": {t: list(r)
                                    for t, r in self._returns.items()},
                "best": None,  # 67.3.2: "best episode" is ambiguous for multi
                "n_updates": int(self.policy.n_updates),
                "epsilon_final": float(self.policy.epsilon),
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "trace": list(self._trace),
                # {task: [returns]} — exactly svg_task_returns' input
                "returns": {t: list(r) for t, r in self._returns.items()},
                "task_names": list(self.tasks),
                "policy_bytes": policy_to_bytes(self.policy),
                "keep_best": False,
                "resumed": self._resumed,
            },
        }

    @staticmethod
    def _baseline_train_seconds(entries) -> float | None:
        return DashboardRunner._baseline_train_seconds(entries)
