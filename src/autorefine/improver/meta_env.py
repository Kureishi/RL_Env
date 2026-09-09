"""AutoRefineEnv: the meta environment (SPEC.md 6).

reset()/step() contract:
  state = env.reset()
  state, reward, done, info = env.step(spec_dict)

reward = candidate_holdout_score - best_score + 1e-3 novelty bonus
The best spec updates only on strict improvement (R2); duplicate specs are
rejected for free (R3); stepping after done raises (R5).

v0.4 (SPEC.md 18): block-bootstrap CI, efficiency-aware scoring, a
generalization-gap penalty, and the unified acceptance rule — all off by
default, so v0.3 behavior is exactly recoverable.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..budget import BudgetManager
from ..config import DEFAULT_SPEC, Budget, ModelSpec, SpecError
from ..evaluator import evaluate_full, kfold_score, score_with_ci
from ..memory import (
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
    KIND_INVALID_SPEC,
    KIND_SCREEN,
    RunMemory,
)
from ..pareto import ParetoFrontier
from ..runconfig import RunConfig  # SPEC.md 37.1 (v0.23, G3)
from ..tasks import TASKS
from ..trainer import train
from .curriculum import ParityCurriculum  # SPEC.md 20.1 (type reference)

NOVELTY_BONUS = 1e-3
HISTORY_DIGEST_LEN = 8

# SPEC.md 18.4: T_CAP — max seconds of training time charged by eff()
TIME_PENALTY_CAP = 10.0
# SPEC.md 18.5: tolerated gen gap = 5% of score (exactly acceptance A3)
GEN_GAP_TOL = 0.05
# SPEC.md 19.3: holdout split size for the ensemble final evaluation
ENSEMBLE_EVAL_POINTS = 200
# SPEC.md 20.2 (stall guard): max consecutive duplicate rejections per
# episode — a collapsed meta-RL policy would otherwise loop on free
# duplicates (R3) until the wall clock
MAX_CONSECUTIVE_DUPLICATES = 32


class _EnsembleModel:
    """SPEC.md 19.3: average of member model outputs (logits for softmax
    tasks, values for mse). Same `forward(x) -> (n, k)` contract, so any
    task can score it. SPEC.md 25.3: members must share one input contract
    (grid vs flat) — the ensemble takes the top member's contract, so the
    averaged forward never mixes (n, d) and (n, C, H, W) inputs."""

    def __init__(self, members: list) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        top_grid = bool(getattr(members[0], "wants_grid", False))
        kept = [m for m in members
                if bool(getattr(m, "wants_grid", False)) == top_grid]
        self.members = kept if kept else list(members)  # >= 1 (top matches)
        self.wants_grid = top_grid  # routed by the task's score() (SPEC.md 25.3)

    def forward(self, x):
        out = self.members[0].forward(x)
        for m in self.members[1:]:
            out = out + m.forward(x)
        return out / len(self.members)


def effective_score(score: float, train_seconds: float, gen_gap: float,
                    efficiency_weight: float, gen_gap_penalty: float) -> float:
    """SPEC.md 18.4 + 18.5: eff = score − η·min(t, T_CAP) − gap_pen,
    with gap_pen = max(0, gen_gap − 0.05·score) · P_GEN.

    Both weights 0 ⇒ exactly the raw score (v0.3 behavior)."""
    time_pen = efficiency_weight * min(max(float(train_seconds), 0.0), TIME_PENALTY_CAP)
    excess = max(0.0, float(gen_gap) - GEN_GAP_TOL * float(score))
    return float(score) - time_pen - excess * float(gen_gap_penalty)


def should_accept(d_eff: float, se: float, z: float) -> bool:
    """SPEC.md 18.6: accept ⟺ Δeff > max(0, z·SE).

    z = 0 (and all other knobs 0) degrades exactly to v1's Δraw > 0."""
    return float(d_eff) > max(0.0, float(z) * float(se))


# --- knob registry (SPEC.md 33.2, C2) -------------------------------------
# One table for every tunable AutoRefineEnv knob with a default: the
# default, the validator (one home for validation — the env's __init__
# routes through it), the spec ref, the CLI subcommands exposing
# `--{kebab-name}`, and the dashboard widget key (None = CLI/env only).
# Presets (33.2.2) and the CLI (33.2.3) are checked against this table by
# the A23 tests; adding a knob is one row here + the env kwarg.

def _v_int(value: Any) -> int:
    return int(value)


def _v_float(value: Any) -> float:
    return float(value)


def _v_top_k(value: Any) -> int:
    return max(0, int(value))


def _v_stall_patience(value: Any) -> Any:
    if value is not None and \
            (not isinstance(value, int) or value < 1):
        raise ValueError(
            f"stall_patience must be an int >= 1 or None (off), "
            f"got {value!r} (SPEC.md 31.1)")
    return value


def _v_kfold(value: Any) -> int:
    if value is None or isinstance(value, bool) \
            or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"kfold must be an int >= 0 (0 = off), got {value!r} "
            f"(SPEC.md 44.1)")
    return value


def _v_screen_frac(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(
            f"screen_frac must be a number in (0, 1] (1.0 = off), "
            f"got {value!r} (SPEC.md 32.2)")
    try:
        frac = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"screen_frac must be a number in (0, 1] (1.0 = off), "
            f"got {value!r} (SPEC.md 32.2)") from None
    if not math.isfinite(frac) or not (0.0 < frac <= 1.0):
        raise ValueError(
            f"screen_frac must be a number in (0, 1] (1.0 = off), "
            f"got {value!r} (SPEC.md 32.2)")
    return frac


@dataclass(frozen=True)
class Knob:
    """One registry row (SPEC.md 33.2). `validate` normalizes + checks a
    single value, raising `ValueError` with the env's message."""
    name: str
    default: Any
    spec_ref: str
    cli: tuple[str, ...] = ()          # subcommands exposing --{kebab}
    app_widget: str | None = None      # dashboard key (None = not in app)
    validate: Callable[[Any], Any] = lambda value: value


# Order = the AutoRefineEnv.__init__ kwarg order (the env defaults are the
# single source of truth the A23 default test compares against)
KNOBS: dict[str, Knob] = {
    "ci_blocks": Knob("ci_blocks", 0, "18.3", (), None, _v_int),
    "z_accept": Knob("z_accept", 0.0, "18.6", (), None, _v_float),
    "efficiency_weight": Knob("efficiency_weight", 0.0, "18.4", (), None, _v_float),
    "gen_gap_penalty": Knob("gen_gap_penalty", 0.0, "18.5", (), None, _v_float),
    "block_size": Knob("block_size", 512, "18.3", (), None, _v_int),
    "ensemble_top_k": Knob("ensemble_top_k", 0, "19.3", (), None, _v_top_k),
    "stall_patience": Knob("stall_patience", None, "31.1",
                           ("run", "fit", "variance"), None, _v_stall_patience),
    "screen_frac": Knob("screen_frac", 1.0, "32.2",
                        ("run", "fit"), None, _v_screen_frac),
    "kfold": Knob("kfold", 0, "44.1", ("run", "fit"), None, _v_kfold),
}


def _preset(name: str, values: dict[str, Any]) -> dict[str, Any]:
    """SPEC.md 33.2 (C2): a preset may only set registry knobs with values
    that pass the registry validator — a typo'd knob name is a
    construction-time error, not a runtime surprise."""
    for key, value in values.items():
        if key not in KNOBS:
            raise KeyError(
                f"preset {name!r} sets unknown knob {key!r} (SPEC.md 33.2)")
        KNOBS[key].validate(value)  # raises ValueError on a bad value
    return dict(values)


def search_quality_v04() -> dict:
    """SPEC.md 18.6: the recommended v0.4 preset as AutoRefineEnv kwargs
    (ci_blocks/z/η/P_GEN per §18.3–§18.5; block_size stays the default).
    SPEC.md 33.2: checked against the KNOBS registry."""
    return _preset("search_quality_v04", dict(
        ci_blocks=8,             # 18.3: 8 blocks × block_size per split
        z_accept=1.0,            # 18.6: ~95% two-sided, deliberately not 1.96
        efficiency_weight=0.5,   # 18.4: η points per training second (T_CAP 10s)
        gen_gap_penalty=0.5,     # 18.5: P_GEN points per excess gap point
    ))


def candidate_screening(frac: float = 0.25) -> dict:
    """SPEC.md 32.2: opt-in two-stage candidate screening as AutoRefineEnv
    kwargs (screen on the first frac·n rows; full-train only strict
    champion beats; the `search_quality_v04()` preset pattern).
    SPEC.md 33.2: checked against the KNOBS registry."""
    return _preset("candidate_screening", dict(screen_frac=frac))


class AutoRefineEnv:
    def __init__(
        self,
        task: str = "cartpole-v1",
        seed: int = 7,
        budget: Budget | None = None,
        runs_dir: str | Path = "runs",
        # SPEC.md 20.3: None (default) falls back to task.default_dataset_size
        # (points for fitting tasks, episodes for episode tasks); an explicit
        # int keeps the legacy size (parameter name unchanged)
        dataset_episodes: int | None = None,
        # --- v0.4 search-quality knobs (SPEC.md 18; legacy defaults) ----------
        ci_blocks: int = 0,
        z_accept: float = 0.0,
        efficiency_weight: float = 0.0,
        gen_gap_penalty: float = 0.0,
        block_size: int = 512,
        # --- v0.5 frontier ensembling (SPEC.md 19.3; 0 = off, v0.4 exact) --
        ensemble_top_k: int = 0,
        # --- v0.6 curriculum (SPEC.md 20.1; None = off, v0.5 exact) ---------
        curriculum: ParityCurriculum | None = None,
        # --- v0.8 task config (SPEC.md 22.1; None = off, pre-v0.8 exact) -----
        task_config: dict | None = None,
        # --- v0.17 stall patience (SPEC.md 31.1; None = off, pre-v0.17 exact)
        stall_patience: int | None = None,
        # --- v0.18 two-stage screening (SPEC.md 32.2; 1.0 = off, pre-v0.18)
        screen_frac: float = 1.0,
        # --- v0.30 k-fold holdout scoring (SPEC.md 44.1; 0 = off, legacy)
        kfold: int = 0,
        # --- v0.23 driver metadata (SPEC.md 37.1.3; None = env-level) ----
        # the env does not own these — the driver (cli fit / dashboard /
        # variance) supplies them so the reset-time recipe is complete
        policy: str | None = None,
        target: float | None = None,
        rl_episodes: int | None = None,
        # --- v0.24 lineage (SPEC.md 38.2; None = a fresh run) -----------------
        parent_run: str | None = None,
    ) -> None:
        if task not in TASKS:  # registry: SPEC.md 15 "more tasks"
            raise ValueError(f"unknown task {task!r} (available: {sorted(TASKS)})")
        # SPEC.md 33.2 (C2): validation routes through the KNOBS registry —
        # one home for the rules, the A21/A22 error messages unchanged
        _sp = KNOBS["stall_patience"].validate(stall_patience)
        _sf = KNOBS["screen_frac"].validate(screen_frac)
        _kf = KNOBS["kfold"].validate(kfold)
        task_cls = TASKS[task]
        self.task_name = task
        self.seed = int(seed)
        self.budget = budget or Budget()
        # SPEC.md 22.1: optional constructor kwargs for data-driven tasks
        # (e.g. csv: {path, label, split_frac}); empty keeps the plain path
        self.task_config = dict(task_config or {})
        if self.task_config:
            self.task = task_cls(seed=self.seed, **self.task_config)
        else:
            self.task = task_cls(seed=self.seed)
        self.bm = BudgetManager(self.budget)
        self.runs_dir = Path(runs_dir)
        # SPEC.md 20.3: the effective train-split size in the task's native
        # units — explicit `dataset_episodes` wins, else the task default
        self.dataset_episodes = dataset_episodes
        self.dataset_size = (int(dataset_episodes) if dataset_episodes is not None
                             else int(task_cls.default_dataset_size))
        # SPEC.md 20.1: opt-in curriculum (adaptive difficulty); None = off
        self.curriculum = curriculum
        self.curriculum_events: list[dict] = []  # per-step-up rows (summary)
        # SPEC.md 18: search-quality knobs (legacy = all off; v0.3 exact)
        self.ci_blocks = int(ci_blocks)
        self.z_accept = float(z_accept)
        self.efficiency_weight = float(efficiency_weight)
        self.gen_gap_penalty = float(gen_gap_penalty)
        self.block_size = int(block_size)
        # SPEC.md 19.3: track the top-k models by holdout score for the
        # opt-in ensemble final evaluation (0 = off; v0.4 runs unchanged)
        self.ensemble_top_k = max(0, int(ensemble_top_k))
        self._top: list[tuple[float, object]] = []

        self.memory: RunMemory | None = None
        self.run_dir: Path | None = None
        self.dataset: Any = None
        self.best_spec: ModelSpec | None = None
        self.best_score = float("-inf")
        self.baseline_score = float("-inf")
        self.best_eff = float("-inf")      # SPEC.md 18.4: eff of current best
        self.baseline_eff = float("-inf")  # SPEC.md 18.4: eff at reset
        self.best_std = 0.0                # SPEC.md 18.3: holdout σ of best
        self.best_model = None
        self.seen: set[str] = set()
        self.history: list[dict] = []
        self.pareto = ParetoFrontier()  # score-vs-train-time frontier (SPEC.md 15)
        self.last: dict | None = None
        self.done = False
        self.done_reason: str | None = None
        self._dup_streak = 0  # SPEC.md 20.2 (stall guard)
        self.stall_patience = _sp  # SPEC.md 31.1 (None = off)
        self._stall_streak = 0  # SPEC.md 31.1: non-improving experiment streak
        # SPEC.md 32.2: opt-in two-stage screening (1.0 = off, pre-v0.18)
        self.screen_frac = _sf
        self.screen_active = self.screen_frac < 1.0
        self._screen_champion: float | None = None  # set at reset (32.2)
        self._screen_baseline_score: float | None = None  # 33.3: reset champion
        # SPEC.md 44.1 (v0.30): k-fold holdout scoring (0 = off, legacy
        # single split exactly — the A1–A33 pins stay green)
        self.kfold = _kf
        # SPEC.md 37.1.3 (v0.23, G3): driver metadata for the canonical
        # recipe (None = env-level, not a driver choice)
        self.policy = policy
        self.target = target
        self.rl_episodes = rl_episodes
        self.run_config: RunConfig | None = None  # built at reset
        # SPEC.md 38.2 (v0.24, T2): the parent run's dir name for a re-run
        # (`fit --from-run`); None = a fresh run
        self.parent_run = parent_run
        self._started = False

    # --- lifecycle ----------------------------------------------------------
    def reset(self) -> dict:
        # a reset starts a fresh episode (R5): clear all per-run state so the
        # env can be re-driven after done (e.g. meta-RL episodes, SPEC.md 15)
        self.done = False
        self.done_reason = None
        self.memory = None
        self.run_dir = None
        self.dataset = None
        self.best_spec = None
        self.best_score = float("-inf")
        self.baseline_score = float("-inf")
        self.best_model = None
        self.seen = set()
        self.history = []
        self.pareto = ParetoFrontier()
        self.last = None
        self.bm.start()
        self._top = []  # SPEC.md 19.3: per-episode frontier tracking
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{self.task_name}-seed{self.seed}-{stamp}"
        run_dir = self.runs_dir / base
        k = 1
        while run_dir.exists():  # never mix two runs into one directory
            k += 1
            run_dir = self.runs_dir / f"{base}-{k}"
        self.memory = RunMemory(run_dir)
        self.run_dir = self.memory.run_dir

        # SPEC.md 37.1.3 (v0.23, G3): the canonical recipe — built at
        # reset (fixed before any experiment) and written so it exists
        # from the first step, even for an interrupted run
        self.run_config = RunConfig.from_env(
            self, policy=self.policy, target=self.target,
            rl_episodes=self.rl_episodes)
        (self.run_dir / "run_config.json").write_text(
            json.dumps(self.run_config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8")

        # one shared train-split dataset (deterministic from the task seed;
        # SPEC.md 20.3: size from the explicit override or the task default)
        self.dataset = self.task.make_dataset(self.dataset_size)

        # baseline: free initialization, not counted against the budget
        result = train(
            self.dataset, DEFAULT_SPEC, self.seed,
            time_limit_seconds=self.bm.train_time_limit(),
            n_out=self.task.n_outputs, head=self.task.head,
        )
        score, std, gen, gen_gap = self._evaluate(result.model)
        eff = self._eff(score, result.train_seconds, gen_gap)
        self.baseline_score = score
        self.best_score = score
        self.baseline_eff = eff
        self.best_eff = eff
        self.best_std = std
        self.best_spec = DEFAULT_SPEC
        self.best_model = result.model
        self.seen.add(DEFAULT_SPEC.fingerprint())
        self.pareto.add(score, result.train_seconds, DEFAULT_SPEC.fingerprint())
        if self.ensemble_top_k > 0:  # SPEC.md 19.3: the baseline is a member
            self._top.append((score, result.model))
        self.memory.log({
            "kind": KIND_BASELINE,  # SPEC.md 35.1 (C4): the kind registry
            "spec": DEFAULT_SPEC.to_dict(),
            "spec_hash": DEFAULT_SPEC.fingerprint(),
            "mutation": None,
            "holdout_score": score,
            "std": std,  # SPEC.md 30.4 (V4): the §18.3 holdout sigma (0.0 legacy)
            "gen_score": gen,
            "gen_gap": gen_gap,
            "train_seconds": result.train_seconds,
            "effective_score": eff,  # SPEC.md 18 (== score in legacy mode)
            "accepted": True,
            "loss_history": result.loss_history,  # SPEC.md 28.1 (C1)
        })
        self._dup_streak = 0  # SPEC.md 20.2: fresh episode, fresh streak
        self._stall_streak = 0  # SPEC.md 31.1: fresh episode, fresh patience
        # SPEC.md 32.2: the screening champion — the baseline spec screened
        # once on the prefix subsample (free, like the reset baseline);
        # K=1 against this fixed champion for the whole episode
        self._screen_champion = None
        self._screen_baseline_score = None
        if self.screen_active:
            sres = train(
                self._subsample(self.dataset), DEFAULT_SPEC, self.seed,
                time_limit_seconds=self.bm.train_time_limit(),
                n_out=self.task.n_outputs, head=self.task.head,
            )
            self._screen_champion = float(self._evaluate(sres.model)[0])
            # SPEC.md 33.3: the summary's baseline_screen_score (32.2) is the
            # reset champion; a curriculum step-up re-pins the champion, and
            # each re-pinned value rides its curriculum event row instead
            self._screen_baseline_score = self._screen_champion
        self._started = True
        return self._state()

    def _check_done(self) -> tuple[bool, str | None]:
        if self.done:
            return True, self.done_reason
        if self.bm.experiments_left <= 0:
            return True, "budget_exhausted"
        if self.bm.wall_exhausted:
            return True, "wall_time_exhausted"
        # SPEC.md 31.1: plateau early stop — K non-improving experiments in
        # a row (checked last: an existing reason on the same step wins)
        if self.stall_patience is not None \
                and self._stall_streak >= self.stall_patience:
            return True, "stalled"
        return False, None

    def step(self, action: dict | ModelSpec) -> tuple[dict, float, bool, dict]:
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError(f"environment is done ({self.done_reason}); call reset()")

        # SPEC.md 25.4: an invalid spec from an external proposer (gym agent,
        # hand-written dict) is a logged rejection, never a crash
        try:
            spec = action if isinstance(action, ModelSpec) else ModelSpec.from_dict(action)
        except SpecError as exc:
            return self._reject_invalid(
                action if isinstance(action, dict) else dict(action), [], str(exc))
        fp = spec.fingerprint()

        # R3: duplicates are rejected without spending budget; reported as a
        # non-improvement so the policy moves on (fail-streak, field switch)
        if fp in self.seen:
            self._dup_streak += 1  # SPEC.md 20.2 (stall guard)
            self.last = {
                "accepted": False,
                "fields": list(spec.diff_fields(self.best_spec)),
            }
            done, reason = self._check_done()
            stalled = (not done) and \
                self._dup_streak >= MAX_CONSECUTIVE_DUPLICATES
            if stalled:
                # a collapsed policy re-proposes seen specs forever; bound
                # the free-duplicate loop (SPEC.md 20.2)
                done, reason = True, "duplicate_stall"
            if done:
                self._finish(reason)
            return (
                self._state(), 0.0, done,
                {"accepted": False,
                 "reason": "duplicate_stall" if stalled else "duplicate",
                 "candidate_score": None},
            )

        self._dup_streak = 0  # SPEC.md 20.2: a fresh proposal resets the streak
        done, reason = self._check_done()
        if done:
            self._finish(reason)
            return (
                self._state(), 0.0, True,
                {"accepted": False, "reason": reason, "candidate_score": None},
            )

        # mutation observability: fields that differ from the current best
        prev_best = self.best_score
        prev_best_eff = self.best_eff
        prev_best_std = self.best_std
        mutation_fields = list(spec.diff_fields(self.best_spec))

        # SPEC.md 25.3: the convnet family trains on the grid dataset; a
        # convnet spec on a flat task raises SpecError here -> SPEC.md 25.4
        try:
            full_dataset = self._dataset_for(spec)
            # SPEC.md 32.2: opt-in two-stage screening — the candidate is
            # first trained on a prefix subsample and scored; only a strict
            # beat of the fixed champion (32.2) spends the full training
            if self.screen_active:
                sres = train(
                    self._subsample(full_dataset), spec, self.seed,
                    time_limit_seconds=self.bm.train_time_limit(),
                    n_out=self.task.n_outputs, head=self.task.head,
                )
                s_score, s_std, s_gen, s_gap = self._evaluate(sres.model)
                if not s_score > self._screen_champion:
                    return self._reject_screened(
                        spec, fp, mutation_fields, sres,
                        s_score, s_std, s_gen, s_gap)
            result = train(
                full_dataset, spec, self.seed,
                time_limit_seconds=self.bm.train_time_limit(),
                n_out=self.task.n_outputs, head=self.task.head,
            )
        except SpecError as exc:
            return self._reject_invalid(
                spec.to_dict(), list(spec.diff_fields(self.best_spec)), str(exc))
        score, std, gen, gen_gap = self._evaluate(result.model)
        eff = self._eff(score, result.train_seconds, gen_gap)

        # SPEC.md 19.3: keep the top-k models by holdout score (deterministic:
        # stable sort, ties keep insertion order)
        if self.ensemble_top_k > 0:
            self._top.append((score, result.model))
            self._top = sorted(self._top, key=lambda m: -m[0])[: self.ensemble_top_k]

        # SPEC.md 18.6: unified acceptance rule —
        # accept ⟺ eff(cand) − eff(best) > max(0, z·SE), SE = √(σ²_cand + σ²_best)
        # (all knobs 0 ⇒ exactly the legacy strict score > prev_best)
        se = math.sqrt(std ** 2 + prev_best_std ** 2)
        accepted = should_accept(eff - prev_best_eff, se, self.z_accept)

        reward = (eff - prev_best_eff) + NOVELTY_BONUS
        if accepted:
            self.best_score = score
            self.best_spec = spec
            self.best_model = result.model
            self.best_eff = eff
            self.best_std = std

        self.bm.spend_experiment()
        self.seen.add(fp)
        self.pareto.add(score, result.train_seconds, fp)  # raw score (SPEC.md 15)
        self.last = {"accepted": accepted, "fields": mutation_fields}
        # SPEC.md 31.1: patience counts scored experiments only — an
        # acceptance (strict improvement) resets, a rejection extends
        if self.stall_patience is not None:
            self._stall_streak = 0 if accepted else self._stall_streak + 1
        self.history.append({
            "spec_hash": fp,
            "score": score,
            "delta": eff - prev_best_eff,
        })
        self.memory.log({
            "kind": KIND_EXPERIMENT,  # SPEC.md 35.1 (C4)
            "spec": spec.to_dict(),
            "spec_hash": fp,
            "mutation": mutation_fields,
            "holdout_score": score,
            "std": std,  # SPEC.md 30.4 (V4): the §18.3 holdout sigma (0.0 legacy)
            "gen_score": gen,
            "gen_gap": gen_gap,
            "train_seconds": result.train_seconds,
            "time_capped": result.time_capped,
            "effective_score": eff,  # SPEC.md 18 (== score in legacy mode)
            "accepted": accepted,
            "loss_history": result.loss_history,  # SPEC.md 28.1 (C1)
        })

        # SPEC.md 20.1: curriculum — when the task saturates, step the
        # difficulty up (rebuild task + dataset, re-baseline the best spec)
        if self.curriculum is not None and \
                self.curriculum.step_up_if_saturated(self.best_score):
            self._advance_curriculum()

        done, reason = self._check_done()
        if done:
            self._finish(reason)
        info = {
            "accepted": accepted,
            "reason": None if not done else reason,
            "candidate_score": score,
            "gen_gap": gen_gap,
            "experiments_left": self.bm.experiments_left,
            "train_seconds": result.train_seconds,
            "time_capped": result.time_capped,
            # SPEC.md 18 (legacy mode: effective_score == candidate_score, se == 0)
            "effective_score": eff,
            "se": se,
            "loss_history": result.loss_history,  # SPEC.md 28.1 (C1)
        }
        return self._state(), reward, done, info

    # --- curriculum (SPEC.md 20.1) --------------------------------------------
    def _advance_curriculum(self) -> None:
        """The task saturated: rebuild it at the next difficulty level (same
        seed), re-baseline the current best spec there (free, like the reset
        baseline), and reset the difficulty-scoped state (dedup, Pareto,
        ensemble top-k) while keeping the full experiment log and the first
        baseline (v0.4 summary contract)."""
        assert self.curriculum is not None and self.best_spec is not None
        self._stall_streak = 0  # SPEC.md 31.1: new level = new ceiling
        best_before = self.best_score
        self.task = self.curriculum.task()
        self.task_name = self.task.name
        self.dataset = self.task.make_dataset(self.dataset_size)
        result = train(
            self.dataset, self.best_spec, self.seed,
            time_limit_seconds=self.bm.train_time_limit(),
            n_out=self.task.n_outputs, head=self.task.head,
        )
        score, std, gen, gen_gap = self._evaluate(result.model)
        eff = self._eff(score, result.train_seconds, gen_gap)
        # carry the best spec over as the new level's champion; the improver
        # re-explores specs against the new difficulty (seen/pareto reset)
        self.best_score = score
        self.best_eff = eff
        self.best_std = std
        self.best_model = result.model
        self.seen = {self.best_spec.fingerprint()}
        self.pareto = ParetoFrontier()
        self.pareto.add(score, result.train_seconds,
                        self.best_spec.fingerprint())
        if self.ensemble_top_k > 0:
            self._top = [(score, result.model)]
        # SPEC.md 33.3 (C3, screening × curriculum): a step-up is a new
        # difficulty, so the screening champion re-pins — the baseline spec
        # is re-screened on the new dataset (free, like the reset champion;
        # 32.2's K=1 semantics are per curriculum level). Without this, a
        # harder level would reject every candidate against a stale
        # easy-level champion.
        if self.screen_active:
            sc = train(
                self._subsample(self.dataset), DEFAULT_SPEC, self.seed,
                time_limit_seconds=self.bm.train_time_limit(),
                n_out=self.task.n_outputs, head=self.task.head,
            )
            self._screen_champion = float(self._evaluate(sc.model)[0])
        event = {
            "level": self.curriculum.level,
            "difficulty": self.curriculum.level_description(),
            # SPEC.md 46.2.2: the level's task parameters — splatted via the
            # curriculum's level_params() (parity keeps its historical
            # n_bits/p_flip keys byte-identical; sine rows carry
            # noise/freq_scale/amplitude, cartpole rows ic_scale)
            **self.curriculum.level_params(),
            "ceiling": self.curriculum.ceiling,
            "best_before_step": best_before,
            "new_baseline_score": score,
            "std": std,  # SPEC.md 30.4 (V4): the §18.3 holdout sigma (0.0 legacy)
            "loss_history": result.loss_history,  # SPEC.md 28.1 (C1)
        }
        if self.screen_active:  # SPEC.md 33.3: additive key (A10 rows unchanged)
            event["screen_champion"] = self._screen_champion
        self.curriculum_events.append(event)
        self.memory.log({"kind": KIND_CURRICULUM, **event})  # 35.1 (C4)

    # --- datasets (SPEC.md 25.3) ---------------------------------------------
    def _dataset_for(self, spec: ModelSpec) -> Any:
        """The train-split dataset a spec family trains on.

        convnet trains on the grid view of a grid_capable task's dataset
        (SPEC.md 25.3); every other family trains on the flat view
        (unchanged). convnet on a non-grid task is a clean SpecError
        (SPEC.md 25.3/25.4) — never a silent fallback."""
        if spec.model_family == "convnet":
            if not getattr(self.task, "grid_capable", False):
                raise SpecError(
                    f"convnet family needs a grid_capable task; task "
                    f"{self.task_name!r} is flat (SPEC.md 25.3/25.4)")
            grid_dataset = getattr(self.task, "grid_dataset", None)
            if grid_dataset is None:
                raise SpecError(
                    f"task {self.task_name!r} is grid_capable but exposes no "
                    "grid_dataset() (SPEC.md 25.3)")
            return grid_dataset(self.dataset_size)
        return self.dataset

    # --- two-stage screening (SPEC.md 32.2) ---------------------------------
    def _subsample(self, dataset: Any) -> Any:
        """SPEC.md 32.2: the first max(1, floor(frac·n)) rows of the (X, y)
        train-split dataset — a prefix slice, no new RNG (G2)."""
        x, y = dataset
        m = max(1, int(math.floor(len(x) * self.screen_frac)))
        return x[:m], y[:m]

    def _reject_screened(self, spec: ModelSpec, fp: str, fields: list[str],
                         sres, s_score: float, s_std: float, s_gen: float,
                         s_gap: float) -> tuple[dict, float, bool, dict]:
        """SPEC.md 32.2: a screened candidate that does not strictly beat
        the fixed champion (the baseline screened at reset) is rejected
        without the full training. One budget experiment is spent (it was
        scored and trained), the fingerprint joins `seen` (R3 still
        applies), and it counts for the v0.17 stall patience (SPEC.md
        31.1: a scored non-improving experiment). Pareto / ensemble /
        curriculum are untouched (32.2)."""
        self.bm.spend_experiment()
        self.seen.add(fp)
        self.last = {"accepted": False, "fields": list(fields)}
        if self.stall_patience is not None:
            self._stall_streak += 1
        self.history.append({
            "spec_hash": fp,
            "score": s_score,
            "delta": s_score - self.best_score,
        })
        self.memory.log({
            "kind": KIND_SCREEN,  # SPEC.md 35.1 (C4)
            "spec": spec.to_dict(),
            "spec_hash": fp,
            "mutation": list(fields),
            "holdout_score": s_score,
            "std": s_std,  # SPEC.md 30.4 (V4): holdout sigma (0.0 legacy)
            "gen_score": s_gen,
            "gen_gap": s_gap,
            "train_seconds": sres.train_seconds,
            "time_capped": sres.time_capped,
            "effective_score": s_score,  # legacy mode: == screen score
            "accepted": False,
            "screen_rejected": True,
            "loss_history": sres.loss_history,  # SPEC.md 28.1 (C1)
        })
        done, reason = self._check_done()
        if done:
            self._finish(reason)
        info = {
            "accepted": False,
            "reason": "screen_rejected" if not done else reason,
            "candidate_score": s_score,
            "gen_gap": s_gap,
            "experiments_left": self.bm.experiments_left,
            "train_seconds": sres.train_seconds,
            "time_capped": sres.time_capped,
            "effective_score": s_score,
            "se": 0.0,
            "loss_history": sres.loss_history,  # SPEC.md 28.1 (C1)
            "screen_rejected": True,
        }
        return self._state(), 0.0, done, info

    # --- invalid-spec rejection (SPEC.md 25.4) --------------------------------
    def _reject_invalid(self, spec_dict: dict, fields: list[str],
                        error: str) -> tuple[dict, float, bool, dict]:
        """SPEC.md 25.4: an invalid spec (e.g. convnet on a flat task, or a
        spec an external gym agent proposed) is rejected without crashing:
        no budget is spent, it is not added to the dedup set (it may still
        be valid on another task), and it is logged in the experiment log."""
        self.last = {"accepted": False, "fields": list(fields)}
        if self.memory is not None:
            self.memory.log({
                "kind": KIND_INVALID_SPEC,  # SPEC.md 35.1 (C4)
                "spec": spec_dict,
                "error": str(error),
                "accepted": False,
                "reason": "invalid_spec",
            })
        done, _reason = self._check_done()
        if done:
            self._finish(_reason)
        return (
            self._state(), 0.0, done,
            {"accepted": False, "reason": "invalid_spec", "candidate_score": None},
        )

    # --- scoring (SPEC.md 18.3/18.4) -----------------------------------------
    def _evaluate(self, model) -> tuple[float, float, float, float]:
        """Score `model` on holdout + gen. Returns (score, std, gen, gen_gap).

        SPEC.md 44.1 (v0.30): with kfold > 0 (checked first) both splits are
        scored over K distinct held-out subsets — the score is the fold mean
        and the std the fold sample std (44.1.2/44.1.3). SPEC.md 18.3:
        with ci_blocks > 0 the scores are block means (each with a sample
        std over the blocks); otherwise the legacy single 200-point score
        with std 0. gen_gap keeps its existing definition (score points)."""
        if self.kfold > 0:
            h = kfold_score(self.task, model, "holdout", k=self.kfold)
            g = kfold_score(self.task, model, "gen", k=self.kfold)
            return h["score"], h["std"], g["score"], h["score"] - g["score"]
        if self.ci_blocks > 0:
            h = score_with_ci(self.task, model, "holdout",
                              n_blocks=self.ci_blocks, block_size=self.block_size)
            g = score_with_ci(self.task, model, "gen",
                              n_blocks=self.ci_blocks, block_size=self.block_size)
            return h["mean"], h["std"], g["mean"], h["mean"] - g["mean"]
        ev = evaluate_full(self.task, model)
        return ev["score"], 0.0, ev["gen_score"], ev["gen_gap"]

    def _eff(self, score: float, train_seconds: float, gen_gap: float) -> float:
        """SPEC.md 18.4/18.5 with this env's configured weights."""
        return effective_score(
            score, train_seconds, gen_gap,
            self.efficiency_weight, self.gen_gap_penalty,
        )

    def _finish(self, reason: str | None) -> None:
        self.done = True
        self.done_reason = reason
        if self.memory is not None and self.best_spec is not None:
            self._write_artifacts(reason)

    def _write_artifacts(self, reason: str | None) -> None:
        assert self.memory is not None
        self.memory.save_best(self.best_spec.to_dict(), self.best_model)
        # per-mutation-type win rate (each field of a mutation counts a trial)
        win_rate: dict[str, dict] = {}
        for entry in self.memory.load_experiments():
            if entry.get("kind") != KIND_EXPERIMENT:  # 35.1 (C4)
                continue
            for fld in entry.get("mutation") or []:
                slot = win_rate.setdefault(fld, {"trials": 0, "wins": 0})
                slot["trials"] += 1
                if entry.get("accepted"):
                    slot["wins"] += 1
        summary = {
            "task": self.task_name,
            "seed": self.seed,
            # SPEC.md 37.1.3 (v0.23, G3): the canonical recipe — the same
            # frozen object written at reset (additive key; 34.2 17 → 18)
            "run_config": self.run_config.to_dict() if self.run_config else None,
            "finished_reason": reason,
            "baseline_score": self.baseline_score,
            "final_best_score": self.best_score,
            "improvement_factor": (
                self.best_score / self.baseline_score if self.baseline_score > 0 else None
            ),
            # SPEC.md 18.4/18.6: effective-score counterparts + active preset
            "baseline_effective": self.baseline_eff,
            "final_best_effective": self.best_eff,
            "effective_improvement_factor": (
                self.best_eff / self.baseline_eff if self.baseline_eff > 0 else None
            ),
            "search_quality": {
                "ci_blocks": self.ci_blocks,
                "z_accept": self.z_accept,
                "efficiency_weight": self.efficiency_weight,
                "gen_gap_penalty": self.gen_gap_penalty,
                "block_size": self.block_size,
            },
            "experiments_run": self.bm.used_experiments,
            # computed once and shared with the 38.1 registry entry (the
            # wall clock keeps advancing; two computations could round
            # differently)
            "wall_seconds": (_wall := round(self.bm.budget.max_wall_seconds
                                            - self.bm.wall_seconds_left, 3)),
            "best_spec": self.best_spec.to_dict(),
            "mutation_win_rate": win_rate,
        }
        # SPEC.md 22.1: data-driven task config (absent for built-in tasks)
        if self.task_config:
            summary["task_config"] = self.task_config
        summary.update(self.pareto.summary())  # SPEC.md 15: efficiency, not just score
        # SPEC.md 20.1: curriculum ladder history (only when a curriculum ran)
        if self.curriculum is not None:
            summary["curriculum"] = {
                "levels": self.curriculum_events,
                "final_difficulty": self.curriculum.level_description(),
                "final_ceiling": self.curriculum.ceiling,
                "levels_left": self.curriculum.levels_left(),
            }
        # SPEC.md 32.2: two-stage screening config (only when active)
        if self.screen_active:
            # SPEC.md 33.3: the reset champion (curriculum step-ups re-pin
            # self._screen_champion; per-level values are in the event rows)
            summary["screening"] = {
                "frac": self.screen_frac,
                "baseline_screen_score": self._screen_baseline_score,
            }
        # SPEC.md 19.3: opt-in ensemble final evaluation (top-k frontier models)
        if self.ensemble_top_k > 0 and len(self._top) >= 1:
            members = [m for _s, m in self._top]
            ens_score = float(
                self.task.score(_EnsembleModel(members), "holdout", ENSEMBLE_EVAL_POINTS)
            )
            summary["ensemble"] = {
                "top_k": len(members),
                "member_scores": [float(s) for s, _m in self._top],
                "ensemble_score": ens_score,
            }
        # SPEC.md 38.2 (v0.24, T2): lineage — additive *conditional* key,
        # absent for a fresh run (the A24 key sets stay untouched)
        if self.parent_run is not None:
            summary["parent_run"] = self.parent_run
        # SPEC.md 44.1.4 (v0.30): k-fold config — additive *conditional*
        # key, absent when kfold = 0 (the A24 key set + the pinned 5-key
        # search_quality dict stay untouched)
        if self.kfold > 0:
            summary["kfold"] = {"k": self.kfold}
        self.memory.save_summary(summary)
        # SPEC.md 38.1 (v0.24, T1): the run registry — one appended entry
        # per finished run (recovery-safe, 38.1.4); the same wall seconds
        # as the summary above
        from ..registry import append_entry, entry_from_env
        append_entry(self.runs_dir,
                     entry_from_env(self, reason, wall_seconds=_wall))

    # --- state ---------------------------------------------------------------
    def _state(self) -> dict:
        digest = [
            (h["spec_hash"][:8], round(h["score"], 4), round(h["delta"], 4))
            for h in self.history[-HISTORY_DIGEST_LEN:]
        ]
        state = {
            "task": self.task_name,  # SPEC.md 20.1/20.2 (active task, may be a
            # curriculum level after a step-up)
            "best_score": self.best_score,
            "best_spec": self.best_spec.to_dict() if self.best_spec else None,
            "baseline_score": self.baseline_score,
            "experiments_left": self.bm.experiments_left,
            "seconds_left": round(self.bm.wall_seconds_left, 3),
            "history_digest": digest,
            "pareto_frontier": self.pareto.frontier(),  # SPEC.md 15
            "best_efficiency": self.pareto.best_score_at(1.0),
            "last": self.last,
            "done": self.done,
        }
        if self.curriculum is not None:  # SPEC.md 20.1
            state["curriculum"] = {
                "level": self.curriculum.level,
                "difficulty": self.curriculum.level_description(),
                "ceiling": self.curriculum.ceiling,
                "levels_left": self.curriculum.levels_left(),
            }
        return state
