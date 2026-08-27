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

import math
import time
from pathlib import Path
from typing import Any

from ..budget import BudgetManager
from ..config import DEFAULT_SPEC, Budget, ModelSpec, SpecError
from ..evaluator import evaluate_full, score_with_ci
from ..memory import RunMemory
from ..pareto import ParetoFrontier
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


def search_quality_v04() -> dict:
    """SPEC.md 18.6: the recommended v0.4 preset as AutoRefineEnv kwargs
    (ci_blocks/z/η/P_GEN per §18.3–§18.5; block_size stays the default)."""
    return dict(
        ci_blocks=8,             # 18.3: 8 blocks × block_size per split
        z_accept=1.0,            # 18.6: ~95% two-sided, deliberately not 1.96
        efficiency_weight=0.5,   # 18.4: η points per training second (T_CAP 10s)
        gen_gap_penalty=0.5,     # 18.5: P_GEN points per excess gap point
    )


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
    ) -> None:
        if task not in TASKS:  # registry: SPEC.md 15 "more tasks"
            raise ValueError(f"unknown task {task!r} (available: {sorted(TASKS)})")
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
            "kind": "baseline",
            "spec": DEFAULT_SPEC.to_dict(),
            "spec_hash": DEFAULT_SPEC.fingerprint(),
            "mutation": None,
            "holdout_score": score,
            "gen_score": gen,
            "gen_gap": gen_gap,
            "train_seconds": result.train_seconds,
            "effective_score": eff,  # SPEC.md 18 (== score in legacy mode)
            "accepted": True,
        })
        self._dup_streak = 0  # SPEC.md 20.2: fresh episode, fresh streak
        self._started = True
        return self._state()

    def _check_done(self) -> tuple[bool, str | None]:
        if self.done:
            return True, self.done_reason
        if self.bm.experiments_left <= 0:
            return True, "budget_exhausted"
        if self.bm.wall_exhausted:
            return True, "wall_time_exhausted"
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
            result = train(
                self._dataset_for(spec), spec, self.seed,
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
        self.history.append({
            "spec_hash": fp,
            "score": score,
            "delta": eff - prev_best_eff,
        })
        self.memory.log({
            "kind": "experiment",
            "spec": spec.to_dict(),
            "spec_hash": fp,
            "mutation": mutation_fields,
            "holdout_score": score,
            "gen_score": gen,
            "gen_gap": gen_gap,
            "train_seconds": result.train_seconds,
            "time_capped": result.time_capped,
            "effective_score": eff,  # SPEC.md 18 (== score in legacy mode)
            "accepted": accepted,
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
        event = {
            "level": self.curriculum.level,
            "difficulty": self.curriculum.level_description(),
            "n_bits": self.curriculum.n_bits,
            "p_flip": self.curriculum.p_flip,
            "ceiling": self.curriculum.ceiling,
            "best_before_step": best_before,
            "new_baseline_score": score,
        }
        self.curriculum_events.append(event)
        self.memory.log({"kind": "curriculum", **event})

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
                "kind": "invalid_spec",
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

        SPEC.md 18.3: with ci_blocks > 0 the scores are block means (each with
        a sample std over the blocks); otherwise the legacy single 200-point
        score with std 0. gen_gap keeps its existing definition (score points)."""
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
            if entry.get("kind") != "experiment":
                continue
            for fld in entry.get("mutation") or []:
                slot = win_rate.setdefault(fld, {"trials": 0, "wins": 0})
                slot["trials"] += 1
                if entry.get("accepted"):
                    slot["wins"] += 1
        summary = {
            "task": self.task_name,
            "seed": self.seed,
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
            "wall_seconds": round(self.bm.budget.max_wall_seconds - self.bm.wall_seconds_left, 3),
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
        self.memory.save_summary(summary)

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
