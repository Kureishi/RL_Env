# AutoRefine — Spec for an Autonomous Iterative Model-Improvement Environment

**Version:** 0.1 (draft for review)
**Date:** 2026-08-23
**Status:** Awaiting approval before implementation

---

## 0. Acceptance Index (v0.21, SPEC.md 35.2)

The single source of truth for milestone → SPEC section → acceptance
number → test file (C5, 35.2). Contract A25: the suite verifies that
every A# defined in this document is cited by ≥ 1 test file, that
every `tests/test_*.py` cites an A#, and that each row's listed test
file exists and cites that row's A#. (M0–M3 predate acceptance
numbers and carry none.)

| M#  | version | SPEC § | A#  | test file(s) |
|-----|---------|--------|-----|--------------|
| M0  | —       | 1–4    | —   | — (scaffold) |
| M1  | —       | 5–8    | —   | — |
| M2  | —       | 6      | —   | — |
| M3  | —       | 11     | —   | — |
| M4  | v0.1    | 12     | A1–A5 | tests/test_meta_env.py |
| M5  | v0.2    | 15     | A6  | tests/test_gym.py, tests/test_meta_env_ext.py, tests/test_pareto.py, tests/test_rl_policy.py |
| M6  | v0.3    | 17     | A7  | tests/test_bandit.py, tests/test_tasks_ext.py |
| M7  | v0.4    | 18     | A8  | tests/test_search_quality.py |
| M8  | v0.5    | 19     | A9  | tests/test_model_spec_space.py, tests/test_trainer.py |
| M9  | v0.6    | 20     | A10 | tests/test_env_loop_design.py |
| M10 | v0.7    | 21     | A11 | tests/test_gym_dqn.py, tests/test_infra_dx.py |
| M11 | v0.8    | 22     | A12 | tests/test_data_cli.py |
| M12 | v0.9    | 23     | A13 | tests/test_dashboard.py |
| M13 | v0.10   | 24     | A14 | tests/test_media_tasks.py |
| M14 | v0.11   | 25     | A15 | tests/test_modality_families.py |
| M15 | v0.12   | 26     | A16 | tests/test_dashboard.py |
| M16 | v0.13   | 27     | A17 | tests/test_dashboard.py |
| M17 | v0.14   | 28     | A18 | tests/test_learning_views.py |
| M18 | v0.15   | 29     | A19 | tests/test_multi_run_policy_views.py |
| M19 | v0.16   | 30     | A20 | tests/test_comprehension_visuals2.py |
| M20 | v0.17   | 31     | A21 | tests/test_optimization_v017.py |
| M21 | v0.18   | 32     | A22 | tests/test_app_perf_screening_v018.py |
| M22 | v0.19   | 33     | A23 | tests/test_coherency_v019.py |
| M23 | v0.20   | 34     | A24 | tests/test_coherency_v020.py |
| M24 | v0.21   | 35     | A25 | tests/test_coherency_v021.py |

---

## 1. Purpose

AutoRefine is a local, reproducible reinforcement-learning-style environment in which an
*improver agent* iteratively improves an ML model through a closed autonomous loop:

```
propose change → train → evaluate → record → (learn from outcome) → propose next change
```

Two levels of "agent" are supported:

1. **Inner model** — a small ML model (v1: a numpy MLP) that is trained on a concrete task.
2. **Improver (outer agent)** — the autonomous loop that decides *how to change* the model
   (hyperparameters, architecture, optimizer, data treatment, training length).

The environment exposes the outer loop as a standard `reset()/step()` Gym-style interface,
so any policy (heuristic search in v1, an RL policy later) can drive it.

## 2. Goals

- **G1.** Fully autonomous improvement loop: given a task + budget, it runs unattended and
  returns the best validated model spec + checkpoint.
- **G2.** Reproducibility: identical seed ⇒ identical sequence of experiments and final scores.
- **G3.** Honest measurement: scoring uses a holdout split the improver never sees, with a
  separate "generalization seed" to catch val-set overfitting.
- **G4.** Bounded resources: hard caps on experiment count, wall time, and per-training time;
  no network access, no shell, no arbitrary code execution.
- **G5.** Extensibility: new tasks and new mutation operators are added by implementing a
  small protocol, with zero changes to the core loop.
- **G6.** Inspectability: every experiment is logged (JSONL) with the full spec, config,
  scores, and the mutation that produced it.

## 3. Non-Goals (v1)

- No distributed/parallel training.
- No deep frameworks (PyTorch/TensorFlow). Core uses **NumPy only**; Gymnasium is an
  *optional* adapter, not a dependency.
- No online learning or model merging.
- No GUI *in the core* (CLI + Python API only). The one visual surface is
  the **optional** Streamlit dashboard (v0.9, §23): an extra package the core
  never imports, launched as a separate process — the same optional-dependency
  pattern as the gymnasium adapter (§21.1).
- No RL-based improver in v1 (the *interface* is ready for one; v1 ships a
  hill-climbing/evolutionary search policy).

## 4. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                      AutoRefineEnv (meta env)                  │
│  state: best spec, budget left, history digest                 │
│  step(action) → (new_state, reward=Δscore, done, info)         │
└───────────────▲────────────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        │   Improver      │  policy.py: search policy (v1: hill-climb +
        │                 │  evolutionary restarts); actions.py: mutation ops
        └───────┬────────┘
                │ proposes candidate ModelSpec
        ┌───────▼────────┐
        │    Trainer      │  deterministic, seeded training of numpy MLP
        └───────┬────────┘
                │
        ┌───────▼────────┐      ┌──────────────┐
        │   Evaluator     │─────▶│  TaskEnv     │  cartpole-v1 (v1 task)
        │ holdout scoring │      │ seeded data  │  + extensible protocol
        └───────┬────────┘      └──────────────┘
                │
        ┌───────▼────────┐
        │ Memory / Budget │  JSONL experiment log, best-spec checkpoint,
        └─────────────────┘  budget manager (experiments, time)
```

### Directory layout (planned)

```
RL_Env/
├── SPEC.md                     ← this document
├── README.md
├── pyproject.toml              # package "autorefine", deps: numpy
├── src/autorefine/
│   ├── __init__.py
│   ├── config.py               # ModelSpec, TaskConfig, Budget (dataclasses, JSON-serializable)
│   ├── tasks/
│   │   ├── base.py             # TaskEnv protocol
│   │   └── cartpole.py         # CartPoleV1
│   ├── models/
│   │   ├── mlp.py              # NumPy MLP (forward/backward, Adam/momentum/SGD)
│   │   └── optimizers.py
│   ├── trainer.py              # deterministic training loop
│   ├── evaluator.py            # holdout + generalization scoring
│   ├── improver/
│   │   ├── actions.py          # mutation operators
│   │   ├── policy.py           # v1 search policy
│   │   └── meta_env.py         # AutoRefineEnv (reset/step)
│   ├── memory.py               # ExperimentLog (JSONL), best-spec store
│   ├── budget.py               # BudgetManager
│   └── cli.py                  # python -m autorefine ...
└── tests/
    ├── test_tasks.py
    ├── test_trainer.py
    └── test_meta_env.py
```

## 5. Core Concepts & Data Model

### 5.1 `ModelSpec` (the object the improver mutates)

A plain, JSON-serializable dataclass. **This is the unit of improvement.**

| Field            | Type            | v1 allowed values                                   | Notes |
|------------------|-----------------|-----------------------------------------------------|-------|
| `architecture`   | `dict`          | hidden layer sizes `[8..128]`, depth `1..3`        | MLP only in v1 |
| `optimizer`      | `str`           | `sgd`, `momentum`, `adam`                          | |
| `learning_rate`  | `float`         | `1e-4 .. 1e-1` (log-uniform sampling)               | |
| `batch_size`     | `int`           | `16, 32, 64, 128`                                   | |
| `weight_decay`   | `float`         | `0 .. 1e-2`                                         | |
| `train_steps`    | `int`           | `200 .. 5000`                                       | capped by per-training time budget |
| `input_noise`    | `float`         | `0 .. 0.1` (Gaussian data augmentation)             | |
| `activation`     | `str`           | `tanh`, `relu`                                      | |

Constraints are enforced in `ModelSpec.validate()`; invalid mutations are rejected
by the improver (not silently clipped) so the policy learns the boundary.

### 5.2 `Budget`

```
Budget(max_experiments: int,       # hard cap on trainer invocations
       max_wall_seconds: float,    # total loop time
       max_train_seconds: float)   # per-training time cap
```

The `BudgetManager` enforces all three; the loop terminates with
`done=True, info={"reason": "budget_exhausted"}` when any is hit.

### 5.3 Scoring protocol (G3 — honesty by construction)

Each task defines three disjoint, seed-derived splits:

1. **train split** — used only by the trainer.
2. **val split** — used by the trainer for its internal early-stopping metric
   (visible to the improver).
3. **holdout split** — used **only** by the `Evaluator` for the official score.
   The improver sees the *score number*, never the data.

**Score definition (CartPoleV1):** mean episode survival steps over 200 seeded
episodes on the holdout split (higher is better). A secondary metric
`gen_score` re-runs the same 200 episodes under a different seed set;
`gen_gap = score - gen_score` is logged to detect val overfitting.

## 6. Meta Environment Contract (the actual deliverable interface)

`AutoRefineEnv` is the object ML models / policies act against.

```python
env = AutoRefineEnv(task="cartpole-v1", seed=7,
                    budget=Budget(30, 900, 30))

state = env.reset()
# state: {
#   "best_score": float,          # best holdout score so far (init = baseline spec)
#   "best_spec": ModelSpec,
#   "experiments_left": int,
#   "seconds_left": float,
#   "history_digest": [ ... ]     # last 8 (spec_hash, score, Δ) tuples
# }

state, reward, done, info = env.step(action)
# action: ModelSpec (a proposed candidate)  — dict or dataclass
# reward: candidate_score - best_score      (can be negative)
#         + 1e-3 * novelty_bonus            (small bonus for untried regions)
# info:   {"candidate_score": ..., "gen_gap": ..., "experiments_left": ...,
#          "accepted": bool, "reason": ...}
```

Rules:

- **R1.** `step` always runs a full train→evaluate cycle; there is no partial state.
- **R2.** The best spec is only updated when `candidate_score > best_score` (strict).
- **R3.** Identical spec hashes are rejected (`info={"accepted": False,
      "reason": "duplicate"}`) without spending a training budget — dedup is free.
- **R4.** `step` must honor the per-training time cap by stepping the trainer in
  chunks and aborting cleanly.
- **R5.** After `done=True`, further `step` calls raise `RuntimeError`.

## 7. Inner Task v1: `CartPoleV1`

A deterministic, fast, classic task (NumPy only):

- **Dynamics:** standard cart-pole ODE with linearized pole, integrated with
  fixed-step semi-implicit Euler (`dt=0.02`). Discrete action `{-1, +1}` (left/right
  force). Episode ends when |pole angle| > 0.25 rad or |cart x| > 2.4 m; max 500 steps.
- **State:** `(x, x_dot, theta, theta_dot)` → 4 floats, clipped to `[-3, 3]`
  before the model sees them (envelope of the training data).
- **Model role:** policy network mapping state → 2 logits; argmax = action.
  Training data: on-trajectory rollouts of a **reference linear controller**
  (train split) **plus random states across the reachable state box, labeled by
  the reference controller** — dense behavior-cloning so the policy is well
  defined everywhere it can drift, not just on the reference manifold.
  (Keeps training deterministic and cheap while still requiring real
  optimization.)
- **Seeding:** `task.reset(seed)` derives all split/episode seeds via a single
  `np.random.Generator` chain (PCG64) — one seed pins everything.
- **Difficulty knobs** (for future tasks / v2): `pole_mass`, `cart_mass`, `dt`
  are exposed in `TaskConfig` but fixed in v1.

**Baseline expectation:** the default `ModelSpec` should score roughly 150–350
mean steps; a well-tuned spec should exceed 400 on 200-episode holdout. These are
targets for the acceptance test, not hard requirements (verified empirically at M2
and pinned in `tests/test_meta_env.py` as soft bounds).

## 8. Improver (v1 policy)

Search over `ModelSpec` space with three mechanisms, all deterministic given a seed:

1. **Hill-climbing:** mutate the current best spec (one random field, sampled from
   the field's allowed distribution); keep if it improves.
2. **Evolutionary restarts:** every `K=5` non-improving steps, spawn a candidate
   from a uniform random point in spec space.
3. **Local refinement:** after a successful improvement, 2 extra mutations in the
   same field neighborhood before switching fields.

This is deliberately *policy-agnostic*: `policy.py` defines
`class SearchPolicy: def propose(self, env_state, rng) -> ModelSpec`, so an RL
agent or Bayesian optimizer can be dropped in later without touching the loop.

## 9. Memory & Artifacts

- `runs/<run_id>/experiments.jsonl` — one line per experiment:
  `{ts, spec, spec_hash, mutation, train_score, holdout_score, gen_gap,
    train_seconds, accepted}`.
- `runs/<run_id>/best_spec.json` — final best spec (JSON).
- `runs/<run_id>/best_model.npz` — numpy weights of the best model (loadable
  without the package: `np.load(..., allow_pickle=False)` + documented shape map).
- `runs/<run_id>/summary.json` — final summary (initial vs final score, budget
  used, per-mutation-type win rate).

## 10. Safety / Sandbox

- **S1.** Stdlib + NumPy only. No `subprocess`, no network, no dynamic code
  eval. Sole exception (v0.9, §23.4): the `dashboard` launcher spawns
  `streamlit run` on the fixed in-repo app file — never on user code.
- **S2.** All wall-time bounds enforced in the trainer loop (check every 256 steps).
- **S3.** All RNG streams derived from a single user seed; no `os.urandom`, no
  wall-clock in scoring paths.
- **S4.** File writes confined to the run directory (CLI flag `--runs-dir`).

## 11. CLI

```
python -m autorefine run   --task cartpole-v1 --seed 7 --experiments 30 \
                           --max-seconds 900 --runs-dir runs/
python -m autorefine report --run runs/<run_id>     # human-readable summary
python -m autorefine eval  --run runs/<run_id> --spec path.json   # rescore a spec
```

## 12. Testing & Acceptance Criteria

Unit tests (pytest):

- **T1.** `CartPoleV1.reset(seed)` is deterministic: same seed ⇒ identical first
  1000 transitions (bit-exact floats).
- **T2.** Trainer is deterministic: same spec+seed ⇒ identical final weights.
- **T3.** Invalid `ModelSpec` values raise `ValueError` (boundary cases both sides).
- **T4.** Dedup: submitting the best spec again returns `reason="duplicate"`,
  consumes no budget.
- **T5.** Budget: `max_experiments=1` ⇒ loop ends after one `step`, `done=True`.
- **T6.** Time cap: a deliberately slow spec (huge `train_steps`, tiny machine)
  aborts cleanly and returns a partial result — no hang, no exception.

Integration / acceptance:

- **A1.** **Improvement property:** with default budget (30 experiments),
  the final best score ≥ baseline score × 1.25, averaged over seeds {1, 2, 3},
  with no single seed regressing below baseline. (Verified in `test_meta_env.py`.)
- **A2.** **Reproducibility:** two full runs with seed 7 produce byte-identical
  `best_spec.json` and `best_model.npz`, and identical `experiments.jsonl`
  (modulo the wall-clock fields `ts` and `train_seconds`, which are
  inherently non-deterministic measurements).
- **A3.** **Generalization:** `gen_gap < 5%` of holdout score for the final best
  spec (guards against val-overfitting).
- **A4.** **Budget discipline:** total wall time of a 900s-budget run ≤ 900s
  (+ small overhead), and `max_experiments` is never exceeded.
- **A5.** Docs: README with 10-line quickstart; SPEC.md stays the source of truth.
- **A6.** **Extension coverage (v0.2):** the new tasks, model families, RL
  improver, and Pareto memory are each exercised by dedicated tests
  (`test_tasks_ext.py`, `test_trainer.py` family tests, `test_rl_policy.py`,
  `test_pareto.py`, `test_gym.py`), and a full-budget run on any registered
  task never regresses below its baseline and writes `pareto_frontier`,
  `best_score_at_1s`, `efficiency_at_1s` into `summary.json`.
- **A7.** **Extension coverage (v0.3):** the `parity-v1` task, the UCB
  field-bandit improver, and the `label_smoothing` spec field are each
  exercised by dedicated tests (`test_tasks_ext.py` parity tests,
  `test_bandit.py`, `test_trainer.py` smoothing tests); a bandit-driven
  run on `parity-v1` finishes within budget and never regresses below its
  baseline; pre-v0.3 spec JSON without `label_smoothing` still loads.
- **A8.** **Search quality (v0.4, §18):** local mutation,
  family-conditioned proposals, block-bootstrap CI acceptance, efficiency-aware
  scoring, and the gen-gap penalty are each exercised by dedicated tests;
  a `bandit(mode="local")` run under the v0.4 preset on `parity-v1` /
  `cartpole-v1` never regresses below its raw baseline and is
  reproducible; legacy mode reproduces the v0.3 acceptance sequence
  bit-exactly, and A1–A7 still pass unchanged.
- **A9.** **Model & spec space (v0.5, §19):** the four new mlp fields
  (`lr_schedule`, `early_stopping_patience`, `init_scale`,
  `gradient_clipping`) are range-validated and back-compatible (pre-v0.5
  JSON loads at legacy defaults), and each has a measurable effect on
  training; the `boost` family trains deterministically, round-trips its
  npz checkpoint, and beats bagged `tree` on `parity-v1` at equal rounds and
  depth; an `ensemble_top_k=2` run records a finite `ensemble_score` in
  `summary.json` while the default (0) keeps v0.4 behavior unchanged; the
  legacy pin (§18.7) is re-pinned to the v0.5 proposal stream (the spec
  space grew), and A1–A8 still pass unchanged.
- **A10.** **Environment & loop design (v0.6, §20):** every task exposes a
  `default_dataset_size` (cartpole/gridnav keep 60; sine 2048; parity 4096),
  `AutoRefineEnv` falls back to it (an explicit `dataset_episodes` still
  wins), and parity/sine baselines at the default size beat their 60-point
  baselines; `ParityTask` is parameterized by (n_bits, p_flip) with
  `Parity4V1` unchanged and `parity_ceiling` matching the Bayes formula;
  `ParityCurriculum` steps up when the best score saturates against the
  current ceiling, re-baselines on the new difficulty, records the ladder in
  state/summary, is deterministic, and is off by default; the env state
  carries `"task"`; the multi-task `MetaRLPolicy(task_names=...)`
  conditions on the task, updates only the active task's bias row, trains
  deterministically across sine→parity→cartpole, and `task_names=None`
  stays the v0.5 single-task policy (bit-identical for mlp-family episodes;
  tree/boost episodes now honor the §18.2 mask at update time, §20.2 fix
  note); the stall guard ends a duplicate-stalled episode with reason
  `duplicate_stall` (32 consecutive duplicates; a fresh proposal resets the
  streak); the §18.7 legacy pin now passes
  `dataset_episodes=60` explicitly, and A1–A9 still pass unchanged.
- **A11.** **Infrastructure & DX (v0.7, §21):** an external-library-style
  DQN agent drives `AutoRefineGymEnv` end-to-end through the gymnasium API
  only (obs in space, valid actions, episode terminates, summary written;
  replay updates move the Q-weights; same-seed action streams are identical,
  G2); the worked `examples/tabular.py` custom task (§21.5) registers into
  `TASKS` and a seed-7 bandit run reaches its 95 target within 20
  experiments from a ≈ 56 baseline; `report --json` prints `summary.json` as
  machine-readable JSON, and
  `report --plot` renders deterministic ASCII score/pareto charts plus valid
  SVG files in the run dir (default report output unchanged); the
  entry-point plugin loader registers external tasks into `TASKS` (validated
  protocol, clear error otherwise) and external policies, deterministically
  and testable without installed packages; `plugins list` reports the
  discovered entry points; and A1–A10 still pass unchanged.
- **A12.** **Data-in CLI & one-page report (v0.8, §22):** the built-in `csv`
  task (`tasks/csv.py`) loads a user CSV through the minimal §15 protocol —
  deterministic per seed (G2), label auto-detection, integer-class labels →
  `softmax` (else `mse`), non-numeric non-label columns ignored, 80/10/10
  seed-split, train-only standardization, `100·accuracy` / `100·R²` scores —
  and `AutoRefineEnv(task_config={...})` passes the data config through to
  the task constructor, records it in `summary.json`, and `eval`
  reconstructs the task from it; `autorefine fit --data sales.csv --label
  churn --target 95.0` runs the improvement loop on the file and gates on the
  user's target (PASS/exit 0 or MISS/exit 2, with budget/spec-space
  guidance); `report --html` writes one self-contained, deterministic
  `report.html` (embedded §21.2 SVGs + summary + experiment table, no
  external assets, default/`--json`/`--plot` behavior unchanged); and
  A1–A11 still pass unchanged.
- **A13.** **Visual dashboard (v0.9, §23):** the streamlit-free
  `DashboardRunner` drives `CsvTask` + `AutoRefineEnv` and yields one JSON-safe
  update dict per experiment (accepted, reason, candidate/gen scores, mutation,
  best score, budget left, done) — a full run's update stream is deterministic
  given the seed (G2), and its final verdict matches the §22.1 gate (PASS iff
  final ≥ target); the Streamlit app (skipped when streamlit is absent, the
  §3/§21.1 pattern) renders the page, takes a CSV (uploader or path), runs the
  loop with live updates, and shows the verdict, the §21.2 SVG plots, and the
  §22.2 `report.html` download; `autorefine dashboard` launches the app (clear
  install hint when streamlit is missing), and `import autorefine` stays
  streamlit-free; and A1–A12 still pass unchanged.

## 13. Milestones

| ID | Deliverable                                                        | Depends on |
|----|--------------------------------------------------------------------|------------|
| M0 | SPEC.md (this doc), layout scaffold                                | —          |
| M1 | `ModelSpec`, `CartPoleV1`, MLP+optimizers, `trainer`, `evaluator` | M0         |
| M2 | `AutoRefineEnv` + v1 `SearchPolicy` + memory + budget (G1, G3, G4)| M1         |
| M3 | CLI + artifacts + `report`/`eval` commands (G6)                    | M2         |
| M4 | Test suite green, A1–A4 verified, README quickstart (A5)           | M3         |
| M5 | v0.2 extension points (§15): tasks, families, meta-RL, gym, Pareto | M4         |
| M6 | v0.3 extension points (§17): `parity-v1`, bandit, `label_smoothing`     | M5         |
| M7 | v0.4 search quality (§18): local mutation, family masking, CI/efficiency acceptance | M6 |
| M8 | v0.5 model & spec space (§19): new mlp fields, `boost` family, frontier ensembling | M7 |
| M9 | v0.6 environment & loop design (§20): curriculum, multi-task meta-RL, task-level dataset sizes | M8 |
| M10 | v0.7 infrastructure & DX (§21): gym DQN agent, `report --plot/--json`, plugin loader | M9 |
| M11 | v0.8 data-in CLI & one-page report (§22): `csv` task + `fit` subcommand, `report --html` | M10 |
| M12 | v0.9 visual dashboard (§23): Streamlit app over `DashboardRunner` (optional dependency) | M11 |

**Estimate:** M1–M2 ≈ core effort; M3–M4 ≈ polish + validation; M5 in v0.2; M6 in v0.3; M7 in v0.4; M8 in v0.5; M9 in v0.6; M10 in v0.7; M11 in v0.8; M12 in v0.9 (all implemented).

## 14. Risks & Mitigations

| Risk                                   | Mitigation |
|----------------------------------------|------------|
| Improver overfits to val split         | Separate holdout + gen_score protocol (§5.3); A3 guard |
| Task too easy → no signal for reward   | Difficulty knobs in `TaskConfig`; v2 tasks (see §15) |
| NumPy backprop bugs corrupt scores     | T2 determinism test + analytic-grad check (finite-diff on small net) |
| Runtime blowup on small machines       | Per-train time cap enforced in-loop (R4, S2); T6 |
| Search space too small for v1 policy   | Mutation operators are data-driven from spec schema; extensible |

## 15. Extension Points (v0.2 — all implemented)

- **More tasks (done):** `SineRegressionV1` (`tasks/sine.py`, unit-scaled
  inputs, mse head, score = 100·R² on fresh points) and `GridNavV1`
  (`tasks/gridnav.py`, 6×6 torus, one-hot states, 4 directions, score =
  100·success_rate + 25·efficiency). Registered in the `TASKS` dict
  (`tasks/__init__.py`); `AutoRefineEnv` is task-agnostic (`--task` on the
  CLI). New tasks: implement the minimal protocol in `tasks/base.py`
  (`make_dataset` + `score`) and register.
- **RL improver (done):** `improver/rl_policy.py` — a linear softmax policy
  over the discrete mutation catalog (`improver/catalog.py`; 37 actions in v0.2, 40 in v0.3),
  trained with REINFORCE (return-to-go, running-mean baseline) via
  `train_policy(env, policy, n_episodes)`. Same `propose(env_state)` contract
  as `SearchPolicy`; deterministic given the seed (A6).
- **More model families (done):** `ModelSpec.model_family` ∈ {`mlp`, `tree`}
  (old spec JSON without the field still loads). An `mlp` with
  `architecture=()` is a **linear model** (depth 0 is now valid); the
  MLP gained an `mse` regression head. `tree` is a bagged CART ensemble
  (`models/trees.py`, `architecture=(depth,)`, `train_steps//100` trees),
  pickle-free npz checkpoints, uniform `forward` contract.
- **Gymnasium adapter (done):** `from autorefine.gym import AutoRefineGymEnv`
  — fixed-dim `Box` observation + `Discrete(40)` actions (v0.3; was 37); `gymnasium` stays an
  optional dependency (clear install hint if absent; `test_gym.py` skips when
  not installed).
- **Pareto memory (done):** `pareto.py` `ParetoFrontier` tracks the
  score-vs-train-time frontier for every run; `summary.json` gains
  `pareto_frontier`, `best_score_at_1s`, `efficiency_at_1s`; state carries
  `pareto_frontier` + `best_efficiency`. Reward formula unchanged (§6).

## 16. Open Questions (please confirm)

1. **Q1.** Is a single v1 task (CartPole) sufficient, or do you want the second
    task (SineRegression) in v1 scope? (Spec currently: one task, protocol ready.)
2. **Q2.** Should `AutoRefineEnv.step` accept a full `ModelSpec` (current design)
    or a *mutation action* (field, operator, value)? Full-spec is simpler and
    policy-agnostic; mutation-action exposes the search structure to the policy.
    **Recommendation: keep full-spec for v1, expose mutations in `info` for
    observability.**
3. **Q3.** Budget defaults: 30 experiments / 900 s total / 30 s per training —
    reasonable for your machine?
4. **Q8.** Text / LLM-embedding inputs as a modality (v0.10 added images +
    audio, §24)? Tokenization / embedding choice is a much bigger surface
    than pixel or mel features — **non-goal for v1**, revisit after a real
    need appears.

## 17. v0.3 Extensions (implemented)

The three "Extending" paths from the README are now each demonstrated by a
concrete, tested implementation (A7, M6):

- **New task (done): `parity-v1`** — `tasks/parity.py` `Parity4V1`,
  registered in `TASKS`. Classification over 4 hidden bits: label = XOR
  parity of the clean bits; each bit is observed with seeded flip noise
  (`P_FLIP = 0.08`); softmax head; `make_dataset(n_points=8192)` balanced by
  construction; score = 100·accuracy on fresh seed-derived points
  (deterministic, G2). Bayes ceiling ≈ 74.9, so the task has a real family
  gradient — linear ≈ 56, small MLP ≈ 57, (64,32)/Adam/3000 ≈ 76, shallow
  trees ≈ 51 (seed 7) — a task linear models provably cannot solve.
- **New policy (done): `BanditPolicy`** — `improver/bandit.py`, a UCB
  field-bandit over the spec *fields*: each step mutates one field of the
  current best spec; the field is chosen by UCB over win rates credited from
  `state["last"]` (accepted + mutated fields); cold start tries every field
  once (UCB = ∞) with a deterministic lexicographic tie-break; values are
  sampled by the existing `mutate_spec_dict` (so guaranteed-different and
  family-coercion safety come for free). `propose(env_state) -> spec_dict`
  contract, deterministic given the seed (G2). CLI: `--policy bandit`.
- **New spec field (done): `label_smoothing`** — `ModelSpec.label_smoothing`
  ∈ [0.0, 0.15], default 0.0 (pre-v0.3 spec JSON still loads, back-compat).
  Softmax-head only: `MLP.loss_and_grads(..., label_smoothing=ε)` computes
  loss and analytic gradients against the smoothed target
  `(1-ε)·onehot + ε/n_out` (mse/tree families ignore it). Sampler in
  `improver/actions.py`; catalog values (0.0, 0.05, 0.1) in
  `improver/catalog.py` — the action catalog grows 37 → 40. Included in
  `fingerprint()` (per-run dedup sets are unaffected). It is a training-time
  parameter, so `MLP.save/load` is unchanged.

**Compatibility note:** `fingerprint()` now includes `label_smoothing`, so
fingerprint strings differ from v0.2 — dedup sets are per-run, so this is
not observable in any persisted artifact.

## 18. Search Quality (v0.4 — implemented)

The loop works, but its *decision rule* is weak: candidates are accepted on a
single noisy holdout score (Δ > 0), the improver spends budget on fields the
current model family ignores, and mutations resample each field over its
*full* range, so "hill climbing" often teleports (e.g. `train_steps`
3484→200). This section specifies five fixes that compose into one stricter,
deterministic, backward-compatible decision rule.

**Design invariants:** G2 holds throughout — every new quantity is
seed-derived. v0.3 behavior is exactly recoverable (all new knobs at 0,
`mode="uniform"`). Spec-JSON and checkpoint formats are unchanged.

### 18.1 Local (neighborhood) mutation

`actions.mutate_spec_dict(spec_dict, field, rng, mode="uniform")` gains a
`mode`:

- **`"uniform"` (default; v0.3 behavior):** resample the full range — unchanged.
- **`"local"`:** move to a *neighborhood* of the current value:
  - ordered numeric fields (`learning_rate`, `batch_size`, `weight_decay`,
    `train_steps`, `input_noise`, `label_smoothing`): anchor the current
    value to its nearest catalog index (extend
    `catalog.index_of_value` with nearest-match), flip a seeded coin, step
    ±1 index (clamped at the ends). `learning_rate` is log-spaced, so this
    is multiplicatively local.
  - `architecture`: 50% depth ±1 (clamped 1..3), 50% the first hidden
    layer's width ±1 step in `HIDDEN_LAYER_SIZES` (mlp, depth ≥ 1);
    if neither is possible, fall back to uniform.
  - categorical fields (`optimizer`, `activation`, `model_family`):
    local ≡ uniform (small finite spaces, no meaningful neighborhood).
  - preserved: guaranteed-different (retry loop), `_coerce_for_family`,
    determinism given the RNG stream.

Consumers: `BanditPolicy(seed, alpha=1.0, mode="local")` — **the v0.4
default is `"local"`** (`mode="uniform"` restores the v0.3 stream; G2
still holds). `SearchPolicy`: its *refinement* steps (re-mutating the
successful field) switch to `mode="local"`; initial field mutations stay
uniform.

### 18.2 Family-conditioned action space

The `tree` family ignores `optimizer`, `learning_rate`, `batch_size`,
`weight_decay`, `activation`, `label_smoothing` — today every such proposal
is a wasted experiment (fingerprint differs, behavior is identical).

- `catalog.py` adds:
  `FAMILY_FIELDS = {"mlp": <all 10 fields>,
                    "tree": ("architecture", "train_steps", "input_noise", "model_family")}`,
  plus `relevant_fields(family) -> tuple[str, ...]` and
  `relevant_actions(family) -> tuple[int, ...]` (catalog indices).
- `BanditPolicy`: UCB is computed over `relevant_fields(family(best))` only;
  cold start tries every *relevant* field once. Trials/wins history is
  **kept** across family switches (no state reset).
- `MetaRLPolicy`: logits of actions outside `relevant_actions(family(best))`
  are masked to `-inf` before the softmax. The action space stays 40, so
  weight/embedding shapes and existing policy state remain loadable.
- `apply_action`, the catalog, and the Gym adapter are unchanged (40
  actions); masking is a proposal-time concern of the policies.

### 18.3 Block-bootstrap score confidence

`evaluate_full` scores a single 200-point holdout — on parity a 4.5-point
"improvement" is ≈1.3σ of sampling noise.

- `evaluator.py` adds
  `score_with_ci(task, model, split="holdout", n_blocks=8, block_size=512)
  -> {"mean", "std", "blocks"}`. Block *i* is
  `task.score(model, f"{split}-b{i}", block_size)` — split names are opaque
  seed-derived strings (G2), so **all four tasks work with zero task
  changes** and the blocks are independent.
- Under the v0.4 preset, holdout *and* gen scores become block means
  (8×512 = 4096 points — also a larger sample than the legacy 200);
  `gen_gap` keeps its existing definition (score points).
- Cost: 16 extra `task.score` calls per experiment (≪ 1 s on all four tasks
  at the default block size; `block_size` is tunable).

### 18.4 Efficiency-aware scoring

- `eff(m) = score(m) − η · min(train_seconds(m), T_CAP)`, recommended
  `η = 0.5` points/s, `T_CAP = 10 s` (max time penalty: 5 points).
  Time-capped candidates use their actual elapsed time — partially trained
  specs are not rewarded.
- **Best-spec tracking and acceptance use `eff`**; raw scores keep being
  reported. `summary.json` retains `baseline_score` / `final_best_score` /
  `improvement_factor` (raw, back-compatible) and adds `*_effective`
  counterparts.

### 18.5 Generalization-gap penalty

- `gap_pen = max(0, gen_gap − 0.05 · score) · P_GEN`, tolerance = 5% of
  score (exactly A3), recommended `P_GEN = 0.5` points per excess gap
  point. Folded into `eff`:
  `eff(m) = score(m) − η·min(train_seconds, T_CAP) − gap_pen(m)`.

### 18.6 Unified acceptance rule

```
accept(cand)  ⟺  eff(cand) − eff(best)  >  max(0, z · SE)
SE = sqrt(σ²_cand + σ²_best)        # sample std over blocks, §18.3
reward     =  (eff(cand) − eff(prev_best)) + NOVELTY_BONUS
```
Recommended `z = 1.0` (≈95% two-sided; deliberately not 1.96, to keep the
search lively). With every knob at 0 the rule degrades **exactly** to v1's
`Δraw > 0`.

**Configuration surface:**

- `AutoRefineEnv(..., ci_blocks=0, z_accept=0.0, efficiency_weight=0.0,
  gen_gap_penalty=0.0, block_size=512)` — defaults are legacy, so existing
  tests and A1–A7 are untouched.
- `autorefine.search_quality_v04()` returns the recommended kwargs
  `dict(ci_blocks=8, z_accept=1.0, efficiency_weight=0.5,
  gen_gap_penalty=0.5)`; the **CLI `run` uses the preset by default**,
  with `--search-quality legacy` to opt out.
- `summary.json` records the active preset (auditability).

### 18.7 Tests and acceptance

- **Local mutation:** for every field the local candidate lies within ±1
  catalog index of the current value (or the documented exceptions);
  guaranteed-different holds; two same-seed runs are identical; the bandit
  in `mode="local"` always proposes valid `ModelSpec`s.
- **Family conditioning:** with a tree best spec the bandit never proposes
  outside `{architecture, train_steps, input_noise, model_family}` across 50
  forced steps; `MetaRLPolicy` never samples a masked action across 1000
  proposals; the action space remains 40.
- **CI acceptance:** a near-tie candidate (Δ within SE) is rejected, a
  clear winner accepted; the acceptance sequence is deterministic.
- **Efficiency:** a 4× faster candidate at equal raw score is accepted under
  the v0.4 preset and rejected under legacy; raw `improvement_factor`
  unchanged.
- **Gen-gap:** a candidate with 15% excess gen gap loses to a clean
  candidate at the same holdout score.
- **Regression:** legacy mode reproduces the v0.3 acceptance sequence
  bit-exactly on `parity-v1`, seed 7, fixed 6-experiment budget.
- **A8 (integration):** under the v0.4 preset, `bandit(mode="local")`
  runs on `parity-v1` and `cartpole-v1` (a) never regress below the raw
  baseline, (b) show A1-style improvement on at least one task, (c) are
  reproducible (A2-style), and (d) leave A1–A7 green in legacy mode.

**Risks:** stricter acceptance → fewer accepts → the A1 1.25× bar gets
harder (mitigations: z=1.0, capped η, A8(d) verified first, every knob
tunable); 16 extra scores per experiment (bounded; tunable); the bandit
default flip changes v0.3 proposal streams (runtime-only; G2 holds;
`mode="uniform"` restores them).

**Open questions (resolved — recommendations approved as-is):**

4. **Q4.** v0.4 preset as the CLI default — **yes** (`--search-quality
   legacy` opts out; the API constructor keeps legacy defaults).
5. **Q5.** η = 0.5 points/s with a 10 s cap — **approved** (A8 verifies
   it on `cartpole-v1`: final 4.71× baseline under the preset).
6. **Q6.** z = 1.0 — **approved** (deliberately not 1.96).
7. **Q7.** Bandit default `mode="local"` — **approved** (`mode="uniform"
   restores the v0.3 stream).

## 19. Model & Spec Space (v0.5 — implemented)

The model & spec space itself gets three upgrades (A9, M8): four new spec
fields with real training effect (each the same extension pattern
`label_smoothing` showed in §17), a third model family (gradient boosting),
and an opt-in ensemble final evaluation over the already-collected frontier.

### 19.1 New spec fields (mlp family; all default to legacy)

| field | space | default | effect |
|---|---|---|---|
| `lr_schedule` | `constant` \| `cosine` \| `warmup_cosine` | `constant` | per-step LR: fixed `learning_rate` (v0.4), cosine decay to 0 over `train_steps`, or linear warmup over the first 10% then cosine decay |
| `early_stopping_patience` | 0..50 (int) | `0` (off) | hold out the last 10% of the train split; stop when val loss fails to improve for `patience` consecutive steps; restore the best-val weights |
| `init_scale` | 0.5..2.0 | `1.0` | multiplicative scale on the Glorot init bound (1.0 = plain Glorot) |
| `gradient_clipping` | 0.0..10.0 | `0.0` (off) | global L2-norm cap on each step's gradients |

Rules:

- Validated for every family; only the `mlp` family consumes them
  (`tree`/`boost` ignore them, exactly like `label_smoothing` in §17).
- The warmup fraction (10%) and early-stopping val fraction (10%) are
  internal constants (`WARMUP_FRACTION`, `VAL_FRACTION` in `trainer.py`),
  not spec fields — the spec surface stays small.
- **Back-compat:** pre-v0.5 spec JSON loads with the legacy defaults;
  all four at defaults reproduces v0.4 training bit-exactly (no val split,
  fixed LR, no clipping, plain Glorot), so the T2/A8 pins are unaffected.
- **Determinism (G2):** the schedule, the clipping, and the early-stopping
  decision (deterministic tail split, no RNG) are pure functions of
  (dataset, spec, seed).

### 19.2 Third model family: `boost` (gradient boosting)

- `model_family="boost"`: a residual (gradient-boosted) CART ensemble,
  `BoostingEnsemble` in `models/trees.py`. Each round fits one regression
  CART per output column to the *residual* of the current model, shrunk by
  a fixed `BOOST_SHRINK = 0.1`; the base estimate is the target mean (the
  class prior for a softmax head).
- Same surface as `tree`: `architecture=(depth,)` (1..3),
  `train_steps // 100` = number of rounds (2..50), `input_noise` applied to
  the bootstrap bag. `FAMILY_FIELDS["boost"]` mirrors `tree` (4 relevant
  fields); `learning_rate` is not a boost knob in v0.5 (the shrink is fixed;
  a dedicated spec field is a documented follow-up).
- Motivation: give the improver a third, genuinely different answer. On
  regression (`sine-v1`) boost is clearly and robustly stronger than bag
  (seed 7, depth 3, 50 rounds, 4000 points: bag ≈ 0 vs boost ≈ 41). On
  `parity-v1` (classification) it is a *competitive* family — better than bag
  in some configs (e.g. 8192 points, depth 2, 50 rounds: bag 46.1 vs boost
  54.9) and worse in others, so the ordering is config-dependent. A9 pins a
  representative config where boost wins rather than asserting a universal
  `boost > bag` on parity.
- Pickle-free npz checkpoints; uniform `forward(x) -> (n, n_out)` contract,
  so evaluator/tasks treat it like any other family.

### 19.3 Frontier ensembling (opt-in final evaluation)

- `AutoRefineEnv(ensemble_top_k=2)` (CLI: `run --ensemble-final`): the env
  tracks the top-`k` models by holdout score (baseline included) and, on
  finish, scores an ensemble — the average of the members' outputs (logits
  for softmax tasks, values for mse) — on the holdout split and records
  `summary["ensemble"] = {"top_k", "member_scores", "ensemble_score"}`.
- Off by default (`ensemble_top_k=0`), so v0.4 runs and A1–A8 are
  unchanged. The ensemble is a reported final evaluation, not an accepted
  candidate: the reward/acceptance contract (§6, §18.6) is untouched.
- This is the soft form of model merging ("no model merging" stays a v1
  non-goal, §3): it averages already-trained frontier models, no new
  training.

### 19.4 Catalog / family wiring

- Catalog: 10 → 14 fields; action space 40 → 54 (`model_family` gains
  `boost`; +3 `lr_schedule`, +4 `early_stopping_patience`, +3 `init_scale`,
  +3 `gradient_clipping` values). Old action indices stay stable (new values
  are appended), so saved v0.4 one-hot states remain index-compatible.
- `apply_action` and `_coerce_for_family` treat `boost` like `tree`
  (architecture forced to a valid depth on family switch).
- Local moves (§18.1): the three ordered numeric fields join
  `ORDERED_FIELDS` (neighbor ±1 in the catalog); `lr_schedule` is
  categorical (local = uniform resample, still guaranteed-different).
- `MetaRLPolicy` / `BanditPolicy` / the gym adapter derive their sizes from
  `len(ACTIONS)` / `FIELD_NAMES`, so they pick up the larger space with no
  code changes (larger embeddings).

### 19.5 Tests (A9)

`tests/test_model_spec_space.py` covers: field validation + back-compat
(old JSON → legacy defaults); cosine / warmup-cosine schedule shapes;
early stopping stops early and restores best-val weights; `init_scale`
scales the Glorot bound; gradient clipping bounds the per-step global
gradient norm; `boost` beats bagged `tree` at equal rounds/depth on a
representative `parity-v1` config (the ordering is config-dependent, §19.2)
and boost is a trained answer above chance; boost trains deterministically
and round-trips via npz; an
`ensemble_top_k=2` run records a finite `ensemble_score` while the default
leaves `summary.json` unchanged.

The §18.7 legacy pin is **re-pinned** to the v0.5 proposal stream: the spec
space grew (14 fields), so the v0.3 cold-start field order shifted — the pin
still guards legacy-mode determinism and the v0.3 acceptance rule. A1–A8
still pass unchanged.

### 19.6 Milestone

**M8** — v0.5 model & spec space (A9).

## 20. Environment & Loop Design (v0.6 — implemented)

Three upgrades to the loop itself (A10, M9): a curriculum that scales task
difficulty with the improver's progress, a task-conditioned multi-task
meta-RL policy, and task-level default dataset sizes. All are opt-in or
backward-compatible: A1–A9 still pass unchanged (the §18.7 legacy pin keeps
its 60-point parity dataset by passing `dataset_episodes=60` explicitly).

### 20.1 Curriculum / adaptive difficulty

A fixed-difficulty task saturates: parity's Bayes ceiling is 74.9 and the
§18.7 run already hits 68.5 in 6 experiments, so the reward delta dies and
the loop stops learning. A curriculum keeps reward informative by stepping
difficulty up as the improver progresses.

- **Parameterized parity:** `ParityTask(seed, n_bits, p_flip)` in
  `tasks/parity.py`; `Parity4V1` (= 4 bits, 8% flip) stays the v0.3
  registry entry with its literal `name`/`state_dim` (back-compat).
  `parity_ceiling(n_bits, p_flip) = 100·(1 + (1−2p)^n) / 2` — the Bayes
  accuracy in score points (the formula `test_tasks_ext.py` already uses).
- **Ladder:** `ParityCurriculum` (`improver/curriculum.py`), defaults:
  sweep p_flip 0.08 → 0.10 → … → 0.16 (step 0.02) at 4 bits, then n_bits 5
  re-sweeps p_flip 0.08 → 0.16 — 10 levels (9 step-ups), capped by
  `max_n_bits`. Level values come from index arithmetic (no float
  accumulation); every decision is a pure function of (score history, seed)
  — G2.
- **Env integration:** `AutoRefineEnv(..., curriculum=...)` (opt-in;
  `None` = off, v0.5 exact). After each experiment, if
  `best_score ≥ trigger_fraction · ceiling(current level)` (default
  trigger_fraction 0.9) and levels remain, the env rebuilds the task at the
  next level (same seed), rebuilds the train-split dataset, re-baselines the
  current best spec on the new level (free, like the reset baseline), and
  resets the difficulty-scoped state (dedup `seen`, Pareto frontier,
  ensemble top-k) while keeping the full experiment log and the *first*
  baseline (v0.4 summary contract).
- **Observability:** `state["curriculum"]` (level, difficulty, ceiling,
  levels left); `summary["curriculum"]` with per-level rows (difficulty,
  ceiling, best-before, new baseline); one `kind="curriculum"` log entry per
  step-up. `state["task"]` reports the active task name at all times.
- **CLI:** `run --curriculum` (parity-v1 for now) prints the ladder
  history; `eval` reconstructs the final difficulty level from the summary
  when the task name is a curriculum level (not in `TASKS`).

### 20.2 Multi-task meta-RL

`train_policy` trains one policy per task. The same REINFORCE machinery
transfers across tasks with a task-conditioned policy — no new machinery:

- `AutoRefineEnv._state()` gains `"task"` (the active task name; a new key,
  existing policies ignore it).
- `MetaRLPolicy(seed, task_names=None)`: with `task_names` set, logits gain
  a per-task bias row — `logits = x·w + b + B[task_idx]` — where the shared
  `w`/`b` are the transfer channel and `B` captures task specificity.
  `task_names=None` stays the v0.5 single-task policy (bit-identical for
  mlp-family episodes; see the v0.5 fix note below).
- **v0.5 fix (found while wiring this):** v0.5's `observe()` replaced the
  last trajectory entry with a bare `(x, a, reward)` 3-tuple, silently
  dropping the stored family mask — so `_update` never actually applied the
  §18.2 family mask it was documented to re-compute. v0.6 preserves the
  full entry (family + task index), so `_update` re-computes exactly the
  masked, task-conditioned probabilities the action was sampled under.
- `train_multi_policy(envs, policy, episodes_per_task)`: round-robin
  episodes over the envs in `task_names` order (deterministic), one shared
  `propose/observe` stream; report = per-task returns + update count.
- **Transfer is via the shared weights**: sine→parity→cartpole episodes all
  push `w`/`b`; each task's `B` row moves only on that task's episodes.
- **Stall guard (v0.6, episode termination):** one REINFORCE episode can
  collapse the 54-way action distribution onto a single spec already in
  `seen` (observed with seed 12 after one sine episode: p ≈ 1.0 on one
  action). Duplicates are free (R3) and spend no budget, so such an episode
  would otherwise loop on free duplicate rejections until the wall clock —
  an unbounded free-work loop. v0.6 ends the episode after 32 consecutive
  duplicate rejections (`done_reason` / `finished_reason` =
  `"duplicate_stall"`, and the stalling step's info reports that reason);
  any non-duplicate proposal resets the streak. A uniform policy stalls
  with probability ~1e-22 per 32-step window, so the guard never fires in
  any A1–A9 run (those stay bit-identical to v0.5).

### 20.3 Task-level default dataset size

`AutoRefineEnv` hard-coded `dataset_episodes=60` into `make_dataset` —
meaningful for cartpole (episodes), but 60 *points* for parity/sine,
which depresses every baseline (the §18.7 run's 45.0 baseline). Now:

- Every task declares `default_dataset_size` in its native units (points
  for fitting tasks, episodes for episode tasks): cartpole 60, gridnav 60,
  sine 2048, parity 4096.
- `AutoRefineEnv(dataset_episodes=None)` falls back to
  `task.default_dataset_size`; an explicit int keeps the legacy size
  (parameter name unchanged). `trainer.train_from_task` follows suit.
- Effect: parity/sine baselines and candidate scores rise noticeably (more
  training data); cartpole/gridnav runs are unchanged; the §18.7 legacy pin
  passes `dataset_episodes=60` explicitly and keeps its exact v0.3 sequence.

### 20.4 Tests (A10)

`tests/test_env_loop_design.py` covers: every task exposes
`default_dataset_size` (cartpole/gridnav unchanged at 60); the env falls
back to it and an explicit override still wins; parity and sine baselines
at the default size beat their 60-point baselines; the parameterized
`ParityTask` (name/state_dim/data shape/determinism) and
`parity_ceiling` (formula, monotone in both axes); the
`ParityCurriculum` ladder order + exhaustion; a curriculum run steps up,
re-baselines on the new difficulty, records the ladder in state/summary,
and is bit-reproducible; curriculum off by default; the env state carries
`"task"`; the multi-task policy conditions on the task (different task ⇒
different logits once trained), updates only the active task's bias row,
trains deterministically across sine→parity→cartpole (each episode
bounded), and the stall guard ends a duplicate-stalled episode with
`duplicate_stall` at the 32-consecutive-duplicate cap (a fresh proposal
resets the streak; a single duplicate is still free, R3);
`task_names=None` stays the v0.5 single-task policy (bit-identical for
mlp-family episodes; tree/boost episodes now honor the §18.2 mask at
update time, §20.2 fix note).

### 20.5 Milestone

**M9** — v0.6 environment & loop design (A10).

## 21. Infrastructure & DX (v0.7 — implemented)

Three infrastructure / developer-experience upgrades (A11, M10). All are
additive and keep the core NumPy-only: gymnasium stays an optional dependency,
plotting is hand-written (no matplotlib), and the plugin loader uses only
`importlib.metadata`. A1–A10 still pass unchanged.

### 21.1 Real RL via the Gymnasium adapter

The §15 adapter was verified only against the discrete catalog (spaces,
reset/step contract). v0.7 proves it carries an external-library-style
agent:

- `examples/gym_dqn.py` — a self-contained, DQN-style agent in NumPy only:
  a 2-layer Q-network (tanh hidden) over the catalog's discrete actions,
  experience-replay buffer, ε-greedy sampling (deterministic ε schedule),
  and hand-written backprop over a replay batch (no target net — a deliberate
  simplification). It touches the environment **only** through the gymnasium
  API (`reset(seed)`, `step(action)`, `observation_space`, `action_space`)
  — no `AutoRefineEnv` internals, no AutoRefine imports beyond the adapter.
  Deterministic given a seed (G2). Runnable: `python examples/gym_dqn.py`.
- `tests/test_gym_dqn.py` — integration (skipped when gymnasium is absent):
  the agent completes a short-budget episode through the adapter only (obs in
  space, valid actions, episode terminates, `summary.json` written); replay
  updates actually move the Q-weights (finite); two same-seed runs produce
  identical action streams (G2).

### 21.2 `report --plot` + `--json`

The data already exists in `experiments.jsonl` / `summary.json`; v0.7 adds
rendering and machine-readable CLI output (no GUI, no new dependency —
ASCII and SVG are generated by hand):

- `src/autorefine/plotting.py` — pure functions of the log rows (deterministic,
  ASCII-safe): `ascii_score_curve(rows)` / `svg_score_curve(rows)` (score vs.
  experiment index; baseline marked `B`, accepted `+`, rejected `o`) and
  `ascii_pareto(points)` / `svg_pareto(points)` (the §15 Pareto frontier:
  score vs training seconds). SVGs are valid XML, escape text, and round
  coordinates; empty inputs return an explanatory string (score curve) or an
  empty-but-valid SVG (pareto).
- CLI `report` (SPEC.md 11): `--json` prints `summary.json` to stdout as JSON
  (machine-readable, `sort_keys`); `--plot` appends both ASCII charts to
  stdout and writes `score_curve.svg` + `pareto_frontier.svg` into the run
  dir; combined (`--plot --json`) stdout is pure JSON and the plots go to the
  SVG files only. Default output is byte-unchanged. Pareto points come from
  `summary["pareto_frontier"]` when present (the §15 contract), else from the
  experiment log.

### 21.3 Plugin loader

`TASKS` and the policy contract (§17) already make registration trivial from
Python; v0.7 adds the no-import path for external packages:

- `src/autorefine/plugins.py` — `discover(group, eps=None)`,
  `register_tasks(eps=None)`, `register_policies(eps=None)`, and
  `PluginError`. Entry-point groups: `autorefine.tasks` and
  `autorefine.policies` (e.g. `[project.entry-points."autorefine.tasks"]
  my-task = "my_pkg.tasks:MyTask"`). Each entry must load to a *class*:
  tasks with callable `make_dataset`/`score` (the §15 protocol), policies
  with callable `propose` — anything else raises `PluginError` naming the
  entry. Loaded tasks are merged into `TASKS`; loaded policies are returned
  as a `{name: cls}` dict (the policy contract is `propose`, not a registry).
  Discovery is deterministic (sorted by entry name), and `eps` injection
  makes the loader testable without installing a package.
- CLI: `run`/`eval` call `register_tasks()` before task selection (so plugin
  tasks appear in `--task` choices and `eval` reconstruction), and a new
  `autorefine plugins list` subcommand reports the discovered entry points
  per group (loaded, for validation) — empty output is a clean no-op.

### 21.4 Tests (A11)

`tests/test_gym_dqn.py` + `tests/test_infra_dx.py` cover: the DQN agent
episode/determinism/update checks above (skipped without gymnasium); the
plotting functions (determinism, ASCII content, valid XML via stdlib
`xml.etree`, point counts, empty-input behavior); `report` (default output
unchanged; `--json` stdout parses and equals `summary.json`; `--plot` writes
both SVGs, parseable; `--plot --json` keeps stdout pure JSON); the plugin
loader (fake entry points register a working task into `TASKS` — driven
through a 1-experiment `AutoRefineEnv` — and a policy dict; invalid entries
raise `PluginError`; `plugins list` output is stable); the worked
  `examples/tabular.py` task (§21.5) registers into `TASKS`, drives a short
  improver run, and the seed-7 bandit run reaches the 95 target within
  20 experiments from its ≈ 56 baseline; and A1–A10 stay green.

### 21.5 Worked example: custom tabular task

The "I have tabular data but don't know which model to fit" story (the first
README "Extending" path) is now demonstrated end-to-end:

- `examples/tabular.py` — `TabularV1`: a synthetic 2-D tabular
  classification (label = XOR of quadrants over `U[-1,1]^2`, clean labels,
  ceiling 100) implementing the minimal §15 protocol (`make_dataset` +
  `score`, unit-scaled inputs). Non-linear by construction, so it carries a
  real family gradient (seed 7: linear ≈ 50.5, tree ≈ 49, boost ≈ 51,
  default spec ≈ 56, tuned MLP ≈ 99) — the same signal that makes the
  improver's search informative. The script (a) registers the task in
  `TASKS`, also showing the §21.3 entry-point alternative, (b) runs the
  bandit improver under a bounded budget, and (c) gates on *the user's*
  target (`--target`, default 95.0) against `final_best_score` — the loop
  maximizes the validated score and does not natively know the target, so
  the gate (and its MISS guidance: raise the budget, or extend the spec
  space per README "Extending") is the user's acceptance step. Deterministic
  given the seed (G2); seed 7 reaches ≈ 99.5 within 20 experiments.

### 21.6 Milestone

**M10** — v0.7 infrastructure & DX (A11).

## 22. Data-in CLI & one-page report (v0.8 — implemented)

Streamlining "input data → trained model" without writing any code (A12, M11).
The only code a user writes today is the `Task` wrapper (§21.5); v0.8 removes
that last step for CSV data and adds a single-file report. Additive, and the
§3 constraints hold: no new dependency (stdlib `csv` + hand-written HTML),
no GUI (a generated file is not an app), NumPy-only core, deterministic
(G2). A1–A11 still pass unchanged.

### 22.1 `fit` subcommand + built-in `csv` task

One command takes a CSV to a gated, validated model:

    python -m autorefine fit --data sales.csv --label churn --target 95.0

- `src/autorefine/tasks/csv.py` — `CsvTask`, a built-in task registered as
  `csv` in `TASKS`. `CsvTask(seed, path, label=None, split_frac=0.2)`:
  - CSV read via stdlib `csv` (header + rows); `label` defaults to the first
    of `label`/`target`/`y`/`class` (case-insensitive), else the last
    column; a named `label` not present is an error. Features are the
    numeric columns (all values float-parseable) except the label, in file
    order; non-numeric non-label columns are ignored; zero feature
    columns, a non-numeric label, or < 3 rows are errors.
  - Head inference (deterministic): all-integer label values with 2–50
    distinct classes → `softmax` with sorted unique values mapped to
    classes 0…K−1; otherwise `mse` regression.
  - Splits: a seed-derived permutation (SeedSequence from the task seed,
    G2) partitions the rows 80/10/10 into train/holdout/gen;
    `split_frac` (default 0.2) is the non-train share, split evenly.
    Features are standardized with **train-only** statistics (std 0 → 1).
  - Protocol: `make_dataset(n)` = the train split (clamped to its length);
    `score(model, split, n)` = 100·accuracy (softmax) or 100·R² clamped at
    0 (mse) on the split's rows (clamped to its length). Split strings are
    matched by prefix: `gen*` → gen, `train*` → train, else holdout — so
    the §18.3 block-bootstrap names (`holdout-b3`, `gen-b2`) work unchanged;
    with a fixed row set every block scores the same rows (σ = 0, the
    §18.6 rule reduces to a strict score comparison).
- `AutoRefineEnv` gains an optional `task_config: dict | None` (SPEC.md
  15 extension, additive): when non-empty the task is constructed as
  `task_cls(seed=seed, **task_config)` (built-in tasks keep the plain
  `task_cls(seed)` path, unchanged). The config is recorded in
  `summary.json` (`task_config`), and `eval` reconstructs the task with it
  (e.g. a `csv` run re-scores against the same file).
- CLI `fit` (SPEC.md 11) = the `run` loop (shared driver: `--policy
  search|bandit|rl`, `--experiments`, `--max-seconds`,
  `--max-train-seconds`, `--seed`, `--runs-dir`, `--search-quality`,
  `--ensemble-final`, `--policy rl`) over `task=csv` with
  `task_config={path, label, split_frac}` and the train split size passed
  explicitly, plus the §21.5 gate: `--target` (default 95.0) checked
  against `final_best_score` — PASS (exit 0) or MISS (exit 2, with the
  "raise the budget / extend the spec space" guidance). The gate is the
  user's acceptance step: the loop maximizes the validated score and does
  not natively know the target.

### 22.2 `report --html`

- `plotting.html_report(summary, entries)` — one self-contained HTML5
  document, hand-written (no new dependency, no JS, no external assets):
  the run summary table, the best spec (`<pre>` JSON), the two §21.2 SVGs
  embedded inline (`svg_score_curve`/`svg_pareto`), and the experiment
  table (kind, accepted, score, gen gap, train seconds, mutation). All
  dynamic text is HTML-escaped; the output is a pure function of
  `(summary, entries)` — no timestamps, byte-identical on rerun (G2).
- CLI `report --html` (SPEC.md 11): writes `report.html` into the run dir
  and prints its path. Default `report` output and the §21.2
  `--plot`/`--json` behavior are unchanged; `--html --json` keeps stdout
  pure JSON (the file is still written).

### 22.3 Tests (A12)

`tests/test_data_cli.py` covers: the `csv` task (G2 determinism of dataset
and scores; split sizes; softmax inference + class mapping on a
non-linear CSV; mse on a regression CSV; non-numeric column ignored;
label auto-detect vs explicit; error cases); `task_config` (a 1-experiment
env on a CSV; the config round-trips through `summary.json`; `eval`
reconstruction re-scores against the file; built-in tasks with
`task_config=None` unchanged); the `fit` CLI end-to-end (classification
CSV, seed-7 bandit, target gate PASS + `task_config` in the summary;
regression CSV with a low target; `fit` on a missing file errors
cleanly); `report --html` (file written and parseable, embeds both
`report --plot` SVGs and the summary values, deterministic; `--html
--json` keeps stdout pure JSON); and A1–A11 stay green.

### 22.4 Milestone

**M11** — v0.8 data-in CLI & one-page report (A12).

## 23. Visual dashboard (v0.9 — implemented)

The "input data → trained model" flow (§22) gains its visual surface (A13,
M12): upload a CSV, watch every experiment happen, and view the plots. The §3
"no GUI" non-goal is amended *explicitly* (see §3): the core stays CLI +
Python API and NumPy-only; the dashboard is an **optional** extra package
(streamlit), launched as a separate process, never imported by the core — the
same pattern as the gymnasium adapter (§21.1). A1–A12 still pass unchanged.

### 23.1 Architecture: pure runner + thin app

- `src/autorefine/dashboard.py` — `DashboardRunner`, **no streamlit import**
  (core-safe, importable in a NumPy-only environment):
  - `DashboardRunner(csv_path, label=None, split_frac=0.2, target=95.0,
    policy="bandit", seed=7, experiments=30, max_seconds=900.0,
    max_train_seconds=30.0, runs_dir="runs", search_quality="v04")` — the
    `fit` parameter surface (SPEC.md 22.1) minus `rl` (the meta-RL trainer
    spans several full-budget episodes and does not map to a live
    per-experiment stream; `rl` stays CLI-only, a clear error names
    `autorefine fit --policy rl`). `search_quality` mirrors the CLI
    (`v04` default, SPEC.md 18.6; `legacy` = the v0.3 rule) — note the v0.4
    eff term is legitimately wall-clock-aware (measured train time, SPEC.md
    18.4), while `legacy` acceptance is bit-deterministic (G2).
  - `start()` — builds the `CsvTask` probe (SPEC.md 22.1) for the inferred
    label/head/splits, constructs `AutoRefineEnv(task="csv",
    task_config={path,label,split_frac}, ...)` with the v0.4 search-quality
    preset (the CLI default, SPEC.md 18.6) and `dataset_episodes` = the
    train-split size, `reset()`s, and returns the initial view: task info
    (label column, head, class values, feature names, split sizes), baseline
    score, run dir.
  - `next()` — one `policy.propose` + `env.step`; returns one JSON-safe
    update dict per experiment: `index`, `accepted`, `reason`,
    `candidate_score`, `gen_gap`, `mutation` fields, `best_score`,
    `train_seconds`, `experiments_left`, `reward`, `done` (in v0.4 mode
    `train_seconds`/`reward` are the wall-clock fields, like A2's
    `ts`/`train_seconds` exclusions). Duplicate rejections and budget
    exhaustion surface as updates, not exceptions.
  - `run_all(on_update=None)` — drives the loop to `done`, calling
    `on_update(update)` per step (the app's live feed), then `finish()`.
  - `finish()` — after `done`: the §22.1 verdict (`"PASS"` iff
    `final_best_score ≥ target`, the same gate `fit` uses), the summary,
    the §21.2 Pareto points, and artifact payloads (`best_spec.json`,
    `best_model.npz`, `experiments.jsonl` bytes, and the §22.2
    `report.html` generated in-memory) plus the §21.2 SVG strings.
  - The UI is a pure function of this stream: nothing in the app influences
    the run, so same-seed runs reproduce the CLI `fit` outcome with the same
    flags (G2).

### 23.2 The app (streamlit, optional)

`src/autorefine/dashboard_app.py` (run with `streamlit run`) — a thin
rendering layer over the runner, streamlit widgets only:

- **Input:** the CSV via `st.file_uploader` (bytes written to the session
  temp dir) *or* a sidebar path field (server-side); the label column is
  pre-filled from `CsvTask` inference (editable); plus target score, policy
  (bandit default / search), seed, experiments, max-train-seconds, runs dir.
  A data preview shows the inferred head, split sizes, and first rows.
- **Live updates** (the loop runs synchronously in the page script, so
  Streamlit re-renders as it goes): a progress bar over the experiment
  budget — counting only budget-spending experiments (`spent = budget −
  experiments_left`), with free duplicate rejections (R3) labeled "dup
  step" — one live experiment table (index, accepted, candidate, gen gap,
  mutation, best) and one live best-score curve, both in placeholders that
  update in place, plus a per-step caption (experiment count + accept/reject
  + score + mutation), including the final step.
- **Persistence:** a widget-triggered re-run (e.g. clicking an artifact
  download, or changing a sidebar value) must not lose the run — the
  finished result (final table, best-score curve, verdict, metrics, SVG
  plots, artifact buttons) is stored in `st.session_state` and re-rendered
  on every subsequent script run; **Run** replaces it, **New run (clear
  result)** drops it.
- **Plots:** while running, the live curve; at the end, the §21.2 hand-
  written SVGs (`svg_score_curve`/`svg_pareto`) rendered inline via
  `st.markdown`, plus `st.download_button` for `report.html` (§22.2),
  `best_spec.json`, `best_model.npz`, `experiments.jsonl`.
- **Verdict:** `st.success`/`st.error` with the §22.1 gate (PASS: final ≥
  target; MISS: raise the budget or extend the spec space), and a
  `st.metric` row (baseline → final, improvement factor, experiments,
  wall time).

### 23.3 Tests (A13)

`tests/test_dashboard.py` covers: the runner (inferred label/head/splits on a
classification CSV; the update stream — every row JSON-safe, finite scores,
`done` exactly once, indices monotonic; the final verdict consistent with the
§22.1 gate on PASS and MISS CSVs; two same-seed `legacy`-quality runs produce
identical update streams except the measured `train_seconds` (G2; A2's
wall-clock exclusion — the v0.4 eff term is wall-clock-aware, §18.4), and the
`v04` preset is the default (SPEC.md 18.6); artifact payloads
— `best_spec.json` parses, `best_model.npz` loads with `allow_pickle=False`, `report.html`
contains both SVGs; error paths: missing file, `next()` before `start()`, rl
policy rejected with the CLI pointer); the app (skipped when streamlit is
absent, §3/§21.1 pattern — `AppTest` renders the page without exceptions,
then a 2-experiment run end-to-end via the run button: verdict shown, live
table + SVG markdown present, artifact download buttons present); the
`dashboard` launcher (the command list is correct and points at the in-repo
app file; the streamlit-absent path returns the install hint); and
`import autorefine` never pulls in streamlit. A1–A12 stay green.

### 23.4 CLI launcher

`autorefine dashboard [--port 8501] [--runs-dir runs]` — the sole
`subprocess` touch point (S1 exception, v0.9): it checks streamlit is
importable (clear `pip install autorefine[gui]` hint + exit 1 when not) and
spawns `sys.executable -m streamlit run <package>/dashboard_app.py`
headless, with `--runs-dir` forwarded through streamlit's `STREAMLIT_ARGS` —
on that fixed in-repo file only, never on user code. `import autorefine` and
every other command remain streamlit-free.

### 23.5 Milestone

**M12** — v0.9 visual dashboard (A13).

---

## 24. Input Modalities (v0.10 — implemented)

Beyond tabular CSV (§22): **images** and **audio clips** become *tasks* (G5).
A modality is just the shape of the features — `ImageTask` and `AudioTask`
implement the same minimal fitting protocol as `CsvTask` (`make_dataset` +
`score` + seed-derived splits), so the improver, spec space, v0.4 acceptance,
frontier, `eval`, `report`, and the dashboard all work on them **unchanged**.
Text / LLM-embedding inputs stay out of scope (v1; §16).

### 24.1 Optional dependencies (S1, the §3/§21.1 pattern)

- **`autorefine[image]`** → Pillow (≥9): decodes PNG / JPEG / BMP / GIF.
  `import autorefine` never imports PIL; constructing `ImageTask` without
  Pillow fails with a clear `pip install autorefine[image]` hint (the §23.4
  pattern, same blocked-import test shape as the streamlit rule).
- **`autorefine[audio]`** → soundfile (≥0.12, i.e. libsndfile ≥1.2): **MP3**
  decode. **WAV (16-bit PCM) uses the stdlib `wave` module — zero
  dependencies**; an MP3 file without soundfile fails with a clear
  `pip install autorefine[audio]` hint. `import autorefine` never imports
  soundfile either.

### 24.2 Shared rules (both media tasks — one source of truth, `tasks/media.py`)

- **Input = one directory.** Labels come from either
  - one **subfolder per class** — `<dir>/<label>/<file>`, or
  - an **`index.csv`** in the directory: column 1 = file path (relative to
    the directory, or absolute), column 2 = label (column names: any of
    `path`/`file` + `label`/`target`/`y`/`class`, else position);
- **head inference — identical to §22.1:** all-integer labels with 2–50
  distinct classes → `softmax` (sorted unique values → classes 0…K−1);
  otherwise `mse` (numeric labels via `index.csv`);
- **splits — identical to §22.1:** seed-derived permutation partitions the
  items into train / holdout / gen (default 80/10/10; `split_frac` is the
  non-train share, split evenly); prefix-matched split names
  (`gen*` / `train*` / holdout) so §18.3 block bootstraps work unchanged;
- features are standardized with **train-only** statistics (std 0 → 1);
- **score** = 100·accuracy (softmax) or 100·R² clamped at 0 (mse);
- `default_dataset_size` = train-split item count (§20.3 units).

### 24.3 `ImageTask` (`tasks/images.py`)

- **features:** deterministic resize to a square grid (default 32×32)
  grayscale → `(grid²,)` vector, in [0, 1]. No augmentation in v1 — the
  improver's spec space is the improvement axis, as for CSV.
- **formats:** `.png .jpg .jpeg .bmp .gif` (via Pillow).
- **errors:** no decodable items → `ValueError` with the label rules; Pillow
  missing → the §24.1 install hint; unreadable file → `ValueError` naming it.

### 24.4 `AudioTask` (`tasks/audio.py`)

- **formats:** `.wav` (16-bit PCM, stdlib `wave`; mono or stereo → mean
  downmix) and `.mp3` (soundfile, §24.1). Other extensions are rejected with
  a hint; non-16-bit WAV is rejected with a hint.
- **features:** a hand-rolled, **NumPy-only log-mel spectrogram** —
  STFT (Hann window, 25 ms window / 10 ms hop at the file's sample rate),
  power spectrum, 26-band mel filterbank (0 Hz → min(sr/2, 8 kHz)),
  log(1+x), mean over time → `(bands,)` vector. Deterministic given the file
  bytes (G2); no new dependency (S1).
- **errors:** too-short signal → `ValueError`; MP3 without soundfile → the
  §24.1 hint.

### 24.5 CLI & dashboard

- **`fit --data` now accepts a directory:** auto-detect image vs audio items
  (an `index.csv` labels either) or force with `--task csv|image|audio`
  (default `auto`; a file is always csv). The §22.1 gate (exit 0/2) and all
  other `fit` flags apply unchanged. `eval` / `report` are unchanged
  (registry + `task_config` round-trip, as for csv).
- **dashboard:** the sidebar path input accepts a **directory** (preview +
  run; `DashboardRunner` auto-detects the modality — `modality` arg to
  force it). File **upload** stays CSV in v0.10 (zip-upload of a folder =
  open question, not a commitment).

### 24.6 Sample data & acceptance (A14)

- `examples/make_sample_media.py` synthesizes runnable samples —
  `tone_clips/` (220 Hz vs 440 Hz WAVs, stdlib `wave`, zero deps) and
  `image_shapes/` (two geometric pattern classes, Pillow when available) —
  mirroring the churn-sample acceptance of §22.6.
- **acceptance:** `autorefine fit --data <either sample dir> --target 90`
  **PASSes** on a small budget; determinism (same seed → same features/
  splits) holds for both tasks; blocked-import hint tests; the core import
  stays PIL/soundfile-free (the §23 import test pattern); `fit --data DIR`
  auto-detects and a mixed directory without `--task` is a clean error.

### 24.7 Milestone

**M13** — v0.10 input modalities: images + audio as tasks (A14).

---

## 25. Modality-aware model families (v0.11)

**Goal.** Give the improver structurally better answers per modality: a
non-parametric family (`knn`) available on every task, and a convolutional
family (`convnet`) that exploits 2-D feature layout — the spatial grid for
images and the time×mel spectrogram for audio (the audio *temporal* model).
Both integrate with the existing loop, scoring, §19.3 ensembling, reports,
and dashboard with no protocol changes beyond two optional task attributes
and one optional task method.

### 25.1 Scope rules
- Zero new dependencies (NumPy only, §3).
- Both families keep the `forward(x) -> (n, n_out)` contract, so task
  scoring (§15/§22.1), the §19.3 ensemble, `eval`, block CI, and the
  dashboard work with no changes.
- Deterministic given the seed (G2); save/load is plain arrays only
  (`allow_pickle=False`, §9).
- The v0.10 flat-feature contracts are unchanged: audio stays 26-d
  time-mean for mlp/tree/boost/knn; image stays 1024-d.

### 25.2 `knn` — k-nearest neighbors (all tasks)
- `models/knn.py::KNN`: "training" = memorizing the standardized train
  split `(X, y)`; forward = k-NN over that split.
  - softmax head: majority vote over the k nearest neighbors → class-vote
    fractions `(n, C)`, with a deterministic ε = 1e-6 smoothing so the
    cross-entropy training loss stays finite;
  - mse head: mean of the k neighbors' targets `(n, 1)`.
- Deterministic ties: `np.argsort(kind="stable")` (equal distances → lower
  dataset index first); vote ties → lower class index (argmax convention).
- New spec field **`knn_k` ∈ {1, 3, 5, 11, 21}** (config `KNN_K_VALUES`),
  default **5** — the default keeps pre-v0.11 spec JSON loadable
  (back-compat, the §17 pattern).
- Family surface: `FAMILY_FIELDS["knn"] = (knn_k, model_family)`;
  `architecture` is ignored for knn (`validate` accepts any valid shape).
- Serialization: `X_train, y_train, k, n_out, head` → `best_model.npz`
  (plain arrays only).

### 25.3 `convnet` — small NumPy convnet over grid-structured features
- **Grid protocol (task side, all optional):**
  - class-level capability flag `grid_capable: bool` and instance-level
    concrete layout `feature_grid: (C, H, W)` — the convnet family is
    offered only for tasks with `grid_capable`;
  - `grid_dataset(n_points) -> (X (n, C, H, W), y)` — the train-split
    data in grid layout, standardized exactly like the flat split;
  - `ImageTask`: `feature_grid = (1, grid, grid)` (default (1, 32, 32));
    `grid_dataset` is a pure reshape of the flat 1024-d train rows — the
    flat path stays bit-identical to v0.10;
  - `AudioTask`: `feature_grid = (1, T, 26)` with
    **T = min(MAX_FRAMES = 128, max frame count over the dataset)** — the
    log-mel **spectrogram** (time × mel) is the audio modality layout and
    this is the audio *temporal* model. `log_mel_features` refactors into
    `log_mel_frames` (per-item (T_i, 26)) + time-mean
    `log_mel_features` (output bit-identical to v0.10). Items shorter
    than T are zero-padded on the time axis (deterministic; a padded frame
    reads as silence). Grid standardization reuses the flat path's
    per-band train mean/std — exactly consistent.
  - A `wants_grid` model's `forward` also accepts flat
    `(n, C·H·W)` input and reshapes it (robustness).
- **Model** (`models/convnet.py::ConvNet`): conv 3×3 (C→c1) + activation +
  2×2 avg-pool → conv 3×3 (c1→c2) + activation + 2×2 avg-pool → flatten →
  FC(32) + activation → linear head `(n, n_out)`. `architecture = (c1, c2)`,
  each in {4, 8, 16, 32} (config `CONV_FILTERS`); pooling applies only to
  dimensions ≥ 2 (guard for very short time axes); im2col forward/backprop;
  Glorot×`init_scale` init. FC hidden size 32 is fixed (deliberately kept
  off the spec surface).
- **Training**: `ConvNet` exposes the same `.layers` (list of `(w, b)`)
  + `.loss_and_grads(x, y, label_smoothing)` interface as `MLP`, so the
  mlp training loop (optimizer, LR schedule, gradient clipping, early
  stopping, weight decay, label smoothing, time cap) is shared verbatim.
  The mlp path itself stays bit-identical (the T2 pin is untouched).
- **Scoring / ensembling**: the task's `score()` routes the input by the
  model's declared contract — a `wants_grid` model gets grid-layout split
  rows; flat models get flat rows (today's behavior, unchanged). The
  §19.3 ensemble forms only from members sharing the top member's input
  contract (mixed contracts → the ensemble degenerates to the matching
  members; deterministic).
- **Rejection semantics**: a convnet spec on a non-grid task is a clean
  `SpecError` at train time — never a silent fallback.

### 25.4 Loop safety: invalid-spec rejection in `step()`
`AutoRefineEnv.step()` catches `SpecError` from spec construction +
training: the step returns `accepted=False, reason="invalid_spec"`,
`candidate_score=None`, spends no budget, is not added to the dedup set
(the spec may still be valid on another task), and is logged in the
experiment log. This closes a pre-existing crash path (an invalid spec
from an external gym agent; convnet-on-flat-task) where the loop used to
die.

### 25.5 Improver integration (the same 4-line pattern)
- `ModelSpec`: `MODEL_FAMILIES += ("knn", "convnet")`; new field
  `knn_k` (default 5); validate: knn ignores `architecture`, convnet
  requires a 2-tuple from `CONV_FILTERS`.
- `catalog.py`: new family values; `knn_k` field values; convnet
  architecture pairs; `FAMILY_FIELDS` for both; `relevant_families(
  task_name)` — task-aware family offering (excludes `convnet` when the
  task class is not `grid_capable`), consulted by the bandit for the
  `model_family` field; search/RL policies may still propose convnet on
  flat tasks → the §25.4 rejection is the safe, logged outcome.
- `actions.py`: `knn_k` sampler; family-consistent `architecture`
  coercion for convnet/knn.
- `gym.py` / `rl_policy.py`: pick up the new values/fields through the
  catalog (embedding follows the catalog, as since v0.4).
- CLI `eval`: the family→loader mapping gains `knn` → `KNN.load`,
  `convnet` → `ConvNet.load`.
- `fit` / dashboard: unchanged — the families are explored autonomously.

### 25.6 Non-goals (v0.11)
>2 conv layers, pooling other than 2×2 avg, dropout / batch-norm, 1-D
convnets for flat tasks, per-modality bandit priors, model merging (v1
non-goal stands), text modality (Q8 stands), attention / seq2seq — the
padded fixed-T grid is the audio temporal surface for v0.11.

### 25.7 Acceptance (A15)
- Protocol: KNN determinism/ties/round-trip; ConvNet determinism/
  round-trip/gradient sanity; both save/load with `allow_pickle=False`.
- Integration: knn branch on a csv fixture; convnet branch on image and
  audio grid fixtures (score ≥ 80 on the bar/tone fixtures); convnet on a
  flat task is a clean `SpecError`; `step()` rejection (no crash, budget
  intact, env usable).
- Contracts: the audio flat path is bit-identical to v0.10 (state_dim 26;
  the existing §24 tests stay green); the image grid reshape equals the
  flat features; the ensemble contract filter is deterministic.
- End-to-end: `fit` on the bundled image sample dir still **PASS**es the
  §22.1 gate; the improver trains a `knn` spec to ≥ 90 on the tone
  sample; a `convnet` spec (the temporal model) reaches ≥ 80 on the tone
  sample; full suite green; the mlp T2 pin untouched.

### 25.8 Milestone

**M14** — v0.11 modality-aware model families: `knn` + `convnet`
(image grid + audio spectrogram) + invalid-spec rejection (A15).

## 26. Dashboard decision views (v0.12)

The v0.9 dashboard shows *what* happened (scores, budget, verdict). This
section adds *why the improver did what it did* — four views derived
entirely from data the loop already records (the runner's update stream,
which extends `experiments.jsonl` with the free duplicate rejections).
**Zero new logging, zero new dependencies**; the core stays streamlit-free
and NumPy-only (SPEC.md 3, §23.1).

### 26.1 D1 — per-field win-rate (both policies)
`field_stats(updates)` (dashboard.py, pure): credit each update's
`mutation` fields with that update's `accepted` outcome — exactly the
(field, accepted) credit pairs the bandit consumes in SPEC.md 17. Output
`{field: {trials, wins, win_rate}}`, keys lexicographically sorted;
`win_rate = wins / trials`. The app renders it as a live bar chart
(streamlit-native `st.bar_chart`, a streamlit dependency — app-only).

### 26.2 D2 — per-step spec diff
Each step's mutated fields as `field: old → new`: `old` is the value in
the best spec *before* the step, `new` the value in the proposed spec;
a value absent from that spec is `None` (rendered as "—"). Carried as
the update's `spec_diff` list of `{field, old, new}`; shown in the
per-step note, as a `diff` column of the experiment table, and in the
restored view (SPEC.md 23.2).

### 26.3 D3 — mutation timeline (SVG)
`svg_mutation_timeline(rows)` (plotting.py, pure, valid XML like the
§21.2 charts): rows = the union of mutated fields (lexicographic),
columns = update index 1..n; a cell is drawn when that field was mutated
in that update — green accepted, red scored-rejected, grey unscored
(duplicate / invalid-spec rejection). Makes the cold-start burst (every
field once) and the shift to exploitation visible at a glance.

### 26.4 D4 — bandit UCB trace (bandit policy only)
`ucb_trace(updates, alpha)` (dashboard.py, pure): replays the bandit's
credit rule (SPEC.md 17) over the update stream. After crediting update
i, every field credited so far gets
`wins[f]/t[f] + sqrt(alpha * ln(max(1, total)) / t[f])` (the bandit's
exact UCB, SPEC.md 17); fields not yet credited are `None` (rendered as
a chart gap). `alpha` is the runner's bandit alpha (default 1.0).
The update's `ucb` key holds the per-field values after that update;
the **search** policy sets `ucb` to `None` (UCB is a bandit concept)
and the app renders the trace only for bandit runs.

### 26.5 App rendering
Live section (after the experiment table / best-score chart): D1 bar
chart, D3 timeline SVG, D4 UCB line chart (bandit only); D2 in the
per-step note and the table's `diff` column. Result section: final D1
bars, D3 SVG, and the full D4 trace; the restored view re-renders all
four from the stored updates — no new run state. All four views are pure
functions of the update stream (G2): same-seed runs produce identical
views, and every update stays JSON-safe (SPEC.md 23.1).

### 26.6 Non-goals (v0.12)
No new rows in `experiments.jsonl` (the views are derived); no
RL-policy views (the app stays bandit/search, SPEC.md 23.1); no
multi-seed variance plots; no matplotlib (NumPy core; hand-rolled SVG
or streamlit-native charts only).

### 26.7 Acceptance (A16)
- The view functions are pure and deterministic: same update stream →
  bit-identical output; every update still JSON-safe.
- `field_stats`: hand-computed trials/wins/win_rate on a fixed stream;
  sorted keys; empty stream → `{}`.
- `spec_diff`: correct old/new per field, absent value → `None`, empty
  mutation → `[]`.
- `ucb_trace`: uncredited fields `None`; credited values match the
  SPEC.md 17 formula by hand computation (alpha = 1.0).
- Timeline SVG: valid XML (starts `<svg`), one cell per (step, field)
  mutation, one fill per outcome class, empty case is header text.
- Runner: every update carries `field_stats` / `spec_diff` / `ucb`
  (bandit: per-field value series; search: `None`); `finish()` carries
  final `field_stats`, `ucb_trace`, `timeline_svg`; the existing
  same-seed determinism test (full update dicts, SPEC.md 23.1) stays
  green.
- App (streamlit optional, §3/§21.1 pattern): the end-to-end run renders
  all four views for bandit and **no** UCB chart for search; A13–A15
  stay green; `import autorefine` never pulls in streamlit.
- Full suite green; the T2 and §18.7 pins untouched.

### 26.8 Milestone

**M15** — v0.12 dashboard decision views: field win-rate, spec diff,
mutation timeline, bandit UCB trace (A16).

---

## 27. Dashboard acceptance-gate views (v0.13)

The v0.12 decision views (SPEC.md 26) explain *why the improver chose
what it chose*. This section explains *how the acceptance gate decided*
(SPEC.md 18.6, §22.1) — three views derived entirely from data the loop
already records: the runner's update stream (per-step candidate score,
outcome, gen gap, SPEC.md 23.1) and, for parity curriculum runs, the
experiment log plus the summary's ladder history (SPEC.md 20.1).
**Zero new logging, zero new dependencies**; the core stays streamlit-free
and NumPy-only (SPEC.md 3, §23.1).

### 27.1 G1 — candidate score strip
`svg_score_strip(rows)` (plotting.py, pure, valid XML like the §21.2/§26.3
charts): one horizontal lane per outcome class — accepted (green),
scored-rejected (red), unscored (grey: the `duplicate` /
`duplicate_stall` / `invalid_spec` rejections, which carry no score,
SPEC.md 6 R3 / §25.4) — and one dot per update: scored candidates at
their exact `candidate_score` on a fixed 0–100 score axis, unscored
rejections stacked in a labelled "no score" zone at the left edge. Each
dot carries a tooltip (step, class; the exact reason for unscored), and
the legend carries the per-class counts. Answers "are the rejections
near-misses or garbage" — i.e. whether the reward signal is still
informative — which the best-score line alone hides. Empty stream →
header text, like the other charts.

### 27.2 G2 — score vs gen-gap scatter
`svg_score_gap_scatter(rows)` (plotting.py, pure): one dot per scored
update at (`candidate_score`, `gen_gap`), colored by outcome (accepted
green, scored-rejected red), on the fixed 0–100 score axis and a
data-driven gap axis. The §18.5 tolerance boundary `gen_gap = 0.05 ·
score` is drawn as a dashed reference line — the region above it is
where the overfit penalty bites — making "improved on holdout but
overfit, so it was rejected" visible at a glance. Unscored updates
contribute no dots. Empty (no scored updates) → header text.

### 27.3 G3 — curriculum ladder curve (parity runs)
`svg_ladder_curve(entries, levels)` (plotting.py, pure): the best-score
curve over an experiment log that includes curriculum step-ups. Sequence
= the `baseline` / `experiment` / `curriculum` rows in log order; the
running best is the baseline's score, a candidate's score when accepted,
unchanged on rejection, and `new_baseline_score` at each step-up (the
re-baselining, SPEC.md 20.1). Each step-up gets a vertical marker labeled
with its `n_bits` / `p_flip` from `summary["curriculum"]["levels"]`
(matched by order — one level row per step-up, SPEC.md 20.1). Shows why
the curve plateaus against the old ceiling, then drops and climbs at
each difficulty change. Rendered only when the run actually has a
ladder: `finish()` sets `ladder_svg` to `None` (not an empty-case SVG)
unless `summary["curriculum"]` is present, `html_report` adds a
"Curriculum ladder" section only then, and `report --plot` writes
`ladder_curve.svg` only then. Non-curriculum runs are byte-identical to
v0.12's output.

### 27.4 Rendering
- **Dashboard app** (SPEC.md 23.2): G1 + G2 as live placeholders,
  recomputed from the update stream so far at each step (the §26.5 live
  pattern; the views are global over the stream, so no new per-update
  keys are added), and again in the result's "Decision views"
  subheader — G3 as well, when present. The restored view re-renders
  everything from the stored result (SPEC.md 23.2 persistence).
- **Report**: `report.html` (§22.2) gains the G3 section when the
  summary carries a ladder; `report --plot` (§21.2) gains
  `ladder_curve.svg` under the same condition.

### 27.5 Non-goals (v0.13)
No new `experiments.jsonl` rows and no new update-stream keys (the views
are derived); no built-in-task mode for the dashboard (G3 renders in the
app only if the run has a ladder — parity runs stay CLI-driven,
SPEC.md 20.1); no per-experiment CI-margin / eff-decomposition charts —
the gate's per-step inputs (Δeff, SE, penalty terms) stay in the log and
summary, this section visualizes the gate's outcomes over the run; no
new dependencies (hand-rolled SVG or streamlit-native, §3/§21.1 pattern).

### 27.6 Acceptance (A17)
- The view functions are pure and deterministic: same inputs →
  bit-identical SVG; every output is valid XML (starts `<svg`); empty
  cases render header text, not an error.
- G1: hand-computed stream → one dot per update, in the right lane
  (accepted / scored-rejected / unscored), unscored dots in the no-score
  zone with the exact reason in the tooltip, legend counts per class;
  scored dots at their hand-computed x position; empty stream → header.
- G2: hand-computed stream → one dot per scored update at its
  hand-computed (score, gen-gap) position with the right outcome color;
  the 5%-of-score tolerance line is present; unscored updates add no
  dots; empty → header.
- G3: hand-computed entries + levels → one point per log row, the
  running-best series matches a hand computation (accepted bumps,
  rejected keeps, step-up resets to `new_baseline_score`); one marker
  per level row with its `n_bits`/`p_flip` in the label; no step-ups →
  empty case. A real parity curriculum run (the SPEC.md 20.1 recipe)
  produces one marker per summary level row, and `html_report` embeds
  the ladder while a non-curriculum summary does not gain the section.
- Runner: `finish()` carries `strip_svg` / `scatter_svg` (always) and
  `ladder_svg` (`None` for non-curriculum runs); same-seed legacy runs →
  bit-identical gate views; the existing A16 assertions (update keys,
  determinism, JSON-safety) stay green.
- App (streamlit optional, §3/§21.1): the end-to-end run renders G1 + G2
  live and in the result; A13–A16 stay green; `import autorefine` never
  pulls in streamlit.
- Full suite green; the T2 and §18.7 pins untouched.

### 27.7 Milestone

**M16** — v0.13 dashboard acceptance-gate views: candidate-score strip,
score-vs-gen-gap scatter, curriculum ladder curve (A17).

---

## 28. Dashboard learning views (v0.14)

The v0.13 gate views (SPEC.md 27) explain *how the gate decided*; these views
explain *what the model is actually learning* — its loss trajectory, its
per-class behavior on the holdout, the items it still gets wrong, and the
architecture that won. Four views, C1–C4 (matching the §26 D* / §27 G* naming
pattern). C1 is the only one with a core change (the trainer returns a bounded
`loss_history`); C2–C4 render over data the run already has — C2 is one extra
forward pass, C3/C4 none. **No new core dependencies** (NumPy only; PIL optional
for C3 image thumbnails, soundfile optional for C3 MP3 — the §3/§21.1 lazy-import
pattern, so `import autorefine` stays clean); **no new JS/GUI** beyond the
existing Streamlit app (SPEC.md 3); the T2 determinism pin (§18.7) stays green
(the C1 probes are forward-only — no RNG, no weight changes).

### 28.1 C1 — per-experiment training curves
`train()` (SPEC.md 4/15) now returns `TrainResult.loss_history`: a bounded
list of `{"step": int, "train": float, "holdout": float}` records, capped at
`LOSS_HISTORY_MAX = 48`. The shape per family:
- **neural (mlp / convnet)** — sampled at deterministic even steps over
  `train_steps` (`T`): `M = min(48, T)`, `M == 1 → {1}`, else
  `t_i = 1 + round(i·(T−1)/(M−1))` for `i in range(M)`. "train" is the batch
  loss at that step; "holdout" is a forward-only probe on the deterministic
tail (`max(1, round(VAL_FRACTION·n))` rows — the same tail early stopping uses,
reusing the val split when patience > 0, else the last `n_tail` rows).
- **tree / boost** — exactly one record `{"step": 1, "train": final_loss,
  "holdout": tail-probe}` (blocking fits have no per-step loop).
- **knn** — one record where both "train" and "holdout" are the loss on the
  first `m = min(n, 256)` rows — the exact rows the reported final loss uses
  (a memorized model, so they are equal by construction).
- degenerate time-cap-0 early returns stay empty (the default `[]`).

The env logs `loss_history` on the baseline row (reset), the experiment row
(step, and the `info` dict), and the curriculum re-baseline row; duplicate and
invalid-spec rows carry none (no key). `svg_loss_curves(history)` (plotting.py,
pure, valid XML): two polylines (train = `_LINE`, holdout = `_BASELINE`),
data-driven loss axis, step axis, a legend, and an empty case → header text.
`finish()` builds `loss_curves: {label: history}` with labels "baseline",
"experiment 1…", "curriculum 1…" (per-kind counters; entries without a
non-empty `loss_history` are skipped). The app renders a selectbox over
`res["loss_curves"]` → the SVG in a "Learning views" subheader (result-only,
SPEC.md 23.2). `report.html`'s experiment table gains a per-row embedded curve
SVG column ONLY when any entry has `loss_history` — old-run reports stay
byte-identical (SPEC.md 22.2).

### 28.2 C2 — final-model diagnostics
New module `diagnostics.py`: `holdout_diagnostics(task, model, n=200) ->
dict | None`. Returns `None` when `task.head != "softmax"`, or the task exposes
no `holdout_rows` (episode tasks: cartpole/parity/sine/gridnav — non-goal,
SPEC.md 28.5), or `class_values` is falsy, or there are 0 holdout rows.
Otherwise `{"n": int, "correct": int, "per_class": [float 0–100],
"confusion": [[int]], "class_counts": [int], "class_labels": [str]}`.
Per-class accuracy is `per_class[i] = 100 · confusion[i][i] / class_counts[i]`
(0.0 for an empty class); invariants: each row sums to `class_counts[i]` and
`correct` = the diagonal sum (the confusion matrix is built with `np.add.at` —
deterministic, correct for repeated indices).

New task protocol method `holdout_rows(n, model=None)` on CsvTask / ImageTask /
AudioTask: clamps `n` like `score()` and returns the holdout `(x, y)`, routing
to grid rows when `model.wants_grid` (mirrors each `score()`'s routing,
SPEC.md 25.3). `fit` output gains per-class accuracy lines + a weakest-class
hint ("the model fails on class X") after the gate line. `report --plot`
(SPEC.md 21.2) writes `per_class.svg` + `confusion.svg`; `report.html` (22.2)
gains a "Holdout diagnostics" section (per-class bars + confusion SVG) when the
data is present. plotting.py: `svg_per_class_bars(diag)`,
`svg_confusion_matrix(diag)` (both pure, valid XML, empty-safe).

### 28.3 C3 — error gallery (media tasks)
`holdout_errors(model, n=200, n_max=8)` on ImageTask / AudioTask: the holdout
items the model misclassifies (argmax mismatch), in holdout order, ≤ `n_max`
items — `{"file", "path", "label", "predicted"}`; audio adds
`{"waveform": [≤512 floats], "sample_rate": int}` (a deterministic block-mean
downsample of the decoded signal via the existing `_load_signal`). Labels come
from the task's canonical `class_values`, not the raw item labels.
plotting.py: `svg_audio_waveform(values, sample_rate=None, title="")` (pure,
data-driven y-axis, a zero line when 0 is in range, empty → header). App: an
image thumbnail (`st.image`, PIL lazy) or an inline waveform SVG, each captioned
`file — true → predicted`. `report.html` (22.2): an "Error gallery" section —
image thumbnails as base64 PNG data URIs (PIL lazy-import; a text-list fallback
when PIL is absent or a decode fails), audio inline waveform SVGs + captions.
CSV tasks carry no gallery (SPEC.md 28.5); an empty gallery (0 errors) renders
no section.

### 28.4 C4 — spec → architecture diagram
plotting.py: `svg_architecture(spec, state_dim, n_out)` — a small horizontal
annotated block diagram: an input block (`state_dim`) → the family body → an
output block (`n_out`). Per family: mlp one rect per hidden layer (depth-0 →
"linear"); convnet the verified conv/pool/FC pipeline with its `c1`/`c2`;
tree "N trees (depth D), bagged"; boost "N rounds (depth D), boosted" (in both
`N = train_steps // 100` clamped 2..50 as in the trainer, `D = architecture[0]`);
knn "k = knn_k (nearest neighbors)". An annotations row carries the optimizer
(always) plus the LR schedule, early stopping, gradient clipping, and label
smoothing when non-default. ASCII-only (e.g. "3x3", "->"). Rendered in the
dashboard result (task from `self.env.task`, spec from `self.env.best_spec`),
in `report.html` (the task reconstructed from the summary's `task`/`seed`/
`task_config` exactly as `eval` does, incl. the curriculum fallback), and
`report --plot` writes `architecture.svg`.

### 28.5 Non-goals (v0.14)
No JS/GUI beyond the existing app; no episode-task or mse diagnostics (C2 is
classification-only — cartpole/parity/sine/gridnav carry no `holdout_rows`);
no CSV error gallery (C3 is media-only); no live per-step curve chart in the
app (C1 is result-only — live would need re-rendering a growing curve every
step, out of scope); no unbounded loss history (cap 48); no new core
dependencies (PIL/soundfile stay optional, §3/§21.1).

### 28.6 Acceptance (A18)
- **C1**: hand-computed `t_i` sampling for a known `train_steps`; records are
  finite with increasing steps; tree/boost/knn each yield exactly one record;
  the env's baseline / experiment / curriculum log rows carry `loss_history`
  (duplicate / invalid rows carry none); the runner's updates carry it
  JSON-safe; `finish()`'s `loss_curves` labels are correct per kind;
  `svg_loss_curves` is valid/deterministic with hand-computed point coordinates
  and an empty case; `report.html` embeds per-row curves when present and stays
  byte-identical when absent; the T2 pin (§18.7) stays green.
- **C2**: `holdout_diagnostics` invariants (row sums = class_counts, correct =
  diag sum, the per-class formula) on a fixed CSV; `None` for an mse task and
  for a task without `holdout_rows`; `fit` output has the per-class lines +
  weakest-class hint; `report --plot` writes `per_class.svg` / `confusion.svg`;
  the html section renders; `html_report(summary, entries)` is byte-identical
  when the new kwargs are `None`.
- **C3**: `holdout_errors` returns ≤ `n_max` items with labels from
  `class_values`; audio waveforms are ≤ 512 floats; `svg_audio_waveform` is
  valid and empty-safe; a media run's html carries the gallery.
- **C4**: `svg_architecture` shows the right blocks + annotations per family
  (mlp / convnet / tree / boost / knn); the html section and `report --plot`
  file are present.
- **Runner/app (A18)**: `finish()` carries all new keys (`loss_curves`,
  `diagnostics`, `per_class_svg`, `confusion_svg`, `error_gallery`,
  `arch_svg`); the app renders the "Learning views" subheader; `import
  autorefine` never pulls in PIL / streamlit / soundfile.
- Full suite green; the T2 and §18.7 pins untouched.

### 28.7 Milestone

**M17** — v0.14 dashboard learning views: training curves, final-model
diagnostics, error gallery, architecture diagram (A18).

---

## 29. Multi-run / policy views (v0.15)

The v0.13/v0.14 views explain one run. These two views answer the two
questions a single run cannot: **is the improvement real** (D1 — the same
budget under N seeds) and **what is the RL policy actually doing** (D2 — the
per-step action distribution and per-task returns). D1 reuses the existing
`DashboardRunner` loop (SPEC.md 23.1); D2 reuses the existing `train_policy` /
`train_multi_policy` machinery (SPEC.md 15/20.2) — no new training, only new
rendering and an opt-in trace. **No new core dependencies** (NumPy only; the
two new SVGs are hand-rolled valid XML like the §21.2/§26/§27/§28 charts); the
D2 trace is **off by default** so every existing `train_policy`/
`train_multi_policy` caller and the §20.2 multi-task pin stay bit-identical;
RL **stays CLI-only** (SPEC.md 23.1) — the app gains only an opt-in
*precomputed* view that renders artifacts a `policy-report` run already wrote
(it never runs RL live).

### 29.1 D1 — seed-variance box plot
`plotting.py`: `svg_seed_variance(seeds, target=None, width=640,
height=360)` (pure, valid XML, deterministic G2, empty → header text, ASCII
text). `seeds` is a list of per-seed dicts, each carrying `baseline` (float),
`final` (float) and optionally `seed` (int). The chart:
- a fixed vertical **0–100 score axis** (`y(s) = T + ph·(1−s/100)`), like the
  §27.1/§27.2 fixed-score charts;
- a **box-and-whisker** over the `final` scores: whiskers to min/max, the box
  body spanning q1–q3, a median line — quartiles from a hand-rolled
  linear-interpolation quantile `_quantile(sorted_vals, p)` (position
  `p·(n−1)`, the NumPy `linear` method; n=1 → that single value). The median,
  q1, q3, min, max are also written as text annotations so the SVG is
  self-documenting and assertable;
- one **paired baseline → final marker per seed** (a diamond at `baseline`, a
  circle at `final`, a connecting line), laid out left-to-right in input order
  and labelled with its `seed` when present — this is the "is the improvement
  real?" half (the spread of the finals plus where each started);
- a dashed **target line** at `y(target)` with a `target X.XX` label when
  `target` is a finite number;
- a summary line `n seeds = N · passing = P/N` (`P` = seeds whose `final >=
  target`; when `target` is None it reads `passing = -`).

`dashboard.py`: `DashboardRunner.seed_sweep(seeds, on_update=None) -> list`
— the same flags re-run under N seeds. For each seed it builds a **fresh**
`DashboardRunner` from `self`'s parameters (a private `_clone(seed)`),
drives `start()` → `next()`…→ `finish()` (the exact §23.1 loop; `on_update`
is forwarded per step), and appends one JSON-safe dict: `{seed, baseline,
final, target, pass, verdict, experiments_run, run_dir}`. Empty `seeds` →
`[]`. The caller's own runner is untouched (each sweep run is isolated, so
this never disturbs a live single-seed run or the §18.7 pin).

`cli.py`: `variance --data PATH [--label COL] [--task auto|csv|image|audio]
[--split-frac F] [--target 95.0] --seeds N [--seed 7] [--experiments 30]
[--policy bandit|search] [--search-quality v04|legacy] [--max-seconds S]
[--max-train-seconds S] [--runs-dir runs]` (SPEC.md 22.1 task resolution,
reusing `_resolve_fit_task`). `--seeds N` means the N distinct seeds
`seed, seed+1, …, seed+N-1` (base `--seed`, default 7). It resolves the task
as `fit` does, builds a `DashboardRunner`, calls `seed_sweep`, prints a
text summary (per-seed `seed baseline final PASS/MISS`; then median/min/max
of the finals, how many beat their own baseline, how many pass the target),
and writes `seed_variance.svg` + `seed_sweep.json` into `--runs-dir`. `--json`
prints the sweep as pure JSON to stdout (SPEC.md 21.2 pattern). MISS seeds
print as MISS but the command exits 0 (a variance report is a measurement,
not a gate — the per-seed `verdict` carries the gate).

The app gains an opt-in **Seed variance** panel (a seed-count input + a
button): when a data path is set it runs `seed_sweep` and renders
`svg_seed_variance`. It is result-only and inert until the button is pressed
(the §23.2 persistence flow is unchanged).

### 29.2 D2 — RL policy view (CLI-surface; RL stays CLI-only)
`rl_policy.py` (SPEC.md 15/18.2/20.2):
- `MetaRLPolicy.probabilities(env_state) -> np.ndarray` — the full
  action-probability vector (length `n_actions`, the 77-action catalog,
  SPEC.md 19.4/25.x) that `propose()` would sample from at this state: the
  exact masked softmax (family mask per SPEC.md 18.2, task bias per
  SPEC.md 20.2). **Pure** — no sampling, no RNG draw, no state change; masked
  actions read ≈ 0 and the relevant actions sum to 1.
- `propose()` additionally records `self.last_proposal = (action_index, probs)`
  (one attribute assignment; the RNG stream and the returned spec are
  unchanged, so A1–A18 and the §20.2 pins stay bit-identical).
- `train_policy(env, policy, n_episodes, verbose=False, trace=None)` and
  `train_multi_policy(envs, policy, episodes_per_task, verbose=False,
  trace=None)` — when `trace is not None`, append one record per step:
  `{"task": str, "action": int, "probs": [float]·n_actions, "reward":
  float}` (the task label from `env_state["task"]`, SPEC.md 20.2). The
  return dicts are **unchanged** when `trace is None` (the default), so every
  existing caller is bit-identical (G2).

`plotting.py` (pure, valid XML, deterministic, ASCII, empty → header):
- `svg_action_probabilities(trace, top_k=8, n_steps=12, width=640)` — the
  last `n_steps` steps of the trace as horizontal bar rows: each row shows
  that step's **top-`top_k` actions by probability** (ties broken by action
  index) as bars plus one `rest` bar (the sum of the tail), on a fixed 0–1
  probability axis. Each bar is labelled with the catalog action
  (`field=value`; tuple values joined, e.g. `architecture=16,8`); the sampled
  action's bar is highlighted and the row caption carries the step index and
  reward.
- `svg_task_returns(returns, width=640, height=360)` — one line per task
  from `returns` (`{task: [episode_return, …]}`, the exact shape of
  `train_multi_policy["episode_returns"]`, SPEC.md 20.2), data-driven axes,
  a per-task legend. A single-task trace is `{task: [..]}` (one line).

`cli.py`: `policy-report --task TASK --seed 7 [--episodes 3] [--experiments
20] [--max-seconds S] [--max-train-seconds S] [--runs-dir runs] [--multi]
[--tasks sine-v1,parity-v1,cartpole-v1] [--top-k 8] [--n-steps 12]`. Without
`--multi` it trains `MetaRLPolicy(seed)` on one task via `train_policy(...,
trace)`; with `--multi` it builds one env per `--tasks` entry and trains
`MetaRLPolicy(seed, task_names=...)` via `train_multi_policy(..., trace)`
(SPEC.md 20.2). It writes `action_probabilities.svg`, `task_returns.svg`, and
the raw `policy_trace.json` + `policy_returns.json` into `--runs-dir`, and
prints a text summary (per-task episode returns, `policy_updates`, trace
length, and the final step's top actions).

The app gains an opt-in **RL policy view (precomputed)** panel: given a
`policy-report` output directory it renders `action_probabilities.svg` +
`task_returns.svg`. It **never runs RL live** — this keeps RL CLI-only
(SPEC.md 23.1); only the precomputed artifacts are shown.

### 29.3 Non-goals (v0.15)
No live RL in the app (the §23.1 boundary is unchanged; the app renders only
precomputed policy artifacts — **a design decision the user should confirm**);
no live per-step RL stream in the app (that would be the §23.1 contract
change); no new core dependencies; no change to `train_policy`/
`train_multi_policy` return dicts when the trace is off (bit-identical);
no box-plot *statistical* tests beyond hand-computed quartile geometry
(confidence bands / outliers are not the point — the paired markers are);
the seed sweep is **opt-in** (default off) so a single-seed `fit`/`run`/live
dashboard and the T2 (§18.7) + §18.7 legacy pins are untouched.

### 29.4 Acceptance (A19)
- **D1 (SVG)**: `svg_seed_variance` is valid XML; hand-computed quartile
  geometry on a known `finals` set (median/q1/q3/min/max via the linear
  quantile) is asserted both in text and by parsing the box rect; the paired
  baseline→final markers and the target line are present when given; the
  `passing = P/N` line is correct; the empty case renders a header + message
  and is valid XML.
- **D1 (runner)**: `seed_sweep([..])` returns one dict per seed with finite
  `baseline`/`final`, the right `target`/`pass`/`verdict`, and JSON-safe
  values; `seed_sweep([])` → `[]`; two sweeps under the same seeds are
  bit-identical in `baseline`/`final` (G2 — the run dirs differ by
timestamp/counter, the scores do not); the caller's own runner is left
  untouched (phase, best score) after a sweep.
- **D1 (CLI)**: `variance` resolves the task as `fit`, exits 0, writes
  `seed_variance.svg` (valid XML) + `seed_sweep.json` (parses; one entry per
  seed); `--json` prints pure JSON; a MISS seed still exits 0 and prints
  MISS; an unknown/missing `--data` exits 1 with a clear error.
- **D2 (policy)**: `probabilities(env_state)` has length `len(ACTIONS)`
  (asserted from the catalog, not hardcoded), is finite, the relevant
  actions sum to 1 (≈), and masked actions ≈ 0; `propose()`'s RNG stream is
  unchanged (two same-seed policies still sample identical action sequences);
  `train_policy`/`train_multi_policy` with `trace=None` return dicts are
  byte-identical to before (the §20.2 multi-task pin and A1–A18 stay green);
  with `trace=[...]` the trace has one record per step, each with `task`, a
  valid `action`, a `probs` vector of the right length summing to 1 over the
  relevant actions, and a finite `reward`.
- **D2 (SVG)**: `svg_action_probabilities` is valid XML, shows the top-K bars
  + a `rest` bar per row with correct labels, highlights the sampled action,
  and is empty-safe; `svg_task_returns` is valid XML, one line per task with
  a legend, and is empty-safe.
- **D2 (CLI)**: `policy-report` (single and `--multi`) trains, captures the
  trace, exits 0, and writes `action_probabilities.svg` + `task_returns.svg`
  (both valid XML) + `policy_trace.json` + `policy_returns.json` (both
  parse); `--multi` with mismatched env/task counts errors clearly (the §20.2
  validation).
- **App (A19)**: the app renders the opt-in **Seed variance** panel (a
  button-driven `seed_sweep` → `svg_seed_variance`, no exceptions) and the
  opt-in **RL policy view (precomputed)** panel (renders precomputed SVGs
  from a given dir; never runs RL live); streamlit-optional (skipped when
  absent); the existing `run`/`clear` flow and the §23.2 persistence stay
  green; `import autorefine` never pulls in streamlit.
- Full suite green; the T2 and §18.7 legacy pins untouched; A1–A18 stay green.

### 29.5 Milestone

**M18** — v0.15 multi-run / policy views: seed-variance box plot (CLI +
runner + app) and the RL policy view (probabilities + trace + action-prob /
task-return SVGs + CLI + precomputed app panel) (A19).

---

## 30. Comprehension visuals II (v0.16)

The v0.12–v0.15 views explain one run, its gate, its learning, and its
multi-run / policy behavior. This section adds six more views — all derived
from data the loop already produces, plus one small logging addition: the
block-bootstrap `std` (SPEC.md 18.3) that the §18.5 CI gate already computes
per experiment is now *logged* with the entry and *shown* as a band on the
score curve. **No new core dependencies** (NumPy only; the new SVGs are
hand-rolled valid XML like the §26/§27/§28/§29 charts); **no search,
training, or acceptance changes**; RL **stays CLI-only** (SPEC.md 23.1) —
the app renders one more precomputed artifact; `import autorefine` stays
clean (SPEC.md 3). Every renderer is pure and deterministic (G2),
ASCII-only in its text, and empty-safe (header + message).

### 30.1 V1 — per-seed best-score curves
`plotting.py`: `svg_seed_curves(seeds, target=None, width=640,
height=360)` — the running best score of each seed over the sweep, so the
variance view (SPEC.md 29.1) answers *when* the seeds diverge (early noise
vs late divergence). Input: a list of dicts each carrying `curve` (a list
of ≥ 1 finite floats) and optionally `seed` (int). The chart:
- a fixed vertical **0–100 score axis** (same layout as `svg_seed_variance`);
- one **polyline per seed** (input order), colored from the §29.2 palette
  (`_LINE, _BASELINE, _LADDER, _ACCEPTED, _SCORED_REJ, _REJECTED`, cycling),
  x = point index `0 … M-1` where `M` = the longest curve (point 0 = the
  baseline);
- a **legend** (swatch + `seed <n>` per line, input order);
- a dashed **target line** + `target X.XX` label when `target` is finite;
- a summary line `n seeds = N - steps = M` (`M` counts the baseline point);
- rows without a finite `curve` are skipped; empty → header +
  `no seed curves in the sweep`.

`dashboard.py`: `DashboardRunner.seed_sweep` extends each per-seed dict
(SPEC.md 29.1 contract) with exactly one new key, `"curve"`:
`[baseline, best after step 1, …]` — the baseline from `start()`'s
`baseline_score`, then each `next()` update's `best_score` (free duplicate
rejections, R3, are steps too, so the curve has one point per `next()` call
plus the baseline). `curve[0] == baseline`, `curve[-1] == final`. The A19
sweep-dict test is extended with this key (the rest of the contract is
unchanged).

`cli.py`: `variance` additionally writes `seed_curves.svg` (and names it in
its output listing; `--json` mode is unchanged). The app's **Seed
variance** panel renders `svg_seed_variance` + `svg_seed_curves`.

### 30.2 V2 — field × value win matrix
`dashboard.py`: `field_value_stats(updates) -> dict[str, dict[str, dict]]`
— the v0.12 per-field win-rate (SPEC.md 26.1) refined to *which value won*
(e.g. `architecture=16,8` vs `8,16`). For each update and each field in its
`mutation`, the new value is read from that update's `spec_diff` (SPEC.md
26.2: `{field, old, new}`); the cell `(field, value)` is credited with the
update's `accepted` outcome. Value keys are normalized strings (`None` →
`"-"`, list/tuple values joined with `,`, else `str(v)`); fields and values
are returned sorted (values numeric-first: a value that float-parses sorts
by its number, else alphabetically). Returns
`{field: {value: {"trials": int, "wins": float, "win_rate": float}}}`.
Pure over the update stream (G2); `field_stats` (SPEC.md 26.1) stays the
per-field rollup — the per-field trial/wins sums of `field_value_stats`
equal `field_stats`' whenever `spec_diff` is present.

`plotting.py`: `svg_field_value_matrix(stats, width=640, row_h=28)` — one
row per field (sorted), one cell per value: a green cell (`_ACCEPTED`) with
`fill-opacity = 0.10 + 0.90·win_rate`, the `wins/trials` text centered
(white when `win_rate ≥ 0.5`, axis color otherwise), and a `<title>`
`field value: wins/trials (rate)`. Field names left of the cells; cell
widths shrink to fit the widest row; a legend line `cell = wins/trials ·
fill = win rate`. Empty → header + `no field values credited`.

`dashboard.py`: `next()` attaches `field_value_stats` to each update (like
`field_stats`, SPEC.md 26.5); `finish()` returns `"field_value_stats"` and
`"field_value_svg"`. The app renders the matrix in the **Decision views**
section (after the per-field bars) and live in the run loop. `report`
CLI output is unchanged (the matrix is a dashboard view).

### 30.3 V3 — decision-boundary scatter (2-feature tabular)
`dashboard.py`: `decision_boundary(task, model, n_grid=24,
n_points=200) -> dict | None` — the flat-task complement of the media error
gallery (SPEC.md 28.3): where does the model fail on a 2-D feature plane?
Applies only when `task.head == "softmax"`, `task.state_dim == 2`,
`task.class_values` is non-empty, `task.holdout_rows` (SPEC.md 28.2) is
callable, and `model is not None` — otherwise `None` (one feature, mse
head, episode tasks). One forward pass over the holdout (the same `n` as
`holdout_diagnostics`) and one over an `n_grid × n_grid` grid spanning the
holdout's per-feature ranges (padded 5%; a degenerate range → ±1);
**no RNG** (G2). Returns a JSON-safe dict:
`{"x0", "x1" (feature names, or `x0`/`x1`), "x0_range", "x1_range",
"n_grid", "grid_preds" (flat, row-major: index `i·n + j` is cell
(i, j), class index), "points" ([{x, y, true, pred} per holdout row]),
"classes" (labels via `class_label_str`, SPEC.md 28.3), "n", "correct"}`;
`correct` = `#{true == pred}`.

`plotting.py`: `svg_decision_boundary(data, width=640, height=560)` — the
grid painted as `n²` class-colored cells (`fill-opacity 0.30`, a
8-color class palette cycling for `k > 8`), each with a `<title>`
`grid <i>,<j> -> class <k>`; the holdout points on top — correct as green
(`_ACCEPTED`) circles, misclassified as red (`_SCORED_REJ`), each with a
`<title>` `true <label> -> pred <label>`; value ticks on both axes; a
legend row (class swatches + correct/misclassified swatches + `x:` / `y:`
feature names) and a summary `n holdout = N - correct = C (P%)`. `data` is
None → header + `no 2-feature classification data`.

`dashboard.py`: `finish()` returns `"boundary"` (or None) and
`"boundary_svg"` (or None); the app renders it in the **Learning views**
section (after the confusion matrix).

### 30.4 V4 — CI band on the score curve
`improver/meta_env.py`: the three scored log entries (baseline,
experiment, curriculum) now record `"std"`: the block-bootstrap holdout σ
already computed by `_evaluate` (SPEC.md 18.3) — `0.0` in legacy mode
(`ci_blocks == 0`), so legacy log rows gain exactly one zero-valued key
and the §18.7 bit-exact pin (which asserts the acceptance sequence and
summary, not the log bytes) stays green.

`plotting.py`: `svg_score_curve` (SPEC.md 21.2) draws, for each scored row
whose `std` is finite and `> 0`, a vertical translucent band
(`#93c5fd`, width 7, round caps) spanning `score ± std`, behind the curve,
with a `<title>` `±std <v>` and a caption `bands = score ± std (SPEC.md
18.3 block-bootstrap; the §18.5 CI gate)`; the y-range expands to cover
the bands. Rows without a positive `std` (legacy mode, old logs) draw
**exactly** the pre-v0.16 SVG (no band elements, no caption) — the §21.2
chart is unchanged for legacy runs, and `ascii_score_curve` is untouched.
`report --plot` picks the band up automatically (it renders from
`experiments.jsonl`).

### 30.5 V5 — model-family score bars
`dashboard.py`: `family_stats(entries) -> list[dict]` — over the scored
entries (`kind` baseline/experiment, finite `holdout_score`), grouped by
the entry's `spec.model_family` (`"unknown"` when absent):
`[{"family", "best_score", "best_seconds" (the cost of the best member),
"trials", "wins" (accepted count)}]`, sorted by `(-best_score,
family)`. Pure (G2).

`plotting.py`: `svg_family_bars(families, width=640, row_h=34)` — one
horizontal bar per family on a fixed 0–100 axis (bar length =
`best_score`), the top family highlighted (`_ACCEPTED`, the rest `_LINE`),
captioned `name — best (cost, wins/trials)`, faint 0/25/50/75/100
gridlines, and a legend `best holdout score per model family (all logged
experiments)`. Each bar's `<title>` is
`family <name>: best <s> in <secs>s (<wins>/<trials> accepted)`. Rows
without a finite `best_score` are skipped; empty → header +
`no scored experiments`.

`dashboard.py`: `finish()` returns `"family_stats"` and
`"family_bars_svg"`; the app renders the bars in the **Decision views**
section.

### 30.6 V6 — RL return-to-go / baseline trace
`plotting.py`: `svg_policy_trace(trace, width=640, height=360)` — beside
the existing action-probability bars (SPEC.md 29.2), *why the policy
explored*: over the existing per-step trace records `{task, action, probs,
reward}` (no `rl_policy.py` change — pure rendering of the v0.15 trace):
- one **reward bar per step** (green `r ≥ 0`, red `r < 0`), `<title>`
  `step <i> · <task> · r=<r>`;
- a **return-to-go** line + points (`_LADDER`): `r2g[t] = Σ rewards[t:]`;
- a dashed **baseline** line (`_BASELINE`): the running mean of rewards up
  to `t` (inclusive);
- a data-driven y-range covering `0`, all rewards, and all return-to-go
  values (all-zero rewards → fixed `[-1, 1]`), x = step index;
- a legend (reward sign, return-to-go, baseline) and a summary `n steps =
  N - final return-to-go = R - mean reward = M`.

`cli.py`: `policy-report` additionally writes `policy_trace_curve.svg` and
names it in its output listing. The app's **RL policy view (precomputed)**
panel renders whichever of `action_probabilities.svg`, `task_returns.svg`,
`policy_trace_curve.svg` exist in the given directory (so a v0.15-era two-
file directory still renders) and warns only when none exist (the A19
warning text is preserved). RL still never runs live (SPEC.md 23.1).

### 30.7 Non-goals (v0.16)
No search/training/acceptance changes (all six views are derived from
logged data; the only core deltas are the logged `std` key and the sweep
dict's `curve` key — both additive); no new core dependencies (NumPy only);
no live RL in the app (the §23.1 boundary is unchanged); no new `report`
sections — `report --plot`'s score curve gains the CI band for free, and
the other five views live in the dashboard/CLI surfaces; no parallelism or
algorithmic changes (the optimization items stay out of v0.16); no change
to `ascii_score_curve`, `field_stats` (SPEC.md 26.1), or any pinned
sequence (A1–A18, §18.7, §20.2, §18.7-legacy).

### 30.8 Acceptance (A20)
- **V1**: `svg_seed_curves` is valid XML, one polyline per seed with a
  legend naming each seed, the target line when given, the `steps = M`
  summary, and is empty-safe; `seed_sweep`'s per-seed dict carries
  `curve` with `curve[0] == baseline`, `curve[-1] == final`, all finite,
  and the rest of the A19 contract (the A19 exact-key test is extended
  with `curve`); `variance` writes `seed_curves.svg` (valid XML) alongside
  `seed_variance.svg`; the app's Seed variance panel renders both.
- **V2**: `field_value_stats` credits `(field, value)` exactly as
  hand-computed on a synthetic stream, normalizes value keys (tuple
  joined, `None` → `-`), sorts fields and values deterministically, and
  its per-field trial/wins sums equal `field_stats`'; `svg_field_value_matrix`
  is valid XML with one cell per credited `(field, value)` (counted via
  their `<title>`s), `wins/trials` text, and is empty-safe; `next()`
  updates carry `field_value_stats`; `finish()` returns `field_value_stats`
  + `field_value_svg`; the app renders the matrix.
- **V3**: `decision_boundary` returns `None` for a 1-feature CSV, an mse-
  head CSV, and `model=None`; on the 2-feature softmax CSV it returns
  `grid_preds` of length `n_grid²`, `points` of length = the holdout size
  with `true`/`pred` in range, `correct == #{true == pred}`, class labels
  `["0", "1"]`; `svg_decision_boundary` is valid XML with `n²` grid-cell
  titles and one `true … -> pred …` title per holdout point, the summary
  line, and is None-safe; `finish()` on a 2-feature CSV run returns
  non-None `boundary` + `boundary_svg`; the app renders it in Learning
  views.
- **V4**: the baseline/experiment log entries carry a finite `std`
  (`== 0.0` in legacy mode, finite in v0.4 mode); `svg_score_curve` on
  rows with `std > 0` draws exactly one band line per such row (counted by
  stroke color) + the §18.3 caption and expands the y-range, while rows
  without a positive `std` render byte-identical SVG to pre-v0.16 (no band
  elements, no caption); `ascii_score_curve` is unchanged; the §18.7
  legacy pin and the §21.2 SVG tests stay green.
- **V5**: `family_stats` on synthetic entries matches hand-computed
  groups/sort; `svg_family_bars` is valid XML with one titled bar per
  family, the top family highlighted, and is empty-safe; `finish()`
  returns `family_stats` + `family_bars_svg`; the app renders the bars.
- **V6**: `svg_policy_trace` is valid XML with one reward bar per step
  (counted by their `<title>`s), hand-computed return-to-go values in the
  text, the dashed baseline line, and is empty-safe; `policy-report`
  writes `policy_trace_curve.svg` (valid XML); the app's RL panel renders
  the third SVG when present, still renders a v0.15 two-file directory,
  and still warns on an empty directory (A19 text preserved).
- Full suite green; A1–A19 stay green (the A19 sweep-dict test updated per
  §30.1); `import autorefine` never pulls in streamlit/PIL/soundfile.

### 30.9 Milestone

**M19** — v0.16 comprehension visuals II: per-seed curves, field × value
win matrix, decision-boundary scatter, CI band on the score curve,
model-family bars, and the RL return-to-go/baseline trace (A20).

## 31. Optimization (v0.17)

Two loop-level optimizations, both opt-in, both G2-safe: the search stops
when it saturates (31.1), and the seed-variance sweep runs its independent
seeds in parallel processes (31.2). Both are off by default at their
existing surfaces, so every pinned run (A1–A20, the §18.7 legacy
sequence) is unchanged.

### 31.1 Plateau early stop (stall patience)
`meta_env.py`: `AutoRefineEnv` gains `stall_patience: int | None = None`
(None = off — pre-v0.17 exactly; opt-in `K ≥ 1`). Semantics:
- every *scored* candidate experiment that is **not accepted** (did not
  clear the §18.6 gate) increments a streak; every **accepted**
  experiment resets it to 0;
- free duplicate rejections (R3) and invalid-spec rejections
  (SPEC.md 25.4) do **not** count — patience counts experiments (budget
  units), matching `experiments_run`; the §20.2 duplicate-stall guard is
  unchanged;
- a curriculum step-up (SPEC.md 20.1) resets the streak — the new
  difficulty level is a new ceiling, so patience re-arms per level;
- `reset()` (a fresh episode) resets the streak;
- when the streak reaches K, `_check_done` reports done with
  `finished_reason="stalled"` (checked *after* `budget_exhausted` /
  `wall_time_exhausted`, so an existing reason on the same step still
  wins); the run finishes exactly like any other (artifacts, summary,
  and the §22.1 gate on `final_best_score` are unchanged — the gate is a
  property of the final score, not of the reason).

CLI: `fit` and `run` gain `--stall-patience K` (default: off); the
`variance` subcommand gains the same flag (passed through
`DashboardRunner`, so each sweep seed can stop at its own plateau).
`DashboardRunner` gains `stall_patience: int | None = None` (default
None = off; `start()` forwards it to `AutoRefineEnv`; `_clone` carries
it to every sweep seed). The `gym.py` adapter contract is unchanged (the
kwarg remains available on `AutoRefineEnv` directly).

### 31.2 Parallel seed sweep
`dashboard.py`: `DashboardRunner.seed_sweep(seeds, on_update=None,
workers=1)`:
- `workers == 1` (the default) keeps the **exact** serial loop — the N=1
  path and every existing caller are byte-identical;
- `workers > 1` runs one full isolated sweep seed per task on a
  `concurrent.futures.ProcessPoolExecutor(max_workers=workers)`: a
  module-level worker builds the same `_clone(seed)` runner and drives
  the same `start()` → `next()`…→ `finish()` loop, returning the same
  JSON-safe per-seed dict (SPEC.md 29.1); results are collected in
  input seed order;
- each seed is a self-contained deterministic run (G2) and the workers
  never share state, so per-seed results are bit-identical to the serial
  path (except `run_dir`, which is timestamped by design);
- `on_update` is a callback and not picklable: passing it with
  `workers > 1` raises `ValueError`; empty `seeds` returns `[]` without
  spawning a pool.

CLI: `variance` gains `--workers N` (default 1 = serial). No new
dependencies (`concurrent.futures` is stdlib); `import autorefine` stays
clean (the pool is created only inside `seed_sweep`).

### 31.3 Non-goals (v0.17)
No change to the acceptance rule, scoring, budget semantics, or any
pinned sequence with the new flags off (A1–A20 stay green as-is); no
live RL; no thread-based parallelism (processes only — seeds stay fully
isolated); no dashboard UI controls for the two new flags (the app keeps
its defaults: patience off, `workers=1`); no `gym.py` contract change;
no change to `fit`/`run` exit codes or the §22.1 gate.

### 31.4 Acceptance (A21)
- **Stall patience**: on a constant-score fake task (no candidate can
  ever be accepted), `stall_patience=2` with a 5-experiment budget stops
  after exactly 2 experiments with `finished_reason="stalled"` and
  `experiments_run=2`; with a scripted 50 → 60 → 55 → 52 score sequence
  and `stall_patience=2`, the 60 is accepted (streak reset) and the run
  stops at experiment 3 with `final_best_score=60`; with `stall_patience`
  unset the same constant-score run burns the full 3-experiment budget
  with `finished_reason="budget_exhausted"` (pre-v0.17 exactly); a
  curriculum step-up resets the streak; `fit`/`run`/`variance` accept
  `--stall-patience`, and a small `run --stall-patience 1` finishes with
  a reason in {stalled, budget_exhausted}.
- **Parallel sweep**: `seed_sweep([7, 8], workers=2)` returns the same
  per-seed dicts (excluding `run_dir`) as `workers=1`, in the same
  order; `workers=1` is the default and matches the pre-v0.17 serial
  path; `on_update` with `workers > 1` raises `ValueError`; empty seeds
  → `[]`; `variance --workers 2` writes the same artifacts (valid SVGs,
  a 2-entry `seed_sweep.json`) and exits 0.
- Full suite green (A1–A20 unchanged); `import autorefine` never pulls
  in streamlit/PIL/soundfile.

### 31.5 Milestone

**M20** — v0.17 optimization: plateau early stop (`stall_patience`,
`finished_reason="stalled"`) and the process-parallel seed sweep
(`seed_sweep(workers=N)`, `variance --workers`) (A21).

## 32. Optimization (v0.18)

Two optimization items: one app-only perceived-performance fix (32.1), and
one opt-in core search optimization (32.2). The third candidate — trainer
float micro-opts (fused MLP forward/backward, vectorized tree bagging) — is
deferred: any float-order change would break the bit-exact pins (A1–A21,
the §18.7 legacy sequence), and it is only worth the pin re-derivation if
screening proves insufficient (32.3).

### 32.1 App perceived perf (app-only)
`dashboard_app.py` (runner/CLI unchanged; SPEC.md 23/29):
- **Streamed seed sweep** — the "Run the seed sweep" button (the SPEC.md
  29.1 panel) opens an `st.progress` bar before calling `seed_sweep` and
  forwards the runner's per-step `on_update` into it (the SPEC.md 31.2
  serial callback path — the app uses `workers=1`). The fraction is
  clamped to `[0, 1]` and the denominator `seeds × experiments` is an
  **upper bound**: free duplicate rejections (R3) may exceed it. On
  success the bar finishes at `1.0` with "complete" text; the after-sweep
  SVG rendering (SPEC.md 29.1/30.1) is unchanged.
- **Cached data preview** — the preview computation moves to a
  `@st.cache_data` function `_preview_data(csv_path, stamp, size)` (pure:
  no `st.*` calls) returning a JSON-safe dict
  `{"kind": "csv" | "media", "caption": str, "header": list | None,
  "rows": ...}` or `{"error": msg}`. `stamp`/`size` are the file's
  `st_mtime_ns`/`st_size` in the cache key — the upload flow reuses one
  temp dir, so a changed file re-probes. The thin `_preview()` renderer
  shows exactly what it did before (same caption, dataframe, and warning
  texts, SPEC.md 23.2/24.5).

### 32.2 Two-stage candidate screening (opt-in)
`meta_env.py`: `AutoRefineEnv` gains `screen_frac: float = 1.0` (default
off = pre-v0.18 exactly; opt-in `0 < frac < 1.0`; valid values are
non-bool finite numbers with `0 < frac <= 1.0`, anything else
`ValueError`). Semantics when active:
- **Screen stage** — before full training, the candidate is trained on the
  first `max(1, floor(frac·n))` rows of the *same* dataset the full stage
  would use (`_dataset_for(spec)` — a convnet spec screens on the grid
  view too, SPEC.md 25.3) and is scored on the usual holdout. The
  subsample is a prefix slice — no new RNG — so per-seed runs stay
  bit-deterministic (G2).
- **Promotion** — a candidate is promoted to the full stage iff
  `screen_score > _screen_champion` (strict). The champion is the
  baseline (`DEFAULT_SPEC`) spec **screened once at `reset()`** — free,
  like the reset baseline. K=1 against this *fixed* champion: the current
  best is not re-screened per step, so no extra training per step.
- **Promoted** → full train on the full dataset → the unchanged §18.6
  gate (accept/reject, budget, Pareto, ensemble, curriculum all as
today).
- **Screen rejections** — logged with `kind="screen"` (all existing
  consumers filter on `baseline/experiment/curriculum`, so charts,
  summaries, and win-rates are unchanged) carrying float
  `holdout_score` (the screen holdout score), `std`, `gen_score`,
  `gen_gap`, `train_seconds`, `time_capped`, `loss_history`,
  `accepted=False`, `screen_rejected=True`. The fingerprint joins `seen`
  (R3 still applies), **one budget experiment is spent**
  (`experiments_run` counts it — it was scored and trained), and it
  counts for the v0.17 stall patience (SPEC.md 31.1: a scored
  non-improving experiment). It is appended to the `history` digest. It
  never touches the Pareto frontier, the ensemble top-k, or the
  curriculum saturation check.
- `report` prints the screen rows in its text table (the float
  `holdout_score`/`gen_gap` satisfy the formatter); `--plot` charts,
  `--json`, and the HTML report are unchanged (kind filters).
- The summary gains `"screening": {"frac": f,
  "baseline_screen_score": s}` when active (additive; absent when off).
- Preset `candidate_screening(frac=0.25)` returns the kwargs dict (the
  `search_quality_v04()` pattern, SPEC.md 18.6).

CLI: `run` and `fit` gain `--screen-frac F` (type=float, default `1.0`
= off). Not added to `variance`, the dashboard app, or `gym.py`
(non-goal, 32.3).

### 32.3 Non-goals (v0.18)
No trainer float micro-opts (fused MLP forward/backward, vectorized tree
bagging) — any float-order change breaks the bit-exact pins (A1–A21,
§18.7); deferred until screening proves insufficient. No screening surface
in `variance`, the dashboard app, or `gym.py`. The app sweep stays
`workers=1`. No change to the acceptance rule, budget semantics, or the
`fit`/`run` exit codes / §22.1 gate.

### 32.4 Acceptance (A22)
- **App**: with streamlit installed the app renders with a CSV; pressing
  "Run the seed sweep" completes the run and the progress bar finishes at
  `1.0` with "complete" text; `_preview_data` is cached — a second call
  with the same `(path, stamp, size)` does not re-probe (a counting
  wrapper on the task `__init__` fires once), while a changed stamp
  re-probes.
- **Screening pin** — scripted fake task, `screen_frac=0.25`, budget 3:
  screen scores 50 (baseline) → 40 → 60 → 55; full scores 50 → 70 → 65.
  Log kinds are exactly `baseline, screen, experiment, experiment`: exp 1
  is screen-rejected (40 ≤ champion 50; no full train), exp 2 is promoted
  and accepted (full 70 > 50), exp 3 is promoted and gate-rejected (65 <
  70). `experiments_run=3`, `finished_reason="budget_exhausted"`,
  `final_best_score=70`, the summary has the `screening` block; `report
  --run <dir>` exits 0 and prints the `screen` row.
- Default off (`1.0`) → no `screen` rows and no `screening` key; invalid
  `screen_frac` (`≤ 0`, `> 1`, str, bool) → `ValueError`; `run`/`fit`
  accept `--screen-frac` and a small screened run finishes with a reason.
- Full suite green (A1–A21); `import autorefine` stays clean
  (subprocess pattern).

### 32.5 Milestone

**M21** — v0.18 optimization: app perceived perf (streamed seed-sweep
progress + cached data preview) and opt-in two-stage candidate screening
(`screen_frac`, `run`/`fit --screen-frac`) (A22).

## 33. Coherency (v0.19)

Feature rounds M1–M21 accumulated knobs, presets, and cross-feature
interactions that were each individually specced but never cross-checked
as a whole. v0.19 closes that gap with three coherency contracts — no new
features, and **no behavior change with any flag off** (A1–A22 stay green
untouched):

- **33.1 (C1)** the version story — one version, one meaning;
- **33.2 (C2)** the knob registry — one table for every tunable knob;
- **33.3 (C3)** the interaction matrix — defined behavior for every pair
  of combinable round features.

Each contract is enforced by a test (A23), so drift that was previously
policed by convention becomes a suite failure.

### 33.1 Version story (C1)

The **package version is the single version number in the project**:
`pyproject.toml` (`[project].version`) and `autorefine.__version__` must
agree, and both are bumped together, one step per feature round — round
v0.19 ⇒ package `0.19.0` (M22). The README/SPEC feature-round labels
("v0.3"…"v0.19") are *milestone* labels, not an independent version
scheme: a round is both an `M#` and a `v0.#`, and the two advance
together. The dashboard caption (`v{__version__}`) shows the same number
automatically.

Enforced: a test asserts `pyproject [project].version ==
autorefine.__version__` (A23). There is deliberately no third copy of the
version anywhere (no `VERSION` constant, no `pip`-only truth).

### 33.2 Knob registry (C2)

Every tunable `AutoRefineEnv` knob with a default lives in one table —
`autorefine.KNOBS: dict[str, Knob]` — with `(name, default, validator,
spec_ref, cli subcommands, app widget)`. Consequences:

1. **One home for validation.** `AutoRefineEnv.__init__` validates every
   knob through the registry's validator. Error messages are unchanged
   (the A21/A22 message pins still hold); the validation *logic* has one
   source of truth.
2. **Presets are checked against the registry.** `search_quality_v04()`
   (18.6) and `candidate_screening(frac)` (32.2) may only set registry
   knobs with values that pass the registry validator; a preset keying an
   unknown knob is a construction-time error, not a runtime surprise.
3. **CLI honesty.** Each registry row lists the subcommands exposing
   `--{kebab-name}`: `stall_patience` → `run`, `fit`, `variance`; `screen_frac`
   → `run`, `fit` (32.3 non-goal: not `variance`/app/gym). A test drives
   the **real argparse parser** (`cli.build_parser()`, extracted from
   `main()` — a pure refactor, identical flags) and asserts presence
   *and absence* per subcommand, plus that parser defaults equal
   registry defaults.
4. **Adding a knob** = one registry row + the env kwarg. The A23 tests
   fail immediately if any surface (env default, preset, CLI flag,
   validator) is forgotten.

Registry (name — default — spec — CLI subcommands):

| knob | default | spec | CLI subcommands |
|---|---|---|---|
| `ci_blocks` | `0` | 18.3 | — (via `--search-quality v04` preset) |
| `z_accept` | `0.0` | 18.6 | — (preset) |
| `efficiency_weight` | `0.0` | 18.4 | — (preset) |
| `gen_gap_penalty` | `0.0` | 18.5 | — (preset) |
| `block_size` | `512` | 18.3 | — (preset) |
| `ensemble_top_k` | `0` | 19.3 | — (`--ensemble-final`) |
| `stall_patience` | `None` | 31.1 | `run`, `fit`, `variance` |
| `screen_frac` | `1.0` | 32.2 | `run`, `fit` |

Knobs consumed only via presets are recorded with an empty CLI list; the
preset test (2.) covers them. `dataset_episodes` (20.3) and `task_config`
(22.1) are task-level, not search knobs — they stay outside the table.

### 33.3 Interaction matrix (C3)

Defined behavior for every pair of round features that can be enabled
together. Each defined pair has a test — an A23 test (new) or a cited
existing pin.

| pair | defined behavior | test |
|---|---|---|
| stall × screening | a screen rejection is a scored non-improving experiment: it extends the stall streak (32.2, 31.1) | A22 (`test_screen_rejections_count_for_stall_patience`) |
| stall × curriculum | a step-up is a new ceiling: it re-arms (resets) the stall streak (31.1, 20.1) | A21 (v0.17 curriculum step-up test) |
| **screening × curriculum** | **new in v0.19**: a step-up **re-pins the screen champion** — the baseline spec is re-screened on the new (harder) dataset, free (no budget spend), exactly like the existing curriculum re-baseline (20.1 pattern). The 32.2 K=1 fixed-champion semantics are therefore **per curriculum level**, not per episode; without this, a harder level would reject every candidate against a stale easy-level champion | A23 (`test_screening_curriculum_repins_champion`) |
| screening × ensemble | a screen-rejected spec never full-trains ⇒ never enters the Pareto frontier or the top-k ensemble (19.3) | A23 (`test_screen_rejected_never_enters_frontier_or_ensemble`) |
| screening × search-quality | the screen gate is the strict champion beat (32.2); the §18 CI/efficiency gate applies only to promoted (full-trained) candidates | 32.2 (A22) |
| curriculum × ensemble | a step-up resets the top-k to the re-baselined best spec (20.1, 19.3) | A10 (v0.6 curriculum run pin) |

Decision recorded (screening × curriculum): the re-pin uses the
**baseline spec** (`DEFAULT_SPEC`), consistent with 32.2's "champion =
the baseline spec" — *not* the current best spec (which is what the
curriculum's own re-baseline carries over). Both champion screenings are
free (reset + each step-up), budget-free and deterministic (G2). The
step-up's curriculum event row gains a `screen_champion` key (only when
screening is active — additive, existing A10 rows unchanged). The
summary's `screening.baseline_screen_score` (32.2) keeps reporting the
**reset** champion in curriculum runs; each re-pinned value rides its
own curriculum event row, so the summary dict stays exactly
`{frac, baseline_screen_score}` (the A22 pin).

### 33.4 Acceptance (A23)
- **Version (33.1)** — `pyproject [project].version ==
  autorefine.__version__`; the dashboard caption renders the same
  number.
- **Knob registry (33.2)** — every registry row's default equals the
  env kwarg default (`inspect`); both presets set only registry knobs
  with validator-passing values; the registry validator and the env agree
  accept/reject on the same value batteries (A21/A22 batteries
  re-run); `cli.build_parser()` exposes each claimed flag per
  subcommand and *not* elsewhere (`--screen-frac` absent from
  `variance` — 32.3), with parser defaults equal to registry defaults;
  registry `app_widget` claims (none today) appear in the dashboard
  source when set.
- **Interaction matrix (33.3)** — the screening × curriculum pin:
  scripted fake task + two-level fake curriculum, `screen_frac=0.25` —
  after the step-up, a candidate that beats the *old* champion but loses
  to the re-pinned one is screen-rejected, one that beats both is
  promoted (full-trained) even when gate-rejected; the champion value
  and the curriculum event's `screen_champion` are exact; bit-
  reproducible (G2). The screening × ensemble leak test: a screen-
  rejected fingerprint is absent from `pareto.points` and the ensemble
  top-k. stall × screening, stall × curriculum, and curriculum ×
  ensemble stay covered by their cited pins.
- **Regression** — full suite green (A1–A22); `import autorefine`
  stays clean (subprocess pattern).

### 33.5 Milestone

**M22** — v0.19 coherency: version story (33.1), knob registry (33.2),
interaction matrix (33.3) (A23).

## 34. Coherency II (v0.20)

C1–C3 (33) closed the env-knob surface (version, KNOBS, CLI honesty,
interactions). v0.20 extends the same cross-check discipline to the two
remaining cross-surface artifacts — the **spec space** (what the improver
may propose) and the **summary contract** (what every downstream surface
reads). Same house rules: no new features, **no behavior change**
(A1–A23 stay green untouched), and the version steps per 33.1
(`0.19.0` → `0.20.0`, both sources together; the A23 round assertion
advances with it: v0.20 ⇒ `0.20.0`).

### 34.1 Spec-space coherency (C4)

The spec space is one object with three surfaces, each enumerated
independently:

- **the law** — `config.py` constants + `ModelSpec.validate` (5.1, 19.1,
  25.2/25.3): what is legal;
- **the catalog** — `improver/catalog.py` `FIELD_CATALOG` (15, 17, 19.4,
  25): the discrete value space shared by the RL improver, the Gym
  adapter, and the bandit;
- **the samplers** — `improver/actions.py` `FIELD_SAMPLERS` (8, 15): what
  the bandit/search actually draw.

Coherency contracts (A24; a violation is a suite failure, not a runtime
surprise):

1. **Exact-value fields equal the config constants** (order included):
   `optimizer == OPTIMIZERS`, `activation == ACTIVATIONS`,
   `batch_size == BATCH_SIZES`, `model_family == MODEL_FAMILIES`,
   `lr_schedule == LR_SCHEDULES`, `knn_k == KNN_K_VALUES`.
2. **Range-field catalog values lie inside the config ranges**:
   `learning_rate`, `weight_decay`, `train_steps`, `input_noise`,
   `label_smoothing`, `early_stopping_patience`, `init_scale`,
   `gradient_clipping`.
3. **Every catalog `architecture` value is legal for at least one
   non-knn family** — mlp widths (15), tree/boost depth (15, 19.2), or
   convnet filter pairs (25.3). The knn family's accept-any-architecture
   escape hatch (25.2) is documented, not a validation path, so it is
   deliberately excluded from this check.
4. **Samplers only draw in the registered space**: a fixed-seed battery
   (100 draws per field, `task=None`) of every `FIELD_SAMPLERS` entry
   lands inside the field's registered space — exact-value fields in
   their catalog values, range fields in their config ranges,
   `architecture` legal for some non-knn family. The field classes are
   exhaustive over `FIELD_NAMES` (a new field that is in none of them
   fails the test).
5. **The field sets are one set**: `set(FIELD_NAMES) ==
   set(CATALOG_FIELDS)`; `SEARCH_FIELDS` is `FIELD_NAMES` minus the
   documented legacy exclusion (25.7: `knn_k`, 14 fields);
   `ORDERED_FIELDS ⊆ FIELD_NAMES`; every `FAMILY_FIELDS` row covers a
   known family and stays inside the catalog fields.

Cited existing pins (unchanged): the 77-action catalog (19.4/25,
`test_search_quality`); every action valid against the mlp and tree best
specs (`test_rl_policy`); per-family relevant fields (18.2,
`test_search_quality`).

### 34.2 Summary contract (C5)

`summary.json` is the cross-surface artifact: the env writes it
(`_write_artifacts`), and `report`, `eval`, the dashboard, and the
plotting surfaces read it. Its keys were documented across ~8 SPEC
sections with no single pin of the whole contract. Contract (A24): the
**exact top-level key set** is pinned for the two canonical
configurations, and each internal dict pins its exact key set:

- **legacy run** (no optional flags) — 17 keys;
- **full-flag run** (`task_config` + `curriculum` + `screen_frac < 1.0`
  + `ensemble_top_k > 0`) — those 17 + the 4 conditional keys.

| key | producer | condition | spec |
|---|---|---|---|
| `task`, `seed`, `finished_reason` | env | always | 5 |
| `baseline_score`, `final_best_score`, `improvement_factor` | env | always | 5.3 |
| `baseline_effective`, `final_best_effective`, `effective_improvement_factor` | env | always | 18.4 |
| `search_quality` `{ci_blocks, z_accept, efficiency_weight, gen_gap_penalty, block_size}` | env | always | 18 |
| `experiments_run`, `wall_seconds` | env | always | 5 |
| `best_spec` (the full 15-field spec dict) | env | always | 5.1 |
| `mutation_win_rate` | env | always | 5.3 |
| `pareto_frontier`, `best_score_at_1s`, `efficiency_at_1s` | pareto | always | 15 |
| `task_config` (the user's kwargs) | env | `task_config` non-empty | 22.1 |
| `curriculum` `{levels, final_difficulty, final_ceiling, levels_left}` | env | `curriculum` given | 20.1 |
| `screening` `{frac, baseline_screen_score}` | env | `screen_frac < 1.0` | 32.2, 33.3 |
| `ensemble` `{top_k, member_scores, ensemble_score}` | env | `ensemble_top_k > 0` and ≥ 1 full-trained member | 19.3 |

Adding a summary key = adding a row here + extending the A24 key sets;
removing/renaming one breaks the A24 pin (and the reading surface) at
test time, not at dashboard runtime.

### 34.3 Acceptance (A24)
- **Spec space (34.1)** — the five contracts tested: exact-value fields
  equal the config constants (order included); range-field catalog values
  inside the config ranges; catalog architectures legal for some non-knn
  family; fixed-seed sampler draws in the registered space for every
  field (exhaustive field classes); the field sets are one set (incl.
  the 25.7 `SEARCH_FIELDS` exclusion and the `FAMILY_FIELDS` rows).
- **Summary (34.2)** — exact top-level key sets for the legacy run and
  the full-flag run (scripted fake task, 4-experiment budgets, flat
  scores); the `search_quality`, `curriculum`, `screening`, and
  `ensemble` internal key sets; `best_spec` carries the full spec dict
  (15 fields); each `pareto_frontier` point carries exactly
  `{score, train_seconds, spec_hash}`.
- **Regression** — full suite green (A1–A23); version stepped to `0.20.0`
  in both sources (33.1) with the A23 round assertion advanced (v0.20 ⇒
  `0.20.0`).

### 34.4 Milestone

**M23** — v0.20 coherency II: spec-space coherency (34.1), summary
contract (34.2) (A24).

---

## 35. Coherency III (v0.21)

34 (v0.20) made the spec space and the summary contract
cross-checkable. v0.21 finishes the loop with the last two artifacts
that were policed by convention only — the **experiment-log kind
strings** (the §9/§21.2 row type every renderer and report formatter
literal-compares) and the **acceptance index** itself (the M# → § → A#
→ test-file map, until now in prose and memory). Same house rules: no
new features, **no behavior change** (A1–A24 stay green untouched; the
logged bytes are byte-identical), and the version steps per 33.1
(`0.20.0` → `0.21.0`, both sources together; the A23 round assertion
advances with it: v0.21 ⇒ `0.21.0`).

### 35.1 Log kind registry (C4)

The `kind` field of an `experiments.jsonl` row (SPEC.md 9) is the row
type every reading surface matches against — `plotting.py` (score
curve / band / ladder, `html_report`), `cli.py` (report table + pareto
points), `dashboard.py` (pareto points, `_loss_curves` labels, family
stats) and the env's own `_write_artifacts` win-rate loop. The five
values (`baseline`, `experiment`, `screen`, `curriculum`,
`invalid_spec`) were bare string literals at every site: a new kind
added to the emitter would be silently missed by the report's table
formatter — the exact gotcha the v0.18 screen row hit (A22).

Contract (A25; a violation is a suite failure, not a runtime
surprise):

1. **The registry is one set** — `memory.py` (the log module, 9)
defines `KIND_BASELINE`, `KIND_EXPERIMENT`, `KIND_SCREEN`,
   `KIND_CURRICULUM`, `KIND_INVALID_SPEC` and
   `LOG_KINDS = frozenset({…})`. The values are byte-identical to the
   historical literals, so existing logs, pins, and the A22
   screen-row tests are untouched.
2. **Emitters route through the registry** — the `reset`/`step` log
calls and `_advance_curriculum` / `_reject_screened` /
   `_reject_invalid` in `improver/meta_env.py` (6.1) never write a raw
   `"kind"` literal; the suite scans the emitter source and fails if
   one is left behind.
3. **Consumers route through the registry** — the kind comparisons in
   `plotting.py`, `cli.py`, `dashboard.py` and `meta_env.py`
   (`_write_artifacts`) use the constants; the scan test fails if a
   consumer still compares against a bare literal.
4. **The registry is closed** — the `KIND_*` constant set equals
   `LOG_KINDS` (exactly five kinds); adding a kind requires a constant,
a row here, and a consumer — any of them missing fails the tests
   instead of the report.

No control-flow change: the five emission sites and every comparison
are value-identical before and after (G2; the A1–A24 pins — including
the A22 `screen` rows and the curriculum-ladder tests — stay green).

### 35.2 Acceptance index (C5)

A1–A24 span §12–§34, and the M# → § → A# → test-file map lived in
prose and memory. The index table (inserted at the top of this
document, immediately after the header block) is now the single
source of truth: each row is `M# | version | SPEC § | A# | test
file(s)`. Contract (A25):

1. **SPEC acceptance numbers are cited by tests** — every `A#`
   defined in SPEC.md (a `- **A#.**` bullet) is cited by ≥ 1
   `tests/test_*.py` file (`\bA\d+\b` matching).
2. **Every test file cites an acceptance number** — every
   `tests/test_*.py` cites ≥ 1 `A#` in its source.
3. **Rows resolve** — each index-table row lists test file(s) that
   exist under `tests/` and cite that row's A# (the M0–M3 scaffold
   rows predate acceptance numbers and carry none).

Closes the loop: an acceptance number added without a test (or a test
file without an acceptance number) now fails the suite — spec/code
drift previously policed by convention alone (A1–A24).

### 35.3 Acceptance (A25)
- **Kind registry (35.1)** — the five `KIND_*` values are
  byte-identical to the historical literals (`baseline`,
  `experiment`, `screen`, `curriculum`, `invalid_spec`) and
  `LOG_KINDS` is exactly their set; the emitter source (`meta_env.py`)
  contains no raw `"kind"` literal and every `KIND_*` name it uses
  resolves to a registry constant; the consumer sources (`plotting.py`,
  `cli.py`, `dashboard.py`) carry no bare kind-literal comparisons;
  and a live run's logged rows all carry kinds in `LOG_KINDS`.
- **Index (35.2)** — every SPEC-defined `A#` (A1–A25) is cited by ≥ 1
  `tests/test_*.py`; every `tests/test_*.py` cites ≥ 1 `A#`; every
  index-table row's listed test file exists and cites that row's A#.
- **Regression** — full suite green (A1–A24); version stepped to
  `0.21.0` in both sources (33.1) with the A23 round assertion
  advanced (v0.21 ⇒ `0.21.0`).

### 35.4 Milestone

**M24** — v0.21 coherency III: log kind registry (35.1), acceptance
index (35.2) (A25).
