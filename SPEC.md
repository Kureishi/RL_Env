# AutoRefine — Spec for an Autonomous Iterative Model-Improvement Environment

**Version:** 0.1 (draft for review)
**Date:** 2026-08-23
**Status:** Awaiting approval before implementation

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
