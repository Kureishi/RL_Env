# AutoRefine

An environment where ML models iteratively improve through autonomous means.
The improver loop proposes model-configuration changes, trains, evaluates on
splits it never saw, records the outcome, and proposes the next change — all
within a hard budget, fully reproducible, NumPy-only.

**Current version: 0.10.0** (input modalities: `autorefine fit --data DIR`
on labelled image/audio directories — see [SPEC.md](SPEC.md) §24).

**Spec (source of truth):** [SPEC.md](SPEC.md)

## Quickstart

```bash
pip install -e .            # installs the `autorefine` package (needs numpy)

# run the autonomous improvement loop (30 experiments, 15 min cap)
python -m autorefine run --task cartpole-v1 --seed 7 --experiments 30

# other tasks (v0.2): sine-v1 (regression), gridnav-v1 (tabular policy)
python -m autorefine run --task sine-v1 --seed 7 --experiments 10

# noisy 4-bit XOR classification (v0.3): a task linear models can't solve
python -m autorefine run --task parity-v1 --seed 7 --experiments 10

# UCB field-bandit improver (v0.3): one-field mutations, credited by outcome
python -m autorefine run --task parity-v1 --policy bandit --seed 7 --experiments 6

# search-quality preset (v0.4) is the default; opt out for the v0.3 rule
python -m autorefine run --task parity-v1 --policy bandit --seed 7 --search-quality legacy

# frontier ensembling (v0.5): also report the top-2 frontier models averaged
# as a final evaluation (off by default; the reward/acceptance is untouched)
python -m autorefine run --task parity-v1 --policy bandit --seed 7 --ensemble-final

# curriculum (v0.6): step parity difficulty up (p_flip, then n_bits) as the
# improver saturates (off by default; parity-v1 only for now)
python -m autorefine run --task parity-v1 --policy bandit --seed 7 --experiments 10 --curriculum

# meta-RL improver: a policy trained on the improvement loop itself
python -m autorefine run --task sine-v1 --policy rl --rl-episodes 5

# inspect a run
python -m autorefine report --run runs/<run_id>

# machine-readable + plotted report (v0.7): JSON stdout, ASCII charts, SVG files
python -m autorefine report --run runs/<run_id> --plot
python -m autorefine report --run runs/<run_id> --json

# one self-contained HTML report (v0.8): summary + embedded SVGs + tables
python -m autorefine report --run runs/<run_id> --html

# re-score the run's best model on fresh splits
python -m autorefine eval  --run runs/<run_id>

# external RL agent through the Gymnasium adapter (v0.7, needs: pip install gymnasium)
python examples/gym_dqn.py --task sine-v1 --experiments 5 --seed 7

# worked example (v0.7): custom tabular task, "which model fits?", gate on >95%
python examples/tabular.py --experiments 20 --target 95.0

# list plugin entry points (v0.7): tasks/policies contributed by other packages
python -m autorefine plugins list

# no-code model fit on your own CSV (v0.8): gate on your target (exit 0/2)
python -m autorefine fit --data sales.csv --label churn --target 95.0
# ...or try the bundled sample (600 rows, 27% churn, nonlinear boundary; baseline ~85 -> PASS):
python -m autorefine fit --data examples/data/churn_sample.csv

# no-code model fit on an image/audio directory (v0.10): one subfolder per
# class (or an index.csv); WAV needs no extra, MP3 needs autorefine[audio],
# images need autorefine[image]
python examples/make_sample_media.py   # synthesize deterministic sample dirs
python -m autorefine fit --data examples/data/tone_clips --target 90.0
python -m autorefine fit --data examples/data/image_shapes --target 90.0

# visual dashboard (v0.9, needs: pip install autorefine[gui];
# v0.10: also accepts an image/audio directory):
# upload the CSV or directory, watch every experiment live, view the plots, download the model
python -m autorefine dashboard          # or: streamlit run src/autorefine/dashboard_app.py

# run the test suite (214 tests, incl. acceptance A1-A14)
python -m pytest tests/
```

## Python API

```python
from autorefine import AutoRefineEnv, Budget, SearchPolicy

env = AutoRefineEnv(task="cartpole-v1", seed=7,
                    budget=Budget(max_experiments=30, max_wall_seconds=900,
                                  max_train_seconds=30))
state = env.reset()                      # trains + scores the baseline spec

policy = SearchPolicy(seed=7)            # any object with propose(state) works
while not env.done:
    action = policy.propose(state)       # a ModelSpec dict (the unit of improvement)
    state, reward, done, info = env.step(action)
    # reward = candidate_holdout_score - best_score + 1e-3 novelty bonus
```

The `step()` contract makes the *improvement loop itself* a Gym-style
environment: `state` = (best score, best spec, budget left, history digest,
Pareto frontier); `action` = a candidate `ModelSpec`; `reward` = validated
score delta. Any policy — the built-in hill-climber, an RL agent, Bayesian
search — can drive it by implementing `propose(env_state) -> spec_dict`.

### Bandit improver (v0.3)

`BanditPolicy` is a second reference policy: a UCB bandit over the spec
*fields*. Each step it mutates one field of the current best spec (values
from the usual samplers), credits that field from the env's
`state["last"]` acceptance feedback, and picks the next field by UCB —
every field is tried once before exploitation starts. Same `propose`
contract, deterministic given the seed.

In v0.4 the bandit mutates by *local* (neighborhood) moves by default —
`BanditPolicy(seed, mode="local")`; `mode="uniform"` restores the v0.3
proposal stream — and its field selection is conditioned on the best
spec's family (a tree best only tries the 4 fields trees actually use,
SPEC.md 18.1/18.2):

```python
from autorefine import AutoRefineEnv, Budget, BanditPolicy

env = AutoRefineEnv(task="parity-v1", seed=7,
                    budget=Budget(max_experiments=10, max_wall_seconds=600,
                                  max_train_seconds=30))
policy = BanditPolicy(seed=7)
state = env.reset()
while not env.done:
    state, reward, done, info = env.step(policy.propose(state))
```

### Search quality (v0.4)

Acceptance in v0.4 is CI-aware and efficiency-aware (SPEC.md 18): holdout
and gen scores are block-bootstrap means (8 blocks × 512 points) with a
sample std, and a candidate is accepted only when
`eff(cand) − eff(best) > z·SE`, where
`eff = score − 0.5·min(train_seconds, 10)` minus a penalty for gen gap
beyond 5% of score. The constructor's defaults stay legacy (v0.3 rule);
`search_quality_v04()` applies the recommended preset — the CLI `run`
command uses it by default (`--search-quality legacy` opts out):

```python
from autorefine import AutoRefineEnv, Budget, BanditPolicy, search_quality_v04

env = AutoRefineEnv(task="parity-v1", seed=7,
                    budget=Budget(max_experiments=10, max_wall_seconds=600,
                                  max_train_seconds=30),
                    **search_quality_v04())  # ci_blocks=8, z=1.0, η=0.5, P_GEN=0.5
policy = BanditPolicy(seed=7)   # v0.4: local mutations + family-conditioned fields
state = env.reset()
while not env.done:
    state, reward, done, info = env.step(policy.propose(state))
# info["effective_score"], info["se"] expose the acceptance quantities;
# summary.json gains baseline_effective / final_best_effective + the active preset
```

### Meta-RL improver (v0.2)

`MetaRLPolicy` is one such policy, trained on `AutoRefineEnv` itself: a
linear softmax over a discrete mutation catalog, learned with REINFORCE.
Same `propose` contract as `SearchPolicy`:

```python
from autorefine import AutoRefineEnv, Budget, MetaRLPolicy, train_policy

env = AutoRefineEnv(task="sine-v1", seed=7,
                    budget=Budget(max_experiments=10, max_wall_seconds=600,
                                  max_train_seconds=30))
policy = MetaRLPolicy(seed=7)
report = train_policy(env, policy, n_episodes=3, verbose=True)
# report -> {"episode_returns": [...], "policy_updates": 3, ...}
```

### Multi-task meta-RL (v0.6)

The same policy transfers across tasks: `MetaRLPolicy(seed, task_names=[...])`
gains a per-task bias row — `logits = x·w + b + B[task]` — so the shared `w`/`b`
carry the transfer channel and `B` captures task specificity (SPEC.md 20.2).
`task_names=None` stays the single-task policy. Episodes alternate round-robin
over the tasks in `task_names` order:

```python
from autorefine import AutoRefineEnv, Budget, MetaRLPolicy, train_multi_policy

envs = [AutoRefineEnv(task=name, seed=7,
                      budget=Budget(3, 600, 30), runs_dir=tmp_dir / name)
        for name in ("sine-v1", "parity-v1", "cartpole-v1")]
policy = MetaRLPolicy(seed=7, task_names=["sine-v1", "parity-v1", "cartpole-v1"])
report = train_multi_policy(envs, policy, episodes_per_task=2)
# report -> {"per_task_returns": {task: [...], ...}, "policy_updates": ...}
```

The env's `state` now carries `"task"` (the active task name); existing
policies ignore it. A stall guard (v0.6) ends an episode after 32 consecutive
duplicate rejections (`finished_reason = "duplicate_stall"`); any fresh
proposal resets the streak — this bounds a REINFORCE episode that collapses
onto a single already-seen action (SPEC.md 20.2).

### External RL agent (v0.7)

`examples/gym_dqn.py` drives `AutoRefineGymEnv` with a DQN-style agent —
2-layer Q-network, experience replay, ε-greedy — written in NumPy only and
touching the environment **only** through the gymnasium API
(`reset(seed)`, `step(action)`, the two spaces). This proves the adapter
carries an external-library-style agent (SPEC.md 21.1):

```bash
pip install gymnasium
python examples/gym_dqn.py --task sine-v1 --experiments 5 --seed 7
```

```python
from autorefine import Budget
from autorefine.gym import AutoRefineGymEnv
from gym_dqn import DQNAgent          # examples/gym_dqn.py

env = AutoRefineGymEnv(task="sine-v1", seed=7,
                       budget=Budget(5, 600, 30), runs_dir="runs")
obs, _ = env.reset(seed=7)
agent = DQNAgent(obs.shape[0], env.action_space.n, seed=7)
while True:
    a = agent.act(obs)
    next_obs, r, terminated, truncated, _ = env.step(a)
    agent.step_learning(obs, a, r, next_obs, terminated)   # replay + update
    obs = next_obs
    if terminated:
        break
```

### Gymnasium adapter (v0.2, optional dependency)

```python
pip install gymnasium
from autorefine.gym import AutoRefineGymEnv

env = AutoRefineGymEnv(task="sine-v1", seed=7)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Fixed-dim `Box` observation, `Discrete` action space (the mutation catalog).

## What improves, and how it's measured

- **Unit of improvement:** `ModelSpec` — model family, architecture,
  optimizer, learning rate, batch size, weight decay, training steps, input
  noise, activation, (v0.3) `label_smoothing` (0.0–0.15, softmax head only),
  and (v0.5) four mlp-family knobs that each have real training effect:
  `lr_schedule` (constant/cosine/warmup_cosine), `early_stopping_patience`
  (0–50, tail-split early stopping with best-val restore), `init_scale`
  (0.5–2.0, scales the Glorot bound), `gradient_clipping` (0.0–10.0, global
  L2-norm cap), and (v0.11) `knn_k` (1/3/5/11/21, consumed only by the
  `knn` family; default 5 keeps pre-v0.11 spec JSON loadable). All default
  to the legacy behavior, so pre-v0.5 spec JSON still loads and legacy
  training stays bit-identical (T2/A8 pins).
- **Tasks:**
  - `cartpole-v1` — balance policy by dense behavior-cloning of a reference
    controller; score = mean episode survival (max 500 steps).
  - `sine-v1` (v0.2) — fit a fixed sum of sinusoids (regression, mse head);
    score = 100·R² on fresh holdout/gen points.
  - `gridnav-v1` (v0.2) — 6×6 toroidal navigation policy (one-hot cell → 4
    directions); score = 100·success_rate + 25·efficiency (max 125).
  - `parity-v1` (v0.3) — XOR parity of 4 hidden bits, observed with seeded
    8% bit-flip noise (softmax head); score = 100·accuracy on fresh points.
    Linear models provably can't solve it; an MLP reaches the ≈75 Bayes
    ceiling — a clean family gradient for the improver.
  - `csv` (v0.8) — `CsvTask(path, seed=0, label=None, split_frac=0.2)`
    wraps your own CSV: numeric columns as features, label inferred (`label`
    arg, else first of `label/target/y/class`, else last column);
    seed-derived 80/10/10 train/holdout/gen splits, train-only standardization;
    score = 100·accuracy (2–50 integer classes → softmax head) or 100·R².
    SPEC.md 22.1.
- **Model families (v0.2, +v0.5, +v0.11):** `mlp` (any depth 0..3; depth 0
  is a *linear* model; `softmax` or `mse` head), `tree` (bagged CART
  ensemble, depth 1..3, `train_steps//100` trees), (v0.5) `boost`
  (gradient-boosted residual CART, `BOOST_SHRINK=0.1`, same surface as
  `tree`), (v0.11) `knn` (non-parametric: memorizes the standardized train
  split and answers by k-NN with deterministic ties; `knn_k` is its knob;
  every task), and (v0.11) `convnet` (small NumPy convnet over
  grid-structured features — the 32×32 image grid and the log-mel
  time×mel spectrogram, the audio *temporal* model; `architecture=(c1,c2)`
  with c ∈ 4/8/16/32; grid-capable tasks only — a convnet spec on a flat
  task is a logged `invalid_spec` rejection, never a crash, SPEC.md
  25.3/25.4). Boost is clearly stronger than bag on regression (`sine-v1`)
  and a competitive third answer on `parity-v1` (the ordering is
  config-dependent, SPEC §19.2). Old spec JSON and checkpoints still load.
- **Honesty:** scoring uses a holdout split the improver never sees as data,
  plus a `gen_score` under different seeds; `gen_gap` is logged to catch
  overfitting (acceptance A3 requires < 5%).
- **Search quality (v0.4):** acceptance is CI-aware (block-bootstrap σ;
  accept only if `Δeff > z·SE`), efficiency-aware (0.5 points per training
  second, 10 s cap), and penalizes gen gap beyond 5% of score — preset
  `search_quality_v04()`, the CLI default (`--search-quality legacy` opts
  out). All knobs 0 reproduces the v0.3 rule exactly (A8).
- **Dedup:** identical specs are rejected for free (`reason="duplicate"`).
- **Budget:** hard caps on experiments, total wall time, and per-training
  time (enforced mid-training, clean abort); the budget is per-episode, so
  `reset()` after `done` starts a fresh run (needed for meta-RL episodes).
- **Efficiency memory (v0.2):** every run tracks the score-vs-train-time
  Pareto frontier; `summary.json` reports `pareto_frontier`,
  `best_score_at_1s`, `efficiency_at_1s`.
- **Frontier ensembling (v0.5, opt-in):** `AutoRefineEnv(ensemble_top_k=2)`
  (CLI `--ensemble-final`) additionally reports the top-2 frontier models
  (baseline included) averaged as a final evaluation:
  `summary.json` gains `ensemble = {top_k, member_scores, ensemble_score}`.
  Off by default, so v0.4 runs and A1–A8 are unchanged; the reward/
  acceptance contract is untouched (the ensemble is a reported evaluation,
  not an accepted candidate).
- **Curriculum (v0.6, opt-in):** `AutoRefineEnv(..., curriculum=ParityCurriculum(seed))`
  (CLI `run --curriculum`, parity-v1 for now) steps difficulty up as the
  improver saturates: sweep `p_flip` 0.08→0.16 (step 0.02) at 4 bits, then
  5 bits re-sweeps — 10 levels, 9 step-ups. A step-up fires when
  `best_score ≥ trigger_fraction · ceiling(level)` (default 0.9), rebuilds
  the task at the next level (same seed), re-baselines the best spec for
  free, and resets difficulty-scoped state (dedup, Pareto, ensemble) while
  keeping the full experiment log. `state["curriculum"]` and
  `summary["curriculum"]` track the ladder (SPEC.md 20.1).
- **Multi-task meta-RL (v0.6):** task-conditioned `MetaRLPolicy` +
  `train_multi_policy` transfer one REINFORCE policy across
  sine→parity→cartpole via shared `w`/`b` and per-task bias rows;
  `state["task"]` is the new conditioning key (SPEC.md 20.2).
- **Default dataset sizes (v0.6):** every task declares
  `default_dataset_size` (cartpole/gridnav 60 episodes; sine 2048, parity
  4096 points); `AutoRefineEnv(dataset_episodes=None)` falls back to it and
  an explicit int still wins. Parity/sine baselines rise noticeably with
  more data; cartpole/gridnav runs are unchanged (SPEC.md 20.3).
- **Report plots + JSON (v0.7):** `report --json` prints `summary.json` as
  machine-readable JSON; `report --plot` appends ASCII score/pareto charts
  and writes `score_curve.svg` + `pareto_frontier.svg` into the run dir
  (hand-written SVG, no new dependency; default output unchanged;
  combined `--plot --json` keeps stdout pure JSON). SPEC.md 21.2.
- **Plugin loader (v0.7):** external packages contribute tasks/policies via
  entry points — `[project.entry-points."autorefine.tasks"] my-task =
  "my_pkg.tasks:MyTask"` — discovered at CLI start (tasks merge into
  `TASKS`, so `--task` and `eval` see them) with protocol validation
  (`make_dataset`/`score`; policies `propose`); `autorefine plugins list`
  reports what was found. SPEC.md 21.3.
- **Data-in CLI (v0.8):** `autorefine fit --data FILE [--label COL]
  [--target T] [run flags]` runs the same improvement loop on your own CSV
  (built-in `csv` task; `--policy bandit` by default) and gates the result:
  exit 0 when the final score ≥ target, exit 2 when it misses — a
  machine-readable gate for pipelines; `task_config` flows through the env,
  `summary.json`, and `eval`. SPEC.md 22.1.
- **One-page HTML report (v0.8):** `report --html` writes a single
  self-contained `report.html` (summary block, embedded score-curve/Pareto
  SVGs, per-experiment and frontier tables; hand-written, no dependency).
  `--html --json` still keeps stdout as pure JSON. SPEC.md 22.2.
- **Visual dashboard (v0.9):** `autorefine dashboard` (or `streamlit run
  src/autorefine/dashboard_app.py`) — the "CSV in → model out" flow as a web
  page: upload (or point at) a CSV, label/head auto-inferred, run the loop
  with a live experiment table + best-score curve, then the §22.1 PASS/MISS
  verdict, the §21.2 SVG plots, and downloads for `report.html`,
  `best_spec.json`, `best_model.npz`, `experiments.jsonl`. The result stays
  on screen after download clicks or sidebar changes (session state) and a
  **New run** button resets it. Optional extra
  (`pip install autorefine[gui]`); the core stays streamlit-free — a pure
  `DashboardRunner` carries the run semantics, the app only renders
  (SPEC.md 23, §3). v0.12 adds four **decision views** explaining the
  improver's choices, derived purely from the existing update stream: a
  per-field win-rate bar chart, per-step spec-diff chips (`field: old → new`)
  in the note and table, a field × step mutation-timeline SVG (accepted /
  scored-rejected / unscored), and the bandit's UCB trace (bandit policy
  only) — rendered live during the run and again in the result section
  (SPEC.md 26, A16). v0.13 adds three **acceptance-gate views** explaining
  how the gate decided, derived purely from the same update stream plus the
  experiment log: a **candidate-score strip** (accepted / scored-rejected /
  unscored lanes — are the rejections near-misses or garbage?), a
  **score-vs-gen-gap scatter** with the S18.5 5%-of-score tolerance line
  (where the overfit penalty bites), and — for curriculum runs only — a
  **ladder curve** with a marker at each difficulty step-up (n_bits /
  p_flip); `report --html` gains the ladder section and `report --plot`
  gains `ladder_curve.svg` when the run has one (SPEC.md 27, A17).
  v0.14 adds four **learning views** explaining what the model actually
  *learned*: per-experiment **training curves** (train/holdout loss over
  steps — the only core change, `train()` now returns a bounded
  `loss_history`, ≤ 48 records, logged per experiment, SPEC.md 28.1);
  **final-model diagnostics** (per-class accuracy bars + a confusion matrix
  on the holdout, turning "96.4%" into "fails on class X", SPEC.md 28.2);
  an **error gallery** for media tasks (inline image thumbnails / audio
  waveforms of the misclassified holdout items, ≤ 8, SPEC.md 28.3); and a
  **spec → architecture diagram** (the best spec rendered as an annotated
  block diagram — layers, activation, family, knn_k, conv filters,
  SPEC.md 28.4). The dashboard renders all four under a "Learning views"
  subheader, `report --html` gains the architecture / diagnostics / gallery
  sections, and `report --plot` writes `architecture.svg` + `per_class.svg`
  + `confusion.svg`. The core stays PIL/soundfile-free (optional extras,
  §3/§21.1) and `import autorefine` never pulls them in (SPEC.md 28, A18).
- **Multi-run / policy views (v0.15):** the two questions a single run
  cannot answer. **Is the improvement real?** — `autorefine variance
  --data PATH --seeds N` re-runs the same `fit` budget under N distinct
  seeds (base `--seed`, default 7) and renders a **seed-variance box
  plot** on a fixed 0–100 axis: min/q1/median/q3/max of the finals
  (hand-rolled linear quantile), one paired baseline→final marker per
  seed, a dashed target line, and `passing = P/N` — writing
  `seed_variance.svg` + `seed_sweep.json` (the per-seed `verdict` carries
  the §22.1 gate; a MISS seed still exits 0 — a variance report is a
  measurement — and `--json` prints pure JSON). **What is the RL policy
  doing?** — `autorefine policy-report --task TASK` (or `--multi --tasks
  sine-v1,parity-v1,cartpole-v1`, SPEC.md 20.2) trains the meta-RL policy
  with the **opt-in per-step trace** (off by default, so existing
  `train_policy`/`train_multi_policy` callers and the §20.2 multi-task pin
  stay bit-identical) and renders **per-step action-probability bars**
  over the 77-action catalog (top-K + `rest`, the sampled action
  highlighted) and **per-task episode-return curves** — writing
  `action_probabilities.svg` + `task_returns.svg` + the raw
  `policy_trace.json`/`policy_returns.json`; `MetaRLPolicy
  .probabilities(env_state)` exposes the exact masked softmax `propose()`
  would sample from (pure, no RNG draw). The dashboard gains the two
  opt-in panels — a button-driven **Seed variance** sweep and an **RL
  policy view (precomputed)** that renders a `policy-report` directory's
  SVGs and **never runs RL live** (RL stays CLI-only, SPEC.md 23.1). Zero
  new core dependencies (NumPy only; both SVGs are hand-rolled valid XML)
  and `import autorefine` never pulls in streamlit (SPEC.md 29, A19, M18).
- **Comprehension visuals II (v0.16):** the six questions the
  v0.12–v0.15 views leave open. **When did the seeds diverge?** — the v0.15
  Seed variance view gains **per-seed best-score curves**: `seed_sweep` now
  carries a `curve` per seed (the baseline + the best score after every step,
  so early noise vs late divergence are visible), and `autorefine variance`
  writes `seed_curves.svg` next to `seed_variance.svg` — one polyline per
  seed on the fixed 0–100 axis, a per-seed legend, the dashed target line,
  and `n seeds / steps`; the app's Seed variance panel renders both (SPEC.md
  30.1). **Which value won?** — the v0.12 per-field win-rate bars gain a
  **field × value win matrix**: `field_value_stats` credits each update's
  `(field, new value)` pair (from its `spec_diff`) with its accepted/
  rejected outcome — value keys normalized (lists joined, `None` → `-`),
  fields and values sorted (values numeric-first) — rendered as one green
  cell per `(field, value)` (`fill-opacity` = win rate, `wins/trials` inside)
  live in the run loop and in the result's Decision views; `finish()`
  returns `field_value_stats` + `field_value_svg` (SPEC.md 30.2).
  **Where does the model fail?** — the flat-task complement of the media
  error gallery: **decision-boundary scatter** for 2-feature
  classification — `decision_boundary` paints an `n_grid × n_grid`
  prediction grid (holdout feature ranges, 5% padding, no RNG) plus the
  holdout points colored correct/misclassified; `finish()` returns
  `boundary` + `boundary_svg` and the app renders it in Learning views
  (one-feature, mse-head, and episode tasks get `None`) (SPEC.md 30.3).
  **How much is the score's noise?** — the **CI band on the score curve**:
  the §18.3 block-bootstrap holdout σ that the §18.5 gate already computes
  is now logged with every baseline/experiment/curriculum entry (`std`,
  `0.0` in legacy mode), and `svg_score_curve` draws a translucent
  `score ± std` band per positive-`std` row — `report --plot` picks it up
  from `experiments.jsonl` for free; legacy rows/old logs render byte-
  identical pre-v0.16 SVG (SPEC.md 30.4).
  **Which family won?** — the **model-family score bars**: `family_stats`
  rolls the scored log entries up per `spec.model_family` (best holdout
  score, that member's cost, trials / accepted count; a missing family is
  `unknown`), rendered as one bar per family on the fixed 0–100 axis with
  the top family highlighted — `finish()` returns `family_stats` +
  `family_bars_svg` and the app renders the bars in Decision views
  (SPEC.md 30.5). **Why did the policy explore?** — the **RL
  return-to-go trace**: `svg_policy_trace` renders the existing per-step
  `policy-report` trace as reward bars (green `r ≥ 0`, red `r < 0`), a
  return-to-go line + points, and a dashed running-mean baseline —
  `autorefine policy-report` now also writes `policy_trace_curve.svg`
  (named in its output listing), and the app's RL policy panel renders
  whichever of the three precomputed SVGs exist (a v0.15-era two-file
  directory still renders; the A19 warning text is preserved when none
  exist) (SPEC.md 30.6). All six are pure over data the loop already
  produces: zero new core dependencies, the SVGs are hand-rolled valid
  XML, and every pinned sequence (A1–A19, §18.7) stays green.
  (SPEC.md 30.1–30.6, A20.)
- **Optimization (v0.17):** two loop-level wins, both opt-in, both
  determinism-safe. **Stop at the plateau** — `run`, `fit`, and `variance`
  accept `--stall-patience K` (or `AutoRefineEnv(stall_patience=K)` /
  `DashboardRunner(stall_patience=K)`): K consecutive non-improving
  experiments end the run early with `finished_reason="stalled"` instead of
  burning the rest of the budget — an acceptance, a fresh `reset()`, or a
  curriculum step-up (a new ceiling) re-arms it; free duplicate/invalid
  rejections don't count. Default is **off**, so every pinned sequence
  (A1–A20, the §18.7 legacy run) is byte-identical, and the §22.1 gate
  still reads only the final score. **Parallel seed sweep** —
  `DashboardRunner.seed_sweep(seeds, workers=N)` and `variance --workers N`
  run the independent seeds on stdlib `concurrent.futures`
  processes: each seed is a self-contained deterministic run, so per-seed
  results are bit-identical to the serial path (except the timestamped
  `run_dir`); `workers=1` (the default) keeps the exact serial loop, and
  `on_update` raises with `workers > 1` (a callback can't cross a process
  boundary). Zero new dependencies; `import autorefine` stays clean
  (SPEC.md 31.1–31.2, A21.)
- **Optimization (v0.18):** **App perceived-perf** (app-only) — the
  "Run the seed sweep" button now streams the per-step `on_update`
  callback (the SPEC.md 31.2 serial path; the app keeps `workers=1`) into
  an `st.progress` bar that finishes at 100% with "complete", and the
  data-preview computation is `st.cache_data`-cached on
  (path, mtime, size) so an unchanged file is never re-probed. **Two-stage
  candidate screening** — opt-in `run`/`fit --screen-frac F` (or
  `AutoRefineEnv(screen_frac=F)` with `0 < F < 1`, preset
  `candidate_screening(frac)`): each candidate is first trained on the
  first `floor(F·n)` rows and only a strict beat of the fixed champion
  (the baseline spec screened once at `reset()`) spends the full
  training; screen rejections are logged `kind="screen"`, spend one budget
  experiment, and count for the v0.17 stall patience. Default
  (`screen_frac=1.0`) is pre-v0.18 exactly — every pinned sequence
  (A1–A21) stays green; the screening path has its own scripted pin (A22).
  **Trainer micro-opts** (fused MLP ops, vectorized bagging) are deferred:
  any float-order change would break bit-identical determinism (G2), and
  they're only worth the pin re-derivation if the loop-level wins fall
  short (SPEC.md 32.3). Zero new dependencies (SPEC.md 32.1–32.3, A22, M21.)
- **Coherency (v0.19):** three cross-check contracts over the accumulated
  knobs and features — no new features, no behavior change with any flag
  off (A1–A22 stay green). **Version story (33.1):** `pyproject.toml`
  `[project].version` and `autorefine.__version__` are one version, bumped
  together each round (v0.19 ⇒ `0.19.0`); the dashboard caption renders the
  same number; there is no third copy. **Knob registry (33.2):**
  `autorefine.KNOBS` is the one table for every tunable `AutoRefineEnv`
  knob (name, default, validator, spec ref, CLI subcommands, app widget);
  env validation and both presets (`search_quality_v04`,
  `candidate_screening`) route through it, and `cli.build_parser()` (a pure
  extraction of the parser from `main()`) is tested for CLI honesty — each
  claimed `--flag` per subcommand present and *absent* elsewhere (no
  `--screen-frac` on `variance`, 32.3) with parser defaults equal to
  registry defaults. **Interaction matrix (33.3):** defined, tested
  behavior for every pair of combinable round features; the one new
  semantic — a curriculum step-up **re-pins the screen champion** (the
  baseline spec re-screened on the harder dataset, free), so the 32.2
  K=1 champion semantics are per curriculum level; a screen-rejected spec
  never enters the Pareto frontier or the ensemble top-k. The summary's
  `baseline_screen_score` keeps reporting the reset champion; each
  re-pinned value rides its curriculum event row (`screen_champion`).
  Enforced by A23 (SPEC.md 33.1–33.3, M22.)
- **Coherency II (v0.20):** the cross-check discipline extends from the
  env knobs (33) to the two remaining cross-surface artifacts — no new
  features, no behavior change (A1–A23 stay green), and the version steps
  per 33.1 (v0.20 ⇒ `0.20.0`, both sources together). **Spec-space
  coherency (34.1):** the spec space is one object with three surfaces —
  the law (`config.py` constants + `ModelSpec.validate`), the
  `FIELD_CATALOG` catalog (RL/Gym/bandit), and the `FIELD_SAMPLERS`
  samplers — and the A24 tests pin their agreement: exact-value fields
  equal the config constants (order included), range-field catalog values
  sit inside the config ranges, every catalog architecture is legal for a
  non-knn family, a fixed-seed battery of every sampler draw stays in the
  registered space (field classes exhaustive), and the field sets are one
  set (`FIELD_NAMES == CATALOG_FIELDS`, the 25.7 `SEARCH_FIELDS`
  exclusion, `ORDERED_FIELDS`, `FAMILY_FIELDS`). **Summary contract
  (34.2):** `summary.json` is the cross-surface artifact (env writes;
  `report`/`eval`/dashboard/plotting read), and A24 pins its **exact
  top-level key set** for the legacy run (17 keys) and the full-flag run
  (+ `task_config`/`curriculum`/`screening`/`ensemble`), plus each
  internal dict's exact key set — a renamed or dropped key is now a suite
  failure, not a dashboard runtime surprise (SPEC.md 34.1–34.2, A24, M23.)
- **Coherency III (v0.21):** the loop closes over the last two
  convention-policed artifacts — no new features, no behavior change
  (A1–A24 stay green, logged bytes byte-identical), version per 33.1
  (v0.21 ⇒ `0.21.0`, both sources together). **Log kind registry (35.1):**
  the five `experiments.jsonl` row kinds (`baseline`, `experiment`,
  `screen`, `curriculum`, `invalid_spec`) are now one constant set —
  `KIND_*` + `LOG_KINDS` in `memory.py` (the log module) — and the
  emitter (`meta_env.py`) plus the consumers (`plotting.py`, `cli.py`,
  `dashboard.py`) route through it; the A25 source scans fail on a raw
  `"kind"` literal, an unregistered `KIND_` name, or a bare
  kind-literal comparison, and a live run's rows all carry registry
  kinds — the next new kind added to the logger is a suite failure
  instead of a silently-missing report row (the v0.18 gotcha).
  **Acceptance index (35.2):** the M# → SPEC § → A# → test-file table
  at the top of SPEC.md is the single source of truth, and the A25
  consistency tests close the spec/code drift loop — every A# defined
  in the SPEC is cited by ≥ 1 test, every `tests/test_*.py` cites an
  A#, and each index row's test file exists and cites that row's A#
  (SPEC.md 35.1–35.2, A25, M24.)
- **Generalization (v0.22):** the last two convention-policed contracts
  become nominal, registered interfaces — no behavior change (the
  77-action catalog, the proposal order, and A1–A25 stay
  byte-identical), version per 33.1 (v0.22 ⇒ `0.22.0`, both sources
  together). **Task ABC (36.1):** `tasks.Task` is a nominal ABC (not the
  former structural Protocol) with exactly two abstract methods
  (`make_dataset`, `score`) and two explicit attributes — `metric`
  (what the task reports: `accuracy`/`r2`/`mean_steps`/`success`,
  declared instead of inferred from `n_outputs`/label dtype; instance-
  set alongside `head` for the data tasks) and `capabilities`
  (`interactive` cartpole/gridnav, `grid`+`media` image/audio) — every
  registered task lists it as its base, and the app preview caption and
  the `fit` gate line now read the declared `metric`; the §23 plugin
  loader, `fit --task auto`, and new metrics (F1, log-loss) get a
  frozen interface to target, with A26 pinning the contract (an AST
  scan asserts no core code performs `isinstance(…, Task)`, so the
  duck-typed test fakes keep working). **ModelSpec field registry
  (36.2):** `improver/specspace.py` holds one
  `SpecField(name, space, validator, families, spec_ref, kind)` row per
  ModelSpec field in catalog order — the single table from which
  `FIELD_CATALOG`/`CATALOG_FIELDS`, the 77-action `ACTIONS` catalog,
  `FAMILY_FIELDS`, `FIELD_NAMES`, `ORDERED_FIELDS`, the `FIELD_SAMPLERS`
  exhaustiveness check, and the app's spec chips all derive — a new
  field is one registry row, and a surface (bandit, search enumerator,
  RL/Gym catalog, dashboard) that misses it fails the A26 suite: the
  model-space analogue of the C2 knob registry and the C4 kind registry
  (SPEC.md 36.1–36.2, A26, M25.)
- **Generalization (v0.23):** every run gets a canonical recipe and the
  acceptance gate becomes a small objective set — no behavior change
  under the defaults (with no `--gate` the gate line, PASS/MISS text,
  and rc 0/2 are byte-identical to pre-v0.23, and A1–A26 stay green),
  version per 33.1 (v0.23 ⇒ `0.23.0`, both sources together).
  **RunConfig (37.1):** a frozen dataclass serialized to
  `run_config.json` at `reset()` and mirrored as an additive
  `summary.json` key — a copy-pasteable full recipe for any run; `fit
  --from-run RUN_DIR` re-runs a finished data run exactly from that
  config (built-in-task runs are pointed at `autorefine run`),
  `report` prints a `reproduce (copy-paste):` line, and the rendered
  recipe parses against the real CLI (`fit_recipe` emits
  `--data`/`--seed`/knobs verbatim). **Objective gates (37.2):**
  `fit --gate NAME OP THRESHOLD` (appendable) and `variance --gate`
  accept an objective set over `score` (vs target), `train` (best
  candidate's train seconds), and `model` (total `best_model.npz`
  values) — all must pass for PASS; the default set is exactly today's
  §22.1 score gate, and the dashboard's result panel shows both the
  reproduce recipe and per-objective gate rows (SPEC.md 37.1–37.2,
  A27, M26.)
- **Tracking (v0.24):** runs over time become comparable data — no
  behavior change under the defaults (the `report --run` path, the
  A24 summary key sets, and A1–A27 stay green), version per 33.1
  (v0.24 ⇒ `0.24.0`, both sources together).
  **Run registry (38.1):** every finished run appends exactly one
  13-key entry (`run_id`, `task`, `seed`, `policy`, `final_score`,
  `target`, `met_target`, `finished_reason`, `experiments_run`,
  `wall_seconds`, `config_fp` — a 12-hex SHA-256 fingerprint of the
  run's recipe, `parent_run`, `timestamp`) to `runs/registry.json` —
  "which of my five attempts was best?" is now one command; a corrupt
  registry is moved aside (bytes preserved) and finish never crashes
  on it. **Lineage (38.2):** `fit --from-run` records the source
  run's dir name as `parent_run` (conditional `summary.json` key +
  registry field), so an iteration chain (`r1 → r2 → r3`) is
  recoverable. **`report --history` (38.3):** the registry as a
  table (or `--json`), with the PASS/MISS gate per run; `--run` and
  `--history` are mutually exclusive. **Past-runs view (38.4):** the
  dashboard renders the registry below the result views plus a
  two-run compare (per-field `best_spec` diff + score delta) — the
  diff is a streamlit-free helper, testable without an app session
  (SPEC.md 38.1–38.4, A28, M27.)
- **Tracking II (v0.25):** the in-flight run and the decisions inside it
  become legible — no behavior change under the defaults (`report --json`,
  the summary key sets, the registry, and the A1–A28 pins stay green),
  version per 33.1 (v0.25 ⇒ `0.25.0`, both sources together).
  **Watch mode (39.1):** `autorefine watch --run RUN_DIR` tails
  `experiments.jsonl` (byte-offset; a partial trailing line waits for the
  next poll) and re-renders the live header + ASCII score curve + Pareto
  until `summary.json` lands — the run dir may not exist yet, `watch`
  waits; `--tail` emits one compact JSON line per row for CI, `--clear`
  refreshes in place, and `--max-polls N` bounds a stuck wait (rc 0 on
  finish, rc 1 on the bounded wait, rc 130 on Ctrl-C). **Decision
  accounting (39.2):** `report --run` gains a "what happened" block —
  accepted vs rejected candidates with the rejected ones bucketed by the
  first failing gate (score → overfit → ci; the curriculum re-pins the
  running best, and free-duplicate rejections are reported as 0 by
  construction since they are unspent and unlogged),
  time-to-first-improvement (first accepted candidate + seconds after the
  baseline), and where the wall time went (baseline / candidates /
  eval+overhead) — pure post-hoc reconstruction from the jsonl + summary,
  no new logged field (SPEC.md 39.1–39.2, A29, M28.)
- **Simulation (v0.26):** answer “what would this run do / what would it
  have done” **without training** — no behavior change under the defaults
  (`fit` without `--dry-run` and `report` without `--what-if` are
  byte-identical to pre-v0.26, A1–A29 stay green), version per 33.1
  (v0.26 ⇒ `0.26.0`, both sources together).
  **Dry-run (40.1):** `autorefine fit --dry-run --data CSV` resolves
  data → task → head/metric → split — the *same* probe that would crash on a
  wrong label column, now surfaced before any epoch — and prints the plan:
  recipe, task (head/metric/n_outputs/state_dim), dataset size, budget, the
  policy’s expected candidate catalog, and a wall-time estimate from the
  registry’s same-task history (median `wall/experiments × budget`, or an
  explicit “no past runs” line) — creating **no** run dir / artifacts (rc 0
  plan; rc 1 on a bad label column, or with `--from-run`).
  **What-if (40.2):** `autorefine report --run DIR --what-if 'score>=97'
  'train<=30'` re-gates the logged history against a *new* objective set and
  reports the counterfactual final (best passing candidate) vs this run’s
  actual best — pure derivation from `experiments.jsonl`, no retraining (rc 0
  PASS / rc 2 MISS / rc 1 on a bad objective; `model` is unsupported — its
  size is only known from the trained artifact). The `--json` / `--history`
  machine and history paths are untouched (SPEC.md 40.1–40.2, A30, M29.)
- **Simulation III (v0.27):** complete the simulation surface — "will we get
  there, how did it go, and show me the loop" — still **without training**
  (`report` without `--project`/`--trace` and `run` without `--demo` are
  byte-identical to pre-v0.27, A1–A30 stay green), version per 33.1
  (v0.27 ⇒ `0.27.0`, both sources together).
  **Projection (41.1):** `autorefine report --run DIR --project
  [--target 95]` fits the Michaelis–Menten saturation curve `score(e) =
  Vmax·e/(Km+e)` (Lineweaver–Burk OLS, deterministic) to the same-task
  history — the registry's finished runs plus this run, deduped by run id —
  and answers "will I hit 95?": **CEILING** when the curve's asymptote does
  not exceed the target, else **~N more experiment(s)** (N = max(0,
  ⌈e_T − e_current⌉)); < 2 usable points degrade gracefully to
  INSUFFICIENT HISTORY (always rc 0 — an informational view; mutually
  exclusive with `--json`/`--history`/`--what-if`).
  **Trace (41.2):** `autorefine report --run DIR --trace` is a terminal
  replay — one decision line per `experiments.jsonl` entry (baseline →
  candidate → mutation → score → ACCEPTED/REJECTED + reason), with the
  documented rejection priority (score gate → overfit → CI gate) and the
  running best reconstructed exactly as in the accounting block; the whole
  loop is auditable without the GUI (rc 0; mutually exclusive with
  `--json`/`--history`/`--what-if`).
  **Demo (41.3):** `autorefine run --demo` runs one tiny, deterministic
  parity-v1 loop (3 experiments / 60 s wall / 10 s per train, v0.4 preset,
  `--seed`/`--runs-dir` apply, the other run flags are ignored) in seconds
  and prints the full loop narrated with the same trace renderer — baseline
  → candidates → gate → best spec — a cheap "here's what this tool does";
  a demo run is a run (normal run dir, artifacts, registry entry; rc 0)
  (SPEC.md 41.1–41.3, A31, M30.)
- **Use the model (v0.28):** close the last gap of the loop — the model
  itself — with three additive, pure commands over an existing run dir:
  **no new data, no training, no writes** (A1–A31 stay green, `fit`/
  `report`/`eval` byte-identical under the defaults), version per 33.1
  (v0.28 ⇒ `0.28.0`, both sources together).
  **Predict (42.1):** `autorefine predict --run DIR` scores **new** rows
  with the run's best model — exactly one input mode: `--row JSON` (an
  object keyed by feature name, case-insensitive, extra keys ignored — or
  a positional array), `--csv FILE` (header auto-detected and matched by
  name, an extra label column is fine; otherwise positional), `--stdin`
  (JSON lines), or `--item FILE` (repeatable; media tasks only — the file
  goes through the task's own `item_features` decode). A new row is
  standardized with the **task's own training stats** (`feature_mean` /
  `feature_std`, the §22.1 rule — `sine-v1` is the identity, already
  unit-scaled) and decoded with one forward pass: the softmax argmax
  mapped back through the user's labels (0/1 values or class names like
  `high`/`low`) + `--json` softmax probabilities; mse → the scalar. Human
  view: one `row_index<TAB>prediction` line per row (rc 0; rc 1 on any
  error — episode tasks point at `eval`, mode exclusivity, missing run
  dir, wrong cell count, undecodable file).
  **Compare (42.2):** `autorefine compare --run A --run B` is the app's
  Past-runs compare widget in the terminal — `diff_two_summaries` now
  lives in **core** `autorefine.dashboard` (no streamlit; the app's name
  *is* the core one, 38.4 unchanged): the final-score delta (B − A) +
  the per-field `best_spec` diff (only differing fields; identical specs
  → an `identical` line); `--json` the pure dict (rc 1 on a missing
  summary).
  **Explain (42.3):** `autorefine explain --run DIR [--target 95]` is a
  one-screen narrative assembled from the run's own artifacts —
  **result** (task/seed/policy, baseline → final, improvement factor,
  finished reason), **tried** (per-field trial/wins/win-rate), **why**
  (the baseline champion + the accepted chain in log order), **weak**
  (per-class holdout diagnostics + the weakest-class hint — “the model
  fails on class X”; an mse/episode run degrades to `n/a`), and **next**
  (the 41.1 budget projection: CEILING / ~N MORE / insufficient
  history) — rc 0 informational, `--json` exactly those five blocks
  (SPEC.md 42.1–42.3, A32, M31.)
- **Ergonomics (v0.33):** four polish items, all additive under the defaults (A1–A36 stay green), version per 33.1 (v0.33 ⇒ `0.33.0`, both sources together). **`--config` on any command (47.1):** every subcommand accepts `--config FILE` — a plain JSON object of kebab-case flag names, or a canonical `run_config.json` (schema `autorefine.run_config/1`) translated into the CLI vocabulary by `RunConfig.from_dict` + `runconfig_to_flags` — so a finished run dir is directly re-runnable: `autorefine fit --config runs/<id>/run_config.json`; explicit CLI flags (including `--flag=value`) beat the file, the file beats the defaults; `store_true` flags take a JSON boolean, `append` flags a JSON array, typed flags the parser's own `type`; unknown keys, bad JSON, and bad values are rc 1 before the command runs (a canonical recipe pairs with the command that owns its flags). **Help polish (47.2):** top-level `autorefine --version` (the 33.1 single source), an `examples:` block per subcommand (the README Quickstart invocations, rendered verbatim), and the exit-code table (`0` success / `1` error / `2` gate MISS / `130` watch Ctrl-C) in the top-level `--help`. **`--quiet` on run/fit (47.3):** `run --quiet` / `fit --quiet` suppress the per-experiment lines, the run-dir/baseline echoes, and fit's diagnostics block; the `=== summary ===` block and fit's gate verdict (and its rc semantics) are always kept; `--demo` and `fit --dry-run` ignore it (already minimal). **`autorefine share` (47.4):** `share --run DIR [--out FILE]` bundles a finished run into one deterministic, self-contained zip — `report.html` (always regenerated in memory, so a share of an old run dir carries a current renderer's HTML), `summary.json`, `run_config.json`, `best_spec.json`, and the run dir's flat SVGs; two shares of the same dir are byte-for-byte equal; a missing run dir is rc 1 (SPEC.md 47.1–47.4, A37, M36.)
- **Probabilistic outputs, curriculum beyond parity, portfolio (v0.32):** three generalizations, all additive under the defaults (A1–A35 stay green), version per 33.1 (v0.32 ⇒ `0.32.0`, both sources together). **Probabilistic CSV outputs (46.1):** `autorefine predict --run DIR --prob` now prints the calibrated **per-class probabilities** alongside every prediction — one `<row>\t<class>: p=…` line per row per class (user labels, class order; `--json` already carried them and is unchanged) — and `fit --metric logloss` scores a softmax-head CSV by **log-loss** instead of accuracy (score = 100·(1 − mean NLL/ln 2), clamped at 0: a perfect model → 100, a constant binary model → 0); the metric is a declared task property (G1), `fit --dry-run --metric logloss` names it in the plan (the accuracy headroom note is omitted — 46.1.3), `fit --metric accuracy` (the default) is byte-identical to no flag, and an mse-head file with `--metric logloss` fails at the probe before any training. **Curriculum beyond parity (46.2):** `run --curriculum` is no longer parity-only — `sine-v1` gets a 3-axis difficulty ladder (label noise → frequency scale → amplitude; 27 levels, `sine_ceiling` = 100·(1 − noise²/var) on a fixed grid) and `cartpole-v1` a single-axis one (initial-condition box scale 1.0 → 2.0 → 3.0; ceiling proxy = 500 mean-steps), both on the *same* curriculum API the §20.1 parity ladder uses — `level_params()` is splatted into the curriculum event rows (parity rows keep their historical `n_bits`/`p_flip` keys byte-identical), `report`/`eval` rebuild the level task from the row, and the ladder curve labels sine/cartpole markers with their difficulty string. **Portfolio mode (46.3):** `fit --tasks A.csv,B.csv` runs one **shared budget** over several CSV datasets — experiments split `ceil(total/N)` per task, one shared wall-clock deadline, tasks in the given order, each fully gated on its own (rc 0 iff *every* task passes, 2 if any misses, 1 on a resolve failure before training); `fit --dry-run --tasks …` prints the per-task plans with zero training; cross-task policy transfer (MetaRL over a shared budget) is the documented follow-up (SPEC.md 46.1–46.3, A36, M35.)
- **Adapts to more use cases (v0.31):** three use-case widenings, all additive under the defaults (A1–A34 stay green), version per 33.1 (v0.31 ⇒ `0.31.0`, both sources together). **Temporal / walk-forward split (45.1):** `fit --temporal` makes the CSV split respect **row order** instead of shuffling — train = first rows, holdout = next block, generalization = last — so time-ordered data (finance, sensor, logs) can be fitted with no leakage; `split_mode` is a `Task`-ABC property defaulting to `"random"` (bit-identical to today's split). **Text modality (45.2):** a fourth modality on the *same* protocol — `fit --task text --data DIR` over a labelled directory of `.txt` files (subfolder per class, or an `index.csv`), pure-numpy hashed word unigram+bigram features (128-dim, crc32 buckets) — so `auto` detection, dry-run, preflight, reports, and the dashboard all work unchanged. **Gradient-boosted trees (45.3):** status note — `BoostingEnsemble` already shipped in v0.5 (SPEC §19.2) and is in the model space; v0.31 **verifies** it (train/score + bandit catalog) rather than re-implementing (SPEC.md 45.1–45.3, A35, M34.)
- **Stable scores, visible features (v0.30):** two opt-in scoring surfaces, both **no behavior change under the defaults** (kfold = 0 is bit-identical to the legacy single-split path — A1–A33 stay green; the A24 summary key set and the `search_quality` 5-key set untouched), version per 33.1 (v0.30 ⇒ `0.30.0`, both sources together). **K-fold holdout scoring (44.1):** `--kfold K` (on `run` and `fit`, default 0 = off) scores each *trained* model on **K distinct held-out subsets** and averages — the score becomes the fold mean and the fold σ feeds the CI gate, so baseline/candidate comparisons are materially more stable for small tabular datasets without K× retraining; CSV folds draw distinct, deterministic random subsets of the non-train pool (documented seed scheme — no leakage), generative tasks fold to distinct fresh point sets; the summary carries a **conditional** `kfold` key (absent at 0), and `RunConfig` / `fit --from-run` round-trip the recipe's K. **Feature importance (44.2):** `report --importance [--importance-repeats R]` (default 5) — pure-numpy **permutation importance** over the run's holdout ("importance = score drop when the column is shuffled"), answering *which columns does my winning model actually use*; sorted feature list, `n/a` line (still rc 0) on episode / non-flat / non-softmax-mse runs (SPEC.md 44.1–44.2, A34, M33.)
- **Trust before you train (v0.29):** catch the bad idea before the 30-minute run — no behavior change under the defaults (A1–A32 stay green; the 40.1 dry-run rc contract unchanged), version per 33.1 (v0.29 ⇒ `0.29.0`, both sources together). **Preflight (43.1):** `fit --dry-run` now also prints a **data-health block** from the probe task it already builds — class balance + a headroom note against your `--target` (“majority class alone scores 90.0 — 5.0 points of headroom below target 95”, or “already meets target”), constant / near-constant (≥ 99% one value) columns named with the offending value/share, per-column missing values (an *ignored* column is named as such), and min/max/mean per feature; media tasks degrade to item count + class balance; bad health is a **warning in the plan, never a failure** (rc 0). **Doctor (43.2):** `autorefine doctor [--runs-dir DIR]` — the friendly first command: version + Python, numpy, the optional extras (via `find_spec` + dist metadata — never imported, A23 invariant), a micro parity-v1 smoke train (< 1 s → ok, slow → warn, error → FAIL, into a throwaway dir), and runs-dir writability; `result: N ok, N warn, N FAIL`; rc 0 unless something FAILs (SPEC.md 43.1–43.2, A33, M32.)
- **Input modalities (v0.10):** `autorefine fit --data DIR` now accepts a
  labelled directory — one subfolder per class (or an `index.csv`) of
  **images** (PNG/JPG/JPEG/BMP/GIF; grayscale 32×32 features; optional
  `autorefine[image]` extra) or **audio clips** (16-bit PCM WAV via stdlib;
  MP3 via optional `autorefine[audio]`; NumPy log-mel features) —
  `ImageTask`/`AudioTask` implement the *same* fitting protocol as
  `CsvTask` (the §22.1 head/split/score rule verbatim), so the improver
  loop, spec space, `eval`, reports, and the dashboard (which also accepts
  a directory) all work unchanged. `fit --task auto` detects the modality
  from the directory; `--task {csv,image,audio}` forces it; a mixed
  directory asks for an explicit choice. The core stays PIL/soundfile-free
  (SPEC.md 24, §3); acceptance A14 runs `fit` on both sample directories
  with the §22.1 gate.
- **Modality-aware model families (v0.11):** the improver now has
  structurally better answers per modality — `knn` on every task and
  `convnet` on grid tasks, where it exploits the 2-D layout of image
  pixels and audio spectrograms that flat-feature families only see
  flattened. Both keep the `forward -> (n, n_out)` contract, so scoring,
  the §19.3 ensemble (mixed grid/flat members filter to the top member's
  contract), `eval`, reports, and the dashboard work unchanged; `eval`
  loads both new families. Invalid specs (a convnet on a flat task, an
  external agent's malformed spec) are rejected in `step()` with
  `reason="invalid_spec"` — no budget spent, not dedup-marked, logged —
  closing a pre-existing crash path. The v0.10 flat features stay
  bit-identical (the audio flat path is the same 26-d time-mean, the image
  grid a pure reshape). SPEC.md 25, acceptance A15.
- **Reproducibility:** one seed pins every RNG stream; two same-seed runs
  produce identical experiment sequences and best specs (acceptance A2).

## Run artifacts

Each run writes to `runs/<task>-seed<seed>-<timestamp>/`:

| File                | Contents |
|---------------------|----------|
| `experiments.jsonl` | every experiment: spec, mutation fields, scores, gen gap, acceptance |
| `best_spec.json`    | final best `ModelSpec` |
| `best_model.npz`    | NumPy weights (loadable with `allow_pickle=False`) |
| `summary.json`      | baseline vs final score (raw + `*_effective`, v0.4), improvement factor, active search-quality preset, per-mutation win rates, Pareto frontier, (v0.6) `curriculum` ladder with `--curriculum`, (v0.23) the additive `run_config` key |
| `run_config.json`   | the canonical frozen `RunConfig` for the run — the single recipe `fit --from-run`, the `report` reproduce line, and the app's copy-paste recipe all read (v0.23) |

## Project layout

```
src/autorefine/
├── config.py          # ModelSpec (validated, hashed; mlp/tree/boost/knn/convnet families), Budget
├── tasks/
│   ├── base.py        # minimal Task protocol (make_dataset + score)
│   ├── cartpole.py    # CartPoleV1: dynamics, reference controller, dataset
│   ├── sine.py        # SineRegressionV1: function fitting (mse head)
│   ├── gridnav.py     # GridNavV1: toroidal navigation policy
│   ├── parity.py      # ParityTask (n_bits, p_flip) + Parity4V1 + parity_ceiling (v0.6)
│   ├── csv.py         # CsvTask: user CSV data (v0.8; head/split inference, seed splits)
│   ├── media.py       # shared labelled-directory rules for image/audio (v0.10)
│   ├── images.py      # ImageTask: labelled image directory (v0.10; optional Pillow)
│   ├── audio.py       # AudioTask: labelled audio directory, WAV/MP3 (v0.10; optional soundfile)
│   └── __init__.py    # TASKS registry
├── models/
│   ├── mlp.py         # NumPy MLP (softmax + mse heads), pickle-free checkpoints
│   ├── trees.py       # TreeEnsemble (bagged) + BoostingEnsemble (boosted, v0.5) CART
│   ├── knn.py         # KNN: memorized-train-split k-NN family (v0.11)
│   ├── convnet.py     # ConvNet: small NumPy convnet over grid features (v0.11)
│   └── optimizers.py  # SGD / momentum / Adam
├── trainer.py         # deterministic training, mid-loop time cap, family dispatch
├── evaluator.py       # holdout + gen scoring, block-bootstrap CI (v0.4), episodes
├── improver/
│   ├── meta_env.py    # AutoRefineEnv — reset()/step(); v0.4 CI/efficiency/gen-gap rule
│   ├── policy.py      # v1 search: hill-climb + restarts + local refinement (v0.4)
│   ├── bandit.py      # BanditPolicy: UCB over spec fields (v0.3; local/family-aware v0.4)
│   ├── actions.py     # per-field mutation samplers (uniform + local modes, v0.4)
│   ├── catalog.py     # discrete mutation catalog (77 actions, v0.11) + family-relevant fields
│   ├── curriculum.py  # ParityCurriculum: adaptive-difficulty ladder (v0.6)
│   └── rl_policy.py   # MetaRLPolicy: REINFORCE on the loop (masked v0.4; multi-task v0.6)
├── pareto.py          # score-vs-train-time frontier (efficiency memory)
├── gym.py             # optional Gymnasium adapter
├── budget.py          # BudgetManager (experiments / wall time / train time)
├── memory.py          # JSONL log + artifact writers/readers
├── plotting.py        # hand-written ASCII/SVG charts + self-contained HTML report
├── plugins.py         # entry-point plugin loader (v0.7)
├── dashboard.py       # DashboardRunner: streamlit-free core of the v0.9 dashboard
├── dashboard_app.py   # Streamlit app (optional extra: pip install autorefine[gui])
└── cli.py             # run / fit (v0.8) / report (--plot, --json, --html) / eval / plugins

examples/
├── gym_dqn.py         # DQN-style agent through the Gymnasium adapter (v0.7)
├── tabular.py         # custom tabular task + improver run + >95% target gate (v0.7)
└── make_sample_media.py  # synthesize sample tone/image directories for `fit` (v0.10)
```

## Extending

All four paths are demonstrated by working, tested examples (SPEC.md 17/21.3):

- **New task:** for tabular CSV data there is *no code to write* (v0.8):
  `autorefine fit --data FILE --label COL --target T` wraps your file in the
  built-in `csv` task and runs the same loop (SPEC.md 22.1) — and the v0.9
  dashboard (`autorefine dashboard`, SPEC.md 23) is that same flow as a web
  page: upload the CSV, watch every experiment live, download the gated
  model + plots. Labelled image/audio directories are equally code-free
  (v0.10): `autorefine fit --data DIR` on one-subfolder-per-class media
  builds the `image`/`audio` task automatically (SPEC.md 24). Otherwise
  implement the minimal `Task` protocol (`tasks/base.py`) —
  `make_dataset` + `score` (interactive tasks add `step_vec`,
  `reference_action`, ...) — and register it in `TASKS`.
  *Worked example:* `parity-v1` (`tasks/parity.py`) — a noisy 4-bit XOR
  classification task added with zero changes to the core loop.
  *Worked example:* `examples/tabular.py` (SPEC.md 21.5) — a custom
  2-D tabular task (quadrant XOR) registered in `TASKS`, a bounded bandit
  run from a 56.0 baseline to 99.5 (seed 7), and a gate on the *user's*
  target (`--target`, default 95.0) against `final_best_score` — the loop
  maximizes the validated score; checking the bar is the user's step.
  From a separate package, skip the import entirely: an entry point in the
  `autorefine.tasks` group is discovered at CLI start, validated, and merged
  into `TASKS` (`plugins.py`; check with `autorefine plugins list`).
- **New policy:** implement `propose(env_state) -> spec_dict`; the env's
  `state["last"]` carries the previous step's acceptance and mutated fields.
  *Worked example:* `BanditPolicy` (`improver/bandit.py`) — a UCB bandit
  over spec fields driven by exactly that feedback; also available as
  `--policy bandit`.
- **More model families / spec fields:** extend `ModelSpec.validate` +
  `actions.py` / `catalog.py`; validation and dedup follow automatically.
  *Worked example:* the `label_smoothing` field (0.0–0.15, softmax head,
  default 0.0 keeps pre-v0.3 spec JSON loadable) — one constant in
  `config.py`, one sampler in `actions.py`, one catalog entry in
  `catalog.py`, consumed by `MLP.loss_and_grads`. The v0.5 fields
  (`lr_schedule`, `early_stopping_patience`, `init_scale`,
  `gradient_clipping`) follow this exact 4-line pattern and are consumed by
  `trainer.py`; the `boost` family adds one entry to `MODEL_FAMILIES`,
  one branch in `trainer.train`, and `FAMILY_FIELDS["boost"]`. The v0.11
  modality-aware families are the same pattern at family scale: `knn`
  (`models/knn.py`, a `knn_k` field, one `FAMILY_FIELDS` entry, one
  trainer branch) and `convnet` (`models/convnet.py` + the optional grid
  protocol on tasks — `grid_capable` / `feature_grid` / `grid_dataset()` —
  reusing the shared neural training loop verbatim); both reach 100 on the
  bundled tone and shape samples, and `eval`, the §19.3 ensemble, and the
  loop's `invalid_spec` rejection (SPEC.md 25.4) pick them up with no
  protocol change (77 catalog actions; SPEC.md 25.5, A15).
