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
  (SPEC.md 23, §3).
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
| `summary.json`      | baseline vs final score (raw + `*_effective`, v0.4), improvement factor, active search-quality preset, per-mutation win rates, Pareto frontier, (v0.6) `curriculum` ladder with `--curriculum` |

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
