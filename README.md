# AutoRefine

An environment where ML models iteratively improve through autonomous means.
The improver loop proposes model-configuration changes, trains, evaluates on
splits it never saw, records the outcome, and proposes the next change — all
within a hard budget, fully reproducible, NumPy-only.

**Current version: 0.51.0.** Input modalities: CSV, plus labelled
image / audio / text directories via `autorefine fit --data PATH`.

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

# run the test suite (acceptance pins A1-A55)
python -m pytest tests/
```

## Quickstart: the visual dashboard (UI)

The same improver loop as a web page — no terminal required:

```bash
pip install autorefine[gui]            # needs streamlit
python -m autorefine dashboard         # or: streamlit run src/autorefine/dashboard_app.py
```

The first screen is the **Quickstart itself** (SPEC §65) — the same four
steps the CLI and this README carry, from one source — with a **Run the
demo now** button: a tiny deterministic parity-v1 loop in seconds (a demo
run is a real run — artifacts + a registry entry, re-openable via
`autorefine report --run <dir>`). Once you pick data, the app is a
five-tab layout driven from the **sidebar**:

1. **Setup** — upload a CSV (or a labelled image/audio/text directory),
   label auto-detect, target score; a **Preset** fills a bundle of settings
   in one click; the *Advanced* expander hides the rest (policy, seed,
   experiments, max train seconds, search quality, runs dir); toggle
   **Narrate the loop** for a plain-English line per experiment.
2. **Run** — every experiment appears as it happens (live table +
   best-score curve); **Stop** any time (keyboard `S`; `R` = Run) for an
   honest partial run with full artifacts; **Copy status** gives a
   self-contained "how's it going?" card; the *reference run* picker draws
   a past run's curve beneath the current one.
3. **Results** — the PASS/MISS verdict, the plots, and the downloads:
   `report.html`, `best_model.npz`, `best_spec.json`,
   `experiments.jsonl`, `run_config.json`, and a self-contained
   **share bundle** (`.zip`).
4. **Compare** — seed sweeps, the RL policy view, and past runs (re-open
   and diff 2–3 runs side by side).
5. **Experiments** — every candidate with the *reason* it was kept or
   rejected.

**Steer, don't just watch.** The *Steer the search* expander (Setup tab)
adds human-in-the-loop rules — **pin** a field, **bias** a mutation toward
one value, or **constrain** an allowed set (SPEC §59.2) — all opt-in. For
the full parameter surface, the Results tab has the *Parameter inspector*
(SPEC §59.1) and the *What-if & comparison* panel (SPEC §60): live what-if
previews, the spec-fingerprint "DNA", the interaction heatmap, and
objective-weight sliders.

> The dashboard is a thin renderer over the same core as the CLI
> (SPEC §23); the `rl` policy stays CLI-only (`autorefine fit --policy
> rl`). The Quickstart panel, `autorefine quickstart`, and this section
> all read from one source (SPEC §65), so they never drift apart.

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

The improver loop is the same for every task and every model family; what
changes is the *space* it searches and the *surfaces* that explain it.

**Loop invariants (always on):**

- **Honesty** — candidates are scored on a holdout split the improver never
  sees, plus a `gen_score` under different seeds; acceptance requires
  `gen_gap < 5%` to catch overfitting (A3).
- **Efficiency memory** — every run tracks the score-vs-train-time Pareto
  frontier (`summary.json`: `pareto_frontier`, `best_score_at_1s`,
  `efficiency_at_1s`).
- **Budget** — hard caps on experiments, total wall time, and per-training
  time (enforced mid-training); the budget is per-episode, so `reset()`
  after `done` starts a fresh run.
- **Dedup** — identical specs are rejected for free (`reason="duplicate"`).
- **Reproducibility** — one seed pins every RNG stream; two same-seed runs
  produce identical experiment sequences and best specs (A2).

**The space being searched:**

- **Unit of improvement** — a `ModelSpec`: model family, architecture,
  optimizer, learning rate, batch size, weight decay, training steps, input
  noise, activation, plus `label_smoothing`, the mlp knobs
  `lr_schedule` / `early_stopping_patience` / `init_scale` /
  `gradient_clipping`, and `knn_k`. All default to the legacy behavior, so
  old spec JSON still loads and legacy training stays bit-identical (A8).
- **Tasks** — `cartpole-v1` (mean episode survival), `sine-v1` (100·R²,
  regression), `gridnav-v1` (navigation policy), `parity-v1` (noisy 4-bit
  XOR — linear models provably can't solve it; an MLP reaches the ≈75 Bayes
  ceiling), `csv` (your own CSV), and labelled image / audio / text
  directories.
- **Model families** — `mlp` (depth 0..3; 0 = linear), `tree` (bagged
  CART), `boost` (gradient-boosted CART — clearly stronger than bag on
  regression), `knn` (memorized k-NN), and `convnet` (grid tasks only:
  image pixels and audio spectrograms; SPEC §25).

**Feature → command.** Every row is opt-in or additive — with the defaults
off, every pinned sequence (A1–A55) stays byte-identical. [SPEC.md](SPEC.md)
carries the full detail; this table is the index.

| Feature | What it does | How to use it | SPEC § |
|---------|--------------|---------------|--------|
| Search quality | CI-aware + efficiency-aware acceptance gate (`Δeff > z·SE`, overfit penalty) | default on `run`/`fit`; `--search-quality legacy` to opt out | 18 |
| Frontier ensembling | Top-2 frontier models averaged, reported as a final evaluation | `run --ensemble-final` | 19.3 |
| Curriculum | Steps task difficulty up as the improver saturates (parity / sine / cartpole ladders) | `run --curriculum` | 20.1, 46.2 |
| Meta-RL improver | REINFORCE policy trained on the loop itself; task-conditioned transfer | `run --policy rl`; `policy-report --multi --tasks ...` | 20.2 |
| Default dataset sizes | Task-declared dataset size when none is passed | (automatic) | 20.3 |
| External RL agent | The loop as a Gymnasium env for outside agents | `python examples/gym_dqn.py` | 21.1 |
| Report outputs | Machine JSON, ASCII + SVG plots, one-page HTML | `report --json` / `--plot` / `--html` | 21.2, 22.2 |
| Plugin loader | External packages contribute tasks/policies via entry points | `plugins list` | 21.3 |
| Data-in fit + gate | No-code fit on your CSV; exit 0/2 as a pipeline gate | `fit --data FILE --label COL --target T` | 22.1 |
| Visual dashboard | Upload data, watch every experiment live, verdict, plots, downloads | `dashboard` (`pip install autorefine[gui]`) | 23 |
| Input modalities | Labelled image / audio directories through the same protocol | `fit --data DIR` | 24 |
| Modality-aware families | `knn` everywhere; `convnet` on grid (image/audio) tasks | (automatic) | 25 |
| Seed variance | Same budget under N seeds — "is the improvement real?" | `variance --data PATH --seeds N [--workers N]` | 29.1, 30.1 |
| RL policy view | Per-step action probabilities + per-task return curves | `policy-report --task T` | 29.2, 30.6 |
| Comprehension visuals | Decision views, acceptance-gate views, training curves, diagnostics, error gallery, architecture diagram | dashboard; `report --html --plot` | 26–30 |
| Stall patience | End the run early after K non-improving experiments | `run/fit --stall-patience K` | 31.1 |
| Parallel seed sweep | Independent seeds on processes, bit-identical per seed | `variance --workers N` | 31.2 |
| Candidate screening | Subsample-screen candidates; full-train only the champions | `run/fit --screen-frac F` | 32.2 |
| Coherency registries | One table for env knobs, spec fields, and log kinds — cross-surface pins | (internal; A23–A26) | 33–36 |
| RunConfig + reproduce | A canonical copy-pasteable recipe; re-run a finished run exactly | `fit --from-run DIR`; the `report` reproduce line | 37.1 |
| Objective gates | PASS on a set of objectives: score / train-time / model-size / constraints | `fit --gate NAME OP THRESHOLD` (appendable) | 37.2, 61.1 |
| Run registry + history | Every finished run logged — "which attempt was best?" | `report --history`; `compare --run A --run B [--run C]` | 38 |
| Watch mode | Tail a live run; re-render charts, or `--tail` JSON for CI | `watch --run DIR` | 39.1 |
| Decision accounting | "What happened": rejections by gate, time-to-first-improvement, wall-time split | `report --run DIR` | 39.2 |
| Dry-run + preflight | The full plan + a data-health report before training (no run created) | `fit --dry-run --data FILE` | 40.1, 43.1 |
| What-if re-gating | Replay the logged history under a different gate — zero training | `report --run DIR --what-if 'score>=97'` | 40.2 |
| Projection | Saturation-curve fit over past runs — "will I hit 95?" | `report --run DIR --project --target 95` | 41.1 |
| Trace | A terminal replay of every decision in the loop | `report --run DIR --trace` | 41.2 |
| Demo | A tiny narrated loop in seconds — "here's what this tool does" | `run --demo` | 41.3 |
| Doctor | Version / numpy / extras / smoke train / runs-dir check | `doctor` | 43.2 |
| Predict | Score NEW rows with the run's best model; `--prob` for calibrated probabilities | `predict --run DIR --row/--csv/--stdin/--item` | 42.1, 46.1 |
| Explain | One-screen narrative: result / tried / why / weak / next | `explain --run DIR` | 42.3 |
| K-fold holdout scoring | K distinct holdout subsets → steadier baseline/candidate comparisons | `run/fit --kfold K` | 44.1 |
| Feature importance | Permutation importance — which columns does the model actually use? | `report --importance` | 44.2 |
| Temporal split | Walk-forward (row-order) split for time-ordered CSVs | `fit --temporal` | 45.1 |
| Text modality | Labelled `.txt` directories; hashed n-gram features | `fit --task text --data DIR` | 45.2 |
| Log-loss metric | Softmax CSV scored by calibrated log-loss instead of accuracy | `fit --metric logloss` | 46.1 |
| Portfolio mode | One shared budget across several CSV datasets, each gated on its own | `fit --tasks a.csv,b.csv` | 46.3 |
| Ergonomics | `--config FILE` on any command, `--quiet`, `--version`, help examples + exit codes | (flags) | 47 |
| Share bundle | One deterministic self-contained .zip of a finished run | `share --run DIR` | 47.4, 50.2 |
| Beginner onboarding | Guided setup, knob glossary, presets, a "what happened" narrative, narrate toggle | dashboard | 48 |
| Advanced analysis | Interactive what-if re-gating, editable best-spec re-score, policy A/B | dashboard | 49 |
| Tabs + live control | Five tabs; Stop mid-run; live rejection reasons; keyboard + colorblind-safe palettes | dashboard | 51 |
| Decision views | Per-candidate gate number-line, live Pareto frontier, live champion spec | dashboard | 53 |
| Live run experience | Opt-in stream mode, mid-run status snapshot, reference-run overlay, wall-time cost strip, hover tooltips | dashboard | 54–55 |
| Professional polish | A provenance "certificate" card, a report cover, explicit app states | `report --certificate`; dashboard | 56 |
| Research surfaces | Spec-lineage DAG, per-field response surfaces, gate-decision region, bandit belief bars, the run dossier | dashboard; `dossier --run DIR` | 57–58 |
| Parameter inspector + steering | Pin / bias / constrain fields; every knob with its space, best-seen value, and measured effect | dashboard | 59.1–59.2 |
| Manual / expert mode | Set the exact spec over the field registry and train it once | `manual --task T --set field=value` | 59.3 |
| What-if & comparison | Live previews, the spec "DNA" fingerprint, interaction heatmap, objective-weight sliders | dashboard | 60 |
| Specialized scenarios | Constraint-aware gates (per-class F1, ECE, cost, monotonicity) + stress scenarios | `fit --gate per_class_f1>=0.9` etc. | 61 |
| Audience reports | exec / domain / technical / regulator re-skins; `verify` independently re-derives every reported number | `report --audience exec`; `verify --run DIR` | 62 |
| Report formats + guides | Markdown / plain-text / PDF; a "what this model does" user guide; the go/no-go decision artifact | `report --format md|txt|pdf`; `report --user`; `report --decision` | 63 |
| Benchmark report | Cross-run leaderboard, generalization matrix, best-score trend; ± uncertainty on every headline | `report --benchmark` | 64 |
| Quickstart in the UI | The same four steps in the README, the CLI, and the app's first screen — one source | `quickstart` | 65 |

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
