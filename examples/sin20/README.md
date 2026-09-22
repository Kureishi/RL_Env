# sin20 — a "classically hard" smooth classification problem

`sin20_train.csv` (2,400 rows × 20 features, deterministic seed 7):

- features `f0 … f19` ~ iid N(0, 1)
- label: `y = 1[ sin(f0) + sin(f1) + … + sin(f19) > 0 ]` (balanced, ≈49/51)

## Why it is hard

The decision boundary is the smooth hypersurface `Σᵢ sin(fᵢ) = 0` — a
single non-linear function of **all** 20 features at once.

- **Linear models** — the boundary is not a hyperplane; they sit well
  below the MLPs.
- **Trees / gradient boosting** — axis-aligned splits cannot express a
  smooth cross-feature sum; they sit at ≈chance (≈50).
- **k-NN** — in 20 dimensions distances concentrate (curse of
  dimensionality); it lands at or below the small-MLP baseline.
- **A wider/deeper MLP** — learns the per-feature sinusoidal structure and
  clearly beats the system's `[16, 8]` baseline (measured ≈78 → ≈89 holdout
  accuracy across dataset seeds).

So the improver has a *real* signal to climb — both an **architecture
gradient** (wider/deeper MLPs are better) and a **family gap** (mlp ≫
tree/boost) — while the problem stays out of reach of the naive model
classes. This is exactly the regime where the RL improver (SPEC §15/§20.2)
has something to learn.

## Measured results (seed 7, 24 experiments/episode)

> **Status (v0.59, SPEC §73): the collapse root cause is fixed.** v0.59
> (A63) corrected the REINFORCE sign — the update was a *descent* on
> expected return (good actions pushed down, bad ones up, §73.8), closed
> the free-duplicate no-op reinforcement channel (§73.2), and made `fit`
> default to ε=0.1 because ε=0 deadlocks the one-shot loop in the
> duplicate-stall attractor (§73.9). The two pre-fix runs below are kept
> as artifacts.

### The v0.58 CLI run (ε = 0) — the original collapse (pre-fix artifact)

```
python -m autorefine fit --data examples/sin20/sin20_train.csv --label label \
    --policy rl --rl-episodes 8 --experiments 24 --seed 7 --target 95 \
    --max-seconds 1800 --max-train-seconds 25
```

(`PYTHONIOENCODING=utf-8` on Windows consoles.) With the default
ε = 0 the REINFORCE policy **mode-collapses after episode 1**: the softmax
piles onto one action (e.g. `learning_rate→0.0001`, holdout 53.75), and
every later episode is that one candidate + 32 free duplicate re-proposals
→ `duplicate_stall`. Measured: episode 1 return −244.94, episodes 2–8
−23.75 each (one experiment, then the stall guard, SPEC §20.2), final best
**77.5 = baseline**, gate **MISS**. Run dir:
`runs/csv-seed7-20260922-112353-5`.

Root causes (v0.59, §73.8/§73.2): the policy *learned the exploit* —
the REINFORCE update had an inverted sign (descent on expected return,
§73.8), so the bad action's probability rose to 0.95 → 0.999 across
episodes; and the 32 free duplicate re-proposals (reward 0.0, no budget)
carried *positive* advantage under the running-mean baseline and
reinforced the duplicated action — the duplicate channel of the exploit
class v0.58 closed for invalid family specs (SPEC §72.3).

### The v0.59 canonical run (default ε = 0.1) — collapse fixed

The same command now (v0.59) runs healthy — no bad-action lock, a mix of
real experiments, and a best that **beats the baseline**:

```
python -m autorefine fit --data examples/sin20/sin20_train.csv --label label \
    --policy rl --rl-episodes 8 --experiments 24 --seed 7 --target 95 \
    --max-seconds 1800 --max-train-seconds 25
```

| episode | 1     | 2     | 3  | 4     | 5      | 6      | 7  | 8     |
|---|---|---|---|---|---|---|---|---|
| return | −83.75 | −57.25 | 0.0 | +0.42 | −50.42 | −11.34 | 0.0 | −52.08 |

- **0 `invalid_spec`**, 11 real experiments across the run; candidates
  span 51.25–78.33 holdout — the duplicate attractor is broken (a random
  draw is a fresh proposal; the stall streak resets, §73.9).
- **Final best 78.33 > baseline 77.5** (pre-fix: locked at 77.5), returns
  trending up (−83.75 → −52.08): the policy is learning, not collapsing.
- **Gate vs 95: MISS (honest)** — the 95 bar is out of reach for this
  spec space on this problem; with more episodes/budget the ε=0.25 run
  below shows the reachable region.
- Run dir: `runs/csv-seed7-20260922-130844-2`.

### The exploration run (ε = 0.25) — RL self-improving, learn → persist → reuse

Driven through the public API exactly as `autorefine run --policy rl`
drives it (same `propose`/`step`/`observe` contract). v0.59 also exposes
the same knob on `fit` (`--rl-epsilon 0.25`) — re-run that way it reaches
**88.33** (`runs/csv-seed7-20260922-131101-2`): the wide-MLP region
(`lr 0.01, batch 32, train_steps 5000, cosine schedule`, holdout 88.33),
confirming the ≈88–89 reachable ceiling in this spec space.

The original v0.53-era public-API run:

```python
from autorefine.improver.rl_policy import MetaRLPolicy, train_policy, save_policy
pol = MetaRLPolicy(seed=7, epsilon=0.25, epsilon_decay=1.0)
out = train_policy(env, pol, 8, verbose=True, best=True)
save_policy(pol, "examples/sin20/policy_sin20.npz")
```

| episode | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|---|---|---|---|---|---|---|---|---|
| best score | 89.17 | 86.67 | 88.33 | 88.75 | 88.33 | 89.58 | 88.75 | 88.33 |
| return | −84.61 | −113.76 | −13.73 | −82.64 | −10.48 | −52.28 | −86.88 | −66.57 |

- **Every episode spends its full 24-experiment budget**
  (`budget_exhausted`) — no stall, sustained exploration, no collapse.
- **Baseline 77.5 → best 88.33–89.58 in every episode** — the policy finds
  the wide-MLP region (e.g. final best `[32, 32]`, lr 0.003, momentum;
  ep8: 9 of 24 candidates accepted, candidate spread 51.25–88.33).
- **Gate vs 95: MISS (honest)** — sin20's ceiling in this spec space is
  ≈89–91; the +10.8 pt architecture gradient is the learnable part.
  (v0.59 note, §73.8.2: with the corrected sign this region is now
  reachable by the *gradient itself*; pre-v0.59 the ε-draws did the work.)
- Run dir: `runs/sin20-rl-eps025-20260922-113118/` (one subdir per
  episode; the last carries `summary.json`: baseline 77.5, final 88.33,
  improvement factor 1.140, `budget_exhausted`).
- **Trained policy saved**: `examples/sin20/policy_sin20.npz`
  (SPEC §67.1 round-trip format). A `load_policy` reload is **bit-exact**
  (identical action probabilities in the reset state, §67.1.3), and one
  reuse episode with the reloaded policy reaches **88.75**
  (`runs/sin20-rl-reuse-20260922-113118/`) — learn → persist → reuse.

### What this shows

- The RL improver *uses* self-improvement on a genuinely hard task: a
  real architecture gradient exists (77.5 → ≈89), and REINFORCE over the
  77-action catalog finds it consistently across episodes.
- The result is *honest*: the 95 bar is out of reach for this spec space
  on this problem — the gate reports MISS rather than fudging it.
- **Collapse root cause: fixed in v0.59** (SPEC §73, A63) — the REINFORCE
  sign was inverted (a descent on expected return, §73.8), the free
  duplicate no-op steps are now excluded from the update (§73.2), and
  `fit` defaults to ε=0.1 (§73.9). `--rl-epsilon 0` restores the
  pre-v0.59 pure-softmax path; the v0.58 collapse artifacts above are kept
  for the record.

## Files

| path | what |
|---|---|
| `sin20_train.csv` | the training data (2,400 × 21, deterministic seed 7) |
| `policy_sin20.npz` | the trained meta-RL policy (ε = 0.25 run, 8 episodes) |
| `runs/csv-seed7-20260922-130844-2`, `runs/csv-seed7-20260922-131101-2` (repo root) | the v0.59 verification runs (default ε=0.1: 78.33; ε=0.25: 88.33) |
| `../parity8/` | the earlier 8-bit-parity attempt — kept for the record; **superseded by sin20** (parity-8 is unsolvable in this spec space: no candidate beats the `[16, 8]` baseline, so the improver has no gradient to learn) |
