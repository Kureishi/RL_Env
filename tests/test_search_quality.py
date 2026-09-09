"""Search-quality (v0.4) tests — SPEC.md 18, acceptance A8.

Covers §18.1 local mutation, §18.2 family-conditioned proposals, §18.3
block-bootstrap CI, §18.4/18.5 efficiency + gen-gap rules, §18.6 the
unified acceptance rule (env-level, with scripted scores), and §18.7's
bit-exact legacy regression pin plus the A8 integration runs.
"""
import csv as _csv
import math
from pathlib import Path

import numpy as np
import pytest

from autorefine import (
    TASKS,
    AutoRefineEnv,
    BanditPolicy,
    Budget,
    DEFAULT_SPEC,
    MetaRLPolicy,
    ModelSpec,
    search_quality_v04,
)
from autorefine.evaluator import score_with_ci
from autorefine.improver.actions import FIELD_NAMES, mutate_spec_dict
from autorefine.improver.catalog import (
    ACTIONS,
    relevant_actions,
    relevant_fields,
)
from autorefine.improver.meta_env import effective_score, should_accept
from autorefine.tasks.csv import CsvTask
from autorefine.tasks.parity import _parity_points
from autorefine.trainer import train_from_task

TREE_FIELDS = ("architecture", "train_steps", "input_noise", "model_family")


# ---------------------------------------------------------------------------
# SPEC.md 18.1: local (neighborhood) mutation
# ---------------------------------------------------------------------------

def test_local_ordered_fields_land_on_catalog_neighbors():
    """Ordered fields step ±1 catalog index around the current value
    (SPEC.md 18.1); DEFAULT_SPEC values anchor to the catalog directly."""
    expected = {
        "learning_rate": {3e-4, 3e-3},     # anchor 1e-3 (index 2)
        "batch_size": {16, 64},            # anchor 32 (index 1)
        "weight_decay": {0.0, 1e-3},       # anchor 1e-4 (index 1)
        "train_steps": {200, 1000},        # anchor 400 (index 1)
        "input_noise": {0.0, 0.05},        # anchor 0.02 (index 1)
        "label_smoothing": {0.05},         # anchor 0.0 (index 0): clamped, flip -> 0.05
    }
    base = DEFAULT_SPEC.to_dict()
    for field, values in expected.items():
        seen = set()
        for seed in range(60):
            rng = np.random.default_rng(seed)
            out = mutate_spec_dict(base, field, rng, mode="local")
            assert out[field] in values, (field, seed, out[field])
            seen.add(out[field])
        assert seen == values  # the coin actually lands on both neighbors


def test_local_non_catalog_values_anchor_to_nearest_index():
    """SPEC.md 18.1: values not in the catalog (e.g. uniform-mutated ones)
    anchor to the nearest catalog index first, then step ±1."""
    cases = {
        ("train_steps", 3484): {1000, 5000},      # nearest 2000 (index 3)
        ("learning_rate", 2e-3): {1e-3, 1e-2},    # log-nearest 3e-3 (index 3)
        ("weight_decay", 0.0): {1e-4},            # boundary: clamped, flip once
        ("input_noise", 0.1): {0.05},             # boundary: clamped, flip once
    }
    for (field, value), values in cases.items():
        base = {**DEFAULT_SPEC.to_dict(), field: value}
        seen = set()
        for seed in range(40):
            rng = np.random.default_rng(seed)
            out = mutate_spec_dict(base, field, rng, mode="local")
            assert out[field] in values, (field, seed, out[field])
            seen.add(out[field])
        assert seen == values


def test_local_architecture_neighborhoods():
    """SPEC.md 18.1: architecture moves are 50% depth ±1 (clamped 1..3),
    50% first-hidden-width ±1 step in HIDDEN_LAYER_SIZES."""
    cases = {
        ("mlp", (16, 8)): {(16,), (16, 8, 32), (8, 8), (32, 8)},
        ("mlp", ()): {(8,)},                    # linear: only move is one layer
        ("tree", (1,)): {(2,)},                 # depth clamped at 1
        ("tree", (2,)): {(1,), (3,)},
        ("tree", (3,)): {(1,)},                 # depth clamped at 3
    }
    for (family, arch), values in cases.items():
        base = {**DEFAULT_SPEC.to_dict(),
                "model_family": family, "architecture": list(arch)}
        seen = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            out = mutate_spec_dict(base, "architecture", rng, mode="local")
            ModelSpec.from_dict(out)  # always a valid spec
            got = tuple(out["architecture"])
            assert got in values, (family, arch, seed, got)
            seen.add(got)
        assert seen == values


def test_local_guaranteed_different_for_every_field():
    """SPEC.md 18.1 (preserved invariant): local mutation always changes
    the requested field — categorical fields fall back to the uniform
    resample, so they stay guaranteed-different too."""
    base = DEFAULT_SPEC.to_dict()
    for field in FIELD_NAMES:
        current = base[field]
        for seed in range(20):
            rng = np.random.default_rng(seed)
            out = mutate_spec_dict(base, field, rng, mode="local")
            ModelSpec.from_dict(out)
            got, cur = out[field], current
            if field == "architecture":
                got, cur = tuple(got), tuple(cur)
            assert got != cur, (field, seed, got, cur)


def test_local_deterministic_given_seed():
    """SPEC.md 18.1 (preserved invariant): same seed + spec + field pins
    the local mutation (G2)."""
    base = DEFAULT_SPEC.to_dict()
    for field in ("learning_rate", "architecture", "batch_size", "train_steps"):
        a = mutate_spec_dict(base, field, np.random.default_rng(5), mode="local")
        b = mutate_spec_dict(base, field, np.random.default_rng(5), mode="local")
        assert a == b


def test_unknown_mode_and_field_raise():
    with pytest.raises(ValueError):
        mutate_spec_dict(DEFAULT_SPEC.to_dict(), "batch_size",
                         np.random.default_rng(0), mode="bogus")
    with pytest.raises(KeyError):
        mutate_spec_dict(DEFAULT_SPEC.to_dict(), "nonsense",
                         np.random.default_rng(0), mode="local")


# ---------------------------------------------------------------------------
# SPEC.md 18.2: family-conditioned action space
# ---------------------------------------------------------------------------

def test_relevant_fields_and_actions():
    """SPEC.md 18.2 + 19.4 + 25: tree/boost ignore optimizer/lr/batch/
    weight-decay/activation/label-smoothing and the mlp-only fields; mlp uses
    all 15 fields (v0.11: +knn_k); the v0.11 action space is 77 (54 v0.5
    + knn_k 5 + convnet arch pairs 16 + family knn/convnet 2)."""
    assert set(relevant_fields("mlp")) == set(FIELD_NAMES)
    assert set(relevant_fields("tree")) == set(TREE_FIELDS)
    assert set(relevant_fields("boost")) == set(TREE_FIELDS)  # SPEC.md 19.2
    assert len(ACTIONS) == 77
    tree_actions = relevant_actions("tree")
    assert len(tree_actions) == 37  # arch 23 + steps 5 + noise 4 + family 5
    assert set(tree_actions) < set(range(len(ACTIONS)))
    assert all(ACTIONS[i][0] in TREE_FIELDS for i in tree_actions)
    assert set(relevant_actions("mlp")) == set(range(77))


def test_bandit_tree_best_stays_in_relevant_fields():
    """SPEC.md 18.2: with a tree best spec the bandit never proposes a
    change outside the family's 4 relevant fields, across 50 forced steps,
    and every proposal is a valid ModelSpec (mode="local" throughout)."""
    pol = BanditPolicy(seed=11, mode="local")
    tree = {**DEFAULT_SPEC.to_dict(), "model_family": "tree", "architecture": [2]}
    state = {"best_spec": tree, "last": None}
    for _ in range(50):
        spec = pol.propose(state)
        ModelSpec.from_dict(spec)  # always valid
        changed = [f for f in FIELD_NAMES if spec[f] != tree[f]]
        assert changed  # guaranteed-different
        assert set(changed) <= set(TREE_FIELDS), changed
        state = {"best_spec": tree, "last": {"accepted": False, "fields": changed}}


def test_meta_rl_policy_masks_tree_irrelevant_actions():
    """SPEC.md 18.2: with a tree best spec MetaRLPolicy never samples one of
    the masked actions across 1000 proposals; the action space stays 40."""
    pol = MetaRLPolicy(seed=0)
    tree = {**DEFAULT_SPEC.to_dict(), "model_family": "tree", "architecture": [2]}
    state = {"best_spec": tree, "best_score": 50.0,
             "experiments_left": 30, "last": None, "done": False}
    rel = set(relevant_actions("tree"))
    for _ in range(1000):
        spec = pol.propose(state)
        ModelSpec.from_dict(spec)
        assert pol._traj[-1][1] in rel  # sampled action index is relevant
    p = pol._probs(pol.embed(state), "tree")
    assert p.sum() == pytest.approx(1.0)
    for i in range(len(ACTIONS)):
        if i not in rel:
            assert p[i] == 0.0


# ---------------------------------------------------------------------------
# SPEC.md 18.3: block-bootstrap score confidence
# ---------------------------------------------------------------------------

def _ci_csv_fixture(path):  # SPEC.md 22.1: the csv task is data-driven
    rng = np.random.default_rng(5)
    n = 80
    x = rng.uniform(-1, 1, n)
    y = ((x > 0) == (2 * x > -1)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x", "y"])
        for i in range(n):
            w.writerow([f"{x[i]:.5f}", int(y[i])])
    return path


def _ci_wav_dir(base: Path) -> Path:
    """Two tone classes, one subfolder each (SPEC.md 24.4, stdlib wave)."""
    import math as _m
    import wave
    d = base / "tones"
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            sr = 8000
            sig = 0.8 * np.sin(2 * _m.pi * freq * np.arange(sr) / sr)
            pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            with wave.open(str(sub / f"c{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
    return d


def _ci_image_dir(base: Path) -> Path:
    """Two bar-orientation classes, one subfolder each (SPEC.md 24.3)."""
    pytest.importorskip("PIL", reason="image fixture needs autorefine[image] (SPEC.md 24.1)")
    from PIL import Image
    d = base / "shapes"
    for name, vertical in (("h", False), ("v", True)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            a = np.zeros((16, 16), dtype=np.uint8)
            if vertical:
                a[:, 4:12] = 255
            else:
                a[4:12, :] = 255
            Image.fromarray(a, mode="L").save(str(sub / f"s{i:02d}.png"))
    return d


def _ci_text_dir(base: Path) -> Path:
    """Two word classes, one subfolder each (SPEC.md 45.2, v0.31)."""
    d = base / "words"
    for name, words in (("up", "up up up rising higher"),
                        ("down", "down down down falling lower")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            (sub / f"t{i:02d}.txt").write_text(words + " ", encoding="utf-8")
    return d


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_score_with_ci_all_tasks(task_name, tmp_path):
    """SPEC.md 18.3: block CI works for every registered task with zero task
    changes; deterministic given the seed (G2); the csv/image/audio/text
    fixtures cover the data-driven tasks (SPEC.md 22.1/24/45.2)."""
    if task_name == "csv":
        task = CsvTask(seed=1, path=str(_ci_csv_fixture(tmp_path / "ci_fixture.csv")))
    elif task_name == "audio":
        task = TASKS[task_name](seed=1, path=str(_ci_wav_dir(tmp_path)))
    elif task_name == "image":
        task = TASKS[task_name](seed=1, path=str(_ci_image_dir(tmp_path)))
    elif task_name == "text":
        task = TASKS[task_name](seed=1, path=str(_ci_text_dir(tmp_path)))
    else:
        task = TASKS[task_name](seed=1)
    spec = ModelSpec.from_dict({**DEFAULT_SPEC.to_dict(), "train_steps": 200})
    result = train_from_task(task, spec, seed=1, time_limit_seconds=30.0)
    ci = score_with_ci(task, result.model, "holdout", n_blocks=4, block_size=64)
    assert len(ci["blocks"]) == 4
    assert math.isfinite(ci["mean"]) and math.isfinite(ci["std"])
    ci_again = score_with_ci(task, result.model, "holdout", n_blocks=4, block_size=64)
    assert ci_again == ci  # split names are seed-derived (G2)
    one = score_with_ci(task, result.model, "gen", n_blocks=1, block_size=64)
    assert one["std"] == 0.0 and len(one["blocks"]) == 1  # n_blocks < 2 -> std 0
    with pytest.raises(ValueError):
        score_with_ci(task, result.model, "holdout", n_blocks=0)


# ---------------------------------------------------------------------------
# SPEC.md 18.4 / 18.5 / 18.6: pure decision rules
# ---------------------------------------------------------------------------

def test_effective_score_values():
    """SPEC.md 18.4/18.5: eff = score − η·min(t, T_CAP) − P_GEN·excess_gap,
    excess gap beyond 5% of score. Both weights 0 ⇒ exactly the raw score."""
    assert effective_score(100.0, 1.0, 0.0, 0.5, 0.5) == pytest.approx(99.5)
    assert effective_score(100.0, 4.0, 0.0, 0.5, 0.5) == pytest.approx(98.0)
    assert effective_score(100.0, 100.0, 0.0, 0.5, 0.0) == pytest.approx(95.0)  # T_CAP
    assert effective_score(100.0, 0.0, 15.0, 0.0, 0.5) == pytest.approx(95.0)
    assert effective_score(100.0, 0.0, 5.0, 0.0, 0.5) == pytest.approx(100.0)  # at tolerance
    # SPEC.md 18.7 (gen-gap): gappy loses to clean at the same holdout score
    assert effective_score(100.0, 0.0, 15.0, 0.0, 0.5) < effective_score(100.0, 0.0, 0.0, 0.0, 0.5)
    assert effective_score(73.2, 5.0, 9.0, 0.0, 0.0) == 73.2  # legacy: raw score


def test_should_accept_rule():
    """SPEC.md 18.6: accept ⟺ Δeff > max(0, z·SE); z = 0 ⇒ Δeff > 0 (v1)."""
    assert should_accept(0.5, 1.0, 1.0) is False   # within 1·SE
    assert should_accept(3.0, 1.0, 1.0) is True    # clear winner
    assert should_accept(0.1, 1.0, 0.0) is True    # z = 0: any positive delta
    assert should_accept(0.0, 0.0, 0.0) is False   # strict improvement
    assert should_accept(0.0, 1.0, 0.0) is False


def test_search_quality_v04_preset():
    """SPEC.md 18.6: the recommended preset's knobs (Q4–Q7 as approved)."""
    assert search_quality_v04() == {
        "ci_blocks": 8,
        "z_accept": 1.0,
        "efficiency_weight": 0.5,
        "gen_gap_penalty": 0.5,
    }


# ---------------------------------------------------------------------------
# SPEC.md 18.6/18.7: env-level acceptance with scripted scores
# ---------------------------------------------------------------------------

class _ScriptedParityTask:
    """Parity-shaped fake task: real training on real parity data (so the
    trainer's timing is genuine), but `score` returns scripted block means
    keyed by the model's first hidden-layer size — so holdout/gen scores,
    block stds, and gen gaps are exactly controlled while the env's
    acceptance machinery (CI, eff, z·SE) runs unmodified.

    Split names follow §18.3: "holdout-b<i>" / "gen-b<i>" for blocks,
    bare "holdout" / "gen" on the legacy single-score path."""

    name = "parity-v1"
    state_dim = 4
    n_outputs = 2
    head = "softmax"
    max_steps = 1

    def __init__(self, seed: int,
                 script: dict[int, tuple[list[float], list[float]]]) -> None:
        self.seed = int(seed)
        self.script = script  # first-hidden size -> (holdout blocks, gen blocks)

    def make_dataset(self, n_points: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        return _parity_points(self.seed, "train", n_points)

    def score(self, model, split: str, n: int) -> float:
        hidden0 = int(model.layers[0][0].shape[1])
        blocks = (self.script[hidden0][0] if split.startswith("holdout")
                  else self.script[hidden0][1])
        key = split.split("-b")[-1]
        if key.isdigit():
            return float(blocks[int(key) % len(blocks)])
        return float(sum(blocks) / len(blocks))  # legacy path: block mean


def _env_with_script(runs_dir, script, candidates, v04: bool, budget_n: int):
    """Reset the env with a scripted task, step the candidate specs, done."""
    kwargs = search_quality_v04() if v04 else {}
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(budget_n, 300, 30),
                        runs_dir=runs_dir, **kwargs)
    env.task = _ScriptedParityTask(7, script)
    state = env.reset()
    rows = []
    for cand in candidates:
        if env.done:
            break
        spec = {**DEFAULT_SPEC.to_dict(), **cand}
        state, _reward, _done, info = env.step(spec)
        rows.append(info)
    return env, rows


def test_ci_acceptance_near_tie_rejected_clear_winner_accepted(tmp_path):
    """SPEC.md 18.7 (CI acceptance): a near-tie candidate (Δ within z·SE) is
    rejected, a clear winner accepted; the sequence is deterministic.
    Baseline (hidden 16) blocks [49,51]×4 → σ ≈ 1.07; near-tie (32) scores
    50.2 (Δ = 0.2 < SE); clear winner (64) scores 53.0 (Δ = 3.0 > SE)."""
    script = {
        16: ([49.0, 51.0] * 4, [49.0, 51.0] * 4),
        32: ([50.2] * 8, [50.2] * 8),
        64: ([53.0] * 8, [53.0] * 8),
    }
    cands = [{"architecture": [32, 8]}, {"architecture": [64, 8]}]
    env, rows = _env_with_script(tmp_path / "ci", script, cands, v04=True, budget_n=3)
    assert [r["accepted"] for r in rows] == [False, True]
    near_delta = rows[0]["effective_score"] - env.baseline_eff
    assert near_delta <= rows[0]["se"]          # z = 1.0: within the threshold
    clear_delta = rows[1]["effective_score"] - env.baseline_eff
    assert clear_delta > rows[1]["se"]
    assert env.best_score == pytest.approx(53.0)
    assert env.best_spec.architecture == (64, 8)


def test_efficiency_faster_candidate_wins_under_v04_not_legacy(tmp_path):
    """SPEC.md 18.7 (efficiency): a 2× faster candidate at equal raw score
    is accepted under the v0.4 preset (Δeff > 0, SE = 0) and rejected
    under legacy (a raw tie is not a strict improvement)."""
    script = {16: ([50.0] * 8, [50.0] * 8)}
    cands = [{"train_steps": 200}, {"train_steps": 800}]

    env, rows = _env_with_script(tmp_path / "v04", script, cands, v04=True, budget_n=3)
    assert [r["accepted"] for r in rows] == [True, False]
    assert env.best_spec.train_steps == 200      # the faster one won
    assert env.best_eff > env.baseline_eff       # efficiency decided it
    assert env.best_score == env.baseline_score  # raw scores are equal

    env_l, rows_l = _env_with_script(tmp_path / "legacy", script, cands, v04=False, budget_n=3)
    assert [r["accepted"] for r in rows_l] == [False, False]
    assert env_l.best_spec.fingerprint() == DEFAULT_SPEC.fingerprint()
    assert env_l.best_eff == env_l.baseline_eff  # eff == raw score in legacy


def test_gen_gap_penalty_rejects_overfit_candidate(tmp_path):
    """SPEC.md 18.7 (gen-gap): a candidate with a 15-point gen gap at the
    same holdout score is rejected under the v0.4 preset (eff 43.75 vs
    the clean baseline's 50, before the small time term)."""
    script = {
        16: ([50.0] * 8, [50.0] * 8),   # clean baseline
        32: ([50.0] * 8, [35.0] * 8),   # gappy candidate (same holdout)
    }
    env, rows = _env_with_script(tmp_path / "gap", script,
                                 [{"architecture": [32, 8]}], v04=True, budget_n=2)
    assert rows[0]["accepted"] is False
    assert rows[0]["gen_gap"] == pytest.approx(15.0)
    expected = 50.0 - 0.5 * rows[0]["train_seconds"] - 0.5 * (15.0 - 0.05 * 50.0)
    assert rows[0]["effective_score"] == pytest.approx(expected, rel=1e-9)
    assert env.best_spec.fingerprint() == DEFAULT_SPEC.fingerprint()


# ---------------------------------------------------------------------------
# SPEC.md 18.7: legacy regression pin + A8 integration
# ---------------------------------------------------------------------------

def test_legacy_regression_sequence_bit_exact(tmp_path):
    """SPEC.md 18.7 (regression): legacy mode + mode="uniform" reproduces
    the v0.3 acceptance sequence bit-exactly on parity-v1, seed 7,
    6-experiment budget.

    SPEC.md 20.3: the v0.3 stream was trained on 60 points; the v0.6 default
    dataset size (4096) would change the scores, so this pin passes
    `dataset_episodes=60` explicitly to stay bit-exact."""
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(6, 300, 30), runs_dir=tmp_path,
                        dataset_episodes=60)
    assert env.ci_blocks == 0 and env.z_accept == 0.0  # legacy defaults
    pol = BanditPolicy(seed=7, mode="uniform")
    state = env.reset()
    assert env.baseline_score == 45.0
    assert env.baseline_eff == env.baseline_score  # eff == raw in legacy
    seq = []
    while not env.done:
        state, _r, _d, info = env.step(pol.propose(state))
        seq.append((info["accepted"], info["candidate_score"], info["reason"]))
    assert seq == [
        (False, 45.0, None),
        (True, 49.5, None),
        (True, 68.5, None),
        (False, 44.5, None),
        (False, 60.0, None),
        (False, 68.5, "budget_exhausted"),
    ]
    assert env.best_score == 68.5
    summary = env.memory.load_summary()
    assert summary["baseline_effective"] == 45.0
    assert summary["final_best_effective"] == 68.5
    assert summary["search_quality"]["ci_blocks"] == 0
    assert summary["search_quality"]["z_accept"] == 0.0


def _a8_run(task: str, budget: Budget, runs_dir):
    """v0.4 preset + bandit (local by default, the v0.4 default) to done."""
    env = AutoRefineEnv(task=task, seed=7, budget=budget,
                        runs_dir=runs_dir, **search_quality_v04())
    pol = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _info = env.step(pol.propose(state))
    return env


def test_A8_v04_parity_never_regresses_and_reproducible(tmp_path):
    """A8 (parity): under the v0.4 preset the run never regresses below its
    raw baseline, and two same-seed runs are bit-reproducible (A2-style):
    identical acceptance flags and final spec."""
    envs = []
    for i in range(2):
        env = _a8_run("parity-v1", Budget(10, 600, 30), tmp_path / f"p{i}")
        assert env.done
        assert env.best_score >= env.baseline_score  # (a) never regresses
        envs.append(env)
    e0, e1 = envs
    # A2-style: identical sequence sans the wall-clock measurements (A2)
    seq = lambda env: [
        {k: e[k] for k in ("kind", "spec_hash", "holdout_score", "gen_score", "accepted")}
        for e in env.memory.load_experiments()
    ]
    assert seq(e0) == seq(e1)
    assert e0.best_spec.fingerprint() == e1.best_spec.fingerprint()
    summary = e0.memory.load_summary()
    assert summary["baseline_effective"] == pytest.approx(e0.baseline_eff)
    assert summary["final_best_effective"] == pytest.approx(e0.best_eff)
    assert summary["search_quality"] == {
        "ci_blocks": 8, "z_accept": 1.0, "efficiency_weight": 0.5,
        "gen_gap_penalty": 0.5, "block_size": 512,
    }


def test_A8_v04_cartpole_improves_over_baseline(tmp_path):
    """A8 (cartpole): under the v0.4 preset the run shows A1-style
    improvement (final >= 1.25x baseline) — 'at least one task' of §18.7."""
    env = _a8_run("cartpole-v1", Budget(6, 600, 30), tmp_path / "c")
    assert env.done
    assert env.best_score >= env.baseline_score
    assert env.best_score >= 1.25 * env.baseline_score
    assert env.best_eff >= env.baseline_eff
