"""SPEC.md 99 (v0.85) — the self-improvement feature batch (A89).

Seventeen additive, opt-in surfaces (SPEC.md 99.1), all of them
deterministic given their inputs (G2) and all of them legacy-safe
(99.2: with the new knobs off, every pinned sequence stays
byte-identical):

  * A1–A5  (A89.1–A89.5) — the improver improving itself: ``meta_loop``
    (the loop tunes its own knobs), ``mine_policy`` / ``pretrain``
    (policies learned from finished runs, the ``--pretrained`` guard),
    ``PopulationPolicy`` (competing improvers over one budget) and
    ``ThompsonPolicy`` (Beta-Bernoulli arm sampling);
  * B1–B5  (A89.6–A89.10) — model & transfer: ``AdapterMLP``
    (frozen backbone + rank-r adapters), the ``aux_pretrain="nextframe"``
    self-supervised stage, ``distill_from_run``, the ``forget_tol``
    forgetting guard, and data augmentation;
  * C1–C3  (A89.11–A89.13) — fusion & robustness: the ``fusion`` /
    ``fusion-v1`` tasks + ``LateFusionMLP`` two-tower, cross-modality
    warm-start adaptation, and the FGSM robustness gate
    (``robust_eps``);
  * D1–D4  (A89.14–A89.17) — RL & infrastructure: the DQN example
    agent, the policy zoo (sha256-provenanced), the benchmark suite,
    and the ``cost_max`` cost-aware gate.

House rules: stdlib + numpy only (PIL / gymnasium optional and lazy);
no new hard dependency; no pinned surface touched (G2). Every fixture
is synthesized in this file (no cross-test imports).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autorefine
from autorefine import (AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec)
from autorefine.adapt import adapt_model, cross_modal_adapt
from autorefine.bench_suite import (SUITE_MULTIMODAL, bench_suite,
                                    render_bench)
from autorefine.cli import build_parser
from autorefine.config import (ADAPTER_RANKS, AUX_PRETRAIN, AUGMENTS,
                               FUSIONS, SpecError)
from autorefine.distill import distill_from_run
from autorefine.improver.bandit import BanditPolicy
from autorefine.improver.meta import LEGACY_META, META_KNOBS, meta_loop
from autorefine.improver.policy_zoo import (INDEX_NAME, export_policy,
                                            import_policy, list_zoo,
                                            zoo_info)
from autorefine.improver.policylab import (PRETRAIN_TASKS,
                                           PopulationPolicy, mine_policy,
                                           pretrain)
from autorefine.improver.rl_policy import MetaRLPolicy
from autorefine.improver.catalog import ACTIONS
from autorefine.improver.thompson import ThompsonPolicy
from autorefine.models.adapters import AdapterMLP
from autorefine.models.augment import augment_batch
from autorefine.models.fusion import LateFusionMLP
from autorefine.models.mlp import MLP
from autorefine.robustness import (fgsm_perturb, input_grads,
                                   robust_accuracy)
from autorefine.tasks import TASKS
from autorefine.tasks.fusion import FusionTask, FusionV1
from autorefine.trainer import train, train_from_task

REPO = Path(__file__).resolve().parents[1]


def _bandit_run(tmp: Path, task: str = "parity-v1", seed: int = 7,
                budget: int = 2) -> Path:
    """One tiny finished run (bandit = single-field mutations, so its
    candidates all map to catalog actions — the mining channel)."""
    env = AutoRefineEnv(task=task, seed=seed, budget=Budget(budget, 900, 30),
                        runs_dir=tmp)
    pol = BanditPolicy(seed=seed)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    assert env.run_dir is not None and (env.run_dir / "summary.json").is_file()
    return Path(env.run_dir)


# --- A1 (A89.1): the meta-loop ------------------------------------------------

def test_a89_1_meta_loop_contract(tmp_path):
    """A89.1 (SPEC.md 99.1 A1): episode 0 is the exact legacy config;
    each later episode mutates one meta-knob and is credited on
    improvement; the result is deterministic (G2)."""
    out = meta_loop("parity-v1", seed=7, episodes=2, budget=2, runs_dir=tmp_path)
    assert {"task", "seed", "episodes", "budget", "best_config",
            "best_score", "history"} <= set(out)
    assert out["task"] == "parity-v1" and out["episodes"] == 2
    assert len(out["history"]) == 2
    assert out["history"][0]["episode"] == 0
    assert out["history"][0]["config"] == dict(LEGACY_META)
    for row in out["history"]:
        assert {"episode", "config", "score", "improved", "run_dir"} <= set(row)
        assert np.isfinite(row["score"])
    # one run directory per episode, all under the caller's runs_dir
    dirs = [p for p in Path(tmp_path).iterdir() if p.is_dir()]
    assert len(dirs) == 2
    # determinism: identical inputs, identical outcome
    out2 = meta_loop("parity-v1", seed=7, episodes=2, budget=2,
                     runs_dir=tmp_path / "again")
    assert out2["best_score"] == out["best_score"]
    assert out2["history"][1]["config"] == out["history"][1]["config"]


def test_a89_1_meta_loop_knobs_and_errors(tmp_path):
    """A89.1: every meta-knob keeps its legacy value (99.2); episodes < 1
    fails loud."""
    for knob, space in META_KNOBS.items():
        assert LEGACY_META[knob] in space, knob
    with pytest.raises(ValueError, match="episodes"):
        meta_loop("parity-v1", episodes=0, budget=2, runs_dir=tmp_path)
    with pytest.raises(ValueError):
        meta_loop("nope-v1", episodes=1, budget=2, runs_dir=tmp_path)


# --- A2 (A89.2): mining a policy from finished runs ---------------------------

def test_a89_2_mine_policy(tmp_path):
    """A89.2 (SPEC.md 99.1 A2): AWR-weighted likelihood ascent over the
    catalog — deterministic, epoch-counted."""
    run = _bandit_run(tmp_path / "run")
    pol = mine_policy([run], seed=7, n_epochs=1)
    assert isinstance(pol, MetaRLPolicy)
    assert pol.n_updates == 1
    assert pol.n_actions == len(ACTIONS)
    a = mine_policy([run], seed=7, n_epochs=3)
    b = mine_policy([run], seed=7, n_epochs=3)
    assert np.array_equal(a.w, b.w) and np.array_equal(a.b, b.b)
    assert a.n_updates == 3


def test_a89_2_mine_policy_errors(tmp_path):
    """A89.2: missing logs / bad hyperparameters / no catalog signal all
    fail loud."""
    with pytest.raises(FileNotFoundError):
        mine_policy([tmp_path / "missing-run"])
    run = _bandit_run(tmp_path / "run")
    with pytest.raises(ValueError):
        mine_policy([run], beta=0.0)
    with pytest.raises(ValueError):
        mine_policy([run], n_epochs=0)
    # a log with only a baseline row carries no (state, action) transition
    fake = tmp_path / "baseline-only"
    fake.mkdir()
    base = DEFAULT_SPEC.to_dict()
    (fake / "experiments.jsonl").write_text(
        json.dumps({"kind": "baseline", "spec": base, "holdout_score": 50.0,
                    "accepted": False}) + "\n", encoding="utf-8")
    (fake / "summary.json").write_text(
        json.dumps({"task": "parity-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no catalog-mapped transitions"):
        mine_policy([fake])


# --- A3 (A89.3): pretraining + the --pretrained guard -------------------------

def test_a89_3_pretrain(tmp_path):
    """A89.3 (SPEC.md 99.1 A3): one task-conditioned policy over several
    tasks, persisted on request; per-task run dirs under the caller's
    runs_dir."""
    policy, saved = pretrain(
        tasks=("parity-v1", "sine-v1"), seed=7, episodes=1, budget=2,
        runs_dir=tmp_path, out_path=tmp_path / "pre.npz")
    assert isinstance(policy, MetaRLPolicy)
    assert policy.task_names == ["parity-v1", "sine-v1"]
    assert saved is not None and Path(saved).is_file()
    assert (tmp_path / "pretrain-parity-v1").is_dir()
    assert (tmp_path / "pretrain-sine-v1").is_dir()
    with pytest.raises(ValueError):
        pretrain(tasks=(), seed=7, episodes=1, budget=2, runs_dir=tmp_path)
    assert "parity-v1" in PRETRAIN_TASKS and "sine-v1" in PRETRAIN_TASKS


def test_a89_3_pretrained_flag_and_guard():
    """A89.3: ``run --pretrained PATH`` parses, and the guard rejects it
    outside the RL policy (the policy must be ``rl`` for a saved policy
    to mean anything)."""
    args = build_parser().parse_args(
        ["run", "--task", "parity-v1", "--pretrained", "x.npz"])
    assert args.pretrained == "x.npz"
    from autorefine.cli import _pretrained_guard
    bad = SimpleNamespace(pretrained="x.npz", policy="bandit")
    msg = _pretrained_guard(bad)
    assert msg is not None and "--pretrained" in msg and "rl" in msg
    assert _pretrained_guard(SimpleNamespace(pretrained="x.npz",
                                             policy="rl")) is None
    assert _pretrained_guard(SimpleNamespace(pretrained=None,
                                             policy="bandit")) is None


# --- A4 (A89.4): the policy population ----------------------------------------

def test_a89_4_population_policy():
    """A89.4 (SPEC.md 99.1 A4): bandit / rl / random compete over one
    budget; the member that proposed the accepted step gets the credit."""
    pop = PopulationPolicy(seed=7)
    assert pop.names == ("bandit", "rl", "random")
    alloc = pop.allocation()
    assert set(alloc) == set(pop.names)
    assert abs(sum(alloc.values()) - 1.0) < 1e-9
    state = {
        "last": None, "best_spec": DEFAULT_SPEC.to_dict(),
        "best_score": 50.0, "experiments_left": 5, "done": False,
        "task": "parity-v1",
    }
    cand = pop.propose(state)
    assert isinstance(cand, dict) and "model_family" in cand
    assert sum(pop.proposals.values()) == 1
    before = sum(pop.accepts.values())
    state2 = dict(state)
    state2["last"] = {"accepted": True, "fields": ["learning_rate"]}
    pop.propose(state2)
    assert sum(pop.accepts.values()) == before + 1
    with pytest.raises(ValueError):
        PopulationPolicy(members=())
    with pytest.raises(ValueError):
        PopulationPolicy(members=("bandit", "bandit"))
    with pytest.raises(ValueError):
        PopulationPolicy(members=("nope",))


# --- A5 (A89.5): Thompson sampling --------------------------------------------

def test_a89_5_thompson_policy():
    """A89.5 (SPEC.md 99.1 A5): Beta(1+w, 1+l) per catalog arm; credit the
    previous arm with the step's outcome; deterministic given the seed."""
    t = ThompsonPolicy(seed=7)
    assert t.n_actions == len(ACTIONS)
    mean = t.posterior_mean()
    assert len(mean) == t.n_actions
    assert bool(np.all(mean > 0.0)) and bool(np.all(mean <= 1.0))
    state = {"last": None, "best_spec": DEFAULT_SPEC.to_dict(),
             "task": "parity-v1"}
    cand = t.propose(state)
    assert isinstance(cand, dict) and "model_family" in cand
    w0 = float(t.wins.sum())
    state2 = dict(state)
    state2["last"] = {"accepted": True, "fields": ["learning_rate"]}
    t.propose(state2)
    assert float(t.wins.sum()) == w0 + 1.0
    l0 = float(t.losses.sum())
    state3 = dict(state)
    state3["last"] = {"accepted": False, "fields": ["batch_size"]}
    t.propose(state3)
    assert float(t.losses.sum()) == l0 + 1.0
    with pytest.raises(ValueError):
        ThompsonPolicy(seed=7, alpha=0.0)
    # deterministic: same seed + same credit stream => same posterior
    a = ThompsonPolicy(seed=11)
    b = ThompsonPolicy(seed=11)
    for st in (state2, state3, state):
        a.propose(st)
        b.propose(st)
    assert np.array_equal(a.wins, b.wins) and np.array_equal(a.losses, b.losses)


# --- B1 (A89.6): adapter fine-tuning ------------------------------------------

def test_a89_6_adapter_mlp():
    """A89.6 (SPEC.md 99.1 B1): frozen backbone + rank-r adapters — the
    trainable surface is exactly the adapter pairs."""
    m = AdapterMLP(8, (16, 8), 3, "tanh", seed=7, rank=2)
    assert m.rank == 2
    assert len(m.backbone) == 3
    # one adapter pair per linear layer (hidden 8→16, 16→8, head 8→3)
    assert m.adapters is m.layers and len(m.adapters) == 3
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 8))
    y = np.array([0, 1, 2, 0, 1])
    out = m.forward(x)
    assert out.shape == (5, 3)
    loss, grads = m.loss_and_grads(x, y)
    assert np.isfinite(loss) and len(grads) == 3
    with pytest.raises(SpecError):
        AdapterMLP(8, (16, 8), 3, "tanh", seed=7, rank=0)


def test_a89_6_adapter_trainer_integration():
    """A89.6: ``adapter_rank`` is a real spec field (the 4-line pattern) —
    rank > 0 trains the adapter model, rank 0 stays the plain MLP."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    base = dict(DEFAULT_SPEC.to_dict())
    res = train((X, y), ModelSpec.from_dict({**base, "adapter_rank": 4,
                                             "train_steps": 200}),
                seed=7, n_out=2, head="softmax")
    assert isinstance(res.model, AdapterMLP) and res.model.rank == 4
    res0 = train((X, y), ModelSpec.from_dict({**base, "train_steps": 200}),
                 seed=7, n_out=2, head="softmax")
    assert not hasattr(res0.model, "adapters")
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**base, "adapter_rank": 3})
    assert 0 in ADAPTER_RANKS and 8 in ADAPTER_RANKS


# --- B2 (A89.7): self-supervised auxiliary pretraining -------------------------

def test_a89_7_aux_pretrain_temporal():
    """A89.7 (SPEC.md 99.1 B2): ``aux_pretrain="nextframe"`` adds a
    self-supervised stage before the supervised loop (temporal families);
    "off" is the exact legacy path."""
    task = TASKS["sequence-motif-v1"](seed=7)
    base = dict(DEFAULT_SPEC.to_dict())
    d_on = {**base, "model_family": "rnn", "train_steps": 200,
            "aux_pretrain": "nextframe"}
    res_on = train_from_task(task, ModelSpec.from_dict(d_on), seed=7)
    assert np.isfinite(res_on.final_loss)
    d_off = {**base, "model_family": "rnn", "train_steps": 200}
    res_off = train_from_task(task, ModelSpec.from_dict(d_off), seed=7)
    assert np.isfinite(res_off.final_loss)
    assert "off" in AUX_PRETRAIN and "nextframe" in AUX_PRETRAIN
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**base, "aux_pretrain": "bogus"})


# --- B3 (A89.8): distillation from a finished run ------------------------------

def test_a89_8_distill_from_run(tmp_path):
    """A89.8 (SPEC.md 99.1 B3): shrink the champion while holding >= keep
    of its score; the report writes back into the run dir."""
    run = _bandit_run(tmp_path / "run")
    out = distill_from_run(run, keep=0.95)
    assert {"spec", "score", "n_params", "champion", "kept",
            "reduction", "tried"} <= set(out)
    assert out["champion"]["n_params"] >= out["n_params"] > 0
    assert out["kept"] >= 0.95
    assert (Path(run) / "distill.json").is_file()
    with pytest.raises(ValueError):
        distill_from_run(run, keep=1.5)
    with pytest.raises(ValueError):
        distill_from_run(run, keep=0.0)
    # an unregistered task fails loud before any training
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "summary.json").write_text(
        json.dumps({"task": "nope-v1", "seed": 7,
                    "final_best_score": 50.0}), encoding="utf-8")
    (fake / "best_spec.json").write_text(
        json.dumps(DEFAULT_SPEC.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="not registered"):
        distill_from_run(fake)


# --- B4 (A89.9): the forgetting guard ------------------------------------------

def _csv4(path: Path, n: int = 80, seed: int = 7) -> Path:
    """A 4-feature / 2-class CSV compatible with a parity champion."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > X[:, 3])).astype(int)
    path.write_text(
        "f0,f1,f2,f3,label\n" + "\n".join(
            ",".join(f"{v:.6f}" for v in row) + f",{yy}"
            for row, yy in zip(X.tolist(), y.tolist())) + "\n",
        encoding="utf-8")
    return path


def test_a89_9_forget_tol_validation():
    """A89.9 (SPEC.md 99.1 B4): the tolerance is a finite number >= 0 —
    bools and bad types fail loud (before any run reconstruction)."""
    with pytest.raises(ValueError):
        adapt_model("runs/nope", "data.csv", forget_tol=True)
    with pytest.raises(ValueError):
        adapt_model("runs/nope", "data.csv", forget_tol=-0.5)
    with pytest.raises(ValueError):
        adapt_model("runs/nope", "data.csv", forget_tol="5")


def test_a89_9_adapt_with_forget_tol(tmp_path):
    """A89.9: ``forget_tol`` adds the measured forgetting block to the
    adapt report (drop vs the champion's recorded score on its own
    holdout); ``None`` keeps the legacy report shape."""
    run = _bandit_run(tmp_path / "run")
    csv = _csv4(tmp_path / "new.csv")
    rep = adapt_model(run, csv, runs_dir=tmp_path / "adapts", seed=7,
                      steps=200, max_seconds=20, forget_tol=5.0)
    assert rep["verdict"] in {"IMPROVED", "NEUTRAL", "DEGRADED"}
    assert "before" in rep and "after" in rep
    f = rep["forgetting"]
    assert f is not None
    assert f["forget_tol"] == 5.0
    assert f["orig_task"] == "parity-v1"
    assert "drop" in f and "ok" in f
    if f["drop"] is not None:
        assert f["ok"] is (f["drop"] <= 5.0)
    # legacy path: no forgetting block at all
    rep0 = adapt_model(run, csv, runs_dir=tmp_path / "adapts0", seed=7,
                       steps=200, max_seconds=20)
    assert rep0["forgetting"] is None


# --- B5 (A89.10): data augmentation --------------------------------------------

def test_a89_10_augment_batch():
    """A89.10 (SPEC.md 99.1 B5): the five modes; "none" is the identity
    (no rng draw), unknown modes fail loud."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 6))
    y = (x[:, 0] > 0).astype(int)
    x0, y0 = augment_batch(x, y, "none", rng)
    assert x0 is x and y0 is y
    for mode in ("noise", "flip", "mixup"):
        xa, ya = augment_batch(x, y, mode, rng)
        assert xa.shape == x.shape and ya.shape == y.shape
        assert not (mode != "none" and np.array_equal(xa, x))
    x3 = rng.normal(size=(8, 5, 4))
    y3 = (x3[:, 0, 0] > 0).astype(int)
    xt, yt = augment_batch(x3, y3, "time_warp", rng)
    assert xt.shape == x3.shape and yt.shape == y3.shape
    assert "none" in AUGMENTS and "time_warp" in AUGMENTS
    with pytest.raises(SpecError):
        augment_batch(x, y, "bogus", rng)


# --- C1 (A89.11): image + text fusion ------------------------------------------

def test_a89_11_fusion_v1_task():
    """A89.11 (SPEC.md 99.1 C1): the synthetic fusion task — rows are
    [image 64-d | text 32-d] with the tower boundary at 64."""
    t = FusionV1(seed=7)
    assert t.name == "fusion-v1"
    assert t.fusion_split == 64 and t.state_dim == 96
    assert t.n_outputs == 3 and t.head == "softmax"
    X, y = t.make_dataset(120)
    assert X.shape == (120, 96)
    assert set(np.unique(y)) == {0, 1, 2}
    # make_dataset is the deterministic train split: same seed ⇒ same stream
    Xh, yh = t.make_dataset(40)
    assert np.array_equal(Xh, X[:40]) and np.array_equal(yh, y[:40])
    assert "fusion-v1" in TASKS and "fusion" in TASKS


def test_a89_11_late_fusion_model():
    """A89.11: ``LateFusionMLP`` — two towers, one head; the row width is
    enforced at the forward boundary."""
    m = LateFusionMLP(64, 32, (16,), 3, "tanh", seed=7)
    rng = np.random.default_rng(0)
    assert m.forward(rng.normal(size=(4, 96))).shape == (4, 3)
    with pytest.raises(SpecError):
        m.forward(rng.normal(size=(4, 95)))
    with pytest.raises(SpecError):
        LateFusionMLP(0, 32, (16,), 3, "tanh", seed=7)
    with pytest.raises(SpecError):
        LateFusionMLP(64, 0, (16,), 3, "tanh", seed=7)


def test_a89_11_fusion_trainer_branch():
    """A89.11: ``fusion="late"`` trains the two-tower model; the
    mutually-exclusive spec combinations fail loud at dispatch."""
    t = FusionV1(seed=7)
    X, y = t.make_dataset(96)
    base = dict(DEFAULT_SPEC.to_dict())
    res = train((X, y),
                ModelSpec.from_dict({**base, "fusion": "late",
                                     "train_steps": 200}),
                seed=7, n_out=3, head="softmax", fusion_split=64)
    assert np.isfinite(res.final_loss)
    acc = float((np.asarray(res.model.forward(X)).argmax(axis=1) == y).mean())
    assert acc > 0.6, f"late fusion should beat 1/3 by a wide margin, got {acc}"
    with pytest.raises(SpecError):
        train((X, y),
              ModelSpec.from_dict({**base, "fusion": "late",
                                   "adapter_rank": 2, "train_steps": 200}),
              seed=7, n_out=3, head="softmax", fusion_split=64)
    with pytest.raises(SpecError):
        train((X, y),
              ModelSpec.from_dict({**base, "fusion": "late",
                                   "fourier_features": 4, "train_steps": 200}),
              seed=7, n_out=3, head="softmax", fusion_split=64)
    with pytest.raises(SpecError):
        train((X, y),
              ModelSpec.from_dict({**base, "fusion": "late",
                                   "train_steps": 200}),
              seed=7, n_out=3, head="softmax", fusion_split=128)
    assert "early" in FUSIONS and "late" in FUSIONS


def test_a89_11_fusion_task_files(tmp_path):
    """A89.11: the file-based fusion task — an index.csv of
    (image, text, label) rows; the path is required."""
    with pytest.raises(ValueError):
        FusionTask(seed=7)
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL is not installed (autorefine[image])")
    d = tmp_path / "fusion_dir"
    d.mkdir()
    rows = ["image,text,label"]
    for i in range(6):
        cls = i % 2
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        if cls == 0:
            img[::2, :, :] = 255  # horizontal stripes
        else:
            img[:, ::2, :] = 255  # vertical stripes
        Image.fromarray(img).save(d / f"img{i}.png")
        (d / f"txt{i}.txt").write_text(
            "red circle small round" if cls == 0 else "blue square large flat",
            encoding="utf-8")
        rows.append(f"img{i}.png,txt{i}.txt,{'cat' if cls == 0 else 'box'}")
    (d / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    t = FusionTask(seed=7, path=d)
    X, y = t.make_dataset()
    # make_dataset() returns the train split (6 items − holdout − gen)
    assert X.shape[1] == 96 and 2 <= len(X) <= 6
    assert set(np.unique(y)) <= {0, 1} and len(np.unique(y)) == 2


# --- C2 (A89.12): cross-modality adaptation -------------------------------------

def test_a89_12_cross_modal_adapt(tmp_path):
    """A89.12 (SPEC.md 99.1 C2): a champion warm-starts a DIFFERENT
    modality's task — the target family is chosen from the target's
    capabilities, the before/after scores are honest (None allowed),
    and the result is a fresh run dir with full artifacts."""
    run = _bandit_run(tmp_path / "run")  # a tabular (mlp) champion
    target = TASKS["sequence-motif-v1"](seed=7)
    rep = cross_modal_adapt(run, target, seed=7, steps=200,
                            max_seconds=20, runs_dir=tmp_path / "xmodal")
    for k in ("parent_run", "src_task", "target_task", "target_family",
              "warm_start", "warm_report", "before", "after", "steps_run",
              "wall_seconds", "run_dir", "run_path"):
        assert k in rep, k
    assert rep["parent_run"] == Path(run).name
    assert rep["src_task"] == "parity-v1"
    assert rep["target_task"] == "sequence-motif-v1"
    assert rep["target_family"] == "conv1d"
    assert rep["warm_start"] is True
    rd = Path(rep["run_path"])
    assert rd.is_dir()
    for art in ("summary.json", "best_spec.json", "best_model.npz",
                "warmstart_report.json"):
        assert (rd / art).is_file(), art
    assert np.isfinite(rep["wall_seconds"])


# --- C3 (A89.13): FGSM robustness ----------------------------------------------

def _mlp() -> MLP:
    return MLP(4, [16, 8], n_out=2, activation="tanh", seed=7, head="softmax")


def test_a89_13_fgsm_primitives():
    """A89.13 (SPEC.md 99.1 C3): input gradient -> one-step FGSM ->
    clean vs robust accuracy; eps = 0 is the identity / gate-off path."""
    m = _mlp()
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 4))
    y = (X[:, 0] > 0).astype(int)
    g = input_grads(m, X, y, "softmax")
    assert g.shape == X.shape and bool(np.all(np.isfinite(g)))
    assert np.array_equal(fgsm_perturb(m, X, y, "softmax", 0.0), X)
    adv = fgsm_perturb(m, X, y, "softmax", 0.5)
    assert not np.array_equal(adv, X)
    r0 = robust_accuracy(m, X, y, 0.0, "softmax")
    assert r0["clean_acc"] == r0["robust_acc"] and r0["eps"] == 0.0
    r1 = robust_accuracy(m, X, y, 0.5, "softmax")
    assert r1["robust_acc"] <= r1["clean_acc"]
    assert r1["n"] == 12 and r1["eps"] == 0.5


def test_a89_13_env_robust_eps_gate(tmp_path):
    """A89.13: ``robust_eps`` requires a classification task, validates
    loudly, and lands in the experiment log when on."""
    with pytest.raises(ValueError, match="classification"):
        AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(2, 900, 30),
                      runs_dir=tmp_path, robust_eps=0.1)
    with pytest.raises(ValueError, match="robust_eps"):
        AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 900, 30),
                      runs_dir=tmp_path, robust_eps=-0.1)
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(1, 900, 30),
                        runs_dir=tmp_path, robust_eps=0.05)
    assert env.robust_eps == 0.05
    pol = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    exp = [e for e in env.memory.load_experiments()
           if e.get("kind") == "experiment"]
    assert exp
    for e in exp:
        assert "robust_acc" in e and "robust_ok" in e and "cost_ok" in e
        assert e["robust_acc"] is not None  # the gate was measured, not skipped
        assert e["robust_acc"] <= 1.0


# --- D1 (A89.14): the DQN example agent ----------------------------------------

def test_a89_14_dqn_agent():
    """A89.14 (SPEC.md 99.1 D1): the hand-rolled DQN — epsilon-greedy
    action selection, decaying exploration, replay-batch learning."""
    pytest.importorskip("gymnasium")
    sys.path.insert(0, str(REPO / "examples"))
    try:
        from gym_dqn import DQNAgent
    except Exception as exc:  # pragma: no cover - import hygiene
        pytest.skip(f"gym_dqn not importable: {exc}")
    a = DQNAgent(obs_dim=8, n_actions=5, seed=7)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=8)
    act = a.act(obs)
    assert 0 <= act < 5
    e0 = a.epsilon()
    assert e0 == 1.0  # eps_start
    for _ in range(30):
        a.t += 1
    assert a.epsilon() < e0
    for i in range(a.batch):
        o = np.random.default_rng(i).normal(size=8)
        a.store(o, i % 5, float(i % 2), o, 1.0 if i == a.batch - 1 else 0.0)
    assert a.update() is True
    assert a.updates == 1


# --- D2 (A89.15): the policy zoo -------------------------------------------------

def test_a89_15_policy_zoo(tmp_path):
    """A89.15 (SPEC.md 99.1 D2): export -> index (sha256 provenance) ->
    import; tampering is detected BEFORE load; bad names / unknown names
    / wrong policy types all fail loud."""
    zoo = tmp_path / "zoo"
    pol = MetaRLPolicy(seed=7)
    entry = export_policy(pol, "parity-v7", zoo_dir=zoo, task="parity-v1",
                          source="meta", note="test policy")
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert entry["n_actions"] == pol.n_actions
    assert (zoo / "parity-v7.npz").is_file()
    assert (zoo / INDEX_NAME).is_file()
    entries = list_zoo(zoo)
    assert [e["name"] for e in entries] == ["parity-v7"]
    info = zoo_info("parity-v7", zoo)
    assert info["sha256_ok"] is True
    pol2, path = import_policy("parity-v7", zoo)
    assert isinstance(pol2, MetaRLPolicy)
    assert pol2.n_actions == pol.n_actions and path.is_file()
    # provenance: one flipped byte breaks the pin before any load
    f = zoo / "parity-v7.npz"
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        import_policy("parity-v7", zoo)
    with pytest.raises(ValueError):
        export_policy(pol, "bad name", zoo_dir=zoo)
    with pytest.raises(ValueError):
        import_policy("ghost", zoo)
    with pytest.raises(TypeError):
        export_policy(BanditPolicy(seed=7), "b", zoo_dir=zoo)


# --- D3 (A89.16): the benchmark suite --------------------------------------------

def test_a89_16_bench_suite(tmp_path):
    """A89.16 (SPEC.md 99.1 D3): one shared budget across several tasks,
    a ranked leaderboard, and the ASCII render — deterministic (G2)."""
    for t in SUITE_MULTIMODAL:
        assert t in TASKS, t
    out = bench_suite(tasks=("parity-v1", "sine-v1"), seed=7, budget=2,
                      runs_dir=tmp_path, wall_seconds=300, train_seconds=20)
    assert tuple(out["tasks"]) == ("parity-v1", "sine-v1")
    assert out["budget"] == 2 and out["seed"] == 7
    assert len(out["rows"]) == 2
    keys = {"task", "seed", "budget", "baseline", "best", "family",
            "architecture", "experiments_run", "finished_reason",
            "wall_seconds", "run_dir"}
    for row in out["rows"]:
        assert keys <= set(row)
        assert np.isfinite(row["baseline"]) and np.isfinite(row["best"])
    # `ranked` is the leaderboard: task names ordered best-first
    by_task = {r["task"]: r for r in out["rows"]}
    scores = [by_task[t]["best"] for t in out["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert out["best"]["task"] == out["ranked"][0]
    assert out["best"]["task"] in {"parity-v1", "sine-v1"}
    text = render_bench(out)
    assert "parity-v1" in text and "sine-v1" in text
    with pytest.raises(ValueError):
        bench_suite(tasks=("nope-v1",), budget=2, runs_dir=tmp_path)
    # `tasks=()` is the documented default-suite fallback (multimodal),
    # not an error


# --- D4 (A89.17): the cost-aware gate ---------------------------------------------

def test_a89_17_cost_max_gate(tmp_path):
    """A89.17 (SPEC.md 99.1 D4): ``cost_max`` bounds a candidate's train
    wall-time in the acceptance rule; a sub-millisecond budget rejects
    every candidate, so the baseline stays the champion."""
    with pytest.raises(ValueError, match="cost_max"):
        AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 900, 30),
                      runs_dir=tmp_path, cost_max=-1.0)
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 900, 30),
                        runs_dir=tmp_path, cost_max=1e-6)
    assert env.cost_max == 1e-6
    pol = BanditPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    exp = [e for e in env.memory.load_experiments()
           if e.get("kind") == "experiment"]
    assert exp
    for e in exp:
        assert e["cost_ok"] is False          # every candidate over budget
        assert e["accepted"] is False
    # the champion never left the baseline
    base_row = next(e for e in env.memory.load_experiments()
                    if e.get("kind") == "baseline")
    assert dict(env.best_spec.to_dict()) == dict(base_row["spec"])
