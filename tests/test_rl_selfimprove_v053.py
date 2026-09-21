"""Self-improving RL — persistence, exploration, reuse (v0.53, A57).

SPEC.md 67: the RL path stops being "train and forget".
  * 67.1 — `save_policy` / `load_policy` round-trip the full trainable
    state (w/b/B/task_names, hyper-parameters, n_updates, ε schedule) as a
    tagged .npz; the load is bit-exact and behavioral.
  * 67.2 — opt-in ε-greedy exploration over the family-legal catalog with
    per-episode geometric decay; the (0.0, 1.0) default keeps A6's pinned
    single-draw proposal stream (67.2.3).
  * 67.3 — opt-in `train_policy(..., best=True)` best-episode checkpoint;
    the default return dict never gains a `"best"` key (67.3.1).
  * 67.4 — the CLI cycle on `run`: `--rl-save` / `--rl-load` /
    `--rl-epsilon` / `--rl-epsilon-decay`.
  * 67.5 — A1–A56 stay green (the default path is bit-identical).

House rules: pure/deterministic (G2), stdlib + numpy, no cross-test imports
(every fixture synthesized here), `dashboard_app` never imported. Envs stay
tiny (parity-v1 / sine-v1, 1-2 experiments, 1-3 episodes) for speed.
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, DEFAULT_SPEC, ModelSpec
from autorefine.cli import build_parser, main as cli_main
from autorefine.improver.catalog import ACTIONS, apply_action, relevant_actions
from autorefine.improver.rl_policy import (
    FORMAT_TAG,
    MetaRLPolicy,
    load_policy,
    save_policy,
    train_multi_policy,
    train_policy,
)

REPO = Path(__file__).resolve().parents[1]


def _state(task: str = "parity-v1", family: str = "mlp") -> dict:
    """One fixed env_state to propose under (the 67.1.3 stream harness)."""
    best = DEFAULT_SPEC.to_dict()
    if family != "mlp":
        best = {**best, "model_family": family, "architecture": [2]}
    return {"task": task, "best_spec": best, "best_score": 50.0,
            "experiments_left": 5, "done": False}


def _propose_stream(policy: MetaRLPolicy, states: list[dict]) -> list[int]:
    """The sampled action indices for a fixed state sequence (G2)."""
    out = []
    for s in states:
        policy.propose(s)
        out.append(policy.last_proposal[0])
    return out


# --- 67.1 persistence: round-trip bit-exactness -------------------------------

def test_roundtrip_single_task_is_bit_exact(tmp_path):
    """A57 (67.1.3): save → load preserves the full state element-equal, and
    the loaded policy's same-seed `propose` stream equals the one of a
    fresh policy constructed with the same seed / hyper-parameters."""
    env = AutoRefineEnv(task="sine-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path / "rt")
    pol = MetaRLPolicy(seed=5, lr=0.3, baseline_decay=0.8)
    train_policy(env, pol, n_episodes=1)
    w_trained, b_trained = pol.w.copy(), pol.b.copy()
    path = save_policy(pol, tmp_path / "single.npz")
    assert path.exists()
    with np.load(path) as z:
        assert str(z["format"]) == FORMAT_TAG  # the tag is in the file

    loaded = load_policy(path, seed=5)
    assert np.array_equal(loaded.w, w_trained)
    assert np.array_equal(loaded.b, b_trained)
    assert loaded.B is None and pol.B is None
    assert loaded.lr == pol.lr and loaded.baseline_decay == pol.baseline_decay
    assert loaded.n_updates == pol.n_updates
    assert loaded.epsilon == 0.0 and loaded.epsilon_decay == 1.0
    assert loaded.task_names is None

    # 67.1.3 behavioral pin: a fresh untrained policy round-tripped is
    # behaviorally identical to an equivalent fresh construction (the
    # untrained state keeps the softmax non-degenerate, so the stream
    # exercises real sampling rather than a collapsed one-action policy)
    states = [_state() for _ in range(10)]
    fresh_pol = MetaRLPolicy(seed=5, lr=0.3, baseline_decay=0.8)
    fresh_path = save_policy(fresh_pol, tmp_path / "fresh.npz")
    loaded_fresh = load_policy(fresh_path, seed=5)
    fresh_ref = MetaRLPolicy(seed=5, lr=0.3, baseline_decay=0.8)
    assert _propose_stream(loaded_fresh, states) == _propose_stream(fresh_ref, states)
    # and loading the same file twice is itself deterministic
    again = load_policy(path, seed=5)
    assert _propose_stream(again, states) == _propose_stream(loaded, states)


def test_roundtrip_multi_task_preserves_b_and_task_conditioning(tmp_path):
    """A57 (67.1.3): the multi-task transfer channel (B + task_names,
    SPEC.md 20.2) survives the round-trip element-equal, and the
    task-conditioned `propose` stream is identical across it."""
    envs = [AutoRefineEnv(task=t, seed=11, budget=Budget(2, 300, 30),
                          runs_dir=tmp_path / f"mt-{t}")
            for t in ("sine-v1", "parity-v1")]
    names = ["sine-v1", "parity-v1"]
    pol = MetaRLPolicy(seed=11, task_names=names)
    train_multi_policy(envs, pol, episodes_per_task=1)
    assert pol.B is not None

    path = save_policy(pol, tmp_path / "multi.npz")
    loaded = load_policy(path, seed=11)
    assert loaded.task_names == names
    assert np.array_equal(loaded.w, pol.w)
    assert np.array_equal(loaded.b, pol.b)
    assert np.array_equal(loaded.B, pol.B)
    assert loaded.n_updates == pol.n_updates

    states = [_state(task=t) for t in names * 3]
    again = load_policy(path, seed=11)
    s1 = _propose_stream(loaded, states)
    s2 = _propose_stream(again, states)
    assert s1 == s2
    for a, s in zip(s1, states):  # every proposal stays a valid spec
        ModelSpec.from_dict(apply_action(s["best_spec"], a))


# --- 67.1 load guards ---------------------------------------------------------

def test_load_rejects_wrong_format_tag(tmp_path):
    """A57 (67.1.2): a foreign / untagged file is a ValueError."""
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, format=np.asarray("some-other-tool/9"))
    with pytest.raises(ValueError, match="format tag"):
        load_policy(bad)
    untagged = tmp_path / "untagged.npz"
    np.savez_compressed(untagged, w=np.zeros((2, 3)), b=np.zeros(3))
    with pytest.raises(ValueError, match="format tag"):
        load_policy(untagged)


def test_load_rejects_incompatible_weight_shape(tmp_path):
    """A57 (67.1.2): a weight matrix that does not match the catalog
    length (SPEC.md 19.4) is a ValueError."""
    bad = tmp_path / "shape.npz"
    np.savez_compressed(
        bad,
        format=np.asarray(FORMAT_TAG),
        w=np.zeros((4, len(ACTIONS) + 7)),  # wrong action dimension
        b=np.zeros(len(ACTIONS) + 7),
        lr=np.float64(0.25), baseline_decay=np.float64(0.9),
        n_updates=np.int64(0), epsilon=np.float64(0.0),
        epsilon_decay=np.float64(1.0),
    )
    with pytest.raises(ValueError, match="action dimension"):
        load_policy(bad)


def test_load_missing_file_raises_file_not_found(tmp_path):
    """A57 (67.1.2): a missing path is a FileNotFoundError (not a silent
    fresh policy, the 67.4 CLI-error rule)."""
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nope.npz")


# --- 67.2 ε-greedy exploration -------------------------------------------------

def test_epsilon_greedy_stays_within_relevant_actions():
    """A57 (67.2.1): with ε = 1 every sampled action is one the family's
    mask allows — for both an mlp and a tree best spec."""
    pol = MetaRLPolicy(seed=1, epsilon=1.0)
    for family, state in (("mlp", _state()), ("tree", _state(family="tree"))):
        rel = set(relevant_actions(family))
        for _ in range(60):
            pol.propose(state)
            assert pol.last_proposal[0] in rel


def test_epsilon_greedy_is_deterministic_and_actually_explores():
    """A57 (67.2.1): same-seed ε-streams are identical, and for some seed
    the ε = 1 stream differs from the ε = 0 stream (exploration samples)."""
    states = [_state() for _ in range(40)]
    a = MetaRLPolicy(seed=42, epsilon=1.0)
    b = MetaRLPolicy(seed=42, epsilon=1.0)
    assert _propose_stream(a, states) == _propose_stream(b, states)

    differed = False
    for seed in range(1, 9):
        greedy = MetaRLPolicy(seed=seed, epsilon=1.0)
        pure = MetaRLPolicy(seed=seed, epsilon=0.0)
        if _propose_stream(greedy, states) != _propose_stream(pure, states):
            differed = True
            break
    assert differed, "ε-greedy never diverged from pure softmax (1..8 seeds)"


def test_epsilon_validation():
    """A57 (67.2.1): the construction guards on the ε schedule."""
    with pytest.raises(ValueError, match="epsilon"):
        MetaRLPolicy(seed=0, epsilon=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        MetaRLPolicy(seed=0, epsilon=1.5)
    with pytest.raises(ValueError, match="epsilon_decay"):
        MetaRLPolicy(seed=0, epsilon_decay=0.0)
    with pytest.raises(ValueError, match="epsilon_decay"):
        MetaRLPolicy(seed=0, epsilon_decay=1.5)
    # the boundaries are legal
    MetaRLPolicy(seed=0, epsilon=0.0, epsilon_decay=1.0)
    MetaRLPolicy(seed=0, epsilon=1.0, epsilon_decay=0.5)


def test_epsilon_decays_geometrically_per_episode(tmp_path):
    """A57 (67.2.2): ε = 0.5 · 0.5ᴺ after N episodes; a default policy's
    ε stays 0.0 through training (67.2.3)."""
    n_episodes = 3
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path / "dec")
    pol = MetaRLPolicy(seed=7, epsilon=0.5, epsilon_decay=0.5)
    train_policy(env, pol, n_episodes=n_episodes)
    assert pol.epsilon == pytest.approx(0.5 * (0.5 ** n_episodes), rel=1e-12)

    env2 = AutoRefineEnv(task="parity-v1", seed=7,
                         budget=Budget(2, 300, 30), runs_dir=tmp_path / "dec0")
    plain = MetaRLPolicy(seed=7)
    train_policy(env2, plain, n_episodes=1)
    assert plain.epsilon == 0.0 and plain.epsilon_decay == 1.0


# --- 67.3 best-episode checkpoint ---------------------------------------------

def test_default_train_policy_return_dict_has_no_best_key(tmp_path):
    """A57 (67.3.1 / 67.5): the default return dict is the pre-v0.53 shape —
    no `"best"` key at all."""
    env = AutoRefineEnv(task="sine-v1", seed=7,
                        budget=Budget(2, 300, 30), runs_dir=tmp_path / "def")
    report = train_policy(env, MetaRLPolicy(seed=7), n_episodes=1)
    assert set(report) == {"episodes", "episode_returns", "policy_updates",
                           "last_return"}
    assert "best" not in report


def test_best_checkpoint_is_the_strictly_greatest_episode(tmp_path):
    """A57 (67.3.1): `best=True` returns the best-episode snapshot — its
    return is the max, and its weights are exactly the policy's at the end
    of that (first) strictly-greatest episode (replayed independently)."""
    n_episodes = 3
    budget = Budget(2, 300, 30)

    env = AutoRefineEnv(task="sine-v1", seed=13, budget=budget,
                        runs_dir=tmp_path / "best")
    pol = MetaRLPolicy(seed=13)
    report = train_policy(env, pol, n_episodes=n_episodes, best=True)
    assert report["best"] is not None
    assert report["best"]["return"] == max(report["episode_returns"])

    # independent replay: one-episode drives, capturing end-of-episode
    # weights — deterministic (G2), so it must match the report exactly
    env2 = AutoRefineEnv(task="sine-v1", seed=13, budget=budget,
                         runs_dir=tmp_path / "replay")
    pol2 = MetaRLPolicy(seed=13)
    rets_local: list[float] = []
    snaps: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_episodes):
        train_policy(env2, pol2, n_episodes=1)
        rets_local.append(pol2.last_episode_return)
        snaps.append((pol2.w.copy(), pol2.b.copy()))
    assert rets_local == report["episode_returns"]
    best_idx = next(i for i, r in enumerate(rets_local)
                    if r == max(rets_local))
    assert report["best"]["episode"] == best_idx
    assert np.array_equal(report["best"]["w"], snaps[best_idx][0])
    assert np.array_equal(report["best"]["b"], snaps[best_idx][1])
    assert report["best"]["B"] is None  # single-task mode
    assert set(report) == {"episodes", "episode_returns", "policy_updates",
                           "last_return", "best"}


# --- 67.4 the CLI cycle ---------------------------------------------------------

def test_cli_run_rl_save_then_load(tmp_path):
    """A57 (67.4): `run --policy rl --rl-save` writes a loadable policy
    (n_updates ≥ 1), and a follow-up `run --rl-load` resumes from it (the
    save → load chain)."""
    runs = tmp_path / "runs"
    pol_path = tmp_path / "pol.npz"
    base = ["run", "--task", "parity-v1", "--policy", "rl",
            "--rl-episodes", "2", "--experiments", "2",
            "--max-seconds", "60", "--max-train-seconds", "10",
            "--runs-dir", str(runs), "--seed", "7"]
    rc = cli_main(base + ["--rl-save", str(pol_path), "--quiet"])
    assert rc == 0
    assert pol_path.exists()
    loaded = load_policy(pol_path, seed=7)
    assert loaded.n_updates >= 1

    rc2 = cli_main(base + ["--rl-load", str(pol_path),
                           "--rl-episodes", "1", "--quiet"])
    assert rc2 == 0


def test_cli_run_accepts_epsilon_flags():
    """A57 (67.4): the ε schedule flags are on `run` (defaults 0.0 / 1.0)."""
    args = build_parser().parse_args([
        "run", "--rl-epsilon", "0.5", "--rl-epsilon-decay", "0.9"])
    assert args.rl_epsilon == 0.5
    assert args.rl_epsilon_decay == 0.9
    assert args.rl_save is None and args.rl_load is None
    dflt = build_parser().parse_args(["run"])
    assert dflt.rl_epsilon == 0.0 and dflt.rl_epsilon_decay == 1.0


# --- 67.5 pin, exports, version ------------------------------------------------

def test_a6_default_construction_and_stream_unchanged():
    """A57 (67.5): the default policy is the pre-v0.53 one — the same
    construction, the same single-draw proposal stream as A6 pins."""
    a = MetaRLPolicy(seed=3)
    b = MetaRLPolicy(seed=3)
    states = [_state() for _ in range(5)]
    sa = [a.propose(s) for s in states]
    sb = [b.propose(s) for s in states]
    for x, y in zip(sa, sb):
        assert ModelSpec.from_dict(x).to_dict() == ModelSpec.from_dict(y).to_dict()
    assert a.epsilon == 0.0 and a.epsilon_decay == 1.0


def test_exports_and_version():
    """A57 (33.1): the new functions are in `__all__` and resolvable; the
    version steps to 0.53.0 in both sources."""
    for name in ("save_policy", "load_policy"):
        assert name in autorefine.__all__
        assert getattr(autorefine, name) is not None
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.56.0"
