"""The RL loop in the dashboard — RLRunner / RLMultiRunner / policy bytes
(v0.54, SPEC.md 68, A58).

SPEC.md 68: the self-improving RL path (SPEC.md 15 / 20.2 / 67) gets a UI
home:
  * 68.1 — `RLRunner` / `RLMultiRunner` implement the policy-agnostic
    worker contract (`start()` → `next()`…→ `finish()`, SPEC.md 51.2.3)
    with the same message shapes as `DashboardRunner`; `RLRunner` and
    `train_policy` are the *same loop*, cross-pinned at 68.1.2;
    keep-best swaps the downloaded policy to the best-episode snapshot
    (68.1.3); the multi-task runner is round-robin over the built-ins with
    one shared policy (68.1.4).
  * 68.2 — `policy_to_bytes` / `policy_from_bytes` are the one home for
    the policy bytes (the 67.1 file format's in-memory twin); a run can
    resume from a saved policy (68.2.2); the ε schedule (67.2.2) is
    exercised UI-side.
  * 68.3 — the app surface: the sidebar RL knobs, the live run, the
    result block's policy download, and the Compare tab's multi-task
    panel (AppTest; streamlit optional).
  * 68.4 — A1–A57 stay green (every addition is additive or opt-in).

House rules: pure / deterministic (G2), stdlib + numpy, no cross-test
imports (every fixture synthesized here), the core stays streamlit-free
(the app tests touch `dashboard_app` only under AppTest). Envs stay tiny
(quadrant-XOR CSV, 2 experiments per episode, 1–2 episodes) for speed.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget
from autorefine.improver.rl_policy import (
    MetaRLPolicy,
    policy_from_bytes,
    policy_to_bytes,
    train_policy,
)
from autorefine.rl_dashboard import (
    BUILTIN_TASKS,
    RLRunner,
    RLMultiRunner,
    describe_policy,
)
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# The tiny per-episode budget: one baseline + one scored candidate.
EXP = 2
WALL = 900.0
TRAIN = 30.0


# --- fixtures (synthesized here — no cross-test imports) ----------------------

def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes)."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _runner(csv: Path, runs_dir: Path, seed: int, episodes: int,
            **kw) -> RLRunner:
    """A legacy-quality `RLRunner` (deterministic; the 68.1.2 semantics)."""
    return RLRunner(
        csv_path=str(csv), seed=seed, experiments=EXP,
        max_seconds=WALL, max_train_seconds=TRAIN,
        runs_dir=str(runs_dir), search_quality="legacy",
        episodes=episodes, target=95.0, **kw)


def _side_b_env(csv: Path, runs_dir: Path, seed: int,
                episodes: int) -> AutoRefineEnv:
    """The runner-independent env (68.1.2 side B): the same task / seed /
    budget / dataset / driver metadata, legacy quality (all knobs off)."""
    task = TASKS["csv"](seed=seed, path=str(csv))
    return AutoRefineEnv(
        task="csv", seed=seed,
        budget=Budget(EXP, WALL, TRAIN),
        runs_dir=str(runs_dir),
        dataset_episodes=int(task.default_dataset_size),
        task_config={"path": str(csv)},
        policy="rl", target=95.0, rl_episodes=episodes,
    )


def _drive(runner) -> tuple[dict, list[dict]]:
    """`start()` → `next()`… → `finish()`; returns (result, updates)."""
    updates: list[dict] = []
    runner.start()
    while not runner.done:
        updates.append(runner.next())
    return runner.finish(), updates


def _drive_from_running(runner) -> dict:
    """Drive an already-started runner to `finish()` (the resume test)."""
    while not runner.done:
        runner.next()
    return runner.finish()


# --- 68.1.1 lifecycle (A58) ----------------------------------------------------

def test_runner_start_info_and_update_shape(tmp_path):
    """A58 (68.1.1): `start()`'s info carries the shared keys + the RL
    knobs; every update carries `episode` / `epsilon` / `task` (and
    `ucb is None`); `done` after exactly `episodes` episodes."""
    csv = _write_csv(tmp_path)
    runner = _runner(csv, tmp_path / "runs", seed=11, episodes=2)
    info = runner.start()
    for key in ("label", "head", "rows", "baseline_score", "target",
                "seed", "budget_experiments", "run_dir", "policy"):
        assert key in info, key
    assert info["label"] == "churn"
    assert info["head"] == "softmax"
    assert info["policy"] == "rl"
    assert info["target"] == 95.0
    assert info["budget_experiments"] == EXP
    # the RL knobs (68.1.1): fresh policy, default ε schedule
    assert info["episodes"] == 2
    assert info["epsilon"] == 0.0
    assert info["epsilon_decay"] == 1.0
    assert info["n_updates"] == 0
    assert info["resumed"] is False
    assert info["rows"]["train"] > 0 and info["rows"]["holdout"] > 0

    updates = []
    while not runner.done:
        updates.append(runner.next())
    assert updates
    for u in updates:
        assert u["ucb"] is None  # UCB is a bandit concept (SPEC.md 26.4)
        assert u["episode"] in (1, 2)
        assert u["epsilon"] == 0.0
        assert u["task"] == "csv"
        assert isinstance(u["reward"], float)
        assert isinstance(u["index"], int)
    # exactly `episodes` episodes complete (68.1.1) — in order
    ended = [u["episode"] for u in updates if u["episode_done"]]
    assert ended == [1, 2]
    assert updates[-1]["done"] is True
    assert runner.done


def test_runner_finish_rl_block_and_policy_bytes(tmp_path):
    """A58 (68.1.1): `finish()` carries the shared result keys + the
    `rl` block; its `policy_bytes` decode via `policy_from_bytes` with
    `n_updates == episodes`, and `describe_policy` previews the state."""
    csv = _write_csv(tmp_path)
    runner = _runner(csv, tmp_path / "runs", seed=11, episodes=2)
    res, updates = _drive(runner)
    for key in ("verdict", "target", "final_best_score", "baseline_score",
                "experiments_run", "best_spec", "run_dir", "summary",
                "updates"):
        assert key in res, key
    assert res["verdict"] in ("PASS", "MISS")
    assert res["target"] == 95.0
    assert len(res["updates"]) == len(updates)

    rl = res["rl"]
    assert rl["episodes"] == 2
    assert len(rl["episode_returns"]) == 2
    assert rl["n_updates"] == 2
    assert rl["trace"], "the D2 trace must be non-empty"
    assert rl["keep_best"] is False
    assert rl["resumed"] is False
    assert rl["task_names"] == ["csv"]
    assert rl["returns"] == {"csv": list(rl["episode_returns"])}
    # the best-episode card (67.3.1) is present after ≥ 1 episode
    assert rl["best"] is not None
    assert rl["best"]["return"] == max(rl["episode_returns"])
    # the trace rows are the policy view's input (SPEC.md 29.2)
    for t in rl["trace"]:
        assert set(t) == {"task", "action", "probs", "reward"}
        assert t["task"] == "csv"

    pol = policy_from_bytes(rl["policy_bytes"])
    assert pol.n_updates == 2
    desc = describe_policy(pol)
    for key in ("n_actions", "state_dim", "task_names", "n_updates",
                "epsilon", "epsilon_decay", "lr", "baseline_decay",
                "has_bias"):
        assert key in desc, key
    assert desc["n_updates"] == 2
    assert desc["task_names"] is None
    assert desc["has_bias"] is False
    assert desc["n_actions"] == len(rl["trace"][0]["probs"])


# --- 68.1.2 one-home cross-pin (A58) ------------------------------------------

def test_one_home_cross_pin_legacy(tmp_path):
    """A58 (68.1.2): `RLRunner`'s per-episode returns (and the per-step
    action/reward stream) equal `train_policy`'s on the same (task,
    seed, budget, N episodes) — the loops are the same loop."""
    csv = _write_csv(tmp_path)
    S, N = 11, 2
    # side A: the runner (the app's live path)
    res_a, _ = _drive(_runner(csv, tmp_path / "a", seed=S, episodes=N))
    rets_a = res_a["rl"]["episode_returns"]
    trace_a = res_a["rl"]["trace"]
    assert res_a["rl"]["n_updates"] == N

    # side B: `train_policy` over an independently built env + policy
    env_b = _side_b_env(csv, tmp_path / "b", seed=S, episodes=N)
    pol_b = MetaRLPolicy(S)
    trace_b: list[dict] = []
    rep_b = train_policy(env_b, pol_b, n_episodes=N, trace=trace_b)
    rets_b = rep_b["episode_returns"]
    assert rep_b["policy_updates"] == N

    assert len(rets_a) == N == len(rets_b)
    for x, y in zip(rets_a, rets_b):
        assert x == y  # exact float equality — same loop, same RNG stream
    # the per-step stream agrees too (stronger than the returns):
    assert len(trace_a) == len(trace_b)
    assert [t["action"] for t in trace_a] == [t["action"] for t in trace_b]
    assert [t["reward"] for t in trace_a] == [t["reward"] for t in trace_b]


# --- 68.1.3 keep-best (A58) ------------------------------------------------------

def test_keep_best_true_downloads_best_checkpoint(tmp_path):
    """A58 (68.1.3): with `keep_best=True` the decoded `policy_bytes`
    equal the strictly-greater best-episode snapshot (replayed
    independently with `train_policy(..., best=True)`)."""
    csv = _write_csv(tmp_path)
    S, N = 13, 2
    res, _ = _drive(_runner(csv, tmp_path / "kb", seed=S, episodes=N,
                            keep_best=True))
    dec = policy_from_bytes(res["rl"]["policy_bytes"])
    assert res["rl"]["keep_best"] is True

    env_b = _side_b_env(csv, tmp_path / "kb2", seed=S, episodes=N)
    pol_b = MetaRLPolicy(S)
    rep = train_policy(env_b, pol_b, n_episodes=N, best=True)
    assert rep["best"] is not None
    assert rep["best"]["return"] == max(rep["episode_returns"])
    assert res["rl"]["best"] == {
        "episode": rep["best"]["episode"],
        "return": rep["best"]["return"],
    }
    assert np.array_equal(dec.w, rep["best"]["w"])
    assert np.array_equal(dec.b, rep["best"]["b"])
    assert dec.n_updates == N


def test_keep_best_false_keeps_final_episode_weights(tmp_path):
    """A58 (68.1.3): with `keep_best=False` (the CLI parity default) the
    decoded `policy_bytes` equal the final episode's weights (replayed
    independently)."""
    csv = _write_csv(tmp_path)
    S, N = 13, 2
    res, _ = _drive(_runner(csv, tmp_path / "kbf", seed=S, episodes=N,
                            keep_best=False))
    dec = policy_from_bytes(res["rl"]["policy_bytes"])
    assert res["rl"]["keep_best"] is False

    env_b = _side_b_env(csv, tmp_path / "kbf2", seed=S, episodes=N)
    pol_b = MetaRLPolicy(S)
    train_policy(env_b, pol_b, n_episodes=N)
    assert np.array_equal(dec.w, pol_b.w)
    assert np.array_equal(dec.b, pol_b.b)
    assert dec.n_updates == N


# --- 68.1.4 multi-task (A58) -----------------------------------------------------

def test_multi_runner_lifecycle(tmp_path):
    """A58 (68.1.4): two built-in tasks, one episode each — `done`; the
    `rl` block's `returns` has both tasks; the decoded policy has
    `B is not None` and `task_names == tasks`; the update `task`
    sequence starts with `tasks[0]` (round-robin)."""
    tasks = ["parity-v1", "sine-v1"]
    assert set(tasks) <= set(BUILTIN_TASKS)
    m = RLMultiRunner(tasks=tasks, episodes_per_task=1, seed=7,
                      experiments=EXP, max_seconds=WALL,
                      max_train_seconds=TRAIN,
                      runs_dir=str(tmp_path / "multi"),
                      search_quality="legacy")
    info = m.start()
    assert info["episodes"] == 2
    assert info["episodes_per_task"] == 1
    assert info["tasks"] == tasks
    assert info["policy"] == "rl"
    assert info["resumed"] is False
    # the built-in tasks carry no class metadata — the display-only info
    # block degrades, it never crashes (68.1.4)
    assert info["class_values"] is None
    assert info["rows"]["train"] > 0

    updates = []
    while not m.done:
        updates.append(m.next())
    assert updates[0]["task"] == tasks[0]  # round-robin order (20.2)
    assert [u["task"] for u in updates if u["episode_done"]] == tasks
    assert sum(1 for u in updates if u["episode_done"]) == 2
    assert updates[-1]["done"] is True

    res = m.finish()
    rl = res["rl"]
    assert len(rl["returns"]["parity-v1"]) == 1
    assert len(rl["returns"]["sine-v1"]) == 1
    assert rl["best"] is None  # 67.3.2: "best episode" is ambiguous for multi
    assert rl["task_names"] == tasks
    assert rl["n_updates"] == 2
    assert rl["keep_best"] is False
    pol = policy_from_bytes(rl["policy_bytes"])
    assert pol.B is not None
    assert pol.task_names == tasks
    assert pol.n_updates == 2
    # the trace covers both tasks (the live policy views' input)
    assert set(t["task"] for t in rl["trace"]) == set(tasks)


def test_multi_runner_guards(tmp_path):
    """A58 (68.1.4): the construction guards fail loud — empty /
    duplicate / unknown tasks, a bad `episodes_per_task`, a bad
    `search_quality`; the single-task runner guards mirror them."""
    with pytest.raises(ValueError, match="non-empty"):
        RLMultiRunner(tasks=[])
    with pytest.raises(ValueError, match="unique"):
        RLMultiRunner(tasks=["parity-v1", "parity-v1"])
    with pytest.raises(ValueError, match="unknown task"):
        RLMultiRunner(tasks=["nope-v1"])
    with pytest.raises(ValueError, match="episodes_per_task"):
        RLMultiRunner(tasks=["parity-v1"], episodes_per_task=0)
    with pytest.raises(ValueError, match="episodes_per_task"):
        RLMultiRunner(tasks=["parity-v1"], episodes_per_task=True)
    for cls, kw in ((RLMultiRunner,
                     {"tasks": ["parity-v1"], "search_quality": "bogus"}),
                    (RLRunner,
                     {"csv_path": "x.csv", "search_quality": "bogus"})):
        with pytest.raises(ValueError, match="search_quality"):
            cls(**kw)
    with pytest.raises(ValueError, match="episodes"):
        RLRunner(csv_path="x.csv", episodes=0)
    with pytest.raises(ValueError, match="episodes"):
        RLRunner(csv_path="x.csv", episodes=True)
    with pytest.raises(ValueError, match="modality"):
        RLRunner(csv_path="x.csv", modality="video")


def test_multi_runner_rejects_incompatible_loaded_policies(tmp_path):
    """A58 (68.1.4): a loaded policy must carry the task bias `B`
    (task_names) and its `task_names` must match the runner's tasks."""
    single = policy_to_bytes(MetaRLPolicy(seed=7))  # B is None
    with pytest.raises(ValueError, match="task_names"):
        RLMultiRunner(tasks=["parity-v1", "sine-v1"],
                      initial_policy=single).start()

    mismatch = policy_to_bytes(
        MetaRLPolicy(seed=7, task_names=["parity-v1"]))
    with pytest.raises(ValueError, match="do not match"):
        RLMultiRunner(tasks=["parity-v1", "sine-v1"],
                      initial_policy=mismatch).start()


# --- 68.2 policy bytes + resume (A58) ------------------------------------------

def test_policy_bytes_roundtrip_single_and_multi():
    """A58 (68.2.1): `policy_from_bytes(policy_to_bytes(p))` is
    bit-equal to `p` — weights, hyper-parameters, n_updates, the ε
    schedule, and the multi-task `B` / `task_names` channel (SPEC.md
    20.2) — complementing A57's file-based pin."""
    p = MetaRLPolicy(seed=5, lr=0.3, baseline_decay=0.8,
                     epsilon=0.5, epsilon_decay=0.9)
    p.n_updates = 7
    p.w = p.w + 0.1  # non-trivial weights, still legal
    p.b = p.b + 0.01
    q = policy_from_bytes(policy_to_bytes(p), seed=5)
    assert np.array_equal(q.w, p.w)
    assert np.array_equal(q.b, p.b)
    assert p.B is None and q.B is None
    assert p.task_names is None and q.task_names is None
    assert q.lr == p.lr and q.baseline_decay == p.baseline_decay
    assert q.n_updates == 7
    assert q.epsilon == p.epsilon and q.epsilon_decay == p.epsilon_decay

    names = ["sine-v1", "parity-v1"]
    m = MetaRLPolicy(seed=9, task_names=names)
    m.n_updates = 3
    m.B[0, 3] = 1.5
    m.B[1, 7] = -2.25
    m2 = policy_from_bytes(policy_to_bytes(m), seed=9)
    assert np.array_equal(m2.B, m.B)
    assert m2.task_names == names
    assert m2.n_updates == 3
    desc = describe_policy(m2)
    assert desc["has_bias"] is True
    assert desc["task_names"] == names


def test_policy_bytes_guards():
    """A58 (68.2): garbage bytes are a `ValueError`; a `str` / `Path` is
    rejected with a pointer at `load_policy` (the file path)."""
    with pytest.raises(ValueError):
        policy_from_bytes(b"definitely not a tagged npz")
    with pytest.raises(ValueError, match="load_policy"):
        policy_from_bytes("some/path.npz")
    with pytest.raises(ValueError, match="load_policy"):
        policy_from_bytes(Path("some/path.npz"))


def test_resume_from_saved_policy(tmp_path):
    """A58 (68.2.2): `start()` with `initial_policy` bytes resumes — the
    policy's weights and `n_updates` equal the saved policy's;
    `info["resumed"]` is True; the run advances `n_updates` from there."""
    csv = _write_csv(tmp_path)
    S = 5
    saved = MetaRLPolicy(seed=S)
    train_policy(_side_b_env(csv, tmp_path / "pre", seed=S, episodes=1),
                 saved, n_episodes=1)
    w0, b0, nu0 = saved.w.copy(), saved.b.copy(), saved.n_updates
    assert nu0 >= 1

    runner = _runner(csv, tmp_path / "resume", seed=S, episodes=2,
                     initial_policy=policy_to_bytes(saved))
    info = runner.start()
    assert info["resumed"] is True
    assert info["n_updates"] == nu0
    assert np.array_equal(runner.policy.w, w0)
    assert np.array_equal(runner.policy.b, b0)

    res = _drive_from_running(runner)
    assert res["rl"]["resumed"] is True
    assert res["rl"]["n_updates"] == nu0 + 2


def test_epsilon_schedule_decays_per_episode(tmp_path):
    """A58 (68.2.2 / 67.2.2 UI-side): with `epsilon=1.0,
    epsilon_decay=0.5`, episode k's updates carry ε = 1.0·0.5^(k−1) and
    the episode's last update carries the post-decay value — after N
    episodes the policy's ε is 1.0·0.5^N. The default stays (0.0, 1.0)."""
    csv = _write_csv(tmp_path)
    N = 2
    runner = _runner(csv, tmp_path / "eps", seed=17, episodes=N,
                     epsilon=1.0, epsilon_decay=0.5)
    runner.start()
    updates = []
    while not runner.done:
        updates.append(runner.next())
    res = runner.finish()

    first_eps: dict[int, float] = {}
    last_eps: dict[int, float] = {}
    for u in updates:
        first_eps.setdefault(u["episode"], u["epsilon"])
        last_eps[u["episode"]] = u["epsilon"]
    for k in range(1, N + 1):
        # the ε in force during episode k (one decay factor per episode)
        assert first_eps[k] == pytest.approx(1.0 * 0.5 ** (k - 1), rel=1e-12)
        # the last update of episode k carries the decayed value
        assert last_eps[k] == pytest.approx(1.0 * 0.5 ** k, rel=1e-12)
    assert runner.policy.epsilon == pytest.approx(1.0 * 0.5 ** N, rel=1e-12)
    assert res["rl"]["epsilon_final"] == pytest.approx(1.0 * 0.5 ** N, rel=1e-12)
    assert res["rl"]["epsilon"] == 1.0
    assert res["rl"]["epsilon_decay"] == 0.5

    # the default path stays the exact single-draw policy (67.2.3)
    plain = _runner(csv, tmp_path / "eps0", seed=17, episodes=1)
    info = plain.start()
    assert info["epsilon"] == 0.0 and info["epsilon_decay"] == 1.0


# --- 68.3 the app surface (A58) ---------------------------------------------------

def test_app_rl_idle_boot_no_raise(tmp_path):
    """A58 (68.3.1): an idle boot with `policy="rl"` (advanced on) does
    not raise — the RL knobs' defaults materialize without a run."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["show_advanced"] = True
    at.session_state["policy"] = "rl"
    at.session_state["rl_episodes"] = 2
    at.run()
    assert not at.exception
    # the 68.3.1 defaults (materialized because the toggle is off by
    # default; here they are already set or materialized)
    ss = at.session_state
    assert ss["rl_epsilon"] == 0.0
    assert ss["rl_epsilon_decay"] == 1.0
    assert ss["rl_keep_best"] is True
    assert ss["rl_episodes"] == 2


def test_app_rl_run_exposes_policy_download(tmp_path):
    """A58 (68.3.3): a tiny `policy="rl"` run completes without
    exception and the result view exposes the policy download button +
    the episode / policy-view markdown."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["show_advanced"] = True
    at.session_state["policy"] = "rl"
    at.session_state["experiments"] = EXP
    at.session_state["max_train"] = 5.0
    at.session_state["rl_episodes"] = 2
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception

    labels = {e.label for e in at.get("download_button")}
    assert "Download the trained policy (.npz)" in labels
    subs = " ".join(e.value for e in at.subheader)
    assert "Reinforcement learning" in subs
    caps = " ".join(e.value for e in at.caption)
    assert "update(s)" in caps  # the n_updates · tasks · ε final line
    assert "csv" in caps  # the run's task name


def test_app_compare_multi_task_panel(tmp_path):
    """A58 (68.3.4): pressing 'Run the multi-task RL' on the Compare
    tab drives `RLMultiRunner` live and exposes the shared-policy
    download."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["experiments"] = EXP
    at.session_state["max_train"] = 5.0
    at.session_state["rlmulti_eps"] = 1
    at.run()
    assert not at.exception
    at.button(key="rlmulti_button").set_value(True).run()
    assert not at.exception
    labels = {e.label for e in at.get("download_button")}
    assert "Download the shared policy (.npz)" in labels
    # NOTE: this Streamlit build's AppTest does not expose the
    # download_button's `data` (`.value` is the pressed state) — the
    # "shared policy" caption is rendered from the very policy object
    # whose bytes the button downloads (the 68.3.4 panel), so it is the
    # faithful check here: the multi-task bias is ON, with the runner's
    # tasks. The deep policy-bytes pin (B + task_names round-trip) lives
    # in test_multi_runner_lifecycle over the same `finish()` block.
    caps = " ".join(c.value for c in at.caption)
    assert "shared policy:" in caps
    assert "bias on" in caps
    assert "tasks: parity-v1, sine-v1" in caps


# --- 33.1 exports + version (A58) --------------------------------------------------

def test_exports_and_version():
    """A58 (33.1): the five new names are in `autorefine.__all__` and
    resolvable (the package re-exports the very objects); the version
    steps to `0.58.0` in both sources."""
    for name in ("RLRunner", "RLMultiRunner", "describe_policy",
                 "policy_to_bytes", "policy_from_bytes"):
        assert name in autorefine.__all__
        assert getattr(autorefine, name) is not None
    import autorefine.improver.rl_policy as rp
    import autorefine.rl_dashboard as rd
    assert autorefine.RLRunner is rd.RLRunner
    assert autorefine.RLMultiRunner is rd.RLMultiRunner
    assert autorefine.describe_policy is rd.describe_policy
    assert autorefine.policy_to_bytes is rp.policy_to_bytes
    assert autorefine.policy_from_bytes is rp.policy_from_bytes
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.62.0"
