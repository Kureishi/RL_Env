"""v0.15 multi-run / policy views (SPEC.md 29, acceptance A19):

- D1 seed-variance box plot — `svg_seed_variance` (SVG geometry),
  `DashboardRunner.seed_sweep` (runner), the `autorefine variance` CLI, and
  the app's opt-in Seed variance panel (SPEC.md 29.1).
- D2 RL policy view — `MetaRLPolicy.probabilities` / opt-in train trace
  (SPEC.md 29.2), the `svg_action_probabilities` / `svg_task_returns`
  renderers, the `autorefine policy-report` CLI, and the app's opt-in
  precomputed policy panel (RL stays CLI-only, SPEC.md 23.1).

House rules: fixtures are duplicated from the other test modules (no
cross-test imports); every SVG is asserted as valid XML; budgets are kept
small (sine-v1 / parity-v1, 1-2 experiments, 1-2 episodes) for speed.
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autorefine import AutoRefineEnv, Budget, DashboardRunner, DEFAULT_SPEC
from autorefine.cli import main as cli_main
from autorefine.improver.catalog import ACTIONS, relevant_actions
from autorefine.improver.rl_policy import (
    MetaRLPolicy,
    train_multi_policy,
    train_policy,
)
from autorefine.plotting import (
    _ACCEPTED,
    _quantile,
    svg_action_probabilities,
    svg_seed_variance,
    svg_task_returns,
)

APP = Path(__file__).resolve().parents[1] / "src" / "autorefine" / "dashboard_app.py"


def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    (Duplicated fixture — tests/test_dashboard.py.)"""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _runner(tmp_path, **kw) -> DashboardRunner:
    kw.setdefault("csv_path", str(_write_csv(tmp_path)))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 3)
    kw.setdefault("policy", "bandit")
    kw.setdefault("max_train_seconds", 10.0)
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    return DashboardRunner(**kw)


def _label(i: int) -> str:
    """Catalog action index -> the renderer's ASCII label (SPEC.md 29.2)."""
    field, value = ACTIONS[i]
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    return f"{field}={value}"


def _rect_by_title(root, title: str):
    """The first <rect> whose <title> child reads `title` — namespace-
    agnostic (the SVG declares xmlns, so plain tag lookups miss)."""
    for el in root.iter():
        if not el.tag.endswith("rect"):
            continue
        for ch in el:
            if ch.tag.endswith("title") and (ch.text or "") == title:
                return el
    return None


def _bars_with_titles(root) -> dict:
    """{title text: fill} over every <rect> that carries a <title> child."""
    bars = {}
    for el in root.iter():
        if not el.tag.endswith("rect"):
            continue
        for ch in el:
            if ch.tag.endswith("title") and ch.text:
                bars.setdefault(ch.text, el.get("fill"))
                break
    return bars


def _synthetic_trace() -> list[dict]:
    """A tiny valid trace: 3 steps, 3 non-zero actions each (top-2 + rest)."""
    n = len(ACTIONS)
    trace = []
    for step in range(3):
        probs = [0.0] * n
        probs[step] = 0.5
        probs[step + 2] = 0.3
        probs[step + 4] = 0.2
        trace.append({"task": "sine-v1", "action": step, "probs": probs,
                      "reward": 0.1 * (step + 1)})
    return trace


def _episode_steps(task: str, seed: int, budget: Budget, runs_dir) -> int:
    """Drive one full episode with the same seed (G2) and count the steps —
    the replay `train_policy(..., trace)` must match (SPEC.md 29.2)."""
    env = AutoRefineEnv(task=task, seed=seed, budget=budget, runs_dir=runs_dir)
    policy = MetaRLPolicy(seed=seed)
    state = env.reset()
    n = 0
    while not env.done:
        spec = policy.propose(state)
        state, reward, done, info = env.step(spec)
        policy.observe(reward, done)
        n += 1
    return n


# --- D1 (SPEC.md 29.1): the quantile helper ----------------------------------

def test_quantile_linear_interpolation():
    """The hand-rolled quantile is the NumPy `linear` method (SPEC.md 29.1)."""
    vals = [60.0, 70.0, 80.0, 90.0]
    assert _quantile(vals, 0.5) == pytest.approx(75.0)
    assert _quantile(vals, 0.25) == pytest.approx(67.5)
    assert _quantile(vals, 0.75) == pytest.approx(82.5)
    assert _quantile(vals, 0.0) == 60.0
    assert _quantile(vals, 1.0) == 90.0
    assert _quantile([42.0], 0.3) == 42.0  # n=1 -> that value
    with pytest.raises(ValueError):
        _quantile([], 0.5)


# --- D1 (SPEC.md 29.1): the seed-variance SVG --------------------------------

def test_seed_variance_svg_quartiles_markers_target_and_passing():
    seeds = [
        {"seed": 1, "baseline": 60.0, "final": 70.0},
        {"seed": 2, "baseline": 65.0, "final": 80.0},
        {"seed": 3, "baseline": 70.0, "final": 90.0},
        {"seed": 4, "baseline": 55.0, "final": 60.0},
    ]
    svg = svg_seed_variance(seeds, target=85.0)
    root = ET.fromstring(svg)  # valid XML (A19)
    # hand-computed quartiles over the finals [60, 70, 80, 90], asserted in
    # text (the SVG is self-documenting, SPEC.md 29.1)
    for name, val in (("median", 75.0), ("q1", 67.5), ("q3", 82.5),
                      ("min", 60.0), ("max", 90.0)):
        assert f"{name} {val:.2f}" in svg
    # ... and by parsing the box rect on the fixed 0-100 axis
    T, B, H = 28.0, 64.0, 360.0  # the renderer's default layout
    ph = H - T - B
    y = lambda s: T + ph * (1.0 - s / 100.0)
    box = _rect_by_title(root, "box q1..q3")
    assert box is not None
    assert float(box.get("y")) == pytest.approx(y(82.5), abs=0.15)
    assert float(box.get("height")) == pytest.approx(y(67.5) - y(82.5), abs=0.15)
    # one paired baseline -> final marker per seed, labelled with its seed
    for s in seeds:
        assert f"baseline {s['baseline']:.2f}" in svg
        assert f"final {s['final']:.2f}" in svg
        assert f"seed {s['seed']}" in svg
    # the dashed target line + the passing summary (only 90.0 >= 85.0)
    assert "target 85.00" in svg
    assert "n seeds = 4 - passing = 1/4" in svg
    # without a target the passing count reads '-' (SPEC.md 29.1)
    svg2 = svg_seed_variance(seeds)
    ET.fromstring(svg2)
    assert "passing = - (no target)" in svg2


def test_seed_variance_svg_single_seed_and_empty():
    svg = svg_seed_variance([{"seed": 7, "baseline": 50.0, "final": 80.0}])
    ET.fromstring(svg)
    for name in ("median", "q1", "q3", "min", "max"):
        assert f"{name} 80.00" in svg
    assert "n seeds = 1 - passing = - (no target)" in svg
    empty = svg_seed_variance([])
    ET.fromstring(empty)  # empty -> header + message, still valid XML
    assert "no seeds in the sweep" in empty


# --- D1 (SPEC.md 29.1): DashboardRunner.seed_sweep ----------------------------

def test_seed_sweep_one_dict_per_seed_json_safe(tmp_path):
    updates = []
    sweep = _runner(tmp_path).seed_sweep([7, 8, 9], on_update=updates.append)
    assert len(sweep) == 3
    assert [s["seed"] for s in sweep] == [7, 8, 9]
    for s in sweep:
        assert set(s) == {"seed", "baseline", "final", "target", "pass",
                          "verdict", "experiments_run", "run_dir"}
        assert np.isfinite(s["baseline"]) and np.isfinite(s["final"])
        assert s["target"] == pytest.approx(95.0)
        assert s["pass"] == (s["final"] >= s["target"])  # the §22.1 gate
        assert s["verdict"] == ("PASS" if s["pass"] else "MISS")
        assert Path(s["run_dir"]).is_dir()
    json.dumps(sweep)  # every value JSON-safe (A19)
    assert len(updates) > 0  # on_update forwarded per step (SPEC.md 29.1)
    for u in updates:
        json.dumps(u)


def test_seed_sweep_empty_returns_empty_list(tmp_path):
    r = _runner(tmp_path)
    assert r.seed_sweep([]) == []
    assert r.seed_sweep(None) == []  # forgiving None, same empty result


def test_seed_sweep_caller_runner_untouched(tmp_path):
    """The sweep never disturbs the caller's own run (SPEC.md 29.1)."""
    r = _runner(tmp_path)
    r.start()
    best_before = r.env.best_score
    caller_dir = str(r.env.run_dir)
    sweep = r.seed_sweep([7, 8])
    assert r.env.best_score == best_before
    assert r.done is False  # still running, exactly as before the sweep
    assert all(s["run_dir"] != caller_dir for s in sweep)
    # the caller can drive its own loop to done and finish cleanly
    while not r.done:
        r.next()
    res = r.finish()
    assert np.isfinite(res["final_best_score"])
    assert Path(res["run_dir"]) == Path(caller_dir)


def test_seed_sweep_bit_identical_for_same_seeds(tmp_path):
    """Two sweeps under the same seeds give identical scores (G2); only the
    run dirs differ (timestamps/counters, SPEC.md 29.1)."""
    a = _runner(tmp_path, runs_dir=str(tmp_path / "s1")).seed_sweep([7, 8])
    b = _runner(tmp_path, runs_dir=str(tmp_path / "s2")).seed_sweep([7, 8])
    for x, y in zip(a, b):
        assert x["baseline"] == y["baseline"]
        assert x["final"] == y["final"]
        assert x["run_dir"] != y["run_dir"]


# --- D1 (SPEC.md 29.1): the variance CLI --------------------------------------

def test_variance_cli_exits_zero_writes_artifacts(tmp_path, capsys):
    csv = _write_csv(tmp_path)
    runs = tmp_path / "vr"
    rc = cli_main(["variance", "--data", str(csv), "--seeds", "3",
                   "--seed", "7", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    svg = (runs / "seed_variance.svg").read_text(encoding="utf-8")
    ET.fromstring(svg)  # valid XML (A19)
    sweep = json.loads((runs / "seed_sweep.json").read_text(encoding="utf-8"))
    assert [s["seed"] for s in sweep] == [7, 8, 9]
    # per-seed verdict lines: PASS/MISS exactly as the §22.1 gate says
    lines = out.splitlines()
    for s in sweep:
        row = next(l for l in lines if l.split() and l.split()[0] == str(s["seed"]))
        assert ("PASS" if s["pass"] else "MISS") in row
    # on this deterministic data at least one seed misses the 95.0 target, so
    # the "a MISS seed still exits 0 and prints MISS" case is exercised (A19)
    assert any(not s["pass"] for s in sweep)
    assert "MISS" in out and "PASS" in out
    assert "pass 95.0" in out
    assert str(runs / "seed_variance.svg") in out


def test_variance_cli_json_is_pure_json(tmp_path, capsys):
    csv = _write_csv(tmp_path)
    runs = tmp_path / "vrj"
    rc = cli_main(["variance", "--data", str(csv), "--seeds", "2",
                   "--seed", "7", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs),
                   "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    sweep = json.loads(out)  # stdout is pure JSON (SPEC.md 21.2 pattern)
    assert [s["seed"] for s in sweep] == [7, 8]
    assert json.loads((runs / "seed_sweep.json").read_text(encoding="utf-8")) == sweep


def test_variance_cli_missing_data_errors(tmp_path, capsys):
    rc = cli_main(["variance", "--data", "/definitely/not/here.csv",
                   "--seeds", "2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no such" in err


# --- D2 (SPEC.md 29.2): probabilities / propose / trace -----------------------

def test_probabilities_length_finite_and_normalized():
    """Length is the catalog's (asserted, not hardcoded); the masked softmax
    sums to 1 over the relevant actions (SPEC.md 29.2, 18.2)."""
    p = MetaRLPolicy(seed=3)
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 50.0,
             "experiments_left": 5, "done": False}
    for family_state in (
        state,  # mlp best spec: every action is relevant
        {**state, "best_spec": {**DEFAULT_SPEC.to_dict(), "model_family": "tree"}},
        {**state, "best_spec": {**DEFAULT_SPEC.to_dict(), "model_family": "knn"}},
    ):
        probs = p.probabilities(family_state)
        assert len(probs) == len(ACTIONS)
        assert np.isfinite(probs).all()
        family = str(family_state["best_spec"]["model_family"])
        rel = relevant_actions(family)
        masked = [i for i in range(len(ACTIONS)) if i not in set(rel)]
        assert sum(probs[i] for i in rel) == pytest.approx(1.0, abs=1e-9)
        assert sum(probs[i] for i in masked) < 1e-9  # masked actions read ~0


def test_probabilities_is_pure():
    """No sampling, no RNG draw, no state change (SPEC.md 29.2, G2)."""
    p = MetaRLPolicy(seed=3)
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 50.0,
             "experiments_left": 5, "done": False}
    w0, b0 = p.weights()
    p1 = p.probabilities(state)
    p2 = p.probabilities(state)
    assert np.array_equal(p1, p2)
    w1, b1 = p.weights()
    assert np.array_equal(w0, w1) and np.array_equal(b0, b1)
    assert p.last_proposal is None  # only propose() sets it
    assert p.n_updates == 0


def test_propose_rng_stream_unchanged_by_last_proposal():
    """`last_proposal` is bookkeeping set after the RNG draw: two same-seed
    policies still sample identical action sequences (A19, G2)."""
    state = {"best_spec": DEFAULT_SPEC.to_dict(), "best_score": 100.0,
             "experiments_left": 12, "done": False}
    a, b = MetaRLPolicy(seed=3), MetaRLPolicy(seed=3)
    seq_a, seq_b = [], []
    for _ in range(6):
        a.propose(state)
        seq_a.append(a.last_proposal[0])
        b.propose(state)
        seq_b.append(b.last_proposal[0])
    assert seq_a == seq_b
    assert all(0 <= i < len(ACTIONS) for i in seq_a)


def test_train_policy_trace_none_return_dict_unchanged(tmp_path):
    """With the trace off (the default) the return dict keeps exactly its
    pre-v0.15 shape and is bit-identical across same-seed runs (A19, G2)."""
    reports = []
    for i in range(2):
        env = AutoRefineEnv(task="sine-v1", seed=11,
                            budget=Budget(2, 300, 30),
                            runs_dir=tmp_path / f"tp-{i}")
        reports.append(train_policy(env, MetaRLPolicy(seed=11), n_episodes=1))
    r0, r1 = reports
    assert set(r0) == {"episodes", "episode_returns", "policy_updates",
                       "last_return"}
    assert r0 == r1  # byte-identical (no trace key when trace is None)


def test_train_multi_policy_trace_none_return_dict_unchanged(tmp_path):
    def run(i: int) -> dict:
        envs = [AutoRefineEnv(task=t, seed=11, budget=Budget(2, 300, 30),
                              runs_dir=tmp_path / f"mp-{i}-{j}")
                for j, t in enumerate(("sine-v1", "parity-v1"))]
        policy = MetaRLPolicy(seed=11, task_names=["sine-v1", "parity-v1"])
        return train_multi_policy(envs, policy, episodes_per_task=1)

    r0, r1 = run(0), run(1)
    assert set(r0) == {"tasks", "episodes_per_task", "episode_returns",
                       "last_returns", "policy_updates"}
    assert r0 == r1  # the §20.2 multi-task pin stays green (A19)


def test_train_policy_trace_one_record_per_step(tmp_path):
    budget = Budget(2, 300, 30)
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=budget,
                        runs_dir=tmp_path / "tr")
    trace: list[dict] = []
    train_policy(env, MetaRLPolicy(seed=7), n_episodes=1, trace=trace)
    # the trace has exactly one record per step (SPEC.md 29.2): count the
    # steps of the identical same-seed replay (G2)
    n_steps = _episode_steps("sine-v1", 7, budget, tmp_path / "replay")
    assert len(trace) == n_steps
    for t in trace:
        assert set(t) == {"task", "action", "probs", "reward"}
        assert t["task"] == "sine-v1"
        assert isinstance(t["action"], int) and 0 <= t["action"] < len(ACTIONS)
        assert len(t["probs"]) == len(ACTIONS)
        assert sum(t["probs"]) == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(t["reward"])


def test_train_multi_policy_trace_records_task_per_step(tmp_path):
    budget = Budget(2, 300, 30)
    envs = [AutoRefineEnv(task=t, seed=7, budget=budget,
                          runs_dir=tmp_path / f"mtr-{t}")
            for t in ("sine-v1", "parity-v1")]
    policy = MetaRLPolicy(seed=7, task_names=["sine-v1", "parity-v1"])
    trace: list[dict] = []
    report = train_multi_policy(envs, policy, episodes_per_task=1, trace=trace)
    assert len(trace) >= 2  # at least one step per episode
    tasks = [t["task"] for t in trace]
    assert set(tasks) == {"sine-v1", "parity-v1"}
    # round-robin order (SPEC.md 20.2): episode 0 all sine-v1, then parity-v1
    first_parity = tasks.index("parity-v1")
    assert all(t == "sine-v1" for t in tasks[:first_parity])
    assert all(t == "parity-v1" for t in tasks[first_parity:])
    for t in trace:
        assert len(t["probs"]) == len(ACTIONS)
        assert np.isfinite(t["reward"])
    assert report["episode_returns"]["sine-v1"]
    assert report["episode_returns"]["parity-v1"]


def test_train_multi_policy_env_task_count_mismatch_raises(tmp_path):
    """The §20.2 validation: env count must equal the task_names count."""
    envs = [AutoRefineEnv(task=t, seed=7, budget=Budget(1, 30, 10),
                          runs_dir=tmp_path / f"mm-{t}")
            for t in ("sine-v1", "parity-v1")]
    three = MetaRLPolicy(seed=7, task_names=["sine-v1", "parity-v1",
                                             "cartpole-v1"])
    with pytest.raises(ValueError, match="expected 3 envs"):
        train_multi_policy(envs, three, episodes_per_task=1)
    plain = MetaRLPolicy(seed=7)  # no task_names -> not a multi-task policy
    with pytest.raises(ValueError, match="task_names"):
        train_multi_policy(envs, plain, episodes_per_task=1)


# --- D2 (SPEC.md 29.2): the two policy-view SVGs ------------------------------

def test_action_probabilities_svg_topk_rest_and_highlight():
    trace = _synthetic_trace()
    svg = svg_action_probabilities(trace, top_k=2, n_steps=10)
    root = ET.fromstring(svg)  # valid XML (A19)
    first = trace[0]
    top = [0, 2]  # 0.5 and 0.3; the 0.2 action falls into `rest`
    for i in top:
        assert f"{_label(i)} {first['probs'][i]:.2f}" in svg
    assert "rest 0.20" in svg
    sampled = _label(0)
    assert f"sampled: {sampled}" in svg  # the row caption names it
    assert "step 0" in svg and "r=+0.100" in svg
    # the sampled action's bar is highlighted (green), the rest are not
    bars = _bars_with_titles(root)
    assert bars[f"{sampled} = {first['probs'][0]:.4f}"] == _ACCEPTED
    assert bars[f"{_label(2)} = {first['probs'][2]:.4f}"] != _ACCEPTED


def test_action_probabilities_svg_n_steps_slicing_and_empty():
    trace = _synthetic_trace()
    svg = svg_action_probabilities(trace, top_k=2, n_steps=2)
    ET.fromstring(svg)
    assert "last 2 of 3 step(s)" in svg  # only the last n_steps rows
    assert "step 1" in svg and "step 0" not in svg
    empty = svg_action_probabilities([])
    ET.fromstring(empty)  # empty-safe (A19)
    assert "no policy trace" in empty


def test_task_returns_svg_one_line_per_task_and_empty():
    returns = {"sine-v1": [1.0, 2.0, 3.0], "parity-v1": [0.5, 1.5]}
    svg = svg_task_returns(returns)
    ET.fromstring(svg)
    assert svg.count("<polyline") == 2  # one line per task
    assert "sine-v1" in svg and "parity-v1" in svg  # the legend names them
    single = svg_task_returns({"a": [1.0]})
    ET.fromstring(single)
    assert single.count("<polyline") == 1  # a single-task trace is one line
    empty = svg_task_returns({})
    ET.fromstring(empty)  # empty-safe (A19)
    assert "no episode returns recorded" in empty


# --- D2 (SPEC.md 29.2): the policy-report CLI ---------------------------------

def test_policy_report_cli_single(tmp_path, capsys):
    runs = tmp_path / "pr"
    rc = cli_main(["policy-report", "--task", "sine-v1", "--seed", "7",
                   "--episodes", "1", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    ET.fromstring((runs / "action_probabilities.svg").read_text(encoding="utf-8"))
    ET.fromstring((runs / "task_returns.svg").read_text(encoding="utf-8"))
    trace = json.loads((runs / "policy_trace.json").read_text(encoding="utf-8"))
    assert len(trace) >= 1
    assert all(len(t["probs"]) == len(ACTIONS) for t in trace)
    returns = json.loads((runs / "policy_returns.json").read_text(encoding="utf-8"))
    assert len(returns["episode_returns"]) == 1
    assert "policy_updates" in out and "trace steps" in out


def test_policy_report_cli_multi(tmp_path, capsys):
    runs = tmp_path / "prm"
    rc = cli_main(["policy-report", "--multi", "--tasks", "sine-v1,parity-v1",
                   "--seed", "7", "--episodes", "1", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    tr_svg = (runs / "task_returns.svg").read_text(encoding="utf-8")
    ET.fromstring(tr_svg)
    assert "sine-v1" in tr_svg and "parity-v1" in tr_svg
    ET.fromstring((runs / "action_probabilities.svg").read_text(encoding="utf-8"))
    returns = json.loads((runs / "policy_returns.json").read_text(encoding="utf-8"))
    assert set(returns["episode_returns"]) == {"sine-v1", "parity-v1"}
    trace = json.loads((runs / "policy_trace.json").read_text(encoding="utf-8"))
    assert {t["task"] for t in trace} == {"sine-v1", "parity-v1"}
    assert "sine-v1:" in out and "parity-v1:" in out


def test_policy_report_cli_unknown_multi_task_errors(tmp_path, capsys):
    rc = cli_main(["policy-report", "--multi", "--tasks", "sine-v1,nope-v1",
                   "--runs-dir", str(tmp_path / "bad")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "unknown task" in err


# --- app (A19): the two opt-in panels (streamlit optional) ---------------------

def _app_setup(tmp_path):
    """Render the app with a CSV configured (the opt-in panels show)."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest
    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    return at


def _app_run_dirs(tmp_path) -> list:
    d = tmp_path / "runs"
    return sorted(d.glob("csv-seed*")) if d.is_dir() else []


def test_app_optin_panels_inert_until_pressed(tmp_path):
    at = _app_setup(tmp_path)
    subs = [h.value for h in at.subheader]
    assert any("Seed variance" in s for s in subs)
    assert any("RL policy view" in s for s in subs)
    md = " ".join(m.value for m in at.markdown)
    assert "final score across" not in md  # D1 SVG not rendered yet
    assert "action probabilities" not in md  # D2 SVG not rendered yet
    assert at.button(key="var_button").value is False
    assert at.button(key="pol_button").value is False
    assert _app_run_dirs(tmp_path) == []  # nothing ran


def test_app_seed_variance_button_runs_sweep(tmp_path):
    at = _app_setup(tmp_path)
    at.number_input(key="var_seeds").set_value(2)
    at.button(key="var_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "final score across 2 seed(s)" in md  # svg_seed_variance header
    caps = " ".join(c.value for c in at.caption)
    assert "2 seeds" in caps  # the passing/beat-baseline summary
    assert len(_app_run_dirs(tmp_path)) == 2  # one isolated run dir per seed


def test_app_policy_view_renders_precomputed_and_never_runs_rl(tmp_path):
    at = _app_setup(tmp_path)
    d = tmp_path / "pol"
    d.mkdir()
    (d / "action_probabilities.svg").write_text(
        svg_action_probabilities(_synthetic_trace(), top_k=2, n_steps=3),
        encoding="utf-8")
    (d / "task_returns.svg").write_text(
        svg_task_returns({"sine-v1": [1.0, 2.0], "parity-v1": [0.5, 1.5]}),
        encoding="utf-8")
    at.text_input(key="pol_dir").set_value(str(d))
    at.button(key="pol_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "action probabilities (last 3" in md
    assert "episode returns per task" in md
    assert _app_run_dirs(tmp_path) == []  # never runs RL live (SPEC.md 23.1)


def test_app_policy_view_warns_when_dir_lacks_artifacts(tmp_path):
    at = _app_setup(tmp_path)
    d = tmp_path / "empty_pol"
    d.mkdir()
    at.text_input(key="pol_dir").set_value(str(d))
    at.button(key="pol_button").set_value(True).run()
    assert not at.exception
    warns = " ".join(w.value for w in at.warning)
    assert "no action_probabilities.svg" in warns


def test_import_autorefine_never_pulls_in_streamlit():
    """The core import stays clean even with the v0.15 views (SPEC.md 3/23.1)."""
    code = (
        "import sys, autorefine; "
        "bad = [m for m in ('streamlit',) if m in sys.modules]; "
        "assert not bad, bad"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
