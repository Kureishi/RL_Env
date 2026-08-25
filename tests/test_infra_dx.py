"""Infrastructure & DX tests (SPEC.md 21.2/21.3, A11)."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autorefine import (
    AutoRefineEnv,
    BanditPolicy,
    Budget,
    SearchPolicy,
    SineRegressionV1,
    ascii_pareto,
    ascii_score_curve,
    plotting,
    plugins,
    svg_pareto,
    svg_score_curve,
)
from autorefine.cli import main as cli_main
from autorefine.tasks import TASKS

_SVG_NS = "http://www.w3.org/2000/svg"

_ROWS = [
    {"kind": "baseline", "holdout_score": 50.0, "gen_gap": 0.0,
     "train_seconds": 0.5, "accepted": True, "mutation": None},
    {"kind": "experiment", "holdout_score": 55.0, "gen_gap": 0.0,
     "train_seconds": 1.0, "accepted": True, "mutation": ["learning_rate"]},
    {"kind": "experiment", "holdout_score": 52.0, "gen_gap": 0.0,
     "train_seconds": 0.8, "accepted": False, "mutation": ["batch_size"]},
    {"kind": "curriculum", "difficulty": "parity-4-p0.10"},  # not scored
]
_PTS = [
    {"score": 50.0, "train_seconds": 0.5, "spec_hash": "a"},
    {"score": 55.0, "train_seconds": 1.0, "spec_hash": "b"},
    {"score": 52.0, "train_seconds": 0.8, "spec_hash": "c"},
]


# --- SPEC.md 21.2: plotting --------------------------------------------------

def test_ascii_score_curve_deterministic_and_content():
    out = ascii_score_curve(_ROWS)
    assert out == ascii_score_curve(_ROWS)  # G2
    assert "B" in out and "+" in out and "o" in out
    assert "55.00" in out and "50.00" in out
    assert "score vs experiment" in out
    assert "curriculum" not in out  # non-scored rows excluded


def test_ascii_score_curve_empty():
    assert ascii_score_curve([]) == "no scored experiments in log"


def test_svg_score_curve_valid_xml_and_points():
    svg = svg_score_curve(_ROWS)
    assert svg == svg_score_curve(_ROWS)  # G2
    root = ET.fromstring(svg)  # valid XML
    assert root.tag == f"{{{_SVG_NS}}}svg"
    assert len(root.findall(f".//{{{_SVG_NS}}}circle")) == 3
    poly = root.find(f".//{{{_SVG_NS}}}polyline")
    assert len(poly.get("points").split()) == 3


def test_svg_score_curve_empty_is_valid():
    ET.fromstring(svg_score_curve([]))


def test_ascii_pareto_content_and_empty():
    out = ascii_pareto(_PTS)
    assert out == ascii_pareto(_PTS)
    assert "*" in out and "seconds" in out and "55.00" in out
    assert ascii_pareto([]) == "no pareto points in log"


def test_svg_pareto_valid_xml_and_points():
    root = ET.fromstring(svg_pareto(_PTS))
    assert len(root.findall(f".//{{{_SVG_NS}}}circle")) == 3
    ET.fromstring(svg_pareto([]))


# --- SPEC.md 21.2: report CLI -------------------------------------------------

def _make_run(tmp_path):
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path / "runs"
    )
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def test_report_default_output_unchanged(tmp_path, capsys):
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "baseline_score" in out and "best spec" in out
    assert "score vs experiment:" not in out  # no plots without --plot
    assert not (run_dir / "score_curve.svg").exists()


def test_report_json_is_machine_readable(tmp_path, capsys):
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    summary = json.loads(out)
    on_disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == on_disk
    assert "best spec" not in out  # stdout is pure JSON


def test_report_plot_writes_svg_and_ascii(tmp_path, capsys):
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir), "--plot"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "score vs experiment" in out and "pareto frontier" in out
    for name in ("score_curve.svg", "pareto_frontier.svg"):
        text = (run_dir / name).read_text(encoding="utf-8")
        ET.fromstring(text)  # valid XML


def test_report_plot_json_stdout_pure_json(tmp_path, capsys):
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir), "--plot", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    json.loads(out)  # stdout still pure JSON
    assert (run_dir / "score_curve.svg").exists()  # SVGs still written


# --- SPEC.md 21.3: plugin loader ---------------------------------------------

class _FakeEp:
    """Entry-point stand-in: `name` + `load()` (SPEC.md 21.3 testability)."""

    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class _NoMethods:
    pass


def test_register_tasks_via_fake_eps_drives_env(tmp_path):
    found = plugins.register_tasks(eps=[_FakeEp("plug-sine", SineRegressionV1)])
    assert found == {"plug-sine": SineRegressionV1}
    assert "plug-sine" in TASKS
    try:
        env = AutoRefineEnv(
            task="plug-sine", seed=7,
            budget=Budget(1, 300, 30), runs_dir=tmp_path,
        )
        state = env.reset()
        assert np.isfinite(state["baseline_score"])
    finally:
        del TASKS["plug-sine"]
    assert "plug-sine" not in TASKS


def test_register_policies_returns_dict():
    from autorefine import BanditPolicy
    found = plugins.register_policies(eps=[_FakeEp("plug-bandit", BanditPolicy)])
    assert found == {"plug-bandit": BanditPolicy}


def test_invalid_plugin_entries_raise():
    with pytest.raises(plugins.PluginError, match="plug-bad"):
        plugins.register_tasks(eps=[_FakeEp("plug-bad", 42)])
    with pytest.raises(plugins.PluginError, match="propose"):
        plugins.register_policies(eps=[_FakeEp("plug-nomethod", _NoMethods)])
    with pytest.raises(plugins.PluginError, match="make_dataset"):
        plugins.register_tasks(eps=[_FakeEp("plug-nomethod", _NoMethods)])


def test_discover_sorted_and_deterministic():
    eps = [_FakeEp("b-task", SineRegressionV1), _FakeEp("a-task", SineRegressionV1)]
    out1 = plugins.discover(plugins.TASKS_GROUP, eps=eps)
    out2 = plugins.discover(plugins.TASKS_GROUP, eps=eps)
    assert list(out1) == ["a-task", "b-task"]
    assert out1 == out2


def test_plugins_list_cli(capsys):
    rc = cli_main(["plugins", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert plugins.TASKS_GROUP in out and plugins.POLICIES_GROUP in out
    assert "(none)" in out  # no external packages register entry points here


# --- SPEC.md 21.5: worked example — custom tabular task ----------------------

def _tabular_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    import tabular  # examples/tabular.py
    return tabular


def test_tabular_example_task_protocol_and_registration(tmp_path):
    mod = _tabular_module()
    TASKS["tabular-v1"] = mod.TabularV1  # the example's registration step
    try:
        t = mod.TabularV1(seed=7)
        assert t.name == "tabular-v1" and t.head == "softmax" and t.n_outputs == 2
        assert t.default_dataset_size == 1024  # SPEC.md 20.3
        x1, y1 = t.make_dataset(64)
        x2, y2 = t.make_dataset(64)
        assert x1.shape == (64, 2) and y1.shape == (64,)
        assert set(np.unique(y1)) <= {0, 1}
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2)  # G2
        env = AutoRefineEnv(
            task="tabular-v1", seed=7,
            budget=Budget(1, 300, 30), runs_dir=tmp_path,
        )
        state = env.reset()
        assert np.isfinite(state["baseline_score"])
        assert state["baseline_score"] < 95.0  # default spec leaves room to improve
    finally:
        del TASKS["tabular-v1"]


def test_tabular_example_bandit_reaches_target(tmp_path):
    """SPEC.md 21.5: seed-7 bandit run clears the 95 target within 20 experiments."""
    mod = _tabular_module()
    TASKS["tabular-v1"] = mod.TabularV1
    try:
        env = AutoRefineEnv(
            task="tabular-v1", seed=7,
            budget=Budget(20, 900, 30), runs_dir=tmp_path,
        )
        state = env.reset()
        assert state["baseline_score"] < 95.0
        policy = BanditPolicy(seed=7)
        while not env.done:
            state, _r, _d, _i = env.step(policy.propose(state))
        assert env.best_score >= 95.0  # probe: ≈ 99.5, spec (16,8)/adam/1e-3/1000
    finally:
        del TASKS["tabular-v1"]
