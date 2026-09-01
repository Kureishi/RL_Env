"""v0.9 visual dashboard: core runner + optional app + launcher (SPEC.md 23, A13),
plus the v0.12 decision views (SPEC.md 26, A16).

The runner is streamlit-free and carries the run semantics (SPEC.md 23.1);
the app tests follow the §3/§21.1 optional-dependency pattern (skipped when
streamlit is absent).
"""
import builtins
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autorefine import DashboardRunner
from autorefine.dashboard import field_stats, spec_diff, ucb_trace
from autorefine.plotting import svg_mutation_timeline


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


def _runner(tmp_path, **kw) -> DashboardRunner:
    kw.setdefault("csv_path", _write_csv(tmp_path))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 4)
    kw.setdefault("policy", "bandit")
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    return DashboardRunner(**kw)


# --- runner: task view (SPEC.md 23.1) ---------------------------------------

def test_runner_start_reports_task(tmp_path):
    r = _runner(tmp_path)
    info = r.start()
    assert info["label"] == "churn"
    assert info["head"] == "softmax"
    assert info["class_values"] == [0.0, 1.0]
    assert info["features"] == ["a", "b"]
    assert info["rows"] == {"train": 48, "holdout": 6, "gen": 6}
    assert 0.0 <= info["baseline_score"] <= 100.0
    assert Path(info["run_dir"]).is_dir()
    assert r.done is False


# --- runner: live update stream (SPEC.md 23.1) -------------------------------

def test_runner_update_stream(tmp_path):
    r = _runner(tmp_path)
    r.start()
    updates = []
    while not r.done:
        u = r.next()
        json.dumps(u, sort_keys=True)  # every row JSON-safe (SPEC.md 23.1)
        updates.append(u)
        assert isinstance(u["accepted"], bool)
        assert np.isfinite(u["best_score"])
        if u["candidate_score"] is not None:
            assert np.isfinite(u["candidate_score"])
        else:
            assert u["accepted"] is False  # duplicates / budget stop
    assert updates
    assert [u["index"] for u in updates] == list(range(1, len(updates) + 1))
    assert updates[-1]["done"] is True
    assert all(not u["done"] for u in updates[:-1])


def test_runner_verdict_matches_the_fit_gate(tmp_path):
    # the §22.1 gate exactly: PASS iff final >= target
    for target in (1.0, 100.0):
        r = _runner(tmp_path, target=target)
        r.start()
        res = r.run_all()
        expected = "PASS" if res["final_best_score"] >= res["target"] else "MISS"
        assert res["verdict"] == expected == (
            "PASS" if res["final_best_score"] >= target else "MISS")
        assert res["final_best_score"] == pytest.approx(r.env.best_score)


def test_runner_deterministic_given_seed(tmp_path):
    # legacy search-quality = the v0.3 rule: strict score comparison with no
    # wall-clock eff term (SPEC.md 18) — so everything is reproducible except
    # the measured train_seconds (A2's wall-clock exclusion)
    streams = []
    for i in range(2):
        r = _runner(tmp_path, runs_dir=str(tmp_path / f"runs{i}"),
                    search_quality="legacy")
        r.start()
        res = r.run_all()
        streams.append([{k: v for k, v in u.items() if k != "train_seconds"}
                        for u in res["updates"]])
    assert streams[0] == streams[1]  # G2 (SPEC.md 23.1)


def test_runner_v04_quality_is_the_default(tmp_path):
    r = _runner(tmp_path)
    r.start()
    # SPEC.md 18.6: the v0.4 preset is active (z_accept > 0, like the CLI)
    assert r.env.z_accept > 0
    r2 = _runner(tmp_path, search_quality="legacy")
    r2.start()
    assert r2.env.z_accept == 0


def test_runner_artifact_payloads(tmp_path):
    r = _runner(tmp_path)
    r.start()
    res = r.run_all()
    spec = json.loads(res["artifacts"]["best_spec.json"])
    assert spec == res["best_spec"]  # the final best ModelSpec, verbatim
    with np.load(io.BytesIO(res["artifacts"]["best_model.npz"]),
                 allow_pickle=False) as npz:  # pickle-free checkpoints (T1/T2)
        assert len(npz.files) >= 1
    lines = [json.loads(x) for x in res["artifacts"]["experiments.jsonl"]
             .decode().splitlines() if x.strip()]
    assert lines and lines[0]["kind"] == "baseline"
    html = res["report_html"]
    assert res["svg_score"] in html and res["svg_pareto"] in html  # §22.2
    assert "<html" in html.lower()
    res["svg_score"]  # §21.2 SVGs are present and valid XML strings
    assert res["svg_score"].startswith("<svg")


# --- runner: error paths (SPEC.md 23.1) --------------------------------------

def test_runner_missing_file(tmp_path):
    r = DashboardRunner(csv_path=tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="no such CSV file"):
        r.start()


def test_runner_phase_errors(tmp_path):
    r = _runner(tmp_path)
    with pytest.raises(RuntimeError, match="start\\(\\)"):
        r.next()
    r.start()
    with pytest.raises(RuntimeError, match="not finished"):
        r.finish()


def test_runner_rejects_rl_policy_with_cli_pointer(tmp_path):
    with pytest.raises(ValueError, match="autorefine fit --policy rl"):
        _runner(tmp_path, policy="rl")


# --- decision views (SPEC.md 26, A16) ----------------------------------------

def test_field_stats_hand_computed():
    # D1 (SPEC.md 26.1): credit = the (mutation fields, accepted) pairs
    stream = [
        {"mutation": ["a"], "accepted": True},
        {"mutation": ["a", "b"], "accepted": False},
        {"mutation": ["b"], "accepted": True},
        {"mutation": [], "accepted": False},  # dup step: credits nothing
    ]
    stats = field_stats(stream)
    assert stats == {
        "a": {"trials": 2, "wins": 1, "win_rate": 0.5},
        "b": {"trials": 2, "wins": 1, "win_rate": 0.5},
    }
    assert list(stats) == ["a", "b"]  # lexicographically sorted keys
    assert field_stats([]) == {}


def test_spec_diff_old_new_and_absent():
    # D2 (SPEC.md 26.2): old = best spec before the step, new = the proposal
    prev = {"lr": 0.1, "hidden": [64]}
    action = {"lr": 0.05, "hidden": [32]}
    assert spec_diff(prev, action, ["lr", "hidden", "dropout"]) == [
        {"field": "lr", "old": 0.1, "new": 0.05},
        {"field": "hidden", "old": [64], "new": [32]},
        {"field": "dropout", "old": None, "new": None},  # absent → None
    ]
    assert spec_diff(prev, action, []) == []
    assert spec_diff(None, None, ["lr"]) == [
        {"field": "lr", "old": None, "new": None}]


def test_ucb_trace_hand_computed():
    # D4 (SPEC.md 26.4): the bandit's UCB (SPEC.md 17) after each credit
    import math
    stream = [
        {"mutation": ["a"], "accepted": True},
        {"mutation": ["a", "b"], "accepted": False},
    ]
    tr = ucb_trace(stream)  # alpha = 1.0
    assert tr["a"][0] == 1.0  # 1/1 + sqrt(ln 1 / 1)
    assert tr["b"][0] is None  # not credited yet → chart gap
    assert tr["a"][1] == pytest.approx(0.5 + math.sqrt(math.log(3) / 2))
    assert tr["b"][1] == pytest.approx(math.sqrt(math.log(3)))
    assert ucb_trace([]) == {}


def test_timeline_svg_cells_and_outcomes():
    # D3 (SPEC.md 26.3): one cell per (step, field) mutation, fill per outcome
    rows = [
        {"mutation": ["a", "b"], "accepted": True, "candidate_score": 70.0},
        {"mutation": ["a"], "accepted": False, "candidate_score": 60.0},
        {"mutation": ["b"], "accepted": False, "candidate_score": None},
    ]
    svg = svg_mutation_timeline(rows)
    assert svg.startswith("<svg")
    assert svg.count("<title>") == 4  # 2 + 1 + 1 cells (one tooltip each)
    assert ">a</text>" in svg and ">b</text>" in svg  # the field rows
    assert "accepted" in svg and "scored-rejected" in svg and "unscored" in svg
    empty = svg_mutation_timeline([])
    assert empty.startswith("<svg")
    assert "no mutations" in empty
    assert empty.count("<title>") == 0


def test_runner_updates_carry_decision_views(tmp_path):
    # A16: every update carries the views; the bandit ucb series grows per step
    r = _runner(tmp_path)  # bandit, 4 experiments
    r.start()
    res = r.run_all()
    updates = res["updates"]
    for i, u in enumerate(updates, start=1):
        json.dumps(u, sort_keys=True)  # still JSON-safe (SPEC.md 23.1)
        assert isinstance(u["field_stats"], dict)
        assert isinstance(u["spec_diff"], list)
        assert isinstance(u["ucb"], dict)  # bandit: per-field values
        # the per-field values *after that update* (SPEC.md 26.4)
        assert u["ucb"] == {f: s[-1] for f, s in ucb_trace(updates[:i]).items()}
    last = updates[-1]
    assert last["field_stats"] == field_stats(updates)
    assert last["ucb"] == {f: s[-1] for f, s in ucb_trace(updates).items()}
    assert res["field_stats"] == field_stats(updates)
    # the result's ucb_trace carries the full aligned per-field series
    assert res["ucb_trace"] == ucb_trace(updates)  # alpha = 1.0 (the default)
    assert all(len(s) == len(updates) for s in res["ucb_trace"].values())
    # D2 trajectory: the last accepted step's new values are the final best
    # spec (any later steps were rejected and could not change it)
    acc = [u for u in updates if u["accepted"]]
    if acc:
        for d in acc[-1]["spec_diff"]:
            assert res["best_spec"][d["field"]] == d["new"]
    assert res["timeline_svg"].startswith("<svg")


def test_runner_search_policy_has_no_ucb(tmp_path):
    # A16: UCB is a bandit concept — search gets None, the other views still
    # render (SPEC.md 26.4)
    r = _runner(tmp_path, policy="search")
    r.start()
    res = r.run_all()
    for u in res["updates"]:
        json.dumps(u, sort_keys=True)
        assert u["ucb"] is None
        assert isinstance(u["field_stats"], dict)
        assert isinstance(u["spec_diff"], list)
    assert res["ucb_trace"] is None
    assert isinstance(res["field_stats"], dict)
    assert res["timeline_svg"].startswith("<svg")


def test_decision_views_deterministic(tmp_path):
    # A16: same-seed runs → bit-identical views (G2)
    views = []
    for i in range(2):
        r = _runner(tmp_path, runs_dir=str(tmp_path / f"dv{i}"),
                    search_quality="legacy")
        r.start()
        res = r.run_all()
        views.append((res["field_stats"], res["ucb_trace"],
                      res["timeline_svg"]))
    assert views[0] == views[1]


# --- app (streamlit optional — §3/§21.1 pattern) -----------------------------

APP = Path(__file__).resolve().parents[1] / "src" / "autorefine" / "dashboard_app.py"


def test_app_renders_and_runs_end_to_end(tmp_path):
    # the app is an optional extra (§3/§21.1 pattern); the runner tests above
    # do not need streamlit
    pytest.importorskip("streamlit", reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest

    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()  # first render: no CSV yet → idle hint, no exception
    assert not at.exception

    # widgets exist after the first render; configure, re-render (data preview)
    at.text_input(key="csv_path").set_value(str(csv))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception

    at.button(key="run_button").set_value(True).run()
    assert not at.exception

    verdicts = [e.value for e in at.success] + [e.value for e in at.error]
    assert any(("PASS" in v) or ("MISS" in v) for v in verdicts)  # §22.1 gate
    md = " ".join(m.value for m in at.markdown)
    assert md.count("<svg") >= 2  # score curve + pareto (SPEC.md 21.2)
    assert len(at.get("download_button")) >= 4  # §23.2 artifact downloads
    assert len(at.metric) == 4
    runs = list((tmp_path / "runs").glob("csv-seed7-*"))
    assert runs and (runs[0] / "summary.json").exists()

    # Persistence (SPEC.md 23.2): a widget-triggered re-run (e.g. a
    # download-button click) must not lose the run — the finished result
    # stays on screen instead of resetting to the pre-run state
    at.run()
    assert not at.exception
    verdicts2 = [e.value for e in at.success] + [e.value for e in at.error]
    assert any(("PASS" in v) or ("MISS" in v) for v in verdicts2)
    md2 = " ".join(m.value for m in at.markdown)
    assert md2.count("<svg") >= 2  # plots re-rendered
    assert len(at.get("download_button")) >= 4  # artifact buttons re-rendered
    assert len(at.dataframe) >= 1  # final experiment table re-rendered
    assert at.button(key="clear_button").value is False  # visible, unpressed

    # and **New run (clear result)** drops the stored result (SPEC.md 23.2)
    at.button(key="clear_button").set_value(True).run()
    assert not at.exception
    verdicts3 = [e.value for e in at.success] + [e.value for e in at.error]
    assert not any(("PASS" in v) or ("MISS" in v) for v in verdicts3)
    assert len(at.get("download_button")) == 0


def _app_chart_marks(at) -> tuple[int, int]:
    """(bar, line) chart counts. streamlit 1.5x's AppTest has no typed
    bar/line elements — both arrive as `arrow_vega_lite_chart`, so the mark
    type is read from the element's Vega-Lite spec."""
    bars = lines = 0

    def walk(node):
        nonlocal bars, lines
        ch = getattr(node, "children", None)
        if not isinstance(ch, dict):
            return
        for el in ch.values():
            if getattr(el, "type", None) == "arrow_vega_lite_chart":
                spec = el.proto.spec
                if '"type": "bar"' in spec:
                    bars += 1
                elif '"type": "line"' in spec:
                    lines += 1
            walk(el)

    walk(at.main)
    return bars, lines


def test_app_renders_decision_views(tmp_path):
    # A16: all four views render live + in the result for bandit; the search
    # policy gets no UCB chart (SPEC.md 26.4/26.5)
    pytest.importorskip("streamlit", reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest

    def run_policy(policy: str):
        csv = _write_csv(tmp_path, name=f"data_{policy}.csv")
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()  # first render materializes the sidebar widgets
        assert not at.exception
        at.text_input(key="csv_path").set_value(str(csv))
        at.text_input(key="runs_dir").set_value(str(tmp_path / f"runs_{policy}"))
        at.selectbox(key="policy").set_value(policy)
        at.number_input(key="experiments").set_value(2)
        at.number_input(key="max_train").set_value(5.0)
        at.run()  # re-render: data preview + the Run button (idle stops earlier)
        assert not at.exception
        at.button(key="run_button").set_value(True).run()
        assert not at.exception
        return at

    b = run_policy("bandit")
    bars, lines = _app_chart_marks(b)
    assert bars >= 2   # D1 win-rate bars: live + result (SPEC.md 26.1/26.5)
    assert lines >= 2  # best-score curve + the bandit UCB trace (SPEC.md 26.4)
    md = " ".join(m.value for m in b.markdown)
    assert md.count("mutation timeline") >= 2  # D3 timeline: live + result
    assert "Decision views" in " ".join(h.value for h in b.subheader)
    assert "→" in " ".join(c.value for c in b.caption)  # D2 chips (SPEC.md 26.2)

    s = run_policy("search")
    sbars, slines = _app_chart_marks(s)
    assert sbars >= 1   # D1 renders for both policies
    assert slines == 1  # best-score curve only — no UCB chart (SPEC.md 26.4)
    assert "mutation timeline" in " ".join(m.value for m in s.markdown)


# --- launcher (SPEC.md 23.4) --------------------------------------------------

def test_dashboard_command_points_at_the_in_repo_app():
    from autorefine.cli import _dashboard_command
    cmd = _dashboard_command("runs", 8502)
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4].endswith("dashboard_app.py")
    assert Path(cmd[4]).is_file()  # the fixed in-repo file only (S1 exception)
    assert "--server.port=8502" in cmd
    assert "--runs-dir=runs" in cmd


def test_dashboard_missing_streamlit_prints_install_hint(tmp_path, monkeypatch,
                                                         capsys):
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    import argparse
    from autorefine.cli import _cmd_dashboard
    rc = _cmd_dashboard(argparse.Namespace(runs_dir="runs", port=8501))
    assert rc == 1
    err = capsys.readouterr().err
    assert "pip install autorefine[gui]" in err


def test_import_autorefine_never_pulls_in_streamlit():
    code = ("import sys; import autorefine; "
            "assert 'streamlit' not in sys.modules")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
