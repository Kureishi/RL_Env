"""v0.9 visual dashboard: core runner + optional app + launcher (SPEC.md 23, A13).

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
