"""v0.76 on-demand UI (the Adapt tab) — SPEC.md 90, A80.

SPEC.md 90 (v0.76): the v0.75 ingestion machinery (89) is re-cut as the
dashboard's **Adapt** tab — on-demand data upload + model adaptation,
rendered as a thin layer over new pure cores (23.1):

  * 90.1 — ``latest_champion`` (the registry's latest finished run, the
    88.3.1 tie-break; ``None`` when there is none);
  * 90.2 — ``score_fresh`` (the champion's accuracy on a fresh CSV, judged
    by the 89.2 drift gate against its own ``final_best_score``);
  * 90.3 — ``refresh_model`` (champion vs a fresh ``DashboardRunner``
    challenger on the fresh CSV — the 89.3 PROMOTE/KEEP decision; optional
    89.7 recency-window training slice);
  * 90.4 — ``csv_preflight`` (a pure usability check: rows / cols / label /
    balance — no snapshot append);
  * 90.5 — ``quality_events`` (the 89.1.3 event log the Adapt tab charts);
  * 90.6 — the Adapt tab (the sixth, appended-last tab; the first five
    render exactly as before; empty state before a finished run);
  * 90.7 — acceptance: the version steps to ``0.83.0`` in both sources,
    the index row M79 resolves to this file, and the new exports land in
    ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (every fixture is
synthesized here), deterministic (G2), tiny budgets (2–3 experiments, a 50
bar where a PASS exit code is wanted), one module-scoped real-run fixture.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    latest_champion,
    csv_preflight,
    quality_events,
    refresh_model,
    score_fresh,
)
from autorefine.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

NEW_EXPORTS = (
    "latest_champion",
    "csv_preflight",
    "quality_events",
    "refresh_model",
    "score_fresh",
)


# --- shared fixtures (test-local; the house rules forbid cross-test imports) ---

def _write_csv(tmp_path: Path, name: str = "data.csv",
               rows: int = 60) -> Path:
    """Deterministic quadrant-XOR classification CSV (2 classes, 60 rows)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_dated_csv(tmp_path: Path, name: str = "dated.csv",
                     rows: int = 40) -> Path:
    """quadrant-XOR rows with a ``date`` column spanning 20 days — the
    89.7 recency-window fixture."""
    lines = ["date,a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        day = i % 20
        lines.append(f"2025-01-{day + 1:02d},{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_drifted_csv(tmp_path: Path, name: str = "drifted.csv",
                       rows: int = 40) -> Path:
    """Rows in the quadrant-XOR class-0 region whose label is always 1 —
    a champion trained on the clean data scores near 0 on them (the
    drift-alert fixture, 89.2)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 10.0          # 0.0 .. 0.4
        b = ((i // 5) % 4) / 10.0  # 0.0 .. 0.3
        lines.append(f"{a:.2f},{b:.2f},1")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run one CLI command, capturing stdout + stderr (the house pattern)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    """One real, tiny, finished ``go`` run — the champion every core test
    scores against (A80.1/A80.2/A80.3)."""
    root = tmp_path_factory.mktemp("ondemand")
    csv = _write_csv(root)
    runs = root / "runs"
    rc, out, err = _run_cmd([
        "go", "--data", str(csv),
        "--runs-dir", str(runs),
        "--experiments", "3", "--target", "50", "--quiet",
    ])
    assert rc == 0, out + err
    run_dirs = sorted(runs.glob("csv-*"))
    assert run_dirs and (run_dirs[0] / "summary.json").is_file()
    return {"root": root, "csv": csv, "runs": runs, "run_dir": run_dirs[0]}


# --- 90.1 the champion (A80.1) ---------------------------------------------------

class TestChampion:
    """SPEC.md 90.1 (A80.1): ``latest_champion`` — the champion run."""

    def test_resolves_the_latest_finished_run(self, champion):
        champ = latest_champion(str(champion["runs"]))
        assert champ is not None
        assert (Path(champ) / "summary.json").is_file()

    def test_none_when_there_is_no_finished_run(self, tmp_path):
        assert latest_champion(str(tmp_path / "none")) is None
        # an empty runs dir (no registry) is also None
        (tmp_path / "empty").mkdir()
        assert latest_champion(str(tmp_path / "empty")) is None


# --- 90.4 the preflight (A80.4) --------------------------------------------------

class TestPreflight:
    """SPEC.md 90.4 (A80.4): ``csv_preflight`` — a pure usability check."""

    def test_shape_and_label(self, champion):
        pf = csv_preflight(champion["csv"])
        assert pf["n_rows"] == 60 and pf["n_cols"] == 3
        assert pf["label_name"] == "churn" and pf["usable"] is True
        assert pf["label_column"] == 2
        assert set(pf["class_balance"]) == {"0", "1"}
        assert sum(pf["class_balance"].values()) == 60

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            csv_preflight(tmp_path / "nope.csv")


# --- 90.2 score the champion on fresh rows (A80.2) -------------------------------

class TestScoreFresh:
    """SPEC.md 90.2 (A80.2): ``score_fresh`` — the champion's drift verdict."""

    def test_stable_and_alert_and_margin(self, champion, tmp_path):
        good = _write_csv(tmp_path, "good.csv", rows=40)
        sf = score_fresh(champion["run_dir"], good, bar=2.0)
        assert sf["n"] == 40
        assert sf["status"] in ("stable", "alert")
        # margin = ref − fresh (the 89.2 definition)
        assert pytest.approx(sf["ref"] - sf["fresh"], abs=1e-6) == sf["margin"]
        assert sf["status"] == (
            "alert" if sf["margin"] > 2.0 else "stable")

    def test_drifted_data_is_an_alert(self, champion, tmp_path):
        bad = _write_drifted_csv(tmp_path)
        sf = score_fresh(champion["run_dir"], bad, bar=2.0)
        assert sf["status"] == "alert"
        assert sf["margin"] > 2.0

    def test_a_broken_champion_run_raises(self, champion, tmp_path):
        good = _write_csv(tmp_path, "g.csv")
        with pytest.raises((ValueError, FileNotFoundError)):
            score_fresh(tmp_path / "no-run", good)

    def test_a_missing_fresh_file_raises(self, champion, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            score_fresh(champion["run_dir"], tmp_path / "nope.csv")


# --- 90.3 champion vs challenger (A80.3) -----------------------------------------

class TestRefresh:
    """SPEC.md 90.3 (A80.3): ``refresh_model`` — the 89.3 PROMOTE/KEEP."""

    def test_champion_vs_challenger(self, champion, tmp_path):
        fresh = _write_csv(tmp_path, "fresh.csv", rows=40)
        r = refresh_model(fresh, champion["runs"], champion["run_dir"],
                          experiments=2, target=50, seed=7, margin=0.0)
        assert r["verdict"] in ("PROMOTE", "KEEP")
        assert r["n"] == 40
        assert Path(r["run_dir"]).is_dir()
        assert "challenger" in r["line"] and "champion" in r["line"]
        assert r["champion_score"] >= 0.0 and r["challenger_score"] >= 0.0

    def test_margin_controls_promotion(self, champion, tmp_path):
        fresh = _write_csv(tmp_path, "f2.csv", rows=40)
        r = refresh_model(fresh, champion["runs"], champion["run_dir"],
                          experiments=2, target=50, seed=7, margin=0.0)
        # a large margin can never be beaten -> KEEP (the 89.3.2 rule)
        r_big = refresh_model(fresh, champion["runs"], champion["run_dir"],
                              experiments=2, target=50, seed=7, margin=50.0)
        assert r_big["verdict"] == "KEEP"
        assert r["verdict"] in ("PROMOTE", "KEEP")

    def test_window_keep_all_when_no_date_column(self, champion, tmp_path):
        fresh = _write_csv(tmp_path, "f3.csv", rows=40)
        r = refresh_model(fresh, champion["runs"], champion["run_dir"],
                          experiments=2, target=50, seed=7, window="7d")
        w = r["window"]
        assert w["kept"] == 40 and w["total"] == 40
        assert w["note"]  # the keep-all note (no date column)

    def test_window_slices_a_dated_csv(self, champion, tmp_path):
        dated = _write_dated_csv(tmp_path)
        r = refresh_model(dated, champion["runs"], champion["run_dir"],
                          experiments=2, target=50, seed=7, window="7d")
        w = r["window"]
        assert w["lo"] is not None and w["hi"] is not None
        assert 0 < w["kept"] < w["total"]


# --- 90.5 the quality-over-time log (A80.5) --------------------------------------

class TestQualityEvents:
    """SPEC.md 90.5 (A80.5): ``quality_events`` — the 89.1.3 event log."""

    def test_missing_log_is_empty(self, champion, tmp_path):
        assert quality_events(str(tmp_path / "none")) == []

    def test_round_trips_a_written_log(self, champion, tmp_path):
        from autorefine.ingest import append_event
        rd = str(tmp_path / "runs")
        append_event(rd, {"poll": 1, "epoch": 1000, "rows": 10,
                          "champion_score": 90.0, "drift": "stable"})
        evs = quality_events(rd)
        assert len(evs) == 1 and evs[0]["champion_score"] == 90.0
        from autorefine.plotting import svg_quality_over_time
        assert svg_quality_over_time(evs).startswith("<svg")


# --- 90.6 the Adapt tab (A80.6) ---------------------------------------------------

class TestAdaptTab:
    """SPEC.md 90.6 (A80.6): the sixth tab renders (thin renderer, 23.1)."""

    @staticmethod
    def _at():
        pytest.importorskip("streamlit",
                            reason="dashboard app is optional")
        from streamlit.testing.v1 import AppTest
        return AppTest

    def test_empty_state_before_a_finished_run(self, tmp_path):
        AppTest = self._at()
        csv = _write_csv(tmp_path)
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()  # first render materializes the sidebar widgets
        assert not at.exception
        at.text_input(key="csv_path").set_value(str(csv))
        at.session_state["runs_dir"] = str(tmp_path / "runs_empty")
        at.session_state["experiments"] = 2
        at.run()  # re-render: the tabs (a data path is set, so no idle stop)
        assert not at.exception
        infos = [i.value for i in at.info]
        assert any("No finished run yet" in v for v in infos)

    def test_champion_and_drift_render(self, champion):
        AppTest = self._at()
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()
        assert not at.exception
        # point at the champion's own training CSV (stable, deterministic)
        at.text_input(key="csv_path").set_value(str(champion["csv"]))
        at.session_state["runs_dir"] = str(champion["runs"])
        at.session_state["experiments"] = 2
        at.run()
        assert not at.exception
        subs = [s.value for s in at.subheader]
        assert any("Adapt the model to new data" in v for v in subs)
        caps = " ".join(c.value for c in at.caption)
        # the champion line (the resolved champion — an earlier refresh
        # test in this module may have promoted a challenger in the same
        # runs dir, so assert against `latest_champion`, not the fixture)
        resolved = latest_champion(str(champion["runs"]))
        assert resolved is not None and resolved.name in caps
        # the drift gate rendered (stable on its own data -> a success line)
        verdicts = ([e.value for e in at.success]
                    + [e.value for e in at.error])
        assert any("Champion scores" in v for v in verdicts)


# --- 90.7 ceremony (A80.7) ---------------------------------------------------------

class TestCeremony:
    """SPEC.md 90.7 (A80.7): the version step + the index row + the exports."""

    def test_version_steps_to_076_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.83.0"' in init
        assert 'version = "0.83.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.83.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 90." in spec
        assert "### 90.7 Acceptance (A80)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M79 |"))
        assert "90" in row and "A80" in row
        assert "tests/test_ondemand_ui_v076.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
            assert hasattr(autorefine, name)
