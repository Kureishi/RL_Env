"""v0.75 time-based and on-demand data ingestion — SPEC.md 89, A79.

SPEC.md 89 (v0.75): the frozen-file assumption is retired one opt-in
surface at a time:

  * 89.1  — ``ingest.poll`` (the first poll always counts as changed;
    the state survives between ``feed`` calls); ``feed --data`` re-scores
    the champion on the fresh rows (the 42.1 leaf) and appends a scored
    event to ``data_events.jsonl``;
  * 89.2  — ``drift_verdict`` (stable inside the bar, alert just
    beyond it, with the margin); a degraded poll prints the ``drift :``
    alert line;
  * 89.3  — ``refresh_verdict`` (PROMOTE at/above the margin, KEEP
    below); ``refresh --data NEW.csv`` runs champion vs challenger
    end-to-end;
  * 89.4  — ``snapshot_dataset`` (deterministic ``ds-<sha12>`` id),
    ``find_snapshot`` (exact or unique prefix), ``datasets`` (the
    registry listing), ``fit --dataset`` (sha256-verified, the entry
    lands as ``dataset.json`` in the run dir; a changed file is rc 1);
  * 89.5  — ``schedule --rounds N`` (sequential rounds, each seeded by
    the previous round's run dir; rc = the last round's gate);
  * 89.6  — ``merge_tables`` (first header wins, mismatched header set
    skipped, exact-dup rows dropped); ``ingest DROP_DIR`` (the three
    lines, the snapshot; a second call is rc 1 — nothing new);
  * 89.7  — ``parse_window`` (``7d``/``24h``/``90m``/``30s`` →
    seconds; bad forms raise), ``window_rows`` (the recency slice),
    ``fit --window`` (the ``window :`` line; no date column is rc 1);
  * 89.8  — ``svg_quality_over_time`` (one titled dot per scored poll,
    alerts marked, ``""`` for no events);
  * 89.9  — ``report --backtest`` (the expanding-window replay, window
    i = the first ``ceil(n*i/K)`` rows; one line per window);
  * 89.10 — ``freshness_view`` / ``render_freshness`` (as-of / age /
    rows / drift); ``report --audience exec`` prints the block;
  * 89.11 — ``measure_ms_per_row`` / ``latency_line``; ``--latency-ms``
    gates the fit/go command (within budget keeps the rc, OVER budget is
    a MISS — rc 2);
  * 89.12 — ``go --feeds A.csv,B.csv`` (one shared budget split evenly,
    the ``portfolio :`` verdict; rc 0 iff every feed PASSes);
  * 89.13 — acceptance: the version steps to ``0.75.0`` in both
    sources, the index row M78 resolves to this file, and the new
    exports land in ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (every fixture
is synthesized here), deterministic (G2), tiny budgets (2–3
experiments, a 50 bar where a PASS exit code is wanted).
"""
from __future__ import annotations

import contextlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    drift_verdict,
    find_snapshot,
    freshness_view,
    latency_line,
    load_events,
    load_snapshots,
    merge_tables,
    parse_window,
    poll,
    refresh_verdict,
    render_freshness,
    snapshot_dataset,
    svg_quality_over_time,
    window_rows,
)
from autorefine.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

NEW_EXPORTS = (
    "find_date_col", "find_snapshot", "hash_file", "latest_snapshot",
    "load_snapshots", "parse_window", "snapshot_dataset", "append_snapshot",
    "window_rows",
    "append_event", "drift_verdict", "file_fingerprint", "hash_csv",
    "load_drop_state", "load_events", "load_state", "merge_tables", "poll",
    "refresh_verdict", "save_drop_state", "save_state",
    "latency_line", "measure_ms_per_row",
    "svg_quality_over_time",
    "freshness_view", "render_freshness",
)


# --- shared fixtures (test-local; the house rules forbid cross-test imports) ---

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


def _write_dated_csv(tmp_path: Path, name: str = "dated.csv") -> Path:
    """60 quadrant-XOR rows with a ``date`` column spanning 14 days
    (5 rows per day, 2025-01-01 onward) — the ``fit --window`` fixture."""
    lines = ["date,a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        day = i // 5  # 0..13
        iso = f"2025-01-{day + 1:02d}"
        lines.append(f"{iso},{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_bad_csv(tmp_path: Path, name: str = "bad.csv") -> Path:
    """60 rows whose features sit in the quadrant-XOR class-0 region
    (a <= 0.4, b <= 0.4) but whose label is always 1 — a champion
    trained on the clean data scores near 0 on it (the drift-alert
    fixture, 89.2)."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 10.0          # 0.0 .. 0.4
        b = ((i // 5) % 4) / 10.0  # 0.0 .. 0.3
        lines.append(f"{a:.2f},{b:.2f},1")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run one CLI command, capturing stdout + stderr (the house
    pattern; a module fixture's registry rows keep the state)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def go_run(tmp_path_factory):
    """One real, tiny, finished ``go`` run (A79.1 / A79.2 / A79.3 /
    A79.4 / A79.10 / A79.11 share it: it is the champion for the
    watcher / drift / refresh / freshness tests)."""
    root = tmp_path_factory.mktemp("ingest")
    csv = _write_csv(root)
    rc, out, _err = _run_cmd([
        "go", "--data", str(csv),
        "--runs-dir", str(root / "runs"),
        "--experiments", "3", "--target", "50", "--quiet",
    ])
    assert rc == 0, out
    runs = sorted((root / "runs").glob("csv-*"))
    assert runs and (runs[0] / "summary.json").is_file()
    return {"root": root, "csv": csv, "runs_dir": root / "runs",
            "run_dir": runs[0], "stdout": out}


# --- 89.1 the watcher (A79.1) --------------------------------------------------

class TestWatcher:
    """SPEC.md 89.1 (A79.1): ``poll`` + ``feed`` — the data watcher."""

    def test_poll_first_touch_change_and_skip(self, tmp_path):
        f = tmp_path / "w.csv"
        f.write_text("a,b\n1,0\n", encoding="utf-8")
        first = poll(f, None)
        assert first["changed"] is True and first["fingerprint"]
        again = poll(f, {"fingerprint": first["fingerprint"]})
        assert again["changed"] is False
        f.write_text("a,b\n1,0\n2,1\n", encoding="utf-8")  # size change
        changed = poll(f, {"fingerprint": first["fingerprint"]})
        assert changed["changed"] is True
        assert changed["fingerprint"] != first["fingerprint"]

    def test_feed_two_polls_picks_up_an_append(self, go_run, tmp_path,
                                              capsys):
        data = tmp_path / "feed.csv"
        data.write_text("a,b,churn\n0.1,0.1,0\n0.2,0.2,1\n0.3,0.2,0\n",
                        encoding="utf-8")
        runs = str(go_run["runs_dir"])
        champ = str(go_run["run_dir"])
        rc, out, err = _run_cmd([
            "feed", "--data", str(data), "--runs-dir", runs,
            "--champion", champ, "--max", "2", "--poll", "0",
        ])
        assert rc == 0, err
        assert "feed : poll 1:" in out and "feed : poll 2:" in out
        events = load_events(runs)
        assert len(events) >= 1
        assert events[0]["poll"] == 1
        for key in ("epoch", "iso", "rows", "champion_score", "drift",
                    "ds_id"):
            assert key in events[0]
        snaps = load_snapshots(runs)
        assert snaps and snaps[0]["ds_id"].startswith("ds-")

    def test_feed_missing_data_is_rc1(self, go_run, capsys):
        rc, _out, err = _run_cmd([
            "feed", "--data", str(go_run["root"] / "nope.csv"),
            "--runs-dir", str(go_run["runs_dir"]), "--champion",
            str(go_run["run_dir"]),
        ])
        assert rc == 1
        assert "no such data file" in err

    def test_feed_needs_a_champion(self, tmp_path, capsys):
        data = tmp_path / "f.csv"
        data.write_text("a,b,churn\n0.1,0.1,0\n0.2,0.2,1\n0.3,0.2,0\n",
                        encoding="utf-8")
        rc, _out, err = _run_cmd([
            "feed", "--data", str(data), "--runs-dir", "runs",
        ])
        assert rc == 1
        assert "--champion" in err and "--latest" in err


# --- 89.2 the drift gate (A79.2) -----------------------------------------------

class TestDrift:
    """SPEC.md 89.2 (A79.2): ``drift_verdict`` + the ``drift :`` line."""

    def test_verdict_stable_inside_bar_alert_beyond(self):
        assert drift_verdict(90.0, 92.0, bar=2.0) == ("stable", 2.0)
        assert drift_verdict(89.99, 92.0, bar=2.0) == ("alert", 2.01)
        status, margin = drift_verdict(80.0, 95.0, bar=2.0)
        assert status == "alert" and margin == pytest.approx(15.0)
        # a fresh score ABOVE the reference is stable (a negative margin)
        assert drift_verdict(97.0, 92.0, bar=2.0)[0] == "stable"

    def test_feed_alert_line_on_degraded_data(self, go_run, tmp_path):
        bad = _write_bad_csv(tmp_path)
        rc, out, err = _run_cmd([
            "feed", "--data", str(bad), "--runs-dir", str(go_run["runs_dir"]),
            "--champion", str(go_run["run_dir"]), "--max", "1",
            "--poll", "0",
        ])
        assert rc == 0, err
        assert "drift alert" in out
        assert "drift :" in out
        evs = load_events(str(go_run["runs_dir"]))
        assert evs and evs[-1]["drift"] == "alert"


# --- 89.3 champion/challenger (A79.3) -------------------------------------------

class TestRefresh:
    """SPEC.md 89.3 (A79.3): ``refresh_verdict`` + the ``refresh`` loop."""

    def test_verdict_promote_and_keep(self):
        v, line = refresh_verdict(90.0, 91.0, margin=0.0)
        assert v == "PROMOTE" and line.startswith("PROMOTE - challenger ")
        v, line = refresh_verdict(90.0, 90.9, margin=1.0)
        assert v == "KEEP" and line.startswith("KEEP - challenger ")
        v, _l = refresh_verdict(90.0, 92.0, margin=2.0)
        assert v == "PROMOTE"  # at the margin

    def test_refresh_end_to_end(self, go_run, tmp_path):
        new = _write_csv(tmp_path, "new.csv")
        rc, out, err = _run_cmd([
            "refresh", "--data", str(new),
            "--runs-dir", str(go_run["runs_dir"]),
            "--champion", str(go_run["run_dir"]),
            "--experiments", "2", "--target", "50",
        ])
        assert rc == 0, err
        assert "champion :" in out and "challenger :" in out
        assert "refresh : " in out
        assert ("PROMOTE" in out) or ("KEEP" in out)
        # a new run dir landed in the registry (the challenger loop)
        reg = json.loads((go_run["runs_dir"] / "registry.json").read_text(
            encoding="utf-8"))
        assert len(reg) >= 2


# --- 89.4 dataset snapshots (A79.4) ---------------------------------------------

class TestSnapshots:
    """SPEC.md 89.4 (A79.4): ``snapshot_dataset`` / ``datasets`` /
    ``fit --dataset``."""

    def test_snapshot_deterministic_and_prefix_resolvable(self, tmp_path):
        f = _write_csv(tmp_path)
        rd = str(tmp_path / "runs")
        s1 = snapshot_dataset(f, rd)
        s2 = snapshot_dataset(f, rd)
        assert s1 == s2  # ds_id, sha256, rows, cols, label, balance
        assert s1["ds_id"] == "ds-" + s1["sha256"][:12]
        assert s1["n_rows"] == 60 and s1["n_cols"] == 3
        prefix = s1["ds_id"][:10]
        assert find_snapshot(rd, s1["ds_id"])["ds_id"] == s1["ds_id"]
        assert find_snapshot(rd, prefix)["ds_id"] == s1["ds_id"]
        assert find_snapshot(rd, "ds-000000000000") is None

    def test_datasets_lists_the_registry(self, tmp_path, capsys):
        rd = str(tmp_path / "runs")
        s = snapshot_dataset(_write_csv(tmp_path), rd)
        rc, out, _err = _run_cmd(["datasets", "--runs-dir", rd])
        assert rc == 0
        assert s["ds_id"] in out and "60" in out
        assert "churn" in out  # the label column name

    def test_datasets_empty_registry(self, tmp_path, capsys):
        rc, out, _err = _run_cmd(["datasets", "--runs-dir",
                                  str(tmp_path / "runs")])
        assert rc == 0 and "none yet" in out

    def test_fit_dataset_verified_and_written(self, go_run, capsys):
        rd = str(go_run["runs_dir"])
        # snapshot the fixture's own 60-row csv (the watcher tests' snapshots
        # are unrelated feed files) and fit from that id
        snap = snapshot_dataset(go_run["csv"], rd)
        ds_id = snap["ds_id"]
        rc, out, err = _run_cmd([
            "fit", "--dataset", ds_id, "--runs-dir", rd,
            "--experiments", "2", "--target", "50", "--quiet",
        ])
        assert rc == 0, err
        assert "dataset :" in out
        # the fit run dir is the one carrying dataset.json (go's share.zip
        # sorts after the run dir, so we can't just take the last csv-*)
        hits = [p for p in go_run["runs_dir"].glob("csv-*/dataset.json")]
        assert hits, "no run dir wrote dataset.json"
        dj = hits[-1]
        assert dj.is_file()
        assert json.loads(dj.read_text(encoding="utf-8"))["ds_id"] == ds_id

    def test_fit_dataset_rejects_a_changed_file(self, go_run, capsys):
        rd = str(go_run["runs_dir"])
        snap = snapshot_dataset(go_run["csv"], rd)
        ds_id = snap["ds_id"]
        path = go_run["csv"]
        backup = path.read_text(encoding="utf-8")
        try:
            path.write_text(backup + "9.9,9.9,1\n", encoding="utf-8")
            rc, _out, err = _run_cmd([
                "fit", "--dataset", ds_id, "--runs-dir", rd,
                "--experiments", "2", "--target", "50", "--quiet",
            ])
            assert rc == 1
            assert "sha256" in err
        finally:
            path.write_text(backup, encoding="utf-8")


# --- 89.5 the schedule (A79.5) ---------------------------------------------------

class TestSchedule:
    """SPEC.md 89.5 (A79.5): ``schedule`` — N compounding rounds."""

    def test_two_rounds_and_closing_line(self, go_run, capsys):
        rc, out, err = _run_cmd([
            "schedule", "--data", str(go_run["csv"]),
            "--runs-dir", str(go_run["runs_dir"]),
            "--rounds", "2", "--every", "0",
            "--experiments", "2", "--target", "50",
        ])
        assert rc == 0, err
        assert "schedule : round 1/2 done" in out
        assert "schedule : round 2/2 done" in out
        assert "schedule : 2 rounds finished - best " in out

    def test_rounds_below_one_is_rc1(self, go_run, capsys):
        rc, _out, err = _run_cmd([
            "schedule", "--data", str(go_run["csv"]),
            "--runs-dir", str(go_run["runs_dir"]), "--rounds", "0",
        ])
        assert rc == 1
        assert "--rounds" in err


# --- 89.6 the drop folder (A79.6) --------------------------------------------------

class TestDropFolder:
    """SPEC.md 89.6 (A79.6): ``merge_tables`` + ``ingest DROP_DIR``."""

    def test_merge_dedupes_and_skips_a_mismatch(self, tmp_path):
        a = tmp_path / "a.csv"
        a.write_text("x,y\n1,2\n3,4\n3,4\n", encoding="utf-8")
        b = tmp_path / "b.csv"
        b.write_text("x,y\n3,4\n5,6\n", encoding="utf-8")
        c = tmp_path / "c.csv"
        c.write_text("x,z\n7,8\n", encoding="utf-8")  # different header set
        header, rows, merged, mismatched, dupes = merge_tables([a, b, c])
        assert header == ["x", "y"]
        assert rows == [["1", "2"], ["3", "4"], ["5", "6"]]
        assert merged == [str(a), str(b)]
        assert mismatched == [str(c)]
        assert dupes == 2  # a's self-dup + b's cross-dup

    def test_ingest_then_nothing_new(self, tmp_path, capsys):
        drop = tmp_path / "drop"
        drop.mkdir()
        (drop / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        (drop / "b.csv").write_text("x,y\n3,4\n", encoding="utf-8")
        rd = str(tmp_path / "runs")
        rc, out, err = _run_cmd(["ingest", str(drop), "--runs-dir", rd])
        assert rc == 0, err
        assert "ingest : 2 new file(s)" in out
        assert "merged 2 rows (0 duplicate row(s) skipped)" in out
        assert "snapshot ds-" in out
        merged = drop / "merged.csv"
        lines = [ln.rstrip(chr(13)) for ln in merged.read_text(
            encoding="utf-8").split(chr(10)) if ln.rstrip(chr(13))]
        assert lines == ["x,y", "1,2", "3,4"]
        assert load_snapshots(rd)
        # the second call: everything is already in the drop state
        rc2, out2, _e2 = _run_cmd(["ingest", str(drop), "--runs-dir", rd])
        assert rc2 == 1
        assert "no new files" in out2

    def test_ingest_skips_a_mismatched_file_with_a_note(self, tmp_path,
                                                        capsys):
        drop = tmp_path / "drop"
        drop.mkdir()
        (drop / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        (drop / "c.csv").write_text("x,z\n7,8\n", encoding="utf-8")
        rc, out, _err = _run_cmd(["ingest", str(drop), "--runs-dir",
                                  str(tmp_path / "runs")])
        assert rc == 0
        assert "c.csv has a different header set - skipped" in out
        _ml = (drop / "merged.csv").read_text(encoding="utf-8")
        _lines = [ln.rstrip(chr(13)) for ln in _ml.split(chr(10))
                  if ln.rstrip(chr(13))]
        assert _lines == ["x,y", "1,2"]


# --- 89.7 the recency window (A79.7) ------------------------------------------------

class TestWindow:
    """SPEC.md 89.7 (A79.7): ``parse_window`` / ``window_rows`` /
    ``fit --window``."""

    def test_parse_window_unit_math_and_bad_forms(self):
        assert parse_window("7d") == 7 * 86400.0
        assert parse_window("24h") == 24 * 3600.0
        assert parse_window("90m") == 90 * 60.0
        assert parse_window("30s") == 30.0
        for bad in ("7", "d7", "7 x", "7w", "-1d", ""):
            with pytest.raises(ValueError):
                parse_window(bad)

    def test_window_rows_keeps_the_recency_slice(self, tmp_path):
        f = _write_dated_csv(tmp_path)
        lines = f.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        rows = [l.split(",") for l in lines[1:]]
        kept, total, lo, hi = window_rows(header, rows, 7 * 86400.0, 0)
        # the window is inclusive (d >= max - window_s): 7 days from the
        # newest day spans 8 calendar days (day 7 .. day 14) = 5 x 8 rows
        assert total == 60 and len(kept) == 40
        assert kept == rows[-40:]
        assert lo is not None and hi is not None
        # a window wider than the data keeps every row (keep-all case)
        kept_all, total_all, lo_all, hi_all = window_rows(
            header, rows, 60 * 86400.0, 0)
        assert len(kept_all) == total_all == 60

    def test_fit_window_end_to_end(self, go_run, capsys):
        dated = _write_dated_csv(go_run["root"], "dated_fit.csv")
        rc, out, err = _run_cmd([
            "fit", "--data", str(dated), "--window", "7d",
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "2", "--target", "50", "--quiet",
        ])
        assert rc == 0, err
        assert "window : kept 40/60 rows" in out

    def test_fit_window_without_a_date_column_is_rc1(self, go_run, capsys):
        rc, _out, err = _run_cmd([
            "fit", "--data", str(go_run["csv"]), "--window", "7d",
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "2", "--target", "50", "--quiet",
        ])
        assert rc == 1
        assert "date column" in err

    def test_window_and_dataset_are_mutually_exclusive(self, go_run,
                                                      capsys):
        rc, _out, err = _run_cmd([
            "fit", "--data", str(go_run["csv"]), "--window", "7d",
            "--dataset", "ds-000000000000", "--runs-dir",
            str(go_run["runs_dir"]),
        ])
        assert rc == 1
        assert "mutually exclusive" in err


# --- 89.8 the quality-over-time chart (A79.8) --------------------------------------

class TestChart:
    """SPEC.md 89.8 (A79.8): ``svg_quality_over_time``."""

    @pytest.fixture()
    def events(self):
        base = int(datetime(2025, 1, 10, tzinfo=timezone.utc).timestamp())
        return [
            {"poll": 1, "epoch": base, "iso": "2025-01-10T00:00:00Z",
             "rows": 10, "champion_score": 90.0, "drift": "stable",
             "ds_id": "ds-1"},
            {"poll": 2, "epoch": base + 3600, "iso": "2025-01-10T01:00:00Z",
             "rows": 12, "champion_score": 70.0, "drift": "alert",
             "ds_id": "ds-2"},
        ]

    def test_one_titled_dot_per_event_and_alert_mark(self, events):
        svg = svg_quality_over_time(events)
        assert svg.startswith("<svg")
        assert svg.count("<circle") == 2
        assert "<title>" in svg
        assert "70.0" in svg  # the alert poll's score
        assert "alert" in svg
        assert "stable" in svg

    def test_empty_log_renders_empty(self):
        assert svg_quality_over_time([]) == ""
        assert svg_quality_over_time(None) == ""
        # a log without a usable score is also empty
        assert svg_quality_over_time([{"poll": 1, "epoch": 1}]) == ""


# --- 89.9 the backtest replay (A79.9) ---------------------------------------------

class TestBacktest:
    """SPEC.md 89.9 (A79.9): the expanding-window replay."""

    def test_cuts_follow_ceil(self):
        import math
        n, k = 60, 3
        cuts = [max(1, min(n, math.ceil(n * i / k))) for i in (1, 2, 3)]
        assert cuts == [20, 40, 60]

    def test_backtest_two_windows(self, go_run, capsys):
        rc, out, err = _run_cmd([
            "report", "--backtest", "--data", str(go_run["csv"]),
            "--runs-dir", str(go_run["runs_dir"]),
            "--windows", "2", "--budget", "2",
        ])
        assert rc == 0, err
        assert "backtest : window 1/2 (rows 1-30)" in out
        assert "backtest : window 2/2 (rows 1-60)" in out
        # the slices landed under <runs-dir>/backtest/
        slices = sorted((go_run["runs_dir"] / "backtest").glob("*.csv"))
        assert len(slices) == 2

    def test_backtest_is_mutually_exclusive(self, go_run, capsys):
        rc, _out, err = _run_cmd([
            "report", "--backtest", "--json", "--data", str(go_run["csv"]),
        ])
        assert rc == 1
        assert "mutually exclusive" in err


# --- 89.10 the freshness block (A79.10) ---------------------------------------------

class TestFreshness:
    """SPEC.md 89.10 (A79.10): ``freshness_view`` / ``render_freshness``
    + the audience-report block."""

    def test_view_and_render_fields(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        summary = {"timestamp": "2025-05-31T00:00:00Z"}
        events = [
            {"poll": 1, "epoch": 1, "iso": "2025-05-31T12:00:00Z",
             "rows": 10, "champion_score": 90.0, "drift": "stable",
             "ds_id": "ds-1"},
            {"poll": 2, "epoch": 2, "iso": "2025-05-31T13:00:00Z",
             "rows": 12, "champion_score": 70.0, "drift": "alert",
             "ds_id": "ds-2"},
        ]
        view = freshness_view(summary, events, now=now)
        assert view["data_as_of"] == "2025-05-31T13:00:00Z"
        assert view["model_age_hours"] == pytest.approx(24.0)
        assert view["rows_seen"] == 22
        assert view["last_drift"] == "alert"
        text = render_freshness(view)
        assert "=== freshness ===" in text
        assert "data as of : 2025-05-31T13:00:00Z" in text
        assert "rows seen  : 22" in text
        assert "last drift : alert" in text

    def test_audience_exec_prints_the_block(self, go_run):
        rc, out, _err = _run_cmd([
            "report", "--run", str(go_run["run_dir"]), "--audience", "exec",
        ])
        assert rc == 0
        assert "=== freshness ===" in out


# --- 89.11 the latency objective (A79.11) ---------------------------------------------

class TestLatency:
    """SPEC.md 89.11 (A79.11): ``--latency-ms`` — the serving-speed gate."""

    def test_latency_line_within_and_over(self):
        ok, line = latency_line(1.0, 10.0)
        assert ok and "within budget" in line
        ok, line = latency_line(11.0, 10.0)
        assert not ok and "OVER budget" in line

    def test_fit_within_budget_is_rc0(self, go_run, capsys):
        rc, out, err = _run_cmd([
            "fit", "--data", str(go_run["csv"]),
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "2", "--target", "50", "--quiet",
            "--latency-ms", "10000",
        ])
        assert rc == 0, err
        assert "latency :" in out and "within budget" in out

    def test_fit_tiny_budget_is_a_miss(self, go_run, capsys):
        rc, out, _err = _run_cmd([
            "fit", "--data", str(go_run["csv"]),
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "2", "--target", "50", "--quiet",
            "--latency-ms", "0.00001",
        ])
        assert rc == 2
        assert "OVER budget" in out


# --- 89.12 the feed portfolio (A79.12) ----------------------------------------------------

class TestFeeds:
    """SPEC.md 89.12 (A79.12): ``go --feeds A.csv,B.csv``."""

    def test_two_feeds_both_pass(self, go_run, capsys):
        a = _write_csv(go_run["root"], "feed_a.csv")
        b = _write_csv(go_run["root"], "feed_b.csv")
        rc, out, err = _run_cmd([
            "go", "--feeds", f"{a},{b}",
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "6", "--target", "50",
        ])
        assert rc == 0, err
        assert "portfolio : 2 feed(s), 2/2 met the bar" in out

    def test_one_miss_is_rc2(self, go_run, capsys):
        a = _write_csv(go_run["root"], "feed_c.csv")
        b = _write_csv(go_run["root"], "feed_d.csv")
        # the miss case: this fixture is trivially separable (the model
        # reaches 100.0), so the bar must sit above it to force a MISS
        rc, out, err = _run_cmd([
            "go", "--feeds", f"{a},{b}",
            "--runs-dir", str(go_run["runs_dir"]),
            "--experiments", "6", "--target", "100.05",
        ])
        assert rc == 2
        assert "portfolio : 2 feed(s), " in out
        assert "met the bar" in out

    def test_feeds_and_data_are_mutually_exclusive(self, go_run, capsys):
        a = _write_csv(go_run["root"], "feed_e.csv")
        b = _write_csv(go_run["root"], "feed_f.csv")
        rc, _out, err = _run_cmd([
            "go", "--feeds", f"{a},{b}", "--data", str(a),
        ])
        assert rc == 1
        assert "mutually exclusive" in err


# --- 89.13 ceremony (A79.13) -----------------------------------------------------------------

class TestCeremony:
    """SPEC.md 89.13.13 (A79.13): the version step + the index row + the
    exports."""

    def test_version_steps_to_075_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.75.0"' in init
        assert 'version = "0.75.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.75.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 89." in spec
        assert "### 89.13 Acceptance (A79)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M78 |"))
        assert "89" in row and "A79" in row
        assert "tests/test_ingest_v075.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
