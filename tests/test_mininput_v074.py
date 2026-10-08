"""v0.74 minimum input, maximum return — SPEC.md 88, A78.

SPEC.md 88 (v0.74): the distance between the user's fewest words and
the run's fullest output closes:

  * 88.1  — ``discover.py``: ``discover_data`` / ``pick_data`` (CSVs
    first, then usable media dirs; deterministic order),
    ``summarize_csv`` / ``label_column``; ``go`` without ``--data``
    finds the data in the current folder (the ``found :`` line), an
    empty folder is rc 1 naming ``--data``;
  * 88.2  — ``goal.extract_data_path`` (drive path or extensioned
    name; bare words rejected); ``resolve_goal`` / ``ask_command``
    carry and embed it; ``ask`` notes a path not found yet (rc 0);
  * 88.3  — ``predict --latest`` (the registry's most recent run) +
    ``--runs-dir``; neither ``--run`` nor ``--latest`` is rc 1;
  * 88.4  — ``goal.auto_target`` (5 above the majority-class ceiling,
    95 floor, 99.5 cap) + ``fair_bar_line``; ``go --auto-target``
    derives the bar from the data's own class structure and gates on
    it; without the flag the 95 default holds;
  * 88.5  — ``goal.measured_per_exp`` / ``experiments_for_time``;
    ``go --time SECONDS`` derives the budget from the measured rate
    (the ``budget :`` line) or falls back with a ``note :`` line;
    explicit ``--experiments`` still wins;
  * 88.6  — ``go`` / ``fit`` ``--continue`` (the most recent same-task
    run becomes the 86 search prior); ``go`` offers the ``earlier :``
    line when the prior exists and the flag is absent; no prior →
    "from scratch" (rc 0);
  * 88.7  — ``go`` leaves the hand-off: report.md / report.txt,
    model_card.txt / model_card.md, report.pdf (reportlab), the
    share bundle ``<run>-share.zip`` + the ``handoff :`` line;
  * 88.8  — ``go --score CSV`` scores a second file with the best
    model (the 42.1 path) + the ``agreement :`` line; a bad file is
    rc 1 (the finished run's artifacts stand);
  * 88.9  — ``briefing.why_this_model`` (the no-diff line / the first
    differing field + the score delta); ``go``'s ``why :`` line;
  * 88.10 — ``go --seeds N`` (the 29.1 sweep over seed+1..N-1) with
    the ``seeds :`` / ``spread :`` lines + the verdict band; the
    default one-line ``note :`` nudge; the gate follows the base run;
  * 88.11 — acceptance: the version steps to ``0.74.0`` in both
    sources, the index row M77 resolves to this file, and the new
    exports land in ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (every fixture
is synthesized here), deterministic (G2), tiny budgets (2–3
experiments, a 50 bar where a PASS exit code is wanted).
"""
from __future__ import annotations

import contextlib
import io
import re
import zipfile
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    auto_target,
    discover_data,
    experiments_for_time,
    extract_data_path,
    fair_bar_line,
    label_column,
    measured_per_exp,
    pick_data,
    resolve_goal,
    summarize_csv,
    why_this_model,
)
from autorefine.cli import main as cli_main
from autorefine.goal import ask_command

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

NEW_EXPORTS = (
    "discover_data", "label_column", "pick_data", "summarize_csv",
    "extract_data_path", "auto_target", "fair_bar_line",
    "measured_per_exp", "experiments_for_time",
    "why_this_model",
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


def _write_imbalanced(tmp_path: Path, name: str = "imbalanced.csv") -> Path:
    """A 90/10 class split: 54 rows of class 0, 6 rows of class 1."""
    lines = ["a,b,churn"]
    for i in range(54):
        lines.append(f"{(i % 5) / 5.0:.2f},0.10,0")
    for i in range(6):
        lines.append(f"0.90,{0.60 + 0.01 * i:.2f},1")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def go_run(tmp_path_factory):
    """One real, tiny, finished ``go`` run (A78.3 / A78.5 / A78.6 /
    A78.7 / A78.9 / A78.10 share it: the registry entry is their
    precondition). Stdout is captured for the verdict-area lines."""
    root = tmp_path_factory.mktemp("mininput")
    csv = _write_csv(root)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main([
            "go", "--data", str(csv),
            "--runs-dir", str(root / "runs"),
            "--experiments", "3", "--target", "50", "--quiet",
        ])
    assert rc == 0, buf.getvalue()
    runs = sorted((root / "runs").glob("csv-*"))
    assert runs and (runs[0] / "summary.json").is_file()
    return {"root": root, "csv": csv, "runs_dir": root / "runs",
            "run_dir": runs[0], "stdout": buf.getvalue()}


# --- 88.1 discovery (A78.1) -----------------------------------------------------

class TestDiscover:
    """SPEC.md 88.1 (A78.1): ``go`` finds the data when it is not named."""

    def test_label_column(self):
        assert label_column(["a", "b", "churn"]) == 2  # last column
        assert label_column(["A", "Label"]) == 1
        assert label_column(["x", "y"]) == 1
        assert label_column(["class", "z"]) == 0
        assert label_column(["target"]) is None  # one column: no label
        assert label_column([]) is None

    def test_summarize_csv_counts_rows_and_label_values(self, tmp_path):
        assert summarize_csv(_write_csv(tmp_path)) == (60, 2)
        hdr = tmp_path / "hdr.csv"
        hdr.write_text("a,b\n", encoding="utf-8")
        assert summarize_csv(hdr) == (0, 0)

    def test_csv_ranks_above_a_media_dir(self, tmp_path):
        csv = _write_csv(tmp_path)
        for cls in ("cat", "dog"):
            (tmp_path / "pics" / cls).mkdir(parents=True)
            (tmp_path / "pics" / cls / "a.png").write_bytes(b"png")
        cands = discover_data(tmp_path)
        assert cands[0] == {"path": str(csv), "task": "csv",
                            "reason": "a CSV table"}
        assert {"path": str(tmp_path / "pics"), "task": "image",
                "reason": "a folder of image items"} in cands
        assert pick_data(tmp_path) == cands[0]  # value-equal (rebuilds the scan)

    def test_empty_dir_is_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_data(empty) == []
        assert pick_data(empty) is None

    def test_go_without_data_in_a_data_folder(self, tmp_path, monkeypatch,
                                              capsys):
        _write_csv(tmp_path)
        monkeypatch.chdir(tmp_path)
        rc = cli_main(["go", "--runs-dir", "runs", "--experiments", "2",
                       "--target", "50", "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "found : data.csv (a CSV table) - using it" in out
        assert list((tmp_path / "runs").glob("csv-*")), "a finished run dir"

    def test_go_without_data_in_an_empty_folder(self, tmp_path, monkeypatch,
                                                capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        assert cli_main(["go", "--runs-dir", "runs"]) == 1
        assert "--data" in capsys.readouterr().err


# --- 88.2 the sentence may carry the path (A78.2) -------------------------------

class TestGoalPath:
    """SPEC.md 88.2 (A78.2): the goal sentence may name the data path."""

    def test_extract_data_path_forms(self):
        assert extract_data_path(
            "reach 96% on C:\\data\\churn.csv, quick") == "C:\\data\\churn.csv"
        assert extract_data_path(
            "reach 96% on C:/data/churn.csv") == "C:/data/churn.csv"
        assert extract_data_path(
            "reach 96% on /data/churn.csv") == "/data/churn.csv"
        assert extract_data_path(
            "reach 96% on churn.csv please") == "churn.csv"
        assert extract_data_path(
            "reach 96% on pics/img.jpg") == "pics/img.jpg"

    def test_bare_words_are_rejected(self):
        assert extract_data_path("reach 96% on my churn table") is None
        assert extract_data_path("make it good please") is None
        assert extract_data_path("   ") is None

    def test_resolve_goal_carries_it_and_ask_embeds_it(self):
        r = resolve_goal("reach 96% on C:\\data\\churn.csv, quick")
        assert r["data"] == "C:\\data\\churn.csv"
        cmd = " ".join(ask_command(r))
        assert "--data C:\\data\\churn.csv" in cmd
        assert "<your data>" not in cmd

    def test_ask_notes_a_path_not_found_yet(self, capsys):
        rc = cli_main(["ask", "reach 96% on C:\\data\\churn.csv, quick"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "--data C:\\data\\churn.csv" in out
        assert "I did not find that path on this machine yet" in out

    def test_ask_no_note_when_the_path_exists(self, tmp_path, capsys):
        p = tmp_path / "real.csv"
        p.write_text("a,b\n1,0\n", encoding="utf-8")
        assert cli_main(["ask", f"reach 96% on {p}"]) == 0
        out = capsys.readouterr().out
        assert str(p) in out
        assert "I did not find that path" not in out


# --- 88.3 predict --latest (A78.3) ----------------------------------------------

class TestPredictLatest:
    """SPEC.md 88.3 (A78.3): the registry's most recent run, by name."""

    def test_predict_latest_scores_new_rows(self, go_run, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["predict", "--latest",
                       "--runs-dir", str(go_run["runs_dir"]),
                       "--csv", str(csv)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "using   : latest run" in out
        assert "0\t" in out and "59\t" in out  # one prediction per row

    def test_neither_flag_is_rc_1(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        assert cli_main(["predict", "--csv", str(csv)]) == 1
        assert "predict needs --run DIR or --latest" \
            in capsys.readouterr().err

    def test_empty_runs_dir_is_rc_1(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["predict", "--latest",
                       "--runs-dir", str(tmp_path / "nope"),
                       "--csv", str(csv)])
        assert rc == 1
        assert "no finished runs" in capsys.readouterr().err


# --- 88.4 the fair bar (A78.4) --------------------------------------------------

class TestAutoTarget:
    """SPEC.md 88.4 (A78.4): the bar from the data's own class structure."""

    def test_formula(self):
        assert auto_target(0.5) == 95.0
        assert auto_target(0.9) == 95.0
        assert auto_target(0.95) == 99.5
        assert auto_target(1.0) == 99.5

    def test_fail_loud_outside_range(self):
        for bad in (0.49, 1.01, True, "0.9"):
            with pytest.raises(ValueError):
                auto_target(bad)

    def test_fair_bar_line(self):
        line = fair_bar_line(0.9, 95.0)
        assert "90.0%" in line
        assert "a fair bar sits 5 points above it" in line
        assert line.endswith("95% (held between the 95 default "
                             "and the 99.5 cap)")

    def test_go_auto_target_derives_and_gates_on_it(self, tmp_path, capsys):
        csv = _write_imbalanced(tmp_path)
        rc = cli_main(["go", "--data", str(csv), "--auto-target",
                       "--target", "99",  # the flag overrides this
                       "--experiments", "2",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc in (0, 2)
        out = capsys.readouterr().out
        assert "target : auto -" in out
        assert "a fair bar sits 5 points above it" in out
        m = re.search(r"it: (\d+(?:\.\d+)?)%", out)
        assert m, out
        # the gate ran on the derived bar, not the --target 99 one
        assert f"vs target {float(m.group(1)):.1f}" in out

    def test_without_the_flag_the_95_default_holds(self, tmp_path, capsys):
        csv = _write_imbalanced(tmp_path)
        rc = cli_main(["go", "--data", str(csv),
                       "--experiments", "2",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc in (0, 2)
        out = capsys.readouterr().out
        assert "target : auto" not in out
        assert "vs target 95.0" in out


# --- 88.5 the budget in minutes (A78.5) -----------------------------------------

class TestTimeBudget:
    """SPEC.md 88.5 (A78.5): ``--time`` from the measured per-exp rate."""

    def test_experiments_for_time_definition(self):
        assert experiments_for_time(10, 2.0) == 5
        assert experiments_for_time(10, 0.05) == 200  # the cap
        assert experiments_for_time(10, 0.001) == 200
        assert experiments_for_time(0.3, 0.05) == 6
        assert experiments_for_time(0, 2.0) == 1
        assert experiments_for_time(-5, 2.0) == 1
        assert experiments_for_time(10, None) == 1
        assert experiments_for_time(10, 0) == 1

    def test_measured_per_exp_definition(self):
        entries = [
            {"task": "csv", "wall_seconds": 10, "experiments_run": 5},
            {"task": "csv", "wall_seconds": 30, "experiments_run": 10},
            {"task": "csv", "wall_seconds": 0, "experiments_run": 4},
            {"task": "image", "wall_seconds": 100, "experiments_run": 2},
        ]
        assert measured_per_exp(entries, "csv") == 2.5  # (2.0 + 3.0) / 2
        assert measured_per_exp(entries, "image") == 50.0
        assert measured_per_exp(entries, "sine-v1") is None
        assert measured_per_exp([], "csv") is None

    def test_go_time_after_a_measured_run(self, go_run, capsys):
        rc = cli_main(["go", "--data", str(go_run["csv"]), "--time", "0.3",
                       "--runs-dir", str(go_run["runs_dir"]), "--quiet"])
        assert rc in (0, 2)
        out = capsys.readouterr().out
        assert "budget :" in out
        assert "s/experiment on csv" in out

    def test_go_time_fallback_without_measurements(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["go", "--data", str(csv), "--time", "10",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc in (0, 2)
        out = capsys.readouterr().out
        assert "note : no measured per-experiment time for csv yet" in out
        assert "--time 10s target was noted" in out
        assert "budget :" not in out

    def test_explicit_experiments_still_wins(self, go_run, capsys):
        rc = cli_main(["go", "--data", str(go_run["csv"]), "--time", "0.3",
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(go_run["runs_dir"]), "--quiet"])
        assert rc == 0
        assert "budget :" not in capsys.readouterr().out


# --- 88.6 the prior run, one flag (A78.6) ----------------------------------------

class TestContinue:
    """SPEC.md 88.6 (A78.6): ``--continue`` = the latest same-task prior."""

    def test_go_continue_after_a_prior_run(self, go_run, capsys):
        rc = cli_main(["go", "--data", str(go_run["csv"]), "--continue",
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(go_run["runs_dir"]), "--quiet"])
        assert rc in (0, 2)
        out = capsys.readouterr().out
        assert "continue : starting from" in out
        assert "its best spec seeds the search prior" in out

    def test_go_continue_without_a_prior_is_scratch(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["go", "--data", str(csv), "--continue",
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "continue : no earlier run of csv in" in out
        assert "starting from scratch" in out

    def test_go_offers_earlier_when_a_prior_exists(self, go_run, capsys):
        rc = cli_main(["go", "--data", str(go_run["csv"]),
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(go_run["runs_dir"]), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "earlier  :" in out
        assert "pass --continue to start from its best spec " \
               "instead of scratch" in out

    def test_fit_continue_after_a_prior_run(self, go_run, capsys):
        rc = cli_main(["fit", "--data", str(go_run["csv"]), "--continue",
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(go_run["runs_dir"]), "--quiet"])
        assert rc in (0, 2)
        assert "continue : starting from" in capsys.readouterr().out


# --- 88.7 the hand-off (A78.7) ----------------------------------------------------

class TestHandoff:
    """SPEC.md 88.7 (A78.7): one command leaves the whole hand-off."""

    def test_go_leaves_the_takeaways(self, go_run):
        run = go_run["run_dir"]
        for name in ("report.md", "report.txt",
                     "model_card.txt", "model_card.md"):
            assert (run / name).is_file(), name
        zip_path = run.parent / f"{run.name}-share.zip"
        assert zip_path.is_file()
        with zipfile.ZipFile(zip_path) as zf:
            assert "summary.json" in zf.namelist()
            assert zf.testzip() is None  # a valid archive
        assert "handoff : everything is in" in go_run["stdout"]
        for name in ("report.md", "report.txt",
                     "model_card.txt", "model_card.md"):
            assert name in go_run["stdout"]


# --- 88.8 the second file, scored (A78.8) -----------------------------------------

class TestScore:
    """SPEC.md 88.8 (A78.8): ``go --score CSV`` + the agreement line."""

    def test_score_a_labelled_file(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["go", "--data", str(csv), "--score", str(csv),
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "scored  : 60 rows with the best model (task csv)" in out
        assert "0\t" in out and "59\t" in out  # one row per line
        assert "agreement :" in out
        assert "rows agree with the label column" in out

    def test_score_a_missing_file_is_rc_1(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["go", "--data", str(csv),
                       "--score", str(tmp_path / "missing.csv"),
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        assert rc == 1
        assert "score:" in capsys.readouterr().err
        # the finished run's artifacts stand
        assert list((tmp_path / "runs").glob("csv-*")), "a finished run dir"


# --- 88.9 the verdict says why (A78.9) --------------------------------------------

class TestWhy:
    """SPEC.md 88.9 (A78.9): ``why_this_model`` + the ``why :`` line."""

    def test_no_diff(self):
        line = "the starter model was already the best - no change beat it"
        assert why_this_model({"a": 1}, {"a": 1}) == line
        assert why_this_model(None, None) == line

    def test_diff_with_the_score_delta(self):
        line = why_this_model({"a": 1}, {"a": 2}, 50.0, 80.0)
        assert line == ("the change that won: a 1 -> 2 "
                        "(final 80.0 vs starter 50.0)")

    def test_diff_without_scores(self):
        assert why_this_model({"a": 1}, {"a": 2}) == \
            "the change that won: a 1 -> 2"

    def test_first_differing_field_in_sorted_order(self):
        line = why_this_model({"z": 1, "a": 1}, {"z": 1, "a": 2})
        assert "a 1 -> 2" in line and "z" not in line

    def test_go_prints_the_why_line(self, go_run):
        assert "why     :" in go_run["stdout"]


# --- 88.10 the headline carries its spread (A78.10) --------------------------------

class TestSeeds:
    """SPEC.md 88.10 (A78.10): ``go --seeds N`` + the one-line nudge."""

    def test_default_one_seed_nudge(self, go_run):
        assert ("note    : one seed was run - pass --seeds 3 for a spread "
                "estimate (how sure to be)") in go_run["stdout"]

    def test_seeds_two_lists_finals_and_the_band(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main(["go", "--data", str(csv), "--seeds", "2",
                       "--experiments", "2", "--target", "50",
                       "--runs-dir", str(tmp_path / "runs"), "--quiet"])
        out = capsys.readouterr().out
        assert "seeds   :" in out and "7 ->" in out and "8 ->" in out
        assert "spread  :" in out and "+/-" in out
        assert "(across 2 seeds)" in out
        assert " - band" in out and "across 2 seeds" in out  # the verdict
        # the gate / exit code follow the base run (88.10.3)
        if "beats the bar" in out:
            assert rc == 0
        else:
            assert rc == 2


# --- 88.11 the ceremony (A78.10) ----------------------------------------------------

class TestCeremony:
    """SPEC.md 88.11 / 88.12 (A78.10): language / version / index."""

    def test_version_steps_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.74.0"' in init
        assert 'version = "0.74.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.74.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 88." in spec
        assert "### 88.11 Acceptance (A78)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M77 |"))
        assert "88" in row and "A78" in row
        assert "tests/test_mininput_v074.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
