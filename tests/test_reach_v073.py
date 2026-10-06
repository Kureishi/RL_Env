"""Broader reach — the environment meets a broader demographic (SPEC.md 87, A77).

SPEC.md 87 (v0.73): a goal in the user's own words becomes a real recipe,
the loop is narrated in plain beats, one command takes data to a plain
verdict, and the results travel in the reader's format. Thin surfaces
over existing machinery (62/63.1/63.2/64.2/66.1), pure and deterministic
(G2), no new dependency, defaults unchanged:

  * 87.1  — ``goal.py``: ``resolve_goal`` (target / task hint / budget
    hint; fail-loud ``ValueError`` on empty or out-of-range),
    ``ask_command`` (defaults omitted), ``go_budget`` (the step
    function); ``autorefine ask`` prints heard / plan / command;
  * 87.2  — ``run --tour``: the 41.3 tiny loop narrated in three plain
    beats + the 87.6 verdict — ASCII-only, jargon-free, rc 0;
  * 87.3  — ``autorefine go --data PATH``: ``fit`` with a data-sized
    default budget + the one-sentence plain verdict; ``fit``'s exit
    codes (0 PASS / 2 MISS / 1 setup error);
  * 87.4  — ``vocab.py``: the one ordered jargon table, ``plain``
    (deterministic, longest-first, idempotent, case-preserving),
    ``plain_free``; the app's "Plain language" toggle (default off);
  * 87.5  — ``briefing.py``: ``so_what`` (headline / met / margin /
    confidence / where-it-fails / next-step; ``None`` without a final
    score);
  * 87.6  — ``confidence_tier`` (robust → high, marginal → medium,
    unassessable → low, ``None`` → ``None``) + ``when_to_distrust``;
  * 87.7  — the app's "Share as…" select (default "(none)") over the 62
    audience views + the 63.1 md / txt / pdf downloads;
  * 87.8  — ``modelcard.py``: ``model_card`` on the 63.2 guide + the
    87.6 chip; text / Markdown renderers byte-identical on re-render;
    ``autorefine card --run DIR`` (rc 1 on a broken dir);
  * 87.9  — the app's "Hand off" grouping (recipe sentence + model-card
    downloads + the existing share bundle / run_config keys);
  * 87.10 — the "View data" companion expander (the table equivalent of
    the hover tooltips);
  * 87.11 — acceptance: the version steps to ``0.73.0`` in both
    sources, the index row M76 resolves to this file, and the new
    exports land in ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (every fixture
is synthesized here), deterministic (G2).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    PLAIN,
    confidence_tier,
    go_budget,
    model_card,
    plain,
    plain_free,
    render_model_card,
    render_model_card_md,
    resolve_goal,
    so_what,
    when_to_distrust,
)
from autorefine.briefing import jargon_free_check, plain_verdict
from autorefine.cli import main as cli_main
from autorefine.goal import ask_command

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

NEW_EXPORTS = (
    "resolve_goal", "ask_command", "go_budget",
    "PLAIN", "plain", "plain_free",
    "so_what", "confidence_tier", "when_to_distrust", "plain_verdict",
    "model_card", "render_model_card", "render_model_card_md",
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


def _csv_run(tmp_path: Path) -> Path:
    """A real, tiny, finished csv run — the card / app source."""
    rc = cli_main([
        "go", "--data", str(_write_csv(tmp_path)),
        "--runs-dir", str(tmp_path / "runs"),
        "--experiments", "2", "--target", "50", "--quiet",
    ])
    assert rc == 0
    runs = sorted((tmp_path / "runs").glob("csv-*"))
    assert runs and (runs[0] / "summary.json").is_file()
    return runs[0]


# --- 87.1 the goal resolver (A77.1) -------------------------------------------

class TestGoalResolver:
    """SPEC.md 87.1 (A77.1): a goal in the user's words -> a recipe."""

    def test_target_reads(self):
        assert resolve_goal("reach 96% on this table")["target"] == 96.0
        assert resolve_goal("at least 90 percent accuracy")["target"] == 90.0

    def test_task_hints(self):
        assert resolve_goal("classify this churn table")["task"] == "csv"
        assert resolve_goal("label these pictures")["task"] == "image"
        assert resolve_goal("sort these speech clips")["task"] == "audio"
        assert resolve_goal("score these reviews")["task"] == "text"

    def test_budget_hints(self):
        assert resolve_goal("quick pass, 90%")["experiments"] == 8
        assert resolve_goal("be thorough, 90%")["experiments"] == 60

    def test_defaults_and_data(self):
        r = resolve_goal("make it good, 90%", data_path="t.csv")
        assert r["data"] == "t.csv"
        assert r["task"] == "auto"
        assert r["experiments"] == 30
        assert r["unrecognized"] == []

    def test_empty_and_out_of_range_fail_loud(self):
        with pytest.raises(ValueError):
            resolve_goal("   ")
        with pytest.raises(ValueError):
            resolve_goal("reach 150% accuracy")
        with pytest.raises(ValueError):
            resolve_goal("reach 0% accuracy")

    def test_no_trigger_reports_unrecognized(self):
        r = resolve_goal("make it good please")
        assert r["unrecognized"] == ["make it good please"]
        assert r["target"] == 95.0  # the fit default

    def test_plain_summary_is_deterministic_and_plain(self):
        a = resolve_goal("reach 96% on this churn table, quick")
        b = resolve_goal("reach 96% on this churn table, quick")
        assert a == b
        assert plain_free(a["plain_summary"])

    def test_ask_command_omits_defaults(self):
        cmd = " ".join(ask_command(resolve_goal("make it 90%")))
        assert cmd == ("autorefine fit --data <your data> --target 90")
        cmd2 = " ".join(ask_command(
            resolve_goal("classify churn rows, quick, 90%")))
        assert "--task csv" in cmd2 and "--experiments 8" in cmd2


class TestAskCli:
    """SPEC.md 87.1.4 (A77.1): ``autorefine ask`` prints the plan."""

    def test_ask_prints_heard_plan_command(self, capsys):
        rc = cli_main(["ask", "reach 96% on this churn table, quick"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "heard" in out and "plan" in out and "command" in out
        assert "--task csv" in out and "--target 96" in out
        assert "--experiments 8" in out

    def test_ask_empty_goal_is_rc_1(self, capsys):
        assert cli_main(["ask", "   "]) == 1
        assert "goal" in capsys.readouterr().err.lower()


# --- 87.2 the tour (A77.2) ------------------------------------------------------

class TestTour:
    """SPEC.md 87.2 (A77.2): the 60-second guided tour."""

    def test_tour_runs_the_tiny_loop_plain_and_ascii(self, tmp_path, capsys):
        rc = cli_main(["run", "--tour", "--runs-dir", str(tmp_path / "tour")])
        assert rc == 0
        out = capsys.readouterr().out
        out.encode("ascii")  # 87.2: ASCII-only (the 48.5 console rule)
        assert "1. Start simple" in out
        assert "2. Try better" in out
        assert "3. Pick the winner" in out
        assert "verdict : We finished at" in out
        assert plain_free(out)
        runs = list((tmp_path / "tour").glob("parity-v1-*"))
        assert runs and (runs[0] / "summary.json").is_file()

    def test_tour_is_deterministic_per_seed(self, tmp_path, capsys):
        def one(tag: str) -> str:
            rc = cli_main(["run", "--tour", "--seed", "3",
                           "--runs-dir", str(tmp_path / tag)])
            assert rc == 0
            lines = [l for l in capsys.readouterr().out.splitlines()
                     if not l.startswith("run dir")]
            return "\n".join(lines)
        assert one("t1") == one("t2")


# --- 87.3 go (A77.3) -------------------------------------------------------------

class TestGo:
    """SPEC.md 87.3 (A77.3): one command from data to verdict."""

    def test_go_budget_step_function(self):
        assert go_budget(None) == 30
        assert go_budget(50) == 15
        assert go_budget(99) == 15
        assert go_budget(100) == 30
        assert go_budget(999) == 30
        assert go_budget(1000) == 40
        assert go_budget(50000) == 40

    def test_go_data_to_verdict(self, tmp_path, capsys):
        csv = _write_csv(tmp_path)
        rc = cli_main([
            "go", "--data", str(csv), "--runs-dir", str(tmp_path / "runs"),
            "--experiments", "2", "--target", "50", "--quiet",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "verdict : You asked for 50." in out
        assert "beats the bar" in out

    def test_go_bad_path_and_missing_data_are_rc_1(self, tmp_path, capsys):
        assert cli_main([
            "go", "--data", str(tmp_path / "missing.csv"),
            "--runs-dir", str(tmp_path / "r")]) == 1
        err = capsys.readouterr().err
        assert "no such file" in err.lower()
        assert cli_main(["go"]) == 1
        assert "--data" in capsys.readouterr().err


# --- 87.4 the vocabulary (A77.4) ---------------------------------------------------

class TestVocab:
    """SPEC.md 87.4 (A77.4): the one ordered jargon table."""

    def test_plain_is_deterministic_and_idempotent(self):
        for t in ("The baseline candidate was rejected by the gate.",
                  "2 epochs, 3 features, 1 mutation",
                  "overfitting beat the holdout signal"):
            assert plain(t) == plain(t)
            assert plain(plain(t)) == plain(t)

    def test_plain_is_longest_first_and_case_preserving(self):
        assert plain("tune the hyperparameters") == "tune the knobs"
        assert plain("tune the hyperparameter") == "tune the knob"
        assert plain("Baseline first") == "Starter model first"

    def test_plain_passes_through_non_strings(self):
        assert plain(42) == 42
        assert plain("") == ""

    def test_plain_free_flags_any_key(self):
        assert not plain_free("the gate said no")
        assert plain_free("the starter model said no")
        assert not plain_free(42)

    def test_table_has_no_spec_or_version_tokens(self):
        for k, v in PLAIN.items():
            assert "SPEC" not in k and "SPEC" not in v
            assert "v0." not in k and "v0." not in v

    def test_replacements_carry_no_keys(self):
        for k, v in PLAIN.items():
            assert plain_free(v), f"replacement {v!r} carries a jargon key"


# --- 87.5 / 87.6 the briefing + chip (A77.5) --------------------------------------

class TestBriefing:
    """SPEC.md 87.5/87.6 (A77.5): the "So what?" card + the trust chip."""

    def test_so_what_forms(self):
        hit = so_what({"final_best_score": 96.0, "target": 95.0})
        assert hit["met"] is True and hit["margin"] == 1.0
        assert "We hit your bar" in hit["headline"]
        miss = so_what({"final_best_score": 90.0, "target": 95.0})
        assert miss["met"] is False and abs(miss["margin"]) == 5.0
        assert "We missed your bar by 5" in miss["headline"]
        nobar = so_what({"final_best_score": 71.25})
        assert nobar["met"] is None and "no bar was set" in nobar["headline"]

    def test_so_what_none_without_a_final_score(self):
        assert so_what({}) is None
        assert so_what({"final_best_score": None}) is None

    def test_where_it_fails_reads_the_diag(self):
        diag = {"per_class": [99.0, 54.0], "class_labels": ["yes", "no"]}
        sw = so_what({"final_best_score": 90.0}, diag=diag)
        assert "weakest: class no" in sw["where_it_fails"]
        assert "54% correct" in sw["where_it_fails"]
        assert "no per-class data was logged" in \
            so_what({"final_best_score": 90.0})["where_it_fails"]

    def test_confidence_tier_mapping(self):
        # the band is best ± (max−min)/2 (the 66.1 read): [95,97] stays
        # above the 90 bar (robust -> high); [83,97] dips below (marginal)
        assert confidence_tier(90, 96, [95.0, 97.0]) == "high"
        assert confidence_tier(90, 96, [83.0, 97.0]) == "medium"
        assert confidence_tier(90, 96, None) == "low"
        assert confidence_tier(90, None, None) is None

    def test_when_to_distrust_lines(self):
        assert "holds across the seed band" in when_to_distrust("high")
        assert "a different random seed could flip" in when_to_distrust("medium")
        assert "one seed was run" in when_to_distrust("low")
        assert "no final score" in when_to_distrust(None)

    def test_plain_verdict_sentences(self):
        beat = plain_verdict({"final_best_score": 96.2}, 95.0)
        assert "You asked for 95" in beat and "beats the bar" in beat
        short = plain_verdict({"final_best_score": 90.0}, 95.0)
        assert "just short of the bar" in short
        nobar = plain_verdict({"final_best_score": 71.25})
        assert "no bar was set" in nobar
        assert "without a final score" in plain_verdict({})

    def test_jargon_free_templates(self):
        assert jargon_free_check() == []


# --- 87.8 the model card (A77.7) ---------------------------------------------------

class TestModelCard:
    """SPEC.md 87.8 (A77.7): the one-pager for the person who uses it."""

    def test_card_carry_examples_and_chip_and_verdict(self, tmp_path):
        run = _csv_run(tmp_path)
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        view = model_card(summary, None, n_examples=3)
        assert len(view["examples"] or []) <= 3
        assert view["trust"] is not None
        assert "We hit your bar" in view["verdict"]
        assert view["how_to_read"]

    def test_renders_are_byte_identical_on_re_render(self, tmp_path):
        run = _csv_run(tmp_path)
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        entries = [json.loads(l) for l in
                   (run / "experiments.jsonl").read_text(
                       encoding="utf-8").splitlines() if l.strip()]
        v1, v2 = model_card(summary, entries), model_card(summary, entries)
        assert render_model_card(v1) == render_model_card(v2)
        assert render_model_card_md(v1) == render_model_card_md(v2)

    def test_card_cli_and_broken_dir(self, tmp_path, capsys):
        run = _csv_run(tmp_path)  # go's output precedes; the card prints last
        assert cli_main(["card", "--run", str(run)]) == 0
        assert "=== Model card - what this model does ===" \
            in capsys.readouterr().out
        assert cli_main(["card", "--run", str(run), "--md"]) == 0
        assert "# Model card - what this model does" \
            in capsys.readouterr().out
        assert cli_main(["card", "--run", str(tmp_path / "missing")]) == 1
        assert "no summary.json" in capsys.readouterr().err


# --- app wiring (A77.4 / A77.6 / A77.7 / A77.8 / A77.9) ---------------------------

class TestAppWiring:
    """SPEC.md 87.4.3 / 87.7 / 87.8.3 / 87.9 / 87.10 (A77): the reader
    surfaces in the app — the default render keeps every existing key,
    the new widgets are additive (the screening precedent)."""

    def test_reader_surfaces(self, tmp_path):
        pytest.importorskip(
            "streamlit", reason="dashboard app is optional (SPEC.md 23)")
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP), default_timeout=300)
        at.run()
        assert not at.exception

        # A77.1 / A77.4: the goal box + the plain-language toggle exist
        # with their defaults (empty goal, toggle off)
        assert at.text_input(key="goal_text").value == ""
        assert at.checkbox(key="plain_language").value is False

        # A77.1: a goal in the sidebar resolves to the plain plan + command
        at.text_input(key="goal_text").set_value(
            "reach 96% on this churn table, quick")
        at.run()
        assert not at.exception
        assert any("You want a model" in str(c.value) for c in at.caption)
        assert any("--task csv --target 96 --experiments 8" in c.value
                   for c in at.code)

        # run the tiny loop (the test_dashboard pattern)
        at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
        at.session_state["runs_dir"] = str(tmp_path / "runs")
        at.session_state["experiments"] = 2
        at.session_state["max_train"] = 5.0
        at.run()
        assert not at.exception
        at.button(key="run_button").set_value(True).run()
        assert not at.exception
        verdicts = [e.value for e in at.success] + [e.value for e in at.error]
        assert any(("PASS" in v) or ("MISS" in v) for v in verdicts)

        # A77.5: the "So what?" card at the top of the result
        md = " ".join(m.value for m in at.markdown)
        assert "So what?" in md

        # A77.4: the plain-language toggle pipes the narration
        at.checkbox(key="plain_language").set_value(True).run()
        assert not at.exception
        md_on = " ".join(m.value for m in at.markdown)
        assert "starter model" in md_on

        # A77.6: "Share as…" defaults to "(none)"; a selection renders
        sel = at.selectbox(key="share_as")
        assert sel.value == "(none)"
        sel.set_value("exec").run()
        assert not at.exception
        assert any("exec view" in c.value for c in at.code)

        # A77.6 / A77.7 / A77.8: the hand-off downloads render. This
        # Streamlit build's AppTest exposes neither `key` nor `data` on
        # download_button elements (`value` is the pressed state) —
        # identify by unique label, as the 36.1 export tests do.
        labels = {e.label for e in at.get("download_button")}
        for want in ("Download report (Markdown)",
                     "Download report (text)",
                     "Download report (PDF)",
                     "Download model card",
                     "Download model card (Markdown)",
                     "Download run_config.json"):
            assert want in labels, want
        assert at.button(key="build_share").value is False
        # the hand-off recipe sentence (87.9.1)
        assert any("In plain words" in str(c.value) for c in at.caption)

        # A77.9: the "View data" companion expander + its table
        assert "View data" in [e.label for e in at.expander]
        assert any(len(df.value) >= 1 for df in at.dataframe)


# --- 87.11 the ceremony (A77.10) ----------------------------------------------------

class TestCeremony:
    """SPEC.md 87.11 / 87.12 (A77.10): language / version / index."""

    def test_version_steps_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.73.0"' in init
        assert 'version = "0.73.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.73.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 87." in spec
        assert "### 87.11 Acceptance (A77)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M76 |"))
        assert "87" in row and "A77" in row
        assert "tests/test_reach_v073.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
