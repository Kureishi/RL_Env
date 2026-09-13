"""v0.48 — "B. Reporting for a wider range of audiences" (SPEC.md 62, A52, M51).

Covers the A52 items as implemented — the **audience axis** re-skins the
already-logged run data for four readers, and ``verify`` independently
re-derives the reported numbers. All additive / opt-in (A1–A51 stay green;
the ``technical`` / default report path is byte-identical, 62.1.4):

- 62.1 **axis** — ``AUDIENCES == ("exec","domain","technical","regulator")``;
  ``build_view`` dispatches each and raises ``ValueError`` on an unknown
  name; ``report --run DIR --audience technical`` is byte-identical to
  ``report --run DIR`` (the pin anchor); a non-technical audience is
  mutually exclusive with ``--json`` (rc 1).
- 62.2 **exec** — ``exec_view`` on a synthetic summary exposes
  ``what_we_built`` / ``target_met`` (met + margin, hand-computed) /
  ``cost`` / ``risk`` / ``go_no_go``; the go/no-go rule is hand-computed
  for all five branches (GO / REVIEW-improved / REVIEW-no-target /
  NO-GO-not-improved / NO-GO-no-score).
- 62.3 **domain** — ``domain_view`` with a synthetic ``diag`` yields the
  per-class rows (count / correct / accuracy), the weakest class, the
  ``ece`` (when provided), and the worked examples (capped); a regression
  run (no ``diag``) degrades to graceful ``None``.
- 62.4 **regulator** — ``regulator_view`` carries the chain-of-custody
  fields (tool / version / seed / target / config_hash / environment /
  data_fingerprint / decision_trace); ``data_fingerprint`` of a temp file
  == ``hashlib.sha256(bytes)``, of a temp dir == the stable manifest hash
  (hand-computed), missing path / non-dict → ``None``.
- 62.5 **verify** — ``verify_run`` on a hand-written, consistent run dir
  passes all 7 checks; a mismatched summary (wrong ``final_best_score``)
  fails ≥ 1 check; a bogus ``kind`` row fails ``kinds_valid``; a missing
  artifact returns ``error``.
- **renderers** — ``render_view`` / ``html_view`` are byte-stable (G2) and
  the HTML page is self-contained with the audience name.
- **CLI** — a real tiny run → ``report --audience exec --run DIR`` rc 0 +
  exec phrases; ``--audience technical`` byte-identical to the default;
  ``--audience exec --json`` rc 1; ``--html`` writes
  ``report_exec.html``; ``verify --run DIR`` rc 0 (consistent) / rc 2
  (mismatch) / rc 1 (missing dir).
- **exports (62.6.7)** — the eleven new top-level names are in
  ``__all__`` (33.1).
- **version** — the version steps to ``0.48.0`` in both sources (33.1).

House rules (A52): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; ``dashboard_app`` is never
imported.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    SearchPolicy,
    AUDIENCES,
    build_view,
    data_fingerprint,
    domain_view,
    exec_view,
    html_view,
    regulator_view,
    render_verify,
    render_view,
    technical_view,
    verify_run,
)
from autorefine.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]

# A52: the eleven new top-level exports (62.6.7).
NEW_EXPORTS = (
    "AUDIENCES", "exec_view", "domain_view", "technical_view",
    "regulator_view", "build_view", "render_view", "html_view",
    "data_fingerprint", "verify_run", "render_verify",
)


# --- small synthesized fixtures (no cross-test imports, A52) -----------------

def _syn_summary(**over) -> dict:
    """A minimal, hand-computed summary the views can consume."""
    s = {
        "task": "parity-v1",
        "seed": 7,
        "baseline_score": 80.0,
        "final_best_score": 97.0,
        "target": 95.0,
        "best_spec": {"model_family": "mlp", "architecture": [64, 32],
                      "activation": "tanh"},
        "experiments_run": 3,
        "wall_seconds": 12.5,
        "finished_reason": "target",
    }
    s.update(over)
    return s


def _syn_diag() -> dict:
    """A 28.2 ``holdout_diagnostics`` dict: 2 classes, 90 holdout rows."""
    return {
        "class_labels": ["cat", "dog"],
        "per_class": [95.0, 70.0],          # cat 19/20, dog 49/70
        "confusion": [[19, 1], [21, 49]],
        "class_counts": [20, 70],
    }


def _write_run(run_dir: Path, summary: dict, rows: list) -> Path:
    """A hand-written run dir (the two artifacts ``verify_run`` reads)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary),
                                          encoding="utf-8")
    with (run_dir / "experiments.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return run_dir


# --- 62.1 the audience axis ----------------------------------------------------

def test_audiences_tuple_and_export():
    """A52 (SPEC.md 62.1.2): the four reader names, documented order —
    the parser's ``choices`` and ``build_view``'s dispatcher share this
    tuple (one source)."""
    assert AUDIENCES == ("exec", "domain", "technical", "regulator")


def test_build_view_dispatches_each_audience():
    """A52 (SPEC.md 62.1): ``build_view`` returns the right view for each
    audience; an unknown name raises ``ValueError`` (the CLI's ``choices``
    is the second guard)."""
    s, e = _syn_summary(), []
    assert "what_we_built" in build_view("exec", s, e)
    assert "per_class" in build_view("domain", s, e)
    assert "experiments" in build_view("technical", s, e)
    assert "chain_of_custody" in build_view("regulator", s, e)
    with pytest.raises(ValueError, match="unknown audience"):
        build_view("board", s, e)


def test_technical_view_is_a_structured_mirror():
    """A52 (SPEC.md 62.1): the technical view mirrors the summary's
    headline keys + the per-entry decision log (pure; the CLI default
    does not render through it)."""
    entries = [{"kind": "baseline", "accepted": None,
                "holdout_score": 80.0, "gen_gap": None, "mutation": []},
               {"kind": "experiment", "accepted": True,
                "holdout_score": 97.0, "gen_gap": 0.5,
                "mutation": ["hidden_dim"]}]
    v = technical_view(_syn_summary(), entries)
    assert v["final_best_score"] == 97.0 and v["task"] == "parity-v1"
    assert v["experiments"][1] == {"kind": "experiment", "accepted": True,
                                   "score": 97.0, "gen_gap": 0.5,
                                   "mutation": ["hidden_dim"]}


# --- 62.2 the exec view (hand-computed, A52) ---------------------------------

def test_exec_view_go_and_margin_hand_computed():
    """A52 (SPEC.md 62.2): target met → GO; the margin is best − target
    (97.0 − 95.0 = 2.0), met True; the cost block mirrors the summary."""
    v = exec_view(_syn_summary(), [])
    assert "A parity-v1 model" in v["what_we_built"]
    assert "64 x 32" in v["what_we_built"]           # the spec in words
    assert v["target_met"] == {"target": 95.0, "met": True, "margin": 2.0}
    assert v["cost"] == {"wall_seconds": 12.5, "experiments_run": 3,
                         "finished_reason": "target"}
    assert v["go_no_go"]["verdict"] == "GO"
    assert isinstance(v["risk"], list) and v["risk"]


def test_exec_view_go_no_go_rule_all_branches():
    """A52 (SPEC.md 62.2.2): the fixed, hand-computable rule —
    no target → REVIEW (exploratory); final ≥ target → GO; final < target
    but improved → REVIEW; final < target and not improved → NO-GO;
    no final score → NO-GO."""
    def verdict(**over) -> str:
        return exec_view(_syn_summary(**over), [])["go_no_go"]["verdict"]

    assert verdict() == "GO"                                   # 97 ≥ 95
    assert verdict(final_best_score=90.0) == "REVIEW"          # 90 > 80 baseline
    assert verdict(target=None) == "REVIEW"                    # exploratory
    assert verdict(final_best_score=75.0) == "NO-GO"           # 75 < 80 baseline
    assert verdict(final_best_score=None) == "NO-GO"           # nothing to gate on


def test_exec_view_no_hyperparameters():
    """A52 (SPEC.md 62.2.3): no knob values and no mutation data appear
    in the exec view (what we built is family + shape only)."""
    v = exec_view(_syn_summary(),
                  [{"kind": "experiment", "accepted": True,
                    "holdout_score": 97.0, "mutation": ["hidden_dim"]}])
    text = render_view(v, "exec")
    assert "hidden_dim" not in text
    assert "64" in text  # shape is allowed; raw knob *fields* are not
    assert "mutation" not in text


# --- 62.3 the domain view -------------------------------------------------------

def test_domain_view_per_class_and_weakest():
    """A52 (SPEC.md 62.3.1): one row per class (label / accuracy / count /
    correct), the weakest class = min per-class accuracy (dog at 70.0)."""
    v = domain_view(_syn_summary(), [], diag=_syn_diag())
    assert v["per_class"] == [
        {"class": "cat", "accuracy_pct": 95.0, "count": 20, "correct": 19},
        {"class": "dog", "accuracy_pct": 70.0, "count": 70, "correct": 49},
    ]
    assert v["weakest_class"] == {"class": "dog", "accuracy_pct": 70.0}


def test_domain_view_calibration_and_worked_examples():
    """A52 (SPEC.md 62.3.1): the ``ece`` (when provided) rounds to 4
    decimals; the worked examples are the gallery items, capped at 10."""
    gallery = [{"file": f"img{i}.png", "label": i % 2, "predicted": 1 - i % 2}
               for i in range(15)]
    v = domain_view(_syn_summary(), [], diag=_syn_diag(),
                    extras={"ece": 0.123456, "gallery": gallery})
    assert v["calibration"] == {
        "ece": 0.1235,
        "note": "expected calibration error (0 = perfectly calibrated)"}
    assert len(v["worked_examples"]) == 10
    assert v["worked_examples"][0] == {"item": "img0.png",
                                       "true": 0, "predicted": 1}


def test_domain_view_regression_degrades_gracefully():
    """A52 (SPEC.md 62.3.2): a regression / non-classification run (no
    ``diag``) degrades to graceful ``None`` (rendered as n/a)."""
    v = domain_view(_syn_summary(final_best_score=12.34), [])
    assert v["per_class"] is None
    assert v["weakest_class"] is None
    assert v["calibration"] is None
    assert v["worked_examples"] is None
    text = render_view(v, "domain")
    assert "per_class" in text and "weakest_class" in text


# --- 62.4 the regulator view + data fingerprint --------------------------------

def test_regulator_view_chain_of_custody():
    """A52 (SPEC.md 62.4.1): the auditor's fields — tool / version / seed /
    target / config_hash / environment / data_fingerprint / the
    chain-of-custody headline / the decision trace."""
    prov = {"tool": "autorefine", "version": "0.48.0", "target": 95.0,
            "config_hash": "cf-1", "python": "3.13"}
    data = {"kind": "file", "path": "d.csv", "sha256": "ab", "bytes": 8}
    trace = ["baseline: 80.0", "exp 1: 97.0 (accepted)"]
    v = regulator_view(_syn_summary(), [], provenance=prov,
                       trace=trace, data=data)
    assert v["tool"] == "autorefine" and v["version"] == "0.48.0"
    assert v["seed"] == 7 and v["target"] == 95.0
    assert v["config_hash"] == "cf-1" and v["environment"]["python"] == "3.13"
    assert v["data_fingerprint"] == data
    assert v["chain_of_custody"] == {
        "experiments_run": 3, "final_best_score": 97.0,
        "finished_reason": "target"}
    assert v["decision_trace"] == trace


def test_regulator_view_all_fields_render_when_sources_absent():
    """A52 (SPEC.md 62.4.1): every field renders even when a source is
    absent (a partial run still produces a legible audit block)."""
    v = regulator_view({"task": "csv"}, [])
    assert v["version"] is None and v["config_hash"] is None
    assert v["data_fingerprint"] is None and v["decision_trace"] == []
    assert "config_hash" in render_view(v, "regulator")


def test_data_fingerprint_file_is_content_sha256(tmp_path):
    """A52 (SPEC.md 62.4.2): a single file → its content sha256 (hand-
    computed), plus the size and the bare name."""
    f = tmp_path / "data.csv"
    payload = b"x,y\n1,2\n3,4\n"
    f.write_bytes(payload)
    fp = data_fingerprint({"path": str(f)})
    assert fp == {"kind": "file", "path": "data.csv",
                  "sha256": hashlib.sha256(payload).hexdigest(),
                  "bytes": len(payload)}
    # G2: a re-fingerprint of the same path is byte-identical
    assert fp == data_fingerprint({"path": str(f)})


def test_data_fingerprint_dir_is_stable_manifest_hash(tmp_path):
    """A52 (SPEC.md 62.4.2): a directory → a manifest sha256 over the
    sorted (relpath, size) pairs — hand-computed, no byte reads."""
    d = tmp_path / "corpus"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("hello", encoding="utf-8")
    (d / "sub" / "b.txt").write_text("world", encoding="utf-8")
    fp = data_fingerprint({"path": str(d)})
    manifest = [["a.txt", 5], ["sub/b.txt", 5]]
    expected = hashlib.sha256(
        json.dumps(manifest, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    assert fp == {"kind": "dir", "path": "corpus", "files": 2,
                  "sha256": expected}
    # G2: the manifest depends only on (relpath, size), not on content
    (d / "a.txt").write_text("HELLO", encoding="utf-8")
    assert data_fingerprint({"path": str(d)}) == fp


def test_data_fingerprint_missing_or_non_dict_is_none(tmp_path):
    """A52 (SPEC.md 62.4.2): a missing / unreadable path or a non-dict
    config degrades to ``None`` (pure and bounded)."""
    assert data_fingerprint({"path": str(tmp_path / "nope.csv")}) is None
    assert data_fingerprint({"path": ""}) is None
    assert data_fingerprint(None) is None
    assert data_fingerprint("not-a-dict") is None


# --- 62.5 verify (hand-written run dirs, A52) ----------------------------------

def _verify_rows() -> list:
    return [
        {"kind": "baseline", "holdout_score": 60.0, "train_seconds": 1.0},
        {"kind": "experiment", "accepted": True, "holdout_score": 70.0,
         "mutation": ["hidden_dim"], "train_seconds": 2.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 65.0,
         "mutation": ["lr"], "train_seconds": 1.0},
    ]


def _consistent_summary(**over) -> dict:
    """A summary exactly supported by ``_verify_rows()`` (hand-computed:
    2 experiments; best 70.0; 70/60 improvement; hidden_dim 1/1, lr 1/0)."""
    s = {
        "task": "parity-v1", "seed": 7,
        "experiments_run": 2,
        "baseline_score": 60.0,
        "final_best_score": 70.0,
        "improvement_factor": 70.0 / 60.0,
        "mutation_win_rate": {"hidden_dim": {"trials": 1, "wins": 1},
                              "lr": {"trials": 1, "wins": 0}},
        "wall_seconds": 10.0,
        "finished_reason": "budget",
    }
    s.update(over)
    return s


def test_verify_run_consistent_passes_all_checks(tmp_path):
    """A52 (SPEC.md 62.5.1): a consistent hand-written run dir passes
    every check — all 7 (experiments_run / baseline_score /
    final_best_score / improvement_factor / mutation_win_rate /
    kinds_valid / wall_seconds_consistency)."""
    rd = _write_run(tmp_path / "ok", _consistent_summary(), _verify_rows())
    r = verify_run(rd)
    assert r["error"] is None and r["passed"] is True
    names = [c["name"] for c in r["checks"]]
    assert len(r["checks"]) == 7, names
    assert all(c["ok"] for c in r["checks"]), [c for c in r["checks"]
                                               if not c["ok"]]
    assert names == ["experiments_run", "baseline_score",
                     "final_best_score", "improvement_factor",
                     "mutation_win_rate", "kinds_valid",
                     "wall_seconds_consistency"]


def test_verify_run_mismatched_final_score_fails(tmp_path):
    """A52 (SPEC.md 62.5.1): a summary whose ``final_best_score`` the log
    does not support (99.0, the log says 70.0) fails ≥ 1 check."""
    rd = _write_run(tmp_path / "bad", _consistent_summary(final_best_score=99.0),
                    _verify_rows())
    r = verify_run(rd)
    assert r["error"] is None and r["passed"] is False
    failed = {c["name"]: c for c in r["checks"] if not c["ok"]}
    assert "final_best_score" in failed
    c = failed["final_best_score"]
    assert c["expected"] == 99.0 and c["actual"] == 70.0


def test_verify_run_bogus_kind_fails_kinds_valid(tmp_path):
    """A52 (SPEC.md 62.5.1): a row with a kind outside ``LOG_KINDS``
    fails ``kinds_valid`` (the C4 registry as a behavioral fact)."""
    rows = _verify_rows() + [{"kind": "bogus-kind", "holdout_score": 1.0}]
    rd = _write_run(tmp_path / "kind", _consistent_summary(), rows)
    r = verify_run(rd)
    assert r["error"] is None and r["passed"] is False
    c = next(c for c in r["checks"] if c["name"] == "kinds_valid")
    assert c["ok"] is False and "bogus-kind" in c["actual"]


def test_verify_run_missing_artifacts_error(tmp_path):
    """A52 (SPEC.md 62.5.2): a missing required artifact returns
    ``error`` (the CLI maps this to rc 1)."""
    d = tmp_path / "half"
    d.mkdir()
    (d / "experiments.jsonl").write_text(
        json.dumps({"kind": "baseline", "holdout_score": 60.0}) + "\n",
        encoding="utf-8")
    r = verify_run(d)
    assert r["passed"] is False
    assert r["error"] == "no summary.json in run dir"
    # and the other way around
    d2 = tmp_path / "half2"
    d2.mkdir()
    (d2 / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    r2 = verify_run(d2)
    assert r2["passed"] is False
    assert r2["error"] == "no experiments.jsonl in run dir"


def test_verify_unreported_numbers_are_never_asserted(tmp_path):
    """A52 (SPEC.md 62.5.1): a check is included only when both sides are
    present — a summary without ``improvement_factor`` /
    ``mutation_win_rate`` / ``wall_seconds`` still passes with exactly
    the 4 supported checks + kinds_valid."""
    s = _consistent_summary()
    del s["improvement_factor"], s["mutation_win_rate"], s["wall_seconds"]
    rd = _write_run(tmp_path / "lean", s, _verify_rows())
    r = verify_run(rd)
    assert r["passed"] is True and r["error"] is None
    assert [c["name"] for c in r["checks"]] == [
        "experiments_run", "baseline_score", "final_best_score",
        "kinds_valid"]


def test_render_verify_headline_and_lines():
    """A52 (SPEC.md 62.5.2): a PASS / FAIL headline + one line per check;
    an error short-circuits to an ERROR line."""
    ok = {"passed": False, "checks": [
        {"name": "a", "ok": True, "expected": 1, "actual": 1},
        {"name": "b", "ok": False, "expected": 2, "actual": 3}],
        "error": None}
    out = render_verify(ok, "runs/x")
    assert out.startswith("verify runs/x: FAIL (1/2 checks passed)")
    assert "[FAIL] b: expected 2 got 3" in out
    err = render_verify({"passed": False, "checks": [],
                         "error": "no summary.json in run dir"}, "runs/y")
    assert err == "verify runs/y: ERROR — no summary.json in run dir"


# --- renderers (G2 byte-stability, A52) ---------------------------------------

def test_render_view_and_html_are_byte_stable():
    """A52 (SPEC.md 62.1.4): a re-render of the same view is byte-identical
    (G2) — no timestamps, no RNG; the HTML page is self-contained and
    carries the audience name."""
    v = exec_view(_syn_summary(), [])
    t1, t2 = render_view(v, "exec"), render_view(v, "exec")
    assert t1 == t2 and t1.startswith("=== AutoRefine report (exec view) ===")
    h1, h2 = html_view("exec", v), html_view("exec", v)
    assert h1 == h2
    assert "<!DOCTYPE html>" in h1 and "</html>" in h1
    assert "exec" in h1 and "what_we_built" in h1
    # the em-dash n/a path renders UTF-8 (the provenance_card convention)
    assert "\u2014" in render_view({"k": None}, "regulator")


# --- exports + version (62.6.7 / 33.1, A52) ------------------------------------

def test_new_names_are_exported():
    """A52 (SPEC.md 62.6.7): the eleven new top-level names are in
    ``__all__`` and resolvable (33.1)."""
    for name in NEW_EXPORTS:
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_version_round_v048():
    """A52 (33.1): the version stepped with the round — ``0.48.0`` in
    both sources (pyproject and the package)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.50.0"


# --- CLI (a real tiny run; A52) -------------------------------------------------

@pytest.fixture
def real_run(tmp_path):
    """A tiny, self-consistent live run (parity-v1, 2 experiments, G2)."""
    env = AutoRefineEnv(seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    assert env.memory.load_summary()
    return Path(env.run_dir)


def test_cli_report_exec_view(real_run, capsys):
    """A52 (SPEC.md 62.6): ``report --audience exec`` rc 0 + the exec
    phrases (the reader-facing re-skin)."""
    rc = cli_main(["report", "--audience", "exec", "--run", str(real_run)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("=== AutoRefine report (exec view) ===")
    assert "what_we_built" in out and "go_no_go" in out and "verdict" in out


def test_cli_report_domain_and_regulator(real_run, capsys):
    """A52 (SPEC.md 62.6): the domain and regulator views render (rc 0)
    on a real run — the classification-free fields degrade to n/a."""
    assert cli_main(["report", "--audience", "domain",
                     "--run", str(real_run)]) == 0
    assert "=== AutoRefine report (domain view) ===" in \
        capsys.readouterr().out
    assert cli_main(["report", "--audience", "regulator",
                     "--run", str(real_run)]) == 0
    assert "=== AutoRefine report (regulator view) ===" in \
        capsys.readouterr().out


def test_cli_report_technical_is_byte_identical(real_run, capsys):
    """A52 (SPEC.md 62.1.4, the pin anchor): ``report --run DIR`` and
    ``report --run DIR --audience technical`` are byte-identical (the
    default path is untouched)."""
    rc1 = cli_main(["report", "--run", str(real_run)])
    out1 = capsys.readouterr().out
    rc2 = cli_main(["report", "--audience", "technical",
                    "--run", str(real_run)])
    out2 = capsys.readouterr().out
    assert rc1 == rc2 == 0
    assert out1 == out2


def test_cli_audience_mutually_exclusive_with_json(real_run, capsys):
    """A52 (SPEC.md 62.1.3): a non-technical audience is a human
    re-skin, mutually exclusive with the machine path (rc 1)."""
    rc = cli_main(["report", "--audience", "exec", "--json",
                   "--run", str(real_run)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mutually exclusive" in err and "--json" in err


def test_cli_report_exec_html(real_run, capsys):
    """A52 (SPEC.md 62.1.3): ``--html`` with an audience writes
    ``report_exec.html`` into the run dir (the self-contained page)."""
    rc = cli_main(["report", "--audience", "exec", "--html",
                   "--run", str(real_run)])
    out = capsys.readouterr().out
    assert rc == 0 and "report_exec.html" in out
    page = (real_run / "report_exec.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in page and "</html>" in page
    assert "exec view" in page


def test_cli_verify_real_run_passes(real_run, capsys):
    """A52 (SPEC.md 62.5.2): ``verify --run DIR`` rc 0 on a real,
    self-consistent run (the reported numbers re-derive from the log)."""
    rc = cli_main(["verify", "--run", str(real_run)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith(f"verify {real_run}: PASS")
    assert "checks passed" in out


def test_cli_verify_rc2_mismatch_rc1_missing(tmp_path, capsys):
    """A52 (SPEC.md 62.5.2): rc 2 when ≥ 1 check FAILs (a number the log
    does not support); rc 1 on a missing / invalid run dir."""
    bad = _write_run(tmp_path / "bad",
                     _consistent_summary(final_best_score=99.0),
                     _verify_rows())
    assert cli_main(["verify", "--run", str(bad)]) == 2
    assert "FAIL" in capsys.readouterr().out
    assert cli_main(["verify", "--run", str(tmp_path / "no-such-dir")]) == 1
    assert "no such run dir" in capsys.readouterr().err
