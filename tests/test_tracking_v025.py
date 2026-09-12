"""v0.25 tracking II — SPEC.md 39, A29:

- 39.1 (T3) watch mode — `read_new_entries` tails by byte offset (partial
  trailing line deferred, missing file `([], 0)`, idempotent re-read);
  `run_finished` keys off `summary.json`; `live_frame` / `tail_line`
  render deterministically; `watch --run DIR` on a pre-finished run dir
  renders the frame and exits 0 (bounded by `--max-polls`); `--tail`
  emits exactly one compact JSON line per logged row; a missing run dir
  with exhausted `--max-polls` → rc 1 (39.1.4).
- 39.2 (T4) decision accounting — `account_run` buckets a known log by
  the documented gate priority (score → overfit → ci; legacy ⇒
  all-score), re-pins the running best on a curriculum step-up, reports
  time-to-first-improvement (index + `ts` delta; "never improved" when
  no accepted candidate), and the wall-time breakdown (baseline +
  candidates + residual = `summary["wall_seconds"]`); free-duplicate is
  0-by-construction with the note; `report --run DIR` shows the block
  and still passes the A11 human invariants, while `--json` stays pure
  and equals `summary.json` (A11).
- Regression — A1–A28 stay green (no run behavior / summary key set /
  registry / artifact change); version stepped to `0.39.0` in both
  sources (33.1, advanced again with v0.32, SPEC.md 45).

House rules: no cross-test imports (the `_make_run` fixture is
duplicated from test_infra_dx.py); stdlib + numpy only.
"""
import json
import tomllib
from pathlib import Path

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.accounting import account_run
from autorefine.cli import main as cli_main
from autorefine.watch import live_frame, read_new_entries, run_finished, tail_line

REPO = Path(__file__).resolve().parents[1]

# --- a known log: every bucket of the 39.2.2 priority (score → overfit → ci)

_LOG = [
    # baseline seeds the running best (50.0); ts 0.0 anchors the TTI clock
    {"kind": "baseline", "accepted": True, "holdout_score": 50.0, "gen_gap": 0.0,
     "train_seconds": 1.0, "ts": 0.0},
    # rejected: 45 ≤ 50 → score
    {"kind": "experiment", "accepted": False, "holdout_score": 45.0, "gen_gap": 0.0,
     "train_seconds": 0.5, "ts": 0.5},
    # rejected: beats 50, gen_gap 4.8 > 0.05·60 = 3.0 → overfit
    {"kind": "experiment", "accepted": False, "holdout_score": 60.0, "gen_gap": 4.8,
     "train_seconds": 0.7, "ts": 1.2},
    # rejected: beats 50, small gen_gap → ci (z·SE / efficiency margin)
    {"kind": "experiment", "accepted": False, "holdout_score": 62.0, "gen_gap": 0.01,
     "train_seconds": 0.6, "ts": 1.8},
    # accepted: raises the running best to 70.0; first improvement (index 4)
    {"kind": "experiment", "accepted": True, "holdout_score": 70.0, "gen_gap": 0.0,
     "train_seconds": 2.0, "ts": 3.5},
    # rejected: 65 ≤ 70 → score (accepted candidates raise the bar)
    {"kind": "experiment", "accepted": False, "holdout_score": 65.0, "gen_gap": 0.01,
     "train_seconds": 0.4, "ts": 4.0},
    # curriculum step-up re-pins the running best to 40.0 (39.2.2)
    {"kind": "curriculum", "difficulty": "parity-4-p0.10",
     "new_baseline_score": 40.0, "ts": 5.0},
    # rejected: 45 > 40 (re-pinned), small gen_gap → ci, not score
    {"kind": "experiment", "accepted": False, "holdout_score": 45.0, "gen_gap": 0.0,
     "train_seconds": 0.3, "ts": 5.3},
    # accepted after the re-pin (accepted count = 2 candidates)
    {"kind": "experiment", "accepted": True, "holdout_score": 42.0, "gen_gap": 0.0,
     "train_seconds": 1.5, "ts": 6.5},
]


# --- 39.1 pure helpers --------------------------------------------------------

def test_read_new_entries_missing_file(tmp_path):
    """A29 (SPEC.md 39.1.2): a missing file returns `([], 0)`."""
    entries, offset = read_new_entries(tmp_path / "no" / "experiments.jsonl")
    assert (entries, offset) == ([], 0)


def test_read_new_entries_tails_by_offset(tmp_path):
    """A29 (SPEC.md 39.1.2): byte-offset tailing; idempotent re-read."""
    p = tmp_path / "experiments.jsonl"
    # write bytes directly: exact `\n` terminators (no platform conversion)
    l1 = (json.dumps({"kind": "baseline", "holdout_score": 50.0}) + "\n").encode("utf-8")
    l2 = (json.dumps({"kind": "experiment", "accepted": True}) + "\n").encode("utf-8")
    p.write_bytes(l1)
    e1, off1 = read_new_entries(p, 0)
    assert len(e1) == 1 and e1[0]["kind"] == "baseline"
    assert off1 == len(l1)
    p.write_bytes(l1 + l2)
    e2, off2 = read_new_entries(p, off1)
    assert len(e2) == 1 and e2[0]["accepted"] is True  # only the new row
    assert off2 == len(l1 + l2)
    e3, off3 = read_new_entries(p, off2)  # idempotent
    assert e3 == [] and off3 == off2


def test_read_new_entries_partial_trailing_line_deferred(tmp_path):
    """A29 (SPEC.md 39.1.2): a non-`\\n`-terminated trailing line is left
    for the next poll; completing it makes it readable."""
    p = tmp_path / "experiments.jsonl"
    complete = (json.dumps({"kind": "baseline", "holdout_score": 50.0}) + "\n").encode("utf-8")
    p.write_bytes(complete + b'{"kind": "experi')  # partial trailing line
    e1, off1 = read_new_entries(p, 0)
    assert len(e1) == 1 and e1[0]["kind"] == "baseline"
    assert off1 == len(complete)  # offset unchanged by the partial
    p.write_bytes(complete + b'{"kind": "experiment", "accepted": false}\n')
    e2, _off2 = read_new_entries(p, off1)
    assert len(e2) == 1 and e2[0]["kind"] == "experiment"


def test_run_finished_keys_off_summary_json(tmp_path):
    """A29 (SPEC.md 39.1.4): terminal state = `summary.json` exists."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "experiments.jsonl").write_text("{}\n", encoding="utf-8")
    assert run_finished(d) is False
    (d / "summary.json").write_text("{}", encoding="utf-8")
    assert run_finished(d) is True


def test_live_frame_deterministic_and_content():
    """A29 (SPEC.md 39.1.2): the human frame — deterministic, live header,
    the existing ASCII curve + Pareto (SPEC 21.2)."""
    f = live_frame(_LOG)
    assert f == live_frame(_LOG)  # deterministic
    assert f.startswith("watching:")
    assert "experiments run: 7" in f  # 7 kind=experiment rows
    assert "score vs experiment" in f  # ascii_score_curve
    assert "pareto frontier (score vs training seconds):" in f
    # empty log: still a valid, deterministic frame
    e = live_frame([])
    assert e == live_frame([]) and "(no score yet)" in e


def test_tail_line_is_the_compact_projection():
    """A29 (SPEC.md 39.1.2): one stable machine line — exactly the six
    projected keys, nothing more (no spec / loss_history bloat)."""
    row = {"kind": "experiment", "accepted": True, "holdout_score": 70.0,
           "gen_gap": 0.0, "train_seconds": 2.0, "mutation": ["learning_rate"]}
    obj = json.loads(tail_line(row, 3))
    assert set(obj) == {"index", "kind", "accepted", "holdout_score",
                        "gen_gap", "train_seconds"}
    assert obj["index"] == 3 and obj["kind"] == "experiment"
    assert "mutation" not in obj  # the projection drops the wide fields


# --- 39.2 accounting ----------------------------------------------------------

def test_account_run_rejection_buckets_by_documented_priority():
    """A29 (SPEC.md 39.2.2): score → overfit → ci, with the running best
    raised by accepted candidates and re-pinned by the curriculum row."""
    r = account_run({"wall_seconds": 10.0}, _LOG)["rejections"]
    # the scored population is kind ∈ {baseline, experiment} (39.2.1):
    # the accepted baseline + the two accepted candidates
    assert r["accepted"] == 3
    assert r["rejected"] == 5
    assert (r["score"], r["overfit"], r["ci"]) == (2, 1, 2)
    # free-duplicate rejections are unspent + unlogged (39.2.1)
    assert r["duplicate"] == 0
    assert "not logged" in r["note"]


def test_account_run_legacy_mode_is_all_score():
    """A29 (SPEC.md 39.2.2): in legacy mode (z_accept=0, penalties 0)
    every rejection is a score rejection; the other buckets stay 0."""
    entries = [
        {"kind": "baseline", "accepted": True, "holdout_score": 50.0, "gen_gap": 0.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 49.0, "gen_gap": 0.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 50.0, "gen_gap": 0.0},
        {"kind": "experiment", "accepted": True, "holdout_score": 55.0, "gen_gap": 0.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 54.0, "gen_gap": 0.0},
    ]
    r = account_run({}, entries)["rejections"]
    assert (r["score"], r["overfit"], r["ci"]) == (3, 0, 0)
    assert r["accepted"] == 2  # baseline + the 55.0 candidate (39.2.1)


def test_account_run_time_to_first_improvement():
    """A29 (SPEC.md 39.2.1): the first accepted candidate's index and the
    wall seconds from the baseline's `ts`."""
    tti = account_run({"wall_seconds": 10.0}, _LOG)["time_to_first_improvement"]
    assert tti["improved"] is True
    assert tti["experiment_index"] == 4  # 4th kind=experiment row
    assert tti["seconds"] == 3.5  # 3.5 (that row's ts) − 0.0 (baseline ts)


def test_account_run_never_improved():
    """A29 (SPEC.md 39.2.1): no accepted candidate → "never improved"."""
    entries = [
        {"kind": "baseline", "accepted": True, "holdout_score": 50.0, "ts": 0.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 49.0, "ts": 1.0},
        {"kind": "experiment", "accepted": False, "holdout_score": 48.0, "ts": 2.0},
    ]
    tti = account_run({}, entries)["time_to_first_improvement"]
    assert tti == {"improved": False, "experiment_index": None, "seconds": None}


def test_account_run_wall_time_breakdown():
    """A29 (SPEC.md 39.2.1): baseline + candidates + the honest residual =
    `summary["wall_seconds"]`."""
    wt = account_run({"wall_seconds": 10.0}, _LOG)["wall_time"]
    assert wt["baseline"] == 1.0
    assert wt["candidates"] == 6.0  # 0.5+0.7+0.6+2.0+0.4+0.3+1.5
    assert wt["total"] == 10.0
    assert wt["eval_overhead"] == 3.0  # the residual (eval + CI + policy + ...)
    assert wt["baseline"] + wt["candidates"] + wt["eval_overhead"] == wt["total"]


def test_account_run_is_pure_and_deterministic():
    """A29 (SPEC.md 39.2.4): a function only of the two artifacts."""
    s = {"wall_seconds": 10.0}
    a = account_run(s, _LOG)
    b = account_run(s, _LOG)
    assert a == b
    assert set(a) == {"rejections", "time_to_first_improvement", "wall_time"}
    # robust to an empty log / missing fields (old runs)
    c = account_run({}, [])
    assert c["rejections"]["rejected"] == 0
    assert c["time_to_first_improvement"]["improved"] is False
    assert c["wall_time"]["total"] == 0.0


# --- CLI: watch (39.1) --------------------------------------------------------

def _make_run(tmp_path):
    """A finished, on-disk run (duplicated from test_infra_dx.py — no
    cross-test imports)."""
    env = AutoRefineEnv(
        task="sine-v1", seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path / "runs"
    )
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def test_watch_human_mode_on_finished_run(tmp_path, capsys):
    """A29 (SPEC.md 39.1.3/39.1.4): pre-finished run dir → final frame +
    "(run finished)", rc 0 (bounded by `--max-polls`)."""
    run_dir = _make_run(tmp_path)
    rc = cli_main(["watch", "--run", str(run_dir),
                   "--interval", "0", "--max-polls", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "watching:" in out
    assert "score vs experiment" in out  # the live frame carries the curve
    assert "(run finished)" in out


def test_watch_tail_mode_emits_one_line_per_row(tmp_path, capsys):
    """A29 (SPEC.md 39.1.3): machine mode — exactly one `tail_line` per
    logged row, in order, each valid JSON with the six keys."""
    run_dir = _make_run(tmp_path)
    rc = cli_main(["watch", "--run", str(run_dir), "--tail",
                   "--interval", "0", "--max-polls", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    n_rows = sum(1 for ln in (run_dir / "experiments.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip())
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == n_rows
    for i, ln in enumerate(lines):
        obj = json.loads(ln)
        assert set(obj) == {"index", "kind", "accepted", "holdout_score",
                            "gen_gap", "train_seconds"}
        assert obj["index"] == i


def test_watch_missing_run_dir_bounded_wait_fails_fast(tmp_path, capsys):
    """A29 (SPEC.md 39.1.4): exhausted `--max-polls` before the run
    appears → rc 1 (a mis-pointed path fails fast, no hang)."""
    rc = cli_main(["watch", "--run", str(tmp_path / "no-such-run"),
                   "--interval", "0", "--max-polls", "2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "run dir not found" in err


# --- CLI: report (39.2) -------------------------------------------------------

def test_report_human_shows_accounting_block(tmp_path, capsys):
    """A29 (SPEC.md 39.2.1): the "what happened" block is in the human
    report, and the A11 invariants (SPEC 21.2) still hold."""
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== what happened ===" in out
    assert "rejected by gate" in out
    assert "first improvement" in out
    assert "wall time" in out
    # A11: the pre-v0.25 invariants are untouched
    assert "baseline_score" in out and "best spec" in out
    assert "score vs experiment:" not in out  # no plots without --plot


def test_report_json_stays_pure(tmp_path, capsys):
    """A29 (SPEC.md 39.2.1 + A11): `--json` is unchanged — pure JSON,
    equal to `summary.json`, no block, no prose."""
    run_dir = _make_run(tmp_path)
    rc = cli_main(["report", "--run", str(run_dir), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    on_disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert json.loads(out) == on_disk
    assert "what happened" not in out
    assert "best spec" not in out


# --- regression ---------------------------------------------------------------

def test_version_round_v025():
    """A29 (SPEC.md 39.3, 33.1): the version stepped to `0.39.0` in both
    sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53 — advanced in place per
    33.1)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"
