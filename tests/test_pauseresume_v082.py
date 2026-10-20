"""v0.82 — pause & resume: the live loop holds between experiments and
continues exactly where it held (SPEC.md 96, A86, M85).

Covers:

- 96.2.1 `DashboardRunner.pause()` / `resume()` / `paused` — before
  `start()` both fail loud; `next()` while paused fails loud (the hold
  is explicit); after `done` both are no-ops; pause is idempotent; a
  resume with no hold is a no-op;
- 96.2.2 trajectory neutrality (G2) — the experiment budget is frozen
  while the run holds, and a same-seed run paused and resumed logs the
  identical decision trajectory (experiments.jsonl minus the
  wall-clock floats, the A2 exclusion 18.4) as the uninterrupted run,
  with the same final best score;
- 96.2.3 the worker protocol (headless, no page render) — the worker
  honors the pause flag between experiments (the `paused` notice posts,
  the loop holds with no further updates), the resume flag releases it
  (the `resumed` notice posts), and the run completes with the full
  result + artifact set; Stop while paused ends the run `stopped`;
- 96.2.3 the app wiring — the app source carries the Pause/Resume
  buttons (bandit/search only, the 96.2.6 RL exclusion), the worker's
  pause/resume flag pair, and the drain's hold notices;
- 96.2.4 `BudgetManager.pause()` / `resume()` — `wall_seconds_left`
  freezes across a sleep, `resume()` does not charge the hold, both are
  idempotent, and a fresh `start()` clears the hold;
- A86.6 the round regression — the version steps to `0.82.0` in both
  sources (33.1), SPEC.md carries the A86 acceptance block, and the
  M85 index row resolves to this file.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy core; streamlit is optional (the worker-protocol tests
skip without it, the 51.2.3 AppTest pattern).
"""
from __future__ import annotations

import json
import re
import threading
import time
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import Budget
from autorefine.budget import BudgetManager
from autorefine.dashboard import DashboardRunner

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# 96.2.2: the wall-clock-derived floats — not bit-identical between any
# two runs (the A2 wall-clock exclusion, 18.4); the decision trajectory
# is everything else in the log
WALL_KEYS = ("train_seconds", "ts", "effective_score",
             "baseline_effective", "final_best_effective",
             "effective_improvement_factor")


# --- fixtures (synthesized here — no cross-test imports) ---------------------

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


def _runner(tmp_path: Path, runs_dir: str, **kw) -> DashboardRunner:
    kw.setdefault("csv_path", str(_write_csv(tmp_path)))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 4)
    kw.setdefault("policy", "bandit")
    # a cap no test train comes close to crossing (the A84.4 flake is a
    # cap crossing; 300s keeps the trajectory wall-clock-stable)
    kw.setdefault("max_train_seconds", 300.0)
    return DashboardRunner(runs_dir=runs_dir, **kw)


def _run_dir(runner: DashboardRunner) -> Path:
    base = Path(runner.env.run_dir)
    if (base / "experiments.jsonl").is_file():
        return base
    return sorted(base.glob("csv*"))[-1]


def _traj(p: Path) -> list:
    """The decision trajectory: every log entry minus the wall-clock
    floats (96.2.2)."""
    out = []
    for line in (p / "experiments.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        for k in WALL_KEYS:
            e.pop(k, None)
        out.append(e)
    return out


# --- A86.1: the core contract (96.2.1) ---------------------------------------

def test_pause_resume_guards(tmp_path):
    """A86.1 (SPEC.md 96.2.1): pause()/resume() before start() fail
    loud; after done both are no-ops; pause is idempotent; a resume with
    no hold is a no-op."""
    r = _runner(tmp_path, str(tmp_path / "runs1"))
    with pytest.raises(RuntimeError, match="before start"):
        r.pause()
    with pytest.raises(RuntimeError, match="before start"):
        r.resume()
    r.start()
    r.pause()
    r.pause()  # idempotent (96.2.1)
    assert r.paused
    r.resume()
    r.resume()  # a resume with no hold is a no-op
    assert not r.paused
    while not r.done:
        r.next()
    r.finish()
    r.pause()  # after done: no-ops (the env already finished)
    r.resume()
    assert r.paused is False
    assert r.done


def test_next_while_paused_fails_loud(tmp_path):
    """A86.1 (SPEC.md 96.2.1): next() while paused fails loud — the hold
    is explicit, never a silent no-op."""
    r = _runner(tmp_path, str(tmp_path / "runs2"))
    r.start()
    r.pause()
    with pytest.raises(RuntimeError, match="paused"):
        r.next()
    r.resume()
    while not r.done:
        r.next()
    r.finish()


# --- A86.2: hold & continue, trajectory neutrality (96.2.2) -------------------

def test_hold_freezes_budget_and_trajectory_is_neutral(tmp_path):
    """A86.2 (SPEC.md 96.2.2): a paused run consumes no experiments
    while it holds (the experiment budget is frozen across a sleep), and
    a same-seed run paused and resumed logs the identical decision
    trajectory as the uninterrupted run, with the same final best
    score.

    The gate runs in legacy mode (SPEC.md 18: all wall-clock knobs 0 =>
    `eff == raw score`), so the comparison is well-defined — under the
    v0.4 preset the acceptance itself is a wall-clock function (the 18.7
    bit-exact pin uses the same device)."""
    rA = _runner(tmp_path, str(tmp_path / "runsA"), search_quality="legacy")
    rA.start()
    while not rA.done:
        rA.next()
    resA = rA.finish()

    rB = _runner(tmp_path, str(tmp_path / "runsB"), search_quality="legacy")
    rB.start()
    rB.next()
    rB.next()
    rB.pause()
    assert rB.paused
    left = rB.env.bm.experiments_left
    time.sleep(0.2)  # the hold
    assert rB.env.bm.experiments_left == left, \
        "the hold must not consume experiments"
    rB.resume()
    assert not rB.paused
    while not rB.done:
        rB.next()
    resB = rB.finish()

    assert resA["final_best_score"] == resB["final_best_score"]
    assert _traj(_run_dir(rA)) == _traj(_run_dir(rB)), \
        "pause/resume is decision-trajectory-neutral (96.2.2)"


# --- A86.3: the worker protocol, headless (96.2.3) -----------------------------

def _worker_record(runner: DashboardRunner, pause: bool = False) -> dict:
    """The worker's shared record (the 51.2.3 shape) with the 96.2.3
    flag pair."""
    import queue
    return {
        "thread": None,
        "stop": {"flag": False},
        "pause": {"flag": pause},
        "resume": {"flag": False},
        "queue": queue.Queue(),
        "messages": [],
        "drained": False,
        "runner": runner,
    }


def _wait_for(record: dict, kind: str, timeout: float = 60.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if any(k == kind for k, _ in record["messages"]):
            return True
        if not record["thread"].is_alive():
            return any(k == kind for k, _ in record["messages"])
        time.sleep(0.02)
    return any(k == kind for k, _ in record["messages"])


def _kinds(record: dict) -> list:
    return [k for k, _ in record["messages"]]


def _n_updates(record: dict) -> int:
    return sum(1 for k, _ in record["messages"] if k == "update")


def test_worker_pause_resume_protocol(tmp_path):
    """A86.3 (SPEC.md 96.2.3): the worker honors the pause flag between
    experiments (the `paused` notice posts, the loop holds with no
    further updates), the resume flag releases it (the `resumed` notice
    posts), and the run completes with the full result + artifact set."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from autorefine.dashboard_app import _worker_loop

    r = _runner(tmp_path, str(tmp_path / "runsW"))
    rec = _worker_record(r, pause=True)
    rec["thread"] = threading.Thread(
        target=_worker_loop, args=(r, rec), daemon=True)
    rec["thread"].start()

    assert _wait_for(rec, "paused"), "the pause flag is honored (96.2.3)"
    assert r.paused
    n0 = _n_updates(rec)
    time.sleep(0.3)  # the hold: nothing trains while the run holds
    assert _n_updates(rec) == n0, "no experiment is consumed while paused"
    assert r.paused

    rec["resume"]["flag"] = True
    rec["thread"].join(timeout=120)
    kinds = _kinds(rec)
    assert "resumed" in kinds, kinds
    assert "result" in kinds and "end" in kinds, kinds
    assert "error" not in kinds, kinds
    assert not r.paused and r.done

    res = next(p for k, p in rec["messages"] if k == "result")
    rd = Path(res["run_dir"])
    assert (rd / "summary.json").is_file()
    assert (rd / "best_spec.json").is_file()
    assert (rd / "best_model.npz").is_file()


def test_worker_stop_while_paused(tmp_path):
    """A86.3 (SPEC.md 96.2.3): Stop while paused releases the hold first
    and ends the run `stopped` (51.2.1 semantics — the run did stop, at
    the user's request)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from autorefine.dashboard_app import _worker_loop

    r = _runner(tmp_path, str(tmp_path / "runsS"))
    rec = _worker_record(r, pause=True)
    rec["thread"] = threading.Thread(
        target=_worker_loop, args=(r, rec), daemon=True)
    rec["thread"].start()

    assert _wait_for(rec, "paused"), "the pause flag is honored"
    rec["stop"]["flag"] = True
    rec["thread"].join(timeout=120)

    kinds = _kinds(rec)
    assert "result" in kinds and "end" in kinds, kinds
    assert "error" not in kinds, kinds
    res = next(p for k, p in rec["messages"] if k == "result")
    summary = json.loads(
        (Path(res["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["finished_reason"] == "stopped"


# --- A86.4: the budget hold (96.2.4) ------------------------------------------

def test_budget_pause_freezes_the_wall_clock():
    """A86.4 (SPEC.md 96.2.4): pause() freezes `wall_seconds_left`
    across a sleep; resume() does not charge the hold; both are
    idempotent; a fresh start() clears the hold."""
    bm = BudgetManager(Budget(5, 1000.0, 10.0))
    assert bm.paused is False
    bm.pause()  # before start(): a no-op (no clock yet)
    assert bm.paused is False
    bm.start()
    assert bm.paused is False  # a fresh episode starts unpaused
    bm.pause()
    bm.pause()  # idempotent
    assert bm.paused
    t = bm.wall_seconds_left
    time.sleep(0.2)
    assert bm.wall_seconds_left > t - 0.05, \
        "the clock is frozen while paused"
    bm.resume()
    bm.resume()  # idempotent
    assert bm.paused is False
    assert bm.wall_seconds_left > t - 0.05, "the hold is not charged"
    bm.pause()
    bm.start()  # a fresh episode clears the hold
    assert bm.paused is False


# --- A86.5: the app wiring (96.2.3) --------------------------------------------

def test_app_wiring_pause_resume():
    """A86.5 (SPEC.md 96.2.3): the app source carries the Pause/Resume
    buttons (gated to runners that own a pause — the 96.2.6 RL
    exclusion), the worker's pause/resume flag pair, and the drain's
    hold notices."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        'key="pause_button"',
        'key="resume_button"',
        '"pause": {"flag": False}',
        '"resume": {"flag": False}',
        'record["pause"]["flag"]',
        'record["resume"]["flag"]',
        "runner.pause()",
        "runner.resume()",
        'if kind == "paused":',
        'if kind == "resumed":',
        'hasattr(runner, "pause")',
        'hasattr(rec["runner"], "pause")',
        "PAUSED — the loop is holding",
        "Resumed — the loop continues where it held.",
    ):
        assert token in src, token


# --- A86.6: the round regression ------------------------------------------------

def test_version_ceremony():
    """A86.6 (SPEC.md 33.1): the version steps to `0.82.0` in both
    sources; SPEC.md carries the A86 acceptance block; the M85 index
    row resolves to this file; the README points at the capability."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.82.0"
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 96.3 Acceptance (A86)" in spec
    assert re.search(
        r"^\s*\| M85 \| v0\.82\s*\|\s*96\s*\|\s*A86\s*\|"
        r"\s*tests/test_pauseresume_v082\.py\s*\|\s*$",
        spec, re.MULTILINE)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "pause()" in readme and "resume()" in readme, \
        "the README points at the capability (A86.6)"
