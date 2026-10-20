"""v0.80 — Live steering: respond to changing parameters, and visualize
the effects (SPEC.md 94, A84, M83).

Covers the nine A items as implemented — all additive / opt-in (the
``steering=None`` default is the byte-identical pre-v0.80 path, G2):

- A84.1 the **history log** — ``SteeringState.record()`` is the only
  history writer (verb / field / experiment validation, tuple→list
  normalization); the builders leave it empty; the ``history`` key is
  *conditional* in ``to_dict`` (the empty state keeps the exact
  pre-v0.80 three-key dict, the A49 pin);
- A84.2 ``AutoRefineEnv.steer()`` — a mid-run change records one event
  (before/after, ``at_experiment = bm.used_experiments``), returns only
  the new events, ``clear`` first with the removed rule set as
  ``before``; a no-arg / unknown-field / bad-value call is a
  ``ValueError`` (fail loud);
- A84.3 the policies' ``reset_field`` — bandit: zero credit ⇒ infinite
  UCB (cold start again), other fields untouched, unknown field raises;
  search: clears the refine queue only when the field is the current
  target, ``knn_k`` is a legal no-op, ``fail_streak`` untouched;
- A84.4 ``DashboardRunner.steer()`` — a pre-``start()`` call is a
  ``RuntimeError``; a mid-run pin re-explores the field first (the
  next candidate's mutation list contains it), the run finishes with
  the ``steering_svg`` + the ``steering_events`` summary key; two
  un-steered same-parameter runs follow the same trajectory (the log
  is byte-identical minus measured wall times; the summary's Pareto
  frontier is stable as a score multiset — tied-score membership is
  decided by measured time, SPEC 15);
- A84.5 the **effect math** — hand-computed pre/post window slicing,
  the i=0 fallback, the flat/delta boundary, fail-loud on bad window /
  eps / curve / event; ``best_score_curve`` mirrors
  ``dashboard.running_best_curve`` exactly;
- A84.6 the **artifacts** — a steered run's ``summary.json`` carries
  the event list; un-steered and pre-run-rules runs keep the key
  *absent* (the A49/A84 pin); ``RunConfig.from_env`` round-trips with a
  history-carrying env;
- A84.7 ``report --run DIR --steer`` — the ASCII effects table
  (cp1252-safe), the pure-JSON 9-key rows with ``--json``, the rc 1
  stderr hint on an un-steered run, and the mutual exclusions;
- A84.8 the **app** — the "Steer now" panel + the "Steering effects"
  result view tokens in the source, and a clean ``AppTest`` run;
- A84.9 the **ceremony** — 0.81.0 in both version sources, the SPEC
  94.3 heading + the M83 index row, the two new exports in
  ``autorefine.__all__``.

House rules: stdlib + numpy only, no cross-test imports (the CSV
fixture is synthesized here), deterministic tiny budgets, module-scoped
fixtures for finished runs.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    SearchPolicy,
    SteeringState,
    best_score_curve,
    effect_windows,
)
from autorefine.improver.bandit import BanditPolicy
from autorefine.dashboard import DashboardRunner, running_best_curve
from autorefine.runconfig import RunConfig
from autorefine.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
APP = ROOT / "src" / "autorefine" / "dashboard_app.py"


def _small_budget() -> Budget:
    return Budget(2, 300.0, 30.0)


def _write_csv(tmp_path: Path, name: str = "data.csv", rows: int = 60) -> Path:
    """Deterministic quadrant-XOR classification CSV (2 classes)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def steered_run(tmp_path_factory) -> dict:
    """A finished parity-v1 run steered once mid-run (A84.6/A84.7)."""
    base = tmp_path_factory.mktemp("livesteer_steered")
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(3, 300.0, 30.0),
                        runs_dir=base / "runs")
    state = env.reset()
    pol = BanditPolicy(seed=7)
    state, _r, _d, _i = env.step(pol.propose(state))
    events = env.steer(pin=("activation", "relu"))
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    assert env.done
    return {"dir": Path(env.run_dir), "events": events}


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory) -> Path:
    """A finished, never-steered parity-v1 run (the A84.6/A84.7 pin)."""
    base = tmp_path_factory.mktemp("livesteer_plain")
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(2, 300.0, 30.0),
                        runs_dir=base / "runs")
    state = env.reset()
    pol = BanditPolicy(seed=7)
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    return Path(env.run_dir)


# --- A84.1 the history log + record() (94.1.1) ------------------------------

def test_builders_keep_history_empty():
    """A84.1 (SPEC.md 94.1.1): the 59.2 builders (pre-run rules) never
    record — ``history`` stays empty."""
    s = (SteeringState()
         .pin("activation", "relu")
         .bias("learning_rate", 1e-3)
         .constrain("train_steps", [200, 400]))
    assert s.history == ()


def test_record_appends_one_event():
    """A84.1 (SPEC.md 94.1.1): ``record()`` appends one JSON-safe event
    dict with exactly the five identity keys."""
    s = SteeringState().record("activation", "pin", None, "relu", 1)
    assert len(s.history) == 1
    assert s.history[0] == {
        "field": "activation", "verb": "pin", "before": None,
        "after": "relu", "at_experiment": 1}


def test_record_normalizes_tuples_to_lists():
    """A84.1 (SPEC.md 94.1.1): tuples are normalized to lists at
    insertion (the JSON-safe round trip)."""
    s = SteeringState().record("train_steps", "constrain",
                               None, (200, 400), 3)
    assert s.history[0]["after"] == [200, 400]
    assert isinstance(s.history[0]["after"], list)


def test_record_validates_verb_field_and_experiment():
    """A84.1 (SPEC.md 94.1.1): a bad verb, a ``clear`` with a field, a
    non-registry field, or a bad ``at_experiment`` is a ValueError
    (fail loud); ``clear`` with ``field=None`` is legal."""
    with pytest.raises(ValueError):
        SteeringState().record("activation", "nudge", None, "relu", 0)
    with pytest.raises(ValueError):
        SteeringState().record("activation", "clear", None, None, 0)
    with pytest.raises(ValueError):
        SteeringState().record("nope", "pin", None, 1, 0)
    for bad in (True, -1, "3", 1.0):
        with pytest.raises(ValueError):
            SteeringState().record("activation", "pin", None, "relu", bad)
    ok = SteeringState().record(None, "clear", None, None, 0)
    assert ok.history[0]["verb"] == "clear"


def test_history_roundtrip_and_empty_state_pin():
    """A84.1 (SPEC.md 94.1.1): ``from_dict({})`` is the empty state, a
    history-carrying state round-trips, and the *empty* state's
    ``to_dict`` keeps the exact pre-v0.80 three-key dict (the A49 pin —
    the history key is conditional)."""
    assert SteeringState.from_dict({}) == SteeringState()
    s = (SteeringState().pin("activation", "relu")
                    .record("activation", "pin", None, "relu", 2))
    d = s.to_dict()
    assert "history" in d
    assert SteeringState.from_dict(d) == s
    empty = SteeringState().to_dict()
    assert set(empty) == {"pins", "biases", "constraints"}
    assert empty == {"pins": [], "biases": [], "constraints": []}


# --- A84.2 AutoRefineEnv.steer() (94.1.3) -----------------------------------

def test_mid_run_steer_records_and_applies(tmp_path):
    """A84.2 (SPEC.md 94.1.3): a mid-run change records one event with
    the before/after values and ``at_experiment``; each call returns
    only its new events; ``clear`` first with the removed rule set as
    ``before`` while the history is kept."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path / "runs")
    assert env.steering is None  # fresh state is created by steer()
    state = env.reset()
    pol = BanditPolicy(seed=7)
    state, _r, _d, _i = env.step(pol.propose(state))
    assert env.bm.used_experiments == 1

    ev = env.steer(pin=("activation", "relu"))
    assert len(ev) == 1
    assert ev[0] == {"field": "activation", "verb": "pin",
                     "before": None, "after": "relu", "at_experiment": 1}

    ev2 = env.steer(pin=("activation", "tanh"))
    assert ev2[0]["before"] == "relu" and ev2[0]["after"] == "tanh"

    ev3 = env.steer(bias=("learning_rate", 1e-3))
    assert ev3[0]["verb"] == "bias" and ev3[0]["after"] == 1e-3

    ev4 = env.steer(constrain=("train_steps", [200, 400]))
    assert ev4[0]["verb"] == "constrain"
    assert ev4[0]["after"] == [200, 400]

    # only the new event is returned each call; the history accumulates
    assert len(env.steering.history) == 4

    ev5 = env.steer(clear=True)
    assert len(ev5) == 1 and len(env.steering.history) == 5
    c = ev5[0]
    assert c["verb"] == "clear" and c["field"] is None
    assert c["after"] is None
    assert c["before"] == {
        "pins": [["activation", "tanh"]],
        "biases": [["learning_rate", 1e-3]],
        "constraints": [["train_steps", [200, 400]]]}
    # the rules are gone; the history (the log) is kept
    assert env.steering.pins == ()
    assert env.steering.biases == ()
    assert env.steering.constraints == ()
    assert len(env.steering.history) == 5


def test_steer_fails_loud(tmp_path):
    """A84.2 (SPEC.md 94.1.3): a no-arg call, an unknown field, or a
    bad value is a ValueError — a silent no-op would be a bug."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path / "runs")
    with pytest.raises(ValueError):
        env.steer()
    with pytest.raises(ValueError):
        env.steer(pin=("nope", 1))
    with pytest.raises(ValueError):
        env.steer(pin=("activation", "leaky_relu"))
    assert env.steering is None  # nothing was recorded


# --- A84.3 the policies' reset_field (94.1.2) -------------------------------

def test_bandit_reset_field_cold_start():
    """A84.3 (SPEC.md 94.1.2): the bandit zeroes the field's credit
    (infinite UCB again ⇒ re-proposed first) and leaves every other
    field's credit untouched; an unknown field is a ValueError."""
    p = BanditPolicy(seed=7)
    p.trials["activation"] = 3
    p.wins["activation"] = 2.0
    p.trials["learning_rate"] = 2
    p.reset_field("activation")
    assert p.trials["activation"] == 0
    assert p.wins["activation"] == 0.0
    assert p._ucb("activation") == float("inf")
    assert p.trials["learning_rate"] == 2
    with pytest.raises(ValueError):
        p.reset_field("nope")


def test_search_reset_field_clears_tracking():
    """A84.3 (SPEC.md 94.1.2): the search clears the refine queue only
    when the field is the current target; a non-target field is a no-op;
    ``knn_k`` (outside the search pool) is a legal no-op; ``fail_streak``
    is untouched; an unknown field is a ValueError."""
    s = SearchPolicy(seed=7)
    s.last_field = "learning_rate"
    s.refine_left = 2
    s.fail_streak = 3
    s.reset_field("learning_rate")
    assert s.last_field is None
    assert s.refine_left == 0
    assert s.fail_streak == 3

    s2 = SearchPolicy(seed=7)
    s2.last_field = "learning_rate"
    s2.refine_left = 2
    s2.reset_field("batch_size")
    assert s2.last_field == "learning_rate" and s2.refine_left == 2

    s3 = SearchPolicy(seed=7)
    s3.last_field = "learning_rate"
    s3.refine_left = 2
    s3.reset_field("knn_k")
    assert s3.last_field == "learning_rate" and s3.refine_left == 2

    with pytest.raises(ValueError):
        s.reset_field("nope")


# --- A84.4 DashboardRunner.steer() (94.1.4) ---------------------------------

def _drive(runner: DashboardRunner) -> dict:
    while not runner.done:
        runner.next()
    return runner.finish()


def test_runner_steer_before_start_raises(tmp_path):
    """A84.4 (SPEC.md 94.1.4): a ``steer()`` before ``start()`` (no
    env/policy yet) is a RuntimeError — fail loud."""
    csv = _write_csv(tmp_path)
    r = DashboardRunner(csv, runs_dir=tmp_path / "runs", experiments=3,
                        target=50, seed=7)
    with pytest.raises(RuntimeError) as exc:
        r.steer(pin=("activation", "relu"))
    assert str(exc.value).startswith("steer() before start()")


def test_runner_mid_run_steer_reexplores_and_reports(tmp_path):
    """A84.4 (SPEC.md 94.1.4): a mid-run pin resets the field's credit
    (the bandit re-explores it first — the pin's value force-enters the
    next candidate's mutation list), and the finished run carries the
    ``steering_svg`` + the ``steering_events`` summary key."""
    csv = _write_csv(tmp_path)
    r = DashboardRunner(csv, runs_dir=tmp_path / "runs", experiments=3,
                        target=50, seed=7)
    r.start()
    r.next()  # experiment 1
    ev = r.steer(pin=("activation", "relu"))
    assert len(ev) == 1
    assert ev[0]["at_experiment"] == 1
    # the changed field is cold-started again (re-explored first)
    assert r.policy.trials["activation"] == 0
    u = r.next()  # experiment 2 — the pin differs from the champion
    assert "activation" in u["mutation"]
    res = _drive(r)
    assert res["steering_svg"] is not None
    summary = json.loads(
        (Path(r.env.run_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["steering_events"] == [
        dict(e) for e in r.env.steering.history]


def test_runner_unsteered_runs_are_deterministic(tmp_path):
    """A84.4 (SPEC.md 94.1.4, the G2 pin): two un-steered, same-parameter
    runs on the legacy (no wall-clock term) acceptance path follow the
    same trajectory — same mutations, same specs, same scores, same
    verdicts: the experiment log is byte-identical except the measured
    wall times (each row's ``train_seconds`` and ``ts`` timestamp).

    The summary is identical except ``wall_seconds`` and the Pareto
    frontier: the frontier is (score, *measured* time) by design
    (SPEC.md 15), and among *tied-score* points (here the 33.3 baseline
    and the rejected 33.3 candidate) dominance is decided by measured
    time, so which spec survives on the frontier is legitimately
    wall-clock dependent. The frontier's *score multiset* is G2-stable
    and is pinned as such.

    (The v0.4 default gate embeds the measured train time in the
    acceptance itself, so it can legitimately flip between runs of the
    same seed; the legacy path is the stable one.)

    The per-train wall-time cap — the loop's only wall-clock knob (a
    spec whose training crosses the cap trains fewer epochs, so its
    score and the acceptance can legitimately differ on a loaded
    machine) — is raised out of reach (600s), so both runs train the
    identical number of epochs and the trajectory pin is exact."""
    csv = _write_csv(tmp_path)
    logs, summaries, frontiers = [], [], []
    for name in ("A", "B"):
        r = DashboardRunner(csv, runs_dir=tmp_path / name, experiments=3,
                            target=50, seed=7, search_quality="legacy",
                            max_train_seconds=600.0)
        r.start()
        _drive(r)
        run_dir = Path(r.env.run_dir)
        log = [json.loads(line) for line in
               (run_dir / "experiments.jsonl").read_text(encoding="utf-8")
               .splitlines() if line.strip()]
        for row in log:
            row.pop("train_seconds", None)  # measured wall time, not G2-stable
            row.pop("ts", None)  # wall-clock log timestamp
        logs.append(log)
        s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        s.pop("wall_seconds", None)
        pts = s.pop("pareto_frontier", None) or []
        frontiers.append(sorted(p["score"] for p in pts))
        summaries.append(s)
    assert logs[0] == logs[1]  # the trajectory pin (the G2 core)
    assert summaries[0] == summaries[1]
    assert frontiers[0] == frontiers[1]
    assert "steering_events" not in summaries[0]


# --- A84.5 best_score_curve + effect_windows (94.1.5) -----------------------

def test_effect_windows_hand_computed():
    """A84.5 (SPEC.md 94.1.5): hand-computed window slicing — the full
    window, the partial window at the start, the i=0 fallback, and the
    flat boundary (|delta| <= eps)."""
    curve = [50, 52, 48, 60, 58, 66]
    mk = lambda i: [{"field": "activation", "verb": "pin",
                     "before": "tanh", "after": "relu",
                     "at_experiment": i}]

    row = effect_windows(curve, mk(2), window=3, eps=0.05)[0]
    assert row["pre_mean"] == pytest.approx(51.0)          # mean(50, 52)
    assert row["post_mean"] == pytest.approx((48 + 60 + 58) / 3)
    assert row["delta"] == pytest.approx(
        (48 + 60 + 58) / 3 - 51.0)
    assert row["direction"] == "up"
    assert row["at_experiment"] == 2
    assert row["field"] == "activation"

    row = effect_windows(curve, mk(1), window=3, eps=0.05)[0]
    assert row["pre_mean"] == pytest.approx(50.0)          # partial: mean(50)
    assert row["post_mean"] == pytest.approx((52 + 48 + 60) / 3)

    row = effect_windows(curve, mk(0), window=3, eps=0.05)[0]
    assert row["pre_mean"] == pytest.approx(50.0)          # i=0 fallback
    assert row["post_mean"] == pytest.approx((50 + 52 + 48) / 3)

    row = effect_windows([50, 50.05], mk(1), window=1)[0]
    assert row["delta"] == pytest.approx(0.05)
    assert row["direction"] == "flat"  # |delta| <= eps (default 0.05)

    row = effect_windows([60, 50], mk(1), window=1)[0]
    assert row["delta"] == pytest.approx(-10.0)
    assert row["direction"] == "down"


def test_effect_windows_fails_loud():
    """A84.5 (SPEC.md 94.1.5): an event past the logged curve, a bad
    window / eps / curve / event is a ValueError; an empty event list is
    the empty list."""
    curve = [50, 52]
    ev = [{"field": "activation", "verb": "pin", "before": None,
           "after": "relu", "at_experiment": 2}]  # == len(curve): no post
    with pytest.raises(ValueError):
        effect_windows(curve, ev)
    for bad_window in (0, True, -1, "3"):
        with pytest.raises(ValueError):
            effect_windows(curve, ev, window=bad_window)
    for bad_eps in (-0.1, float("nan"), "x", True):
        with pytest.raises(ValueError):
            effect_windows(curve, ev, eps=bad_eps)
    for bad_curve in ([], [1.0, float("nan")], "abc"):
        with pytest.raises(ValueError):
            effect_windows(bad_curve, ev)
    with pytest.raises(ValueError):
        effect_windows(curve, [["x"]])
    assert effect_windows(curve, []) == []


def test_best_score_curve_running_best_semantics():
    """A84.5 (SPEC.md 94.1.5): point 0 is the baseline, the running best
    never drops (a rejected dip is skipped), non-baseline/experiment
    kinds and non-finite scores are excluded, ``[]`` when nothing is
    scored, non-dict rows skipped — and it mirrors
    ``dashboard.running_best_curve`` exactly."""
    entries = [
        "not-a-dict",
        {"kind": "baseline", "holdout_score": 50.0},
        {"kind": "experiment", "holdout_score": 45.0},   # a dip
        {"kind": "curriculum", "holdout_score": 99.0},   # not scored
        {"kind": "experiment", "holdout_score": float("nan")},
        {"kind": "experiment", "holdout_score": 60.0},
    ]
    assert best_score_curve(entries) == [50.0, 50.0, 60.0]
    assert best_score_curve([{"kind": "baseline"}]) == []
    assert best_score_curve([]) == []
    # the mirror check: the dashboard's curve has identical semantics
    assert running_best_curve(entries) == best_score_curve(entries)


# --- A84.6 the steering_events summary artifact (94.1.6) --------------------

def test_steered_run_summary_carries_events(steered_run):
    """A84.6 (SPEC.md 94.1.6): a steered run's ``summary.json`` carries
    the event list (JSON-safe dicts)."""
    summary = json.loads(
        (steered_run["dir"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["steering_events"] == steered_run["events"]
    assert len(summary["steering_events"]) == 1
    row = summary["steering_events"][0]
    assert set(row) == {"field", "verb", "before", "after",
                        "at_experiment"}


def test_unsteered_and_prerun_runs_keep_key_absent(plain_run, tmp_path):
    """A84.6 (SPEC.md 94.1.6, the A49/A84 pin): un-steered runs and
    pre-run-rules runs (the constructor, never ``steer()``) keep the
    ``steering_events`` key *absent* — the pre-v0.80 key set intact."""
    plain = json.loads(
        (plain_run / "summary.json").read_text(encoding="utf-8"))
    assert "steering_events" not in plain

    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path / "runs",
                        steering=SteeringState().pin("activation", "relu"))
    state = env.reset()
    pol = BanditPolicy(seed=7)
    while not env.done:
        state, _r, _d, _i = env.step(pol.propose(state))
    pruned = json.loads(
        (Path(env.run_dir) / "summary.json").read_text(encoding="utf-8"))
    assert "steering_events" not in pruned
    assert env.steering is not None and env.steering.history == ()


def test_runconfig_roundtrips_with_history(tmp_path):
    """A84.6 (SPEC.md 94.1.6): ``RunConfig.from_env`` reads the rules
    (the conditional history key is harmless) and the recipe
    round-trips with a history-carrying env."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path / "runs")
    env.reset()
    env.steer(pin=("activation", "relu"))  # history non-empty
    cfg = RunConfig.from_env(env, policy="bandit", target=95.0)
    assert cfg.pins == [["activation", "relu"]]
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


# --- A84.7 report --run DIR --steer (94.1.7) --------------------------------

def test_report_steer_ascii_table(steered_run, capsys):
    """A84.7 (SPEC.md 94.1.7): the human form is an ASCII effects table
    (cp1252-safe: ``->`` and ``delta``), rc 0."""
    rc = cli_main(["report", "--run", str(steered_run["dir"]), "--steer"])
    out = capsys.readouterr().out
    assert rc == 0
    out.encode("cp1252")  # console-safe
    assert "steering effects" in out
    assert "delta" in out and "->" in out


def test_report_steer_json_rows(steered_run, capsys):
    """A84.7 (SPEC.md 94.1.7): with ``--json`` the pure 9-key effect-row
    list is on stdout (``--steer`` and ``--json`` are compatible)."""
    rc = cli_main(["report", "--run", str(steered_run["dir"]),
                   "--steer", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    rows = json.loads(out)
    assert isinstance(rows, list) and len(rows) == 1
    for row in rows:
        assert set(row) == {"field", "verb", "before", "after",
                            "at_experiment", "pre_mean", "post_mean",
                            "delta", "direction"}


def test_report_steer_requires_a_steered_run(plain_run, capsys):
    """A84.7 (SPEC.md 94.1.7): an un-steered run is rc 1 with the
    stderr hint (the view is for steered runs)."""
    rc = cli_main(["report", "--run", str(plain_run), "--steer"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("no live steering events")


def test_report_steer_mutual_exclusions(steered_run, capsys):
    """A84.7 (SPEC.md 94.1.7): ``--steer`` is mutually exclusive with
    the other human views (``--history``, ``--importance``, ...)."""
    base = ["report", "--run", str(steered_run["dir"]), "--steer"]
    for flag in (["--history"], ["--importance"]):
        rc = cli_main(base + flag)
        capsys.readouterr()
        assert rc == 1


# --- A84.8 the app (94.2) ----------------------------------------------------

def test_app_live_steer_source_tokens():
    """A84.8 (SPEC.md 94.2.1/94.2.2): the app wires the "Steer now"
    panel (``runner.steer``) and the "Steering effects" result view
    (``best_score_curve`` + ``effect_windows``)."""
    src = APP.read_text(encoding="utf-8")
    for tok in (
        "stg_live_verb", "stg_live_apply", "runner.steer",
        "effect_windows", "best_score_curve",
        "Steer now", "Steering effects", "cleared all steering rules",
    ):
        assert tok in src, tok


def test_app_renders_with_steering():
    """A84.8 (SPEC.md 94.2): the app boots clean with the v0.80
    steering surfaces in place (streamlit is optional)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception


# --- A84.9 the ceremony -------------------------------------------------------

def test_version_ceremony():
    """A84.9 (SPEC.md 94.3): 0.81.0 in both version sources, the
    SPEC 94.3 heading + the M83 index row, and the two new exports in
    ``autorefine.__all__``."""
    assert autorefine.__version__ == "0.81.0"
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == "0.81.0"
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 94.3 Acceptance (A84)" in spec
    assert re.search(r"^\s*\|\s*M83\s*\|\s*v0\.80\s*\|\s*94\s*\|\s*A84\s*\|",
                     spec, flags=re.MULTILINE)
    assert "**M83**" in spec
    assert "best_score_curve" in autorefine.__all__
    assert "effect_windows" in autorefine.__all__
