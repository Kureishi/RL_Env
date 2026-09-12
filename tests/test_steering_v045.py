"""v0.45 — "C. Parameters: visually interpreted and modified" (SPEC.md 59,
A49, M48).

Covers the three A items as implemented — all *additive / opt-in* over the
machinery that already exists (the ``SPEC_FIELDS`` registry, the ``KNOBS``
registry, the ``research`` §57 derivations), the pre-v0.45 default path is
byte-identical (G2), and the A1–A4 seeded proposal stream is unchanged
(``exclude_fields=()`` default):

- 59.1 the **parameter inspector** — ``parameter_inspection`` (24 rows: 15
  architecture + 9 loop, in registry order; the architecture row's
  best-seen / Δ recomposed from ``research.field_response_stats``, the
  win-rate / Wilson CI / UCB from ``research.bandit_beliefs``; the loop row
  carries the ``KNOBS`` cli / app_widget and ``None`` effect fields; empty
  entries degrade every row to ``None`` effect fields);
- 59.2 the **steering verbs** — ``SteeringState`` (registry-level
  validation, last-one-wins, the ``to_dict`` / ``from_dict`` round-trip,
  the ``active`` fast path) + ``apply_steering`` (pins force-set, biases
  redirect only a *mutated* field, constraints yield a violation string,
  ``None`` / empty → the input unchanged) + the env (pin changes the reset
  baseline; constraint rejects out-of-set as ``invalid_spec``; ``None``
  steering is the pre-v0.45 baseline) + the policies (``exclude_fields``
  drops the pinned field from the mutation pool, default stream unchanged);
- 59.3 the **manual / expert mode** — ``manual_spec`` (empty =
  ``DEFAULT_SPEC``; unknown field → ``ValueError``; a bad value →
  ``ValueError``; a bad *combination* → ``SpecError``) + ``train_manual``
  (unknown task → ``ValueError``; one tiny parity train returns the 7-key
  dict);
- 59.4 the **surfaces + exports** — the CLI wires the ``manual`` subcommand
  + the ``--pin/--bias/--constrain`` flags, the dashboard wires the
  ``steering`` kwarg + the ``exclude_fields`` pass-through, the app wires
  the inspector / steering / manual surfaces, ``runconfig`` round-trips
  (an old dict missing the fields loads ``[]``; ``from_env`` reads
  ``env.steering``; ``fit_recipe`` / ``runconfig_to_flags`` emit the flags),
  and the five new names are in ``__all__``;
- the A49 round regression — the version stepped to ``0.45.0`` in both
  sources (33.1) and SPEC carries the A49 block + M48 row.

House rules (A49): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; the app / dashboard / CLI
wiring is tested by source-token assertions + one light AppTest (streamlit
optional). The loop integration uses a tiny parity budget.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    DEFAULT_SPEC,
    SearchPolicy,
    SteeringState,
    apply_steering,
    manual_spec,
    parameter_inspection,
    train_manual,
)
from autorefine.config import SpecError
from autorefine.improver.bandit import BanditPolicy
from autorefine.research import bandit_beliefs
from autorefine.runconfig import RunConfig, fit_recipe, runconfig_to_flags

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
DASH = REPO / "src" / "autorefine" / "dashboard.py"
CLI = REPO / "src" / "autorefine" / "cli.py"


def _small_budget() -> Budget:
    return Budget(max_experiments=2, max_wall_seconds=300.0,
                  max_train_seconds=30.0)


# --- 59.2 SteeringState (A49) -------------------------------------------------

def test_steering_unknown_field_raises():
    """A49 (SPEC.md 59.2): an unknown field is a ValueError naming the
    registry — on all three verbs."""
    for verb in ("pin", "bias", "constrain"):
        with pytest.raises(ValueError):
            getattr(SteeringState(), verb)("nope", 1)


def test_steering_bad_value_raises():
    """A49 (SPEC.md 59.2): a value the field's registry validator rejects is
    a ValueError (the per-field 'why is this rejected')."""
    with pytest.raises(ValueError):
        SteeringState().pin("activation", "leaky_relu")
    with pytest.raises(ValueError):
        SteeringState().constrain("activation", ["tanh", "leaky_relu"])


def test_steering_constrain_empty_or_noniterable_raises():
    """A49 (SPEC.md 59.2): the allowed set must be a non-empty iterable."""
    with pytest.raises(ValueError):
        SteeringState().constrain("activation", [])
    with pytest.raises(ValueError):
        SteeringState().constrain("activation", 3)  # an int is not iterable


def test_steering_last_one_wins_and_active():
    """A49 (SPEC.md 59.2): a second rule for the same field replaces the
    first (last one wins); ``active`` reflects whether any rule exists."""
    assert not SteeringState().active
    s = SteeringState().pin("activation", "tanh").pin("activation", "relu")
    assert s.pins == (("activation", "relu"),)
    assert s.active
    s2 = s.bias("learning_rate", 1e-3)
    assert s2.pins == (("activation", "relu"),)
    assert s2.biases == (("learning_rate", 1e-3),)
    s3 = s2.constrain("train_steps", [200, 400])
    assert s3.constraints == (("train_steps", (200, 400)),)


def test_steering_to_from_dict_roundtrip():
    """A49 (SPEC.md 59.2): the JSON-safe pair round-trips and re-validates."""
    s = (SteeringState()
         .pin("activation", "relu")
         .bias("learning_rate", 1e-3)
         .constrain("train_steps", [200, 400]))
    d = s.to_dict()
    assert d == {
        "pins": [["activation", "relu"]],
        "biases": [["learning_rate", 1e-3]],
        "constraints": [["train_steps", [200, 400]]],
    }
    assert SteeringState.from_dict(d) == s


def test_steering_from_dict_rejects_bad_input():
    """A49 (SPEC.md 59.2): ``from_dict`` re-validates — a non-dict, or a dict
    with an unknown field / bad value, is a ValueError."""
    with pytest.raises(ValueError):
        SteeringState.from_dict("nope")
    with pytest.raises(ValueError):
        SteeringState.from_dict({"pins": [["nope", 1]]})
    with pytest.raises(ValueError):
        SteeringState.from_dict({"pins": [["activation", "leaky_relu"]]})
    # an empty / missing-keys dict is the empty state (back-compat)
    assert SteeringState.from_dict({}) == SteeringState()


# --- 59.2 apply_steering (A49) ------------------------------------------------

def test_apply_steering_none_and_empty_unchanged():
    """A49 (SPEC.md 59.2.2): ``None`` / an empty state returns the input
    unchanged (a new dict) with no violations — the pre-v0.45 path (G2)."""
    spec = {"activation": "tanh", "learning_rate": 1e-3}
    for state in (None, SteeringState()):
        out, violations = apply_steering(spec, state)
        assert out == spec
        assert violations == []
    # the input is never mutated
    spec["activation"] = "relu"
    out, _ = apply_steering(spec, SteeringState().pin("activation", "tanh"))
    assert spec["activation"] == "relu"  # input intact
    assert out["activation"] == "tanh"   # the copy is rewritten


def test_apply_steering_pins_force_set():
    """A49 (SPEC.md 59.2.2): pins force-set the field unconditionally, even
    when the candidate already holds a different value."""
    state = SteeringState().pin("activation", "tanh")
    out, violations = apply_steering({"activation": "relu"}, state)
    assert out["activation"] == "tanh"
    assert violations == []


def test_apply_steering_bias_redirects_only_mutated():
    """A49 (SPEC.md 59.2.2): a bias redirects a *mutated* field (one whose
    value differs from the champion) to the biased value; a candidate equal
    to the champion, or already the biased value, is untouched."""
    champ = {"activation": "tanh", "learning_rate": 1e-3}
    state = SteeringState().bias("learning_rate", 1e-4)

    # mutated (differs from the champion) -> redirected
    out, _ = apply_steering({"learning_rate": 1e-2}, state, champ)
    assert out["learning_rate"] == 1e-4

    # already the biased value -> untouched (still 1e-4)
    out, _ = apply_steering({"learning_rate": 1e-4}, state, champ)
    assert out["learning_rate"] == 1e-4

    # equal to the champion (not mutated) -> untouched (stays the champion's)
    out, _ = apply_steering({"learning_rate": 1e-3}, state, champ)
    assert out["learning_rate"] == 1e-3


def test_apply_steering_constraint_violation_string():
    """A49 (SPEC.md 59.2.2): a constraint violation yields the exact string;
    an in-set value does not."""
    state = SteeringState().constrain("activation", ["tanh", "relu"])
    out, violations = apply_steering({"activation": "leaky_relu"}, state)
    assert len(violations) == 1
    assert violations[0] == (
        "constrain(activation): 'leaky_relu' outside the allowed set "
        "['tanh', 'relu']")
    # an in-set value produces no violation
    out, violations = apply_steering({"activation": "tanh"}, state)
    assert violations == []


# --- 59.3 manual_spec / train_manual (A49) ------------------------------------

def test_manual_spec_empty_is_default():
    """A49 (SPEC.md 59.3.1): an empty override map is DEFAULT_SPEC itself."""
    assert manual_spec({}).to_dict() == DEFAULT_SPEC.to_dict()


def test_manual_spec_unknown_field_raises():
    """A49 (SPEC.md 59.3.1/59.2.1): an unknown field is a ValueError."""
    with pytest.raises(ValueError):
        manual_spec({"nope": 1})


def test_manual_spec_bad_value_raises():
    """A49 (SPEC.md 59.3.1/59.2.1): a value the registry rejects is a
    ValueError (before the combination check)."""
    with pytest.raises(ValueError):
        manual_spec({"activation": "leaky_relu"})


def test_manual_spec_bad_combo_is_spec_error():
    """A49 (SPEC.md 59.3.1): a value legal for *some* family (the registry
    accepts a tree/boost depth) but invalid for the default mlp family is a
    SpecError — the same typed error the loop rejects (25.4)."""
    # arch (1,) is legal as a tree/boost depth; with the default mlp family
    # it is an invalid hidden width -> SpecError
    with pytest.raises(SpecError):
        manual_spec({"architecture": (1,)})


def test_manual_spec_valid_override():
    """A49 (SPEC.md 59.3.1): a valid override is applied on top of the
    defaults (the rest unchanged)."""
    spec = manual_spec({"activation": "relu"})
    d = spec.to_dict()
    assert d["activation"] == "relu"
    for k, v in DEFAULT_SPEC.to_dict().items():
        if k != "activation":
            assert d[k] == v


def test_train_manual_unknown_task_raises():
    """A49 (SPEC.md 59.3.2): an unknown task name is a ValueError (the
    registry, 15)."""
    with pytest.raises(ValueError):
        train_manual("nope-v1", {})


def test_train_manual_tiny_parity_returns_seven_keys():
    """A49 (SPEC.md 59.3.2): one tiny parity train returns the 7-key dict —
    the researcher knows the architecture, the loop validates + reports."""
    out = train_manual("parity-v1", {"train_steps": 200},
                       seed=7, max_train_seconds=5)
    assert set(out) == {"spec", "score", "gen_score", "gen_gap",
                        "train_seconds", "final_loss", "time_capped"}
    assert out["spec"]["activation"] == DEFAULT_SPEC.to_dict()["activation"]
    for key in ("score", "gen_score", "gen_gap", "train_seconds",
                "final_loss"):
        assert isinstance(out[key], float)
    assert isinstance(out["time_capped"], bool)


# --- 59.1 parameter_inspection (A49) ------------------------------------------

def test_inspection_row_count_and_layers():
    """A49 (SPEC.md 59.1): 24 rows — the 15 architecture rows (registry
    order) followed by the 9 loop rows (KNOBS order)."""
    rows = parameter_inspection([])
    assert len(rows) == 24
    layers = [r["layer"] for r in rows]
    assert layers[:15] == ["architecture"] * 15
    assert layers[15:] == ["loop"] * 9


def test_inspection_arch_row_shape_and_effect():
    """A49 (SPEC.md 59.1): an architecture row carries the registry metadata
    (kind / space / families / spec_ref) + current / baseline, and — over a
    scored fixture — the hand-computed best-seen / Δ, with the win-rate /
    CI / UCB wired from ``research.bandit_beliefs``."""
    entries = [
        {"kind": "baseline", "spec": {"activation": "tanh"},
         "holdout_score": 50.0},
        {"kind": "experiment", "spec": {"activation": "relu"},
         "holdout_score": 80.0, "accepted": True},
    ]
    field_stats = {"activation": {"trials": 4, "wins": 3, "win_rate": 0.75}}
    ucb = {"activation": 0.9}
    rows = parameter_inspection(entries, best_spec={"activation": "relu"},
                                field_stats=field_stats, ucb=ucb)
    act = {r["name"]: r for r in rows}["activation"]
    assert act["layer"] == "architecture"
    assert act["current"] == "relu"
    assert act["baseline"] == "tanh"          # DEFAULT_SPEC activation
    assert act["best_seen"] == "relu"         # the highest-mean value
    assert act["best_seen_mean"] == 80.0
    assert act["best_seen_n"] == 1
    assert act["delta"] == pytest.approx(30.0)  # 80 (relu) - 50 (baseline)
    # the belief fields are wired from the §57.4 derivation
    bel = bandit_beliefs(field_stats, ucb)["activation"]
    assert act["win_rate"] == bel["win_rate"]
    assert act["ci_lo"] == bel["lo"]
    assert act["ci_hi"] == bel["hi"]
    assert act["ucb"] == bel["ucb"]
    # the registry metadata is present (the G2 field, 36.2)
    assert act["kind"] and act["families"] and act["spec_ref"]
    assert isinstance(act["space"], list)


def test_inspection_loop_row_shape():
    """A49 (SPEC.md 59.1): a loop row carries the ``KNOBS`` cli / app_widget
    + current (from run_config) / baseline (the knob default), with every
    bandit-derived effect field ``None``."""
    rows = parameter_inspection([], run_config={"stall_patience": 3})
    loop = {r["name"]: r for r in rows if r["layer"] == "loop"}
    assert "stall_patience" in loop
    sp = loop["stall_patience"]
    assert sp["current"] == 3
    assert sp["baseline"] is None          # the knob default is None
    assert isinstance(sp["cli"], list)
    for key in ("win_rate", "ci_lo", "ci_hi", "ucb", "best_seen", "delta",
                "best_seen_mean", "best_seen_n"):
        assert sp[key] is None


def test_inspection_empty_entries_degrades():
    """A49 (SPEC.md 59.1): with no scored entries every row is still present
    and every effect field is None (an old run without field_stats still
    renders every row)."""
    rows = parameter_inspection([])
    assert len(rows) == 24
    for r in rows:
        assert r["best_seen"] is None
        assert r["delta"] is None
        assert r["win_rate"] is None


# --- 59.2 environment + policy integration (A49) ------------------------------

def test_env_pin_changes_reset_baseline(tmp_path):
    """A49 (SPEC.md 59.2): a pin force-sets the field on the reset baseline
    (DEFAULT_SPEC activation is tanh; pinning relu changes it)."""
    state = SteeringState().pin("activation", "relu")
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path, steering=state)
    s = env.reset()
    assert s["best_spec"]["activation"] == "relu"


def test_env_none_steering_baseline_is_default(tmp_path):
    """A49 (SPEC.md 59.2/G2): no steering is the pre-v0.45 baseline exactly
    (DEFAULT_SPEC)."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path)
    assert env.reset()["best_spec"] == DEFAULT_SPEC.to_dict()


def test_env_non_steering_state_raises(tmp_path):
    """A49 (SPEC.md 59.2): a non-SteeringState non-None steering is a
    construction-time ValueError (the stop_check fail-loud style)."""
    with pytest.raises(ValueError):
        AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                      runs_dir=tmp_path, steering="nope")


def test_env_constraint_rejects_out_of_set(tmp_path):
    """A49 (SPEC.md 59.2): a constraint rejects an out-of-set candidate as
    ``invalid_spec`` (logged, 25.4) — the baseline keeps the default (tanh),
    so a relu mutation is outside the allowed set {"tanh"}."""
    state = SteeringState().constrain("activation", ["tanh"])
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path, steering=state)
    s = env.reset()
    assert s["best_spec"]["activation"] == "tanh"  # the baseline is unchanged
    cand = dict(s["best_spec"])
    cand["activation"] = "relu"  # outside the allowed set {"tanh"}
    s, r, d, info = env.step(cand)
    assert info["accepted"] is False
    assert info["reason"] == "invalid_spec"


def test_policy_exclude_fields_never_mutates():
    """A49 (SPEC.md 59.2): the policies' ``exclude_fields`` drop the pinned
    field from the mutation pool — the excluded field is never the mutation
    target, so the candidate always keeps the champion's value. Both policies
    store the tuple; the empty default is the pre-v0.45 stream."""
    base = DEFAULT_SPEC.to_dict()
    state = {"last": None, "best_spec": base, "task": None}
    pol = SearchPolicy(seed=7, exclude_fields=("activation",))
    for _ in range(60):
        cand = pol.propose(state)
        assert cand["activation"] == base["activation"]
    # both policies store the exclusion; the default is empty
    assert BanditPolicy(seed=7, exclude_fields=("activation",)).exclude_fields \
        == ("activation",)
    assert SearchPolicy(seed=7).exclude_fields == ()
    assert BanditPolicy(seed=7).exclude_fields == ()


# --- 59.4 runconfig round-trip (A49) ------------------------------------------

def test_runconfig_backcompat_missing_steering(tmp_path):
    """A49 (SPEC.md 59.4): an old run_config.json (missing the steering
    fields) still loads with empty lists (the _OPTIONAL_FIELDS default)."""
    d = RunConfig(task="parity-v1", seed=7).to_dict()
    del d["pins"], d["biases"], d["constraints"]
    cfg = RunConfig.from_dict(d)
    assert cfg.pins == []
    assert cfg.biases == []
    assert cfg.constraints == []


def test_runconfig_from_env_reads_steering(tmp_path):
    """A49 (SPEC.md 59.4): ``from_env`` reads ``env.steering.to_dict()`` into
    the recipe's pins / biases / constraints."""
    state = (SteeringState()
             .pin("activation", "relu")
             .bias("learning_rate", 1e-3)
             .constrain("train_steps", [200, 400]))
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=_small_budget(),
                        runs_dir=tmp_path, steering=state)
    cfg = RunConfig.from_env(env, policy="search")
    assert cfg.pins == [["activation", "relu"]]
    assert cfg.biases == [["learning_rate", 1e-3]]
    assert cfg.constraints == [["train_steps", [200, 400]]]


def test_runconfig_recipe_and_flags_emit_steering(tmp_path):
    """A49 (SPEC.md 59.4): ``fit_recipe`` emits the ``--pin/--bias/--
    constrain`` flags and ``runconfig_to_flags`` emits ``field=value``
    lists for the CLI's repeatable append flags."""
    cfg = RunConfig(
        task="parity-v1", seed=7,
        pins=[["activation", "relu"]],
        biases=[["learning_rate", 1e-3]],
        constraints=[["train_steps", [200, 400]]],
    )
    recipe = fit_recipe(cfg)
    assert "--pin" in recipe and "activation=relu" in recipe
    assert "--bias" in recipe and "learning_rate=0.001" in recipe
    assert "--constrain" in recipe and "train_steps=200,400" in recipe
    flags = runconfig_to_flags(cfg)
    assert flags["pin"] == ["activation=relu"]
    assert flags["bias"] == ["learning_rate=0.001"]
    assert flags["constrain"] == ["train_steps=200,400"]


# --- 59.4 the surfaces (app / dashboard / CLI) (A49) --------------------------

def test_cli_wires_steering_and_manual():
    """A49 (SPEC.md 59.4): the CLI wires the ``manual`` subcommand + the
    ``--pin/--bias/--constrain`` flags (the helpers, the handler, the
    epilog, the imports)."""
    src = CLI.read_text(encoding="utf-8")
    for token in (
        "from .steering import SteeringState, train_manual",
        "def _steering_from_args",
        "def _cmd_manual",
        "_EP_MANUAL",
        '"--pin"',
        '"--bias"',
        '"--constrain"',
        '"--set"',
    ):
        assert token in src, token


def test_dashboard_wires_steering():
    """A49 (SPEC.md 59.4): the dashboard accepts the ``steering`` kwarg,
    passes ``exclude_fields`` from the pins to the policies, and carries it
    in ``_sweep_params`` (the seed-sweep, 31.2)."""
    dash = DASH.read_text(encoding="utf-8")
    for token in (
        "from .steering import SteeringState",
        "steering: SteeringState | None = None",
        "exclude_fields=excl",
        "steering=self.steering",
    ):
        assert token in dash, token


def test_app_wires_inspector_steering_manual():
    """A49 (SPEC.md 59.4): the app imports the steering API, renders the
    inspector (the subheader + the row helper), the steering expander
    (verb / field), and the manual expander (button + early validation +
    train)."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "SteeringState",
        "manual_spec",
        "parameter_inspection",
        "train_manual",
        "def _inspector_field",
        '"Parameter inspector (SPEC.md 59.1)"',
        "key=\"stg_verb\"",
        "key=\"stg_field\"",
        "key=\"manual_button\"",
        "manual_spec(_mvals)",
        "train_manual(",
    ):
        assert token in src, token


# --- 59.4 the default path end-to-end (A49) ------------------------------------

def test_app_default_path_renders_inspector(tmp_path):
    """A49 (SPEC.md 59.4): the default path runs end-to-end and the Results
    tab renders the parameter inspector (AppTest-safe; no new widget)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    # a small classification CSV (quadrant XOR, 2 classes)
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    assert at.session_state["result"] is not None
    sub_headers = [str(s.value) for s in at.subheader]
    assert "Parameter inspector (SPEC.md 59.1)" in sub_headers, sub_headers


# --- A49 round regression ------------------------------------------------------

def test_version_round_v045():
    """A49 (SPEC.md 59.6, 33.1): the version stepped to ``0.45.0`` in both
    sources (v0.45 ⇒ ``0.45.0``, M48, SPEC.md 59)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"


def test_all_exports_steering_round():
    """A49 (SPEC.md 59.4, 33.1): the five new names are exported from the
    package top level and in ``__all__``."""
    for name in ("SteeringState", "apply_steering", "manual_spec",
                 "parameter_inspection", "train_manual"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_spec_cites_a49_and_round():
    """A49 (SPEC.md 59.6): SPEC.md defines the A49 acceptance block and the
    M48 index row + milestone — the A25 index machinery reads both
    (``defined == set(range(1, 50))`` includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 59.5 Acceptance (A49)" in spec
    import re
    assert re.search(r"^\s*\| M48 \| v0\.45\s*\|\s*59\s*\|\s*A49\s*\|",
                     spec, re.MULTILINE)
    assert "**M48**" in spec
