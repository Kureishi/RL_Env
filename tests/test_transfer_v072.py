"""Transfer to similar tasks — the search prior (SPEC.md 86, A76).

SPEC.md 86 (v0.72): a finished run leaves behind its *search knowledge* —
the best spec it found and the per-field trial/win credit of its search.
Carrying that to a new run on a **similar** task (new data, possibly
different input dimensions) means the new run starts informed:

  * 86.1  — ``transfer.py``: the frozen ``SearchPrior`` (to_dict /
    from_dict round-trip, ``to_bandit_prior``), ``read_search_prior``
    (a pure reader over ``summary.json`` + ``best_spec.json``; finished
    runs only; fail-loud ``ValueError`` on a missing / malformed
    artifact), ``prior_spec``;
  * 86.2  — ``BanditPolicy(prior=...)`` seeds the trial/win tables —
    a field with prior credit has a finite UCB, a zero-trial field
    keeps the cold-start priority; ``prior=None`` is the bit-identical
    legacy proposal stream (A1–A4 pins, G2);
  * 86.3  — ``AutoRefineEnv(prior_run=...)``: the baseline is the
    prior's best spec **retrained on the current task** (the baseline
    row and the summary carry the conditional ``from_prior_run`` key);
    pins override the prior's fields; mutually exclusive with
    ``initial_model``; a bad directory is a fail-loud ``ValueError``;
    the default (``prior_run=None``) keeps the exact legacy reset;
  * 86.4  — the CLI ``--prior-run`` on ``run`` / ``fit`` with the
    mutual exclusions (× ``--from-model``, × ``--from-run``,
    × ``--tasks``) refusing at rc 1;
  * 86.5  — ``RunConfig.prior_run`` (optional in ``from_dict`` —
    pre-v0.72 configs load; ``fit_recipe`` / ``runconfig_to_flags``
    emit the flag only when set);
  * 86.6  — acceptance: the reader, the bandit, the env, the recipe,
    the CLI, and the version step / A-index advance.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    BanditPolicy,
    Budget,
    DEFAULT_SPEC,
    ModelSpec,
    SearchPolicy,
)
from autorefine.cli import main as cli_main
from autorefine.runconfig import RunConfig, fit_recipe, runconfig_to_flags
from autorefine.steering import SteeringState
from autorefine.transfer import SearchPrior, prior_spec, read_search_prior

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"


# --- shared fixtures (test-local; the house rules forbid cross-test imports) ---

def _drive(env: AutoRefineEnv, policy) -> None:
    """Run the loop to completion (the A70 loop pattern)."""
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))


def _read_rows(env: AutoRefineEnv) -> list[dict]:
    lines = (env.run_dir / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _prior_run(tmp_path: Path) -> Path:
    """A real, tiny, **finished** parity-v1 run — the prior's source."""
    env = AutoRefineEnv(
        task="parity-v1", seed=1,
        budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "source")
    _drive(env, SearchPolicy(seed=1))
    assert env.done
    return env.run_dir


# --- 86.1 the reader (A76.1) -----------------------------------------------------

class TestReader:
    """SPEC.md 86.1 (A76.1): ``read_search_prior`` folds a finished
    run's two artifacts into a frozen, round-trippable prior."""

    def test_reader_returns_the_run_knowledge(self, tmp_path):
        src = _prior_run(tmp_path)
        summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        p = read_search_prior(src)
        assert isinstance(p, SearchPrior)
        assert p.task == "parity-v1"
        assert p.source_run == src.name
        assert p.best_score == summary["final_best_score"]
        assert p.n_experiments == summary["experiments_run"]
        # the best spec is the run's best spec and a valid ModelSpec
        assert p.best_spec == summary["best_spec"]
        spec = prior_spec(p)
        assert isinstance(spec, ModelSpec)
        # per-field credit is the summary's canonical mutation_win_rate
        assert set(p.trials) == set(p.wins) == set(summary["mutation_win_rate"])
        for field, slot in summary["mutation_win_rate"].items():
            assert p.trials[field] == slot["trials"]
            assert p.wins[field] == float(slot["wins"])

    def test_two_reads_are_bit_identical(self, tmp_path):
        src = _prior_run(tmp_path)
        assert read_search_prior(src).to_dict() == read_search_prior(src).to_dict()

    def test_missing_summary_is_a_fail_loud_value_error(self, tmp_path):
        src = _prior_run(tmp_path)
        d = tmp_path / "half"
        d.mkdir()
        (d / "best_spec.json").write_text(
            (src / "best_spec.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        with pytest.raises(ValueError, match="summary.json"):
            read_search_prior(d)

    def test_missing_best_spec_is_a_fail_loud_value_error(self, tmp_path):
        src = _prior_run(tmp_path)
        d = tmp_path / "half2"
        d.mkdir()
        (d / "summary.json").write_text(
            (src / "summary.json").read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(ValueError, match="best_spec.json"):
            read_search_prior(d)

    def test_malformed_artifacts_are_fail_loud_value_errors(self, tmp_path):
        d1 = tmp_path / "bad_summary"
        d1.mkdir()
        (d1 / "summary.json").write_text("[1, 2, 3]", encoding="utf-8")
        (d1 / "best_spec.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_search_prior(d1)

        d2 = tmp_path / "keyless_summary"
        d2.mkdir()
        (d2 / "summary.json").write_text('{"task": "parity-v1"}',
                                        encoding="utf-8")
        (d2 / "best_spec.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            read_search_prior(d2)

    def test_to_dict_from_dict_round_trips(self, tmp_path):
        src = _prior_run(tmp_path)
        p = read_search_prior(src)
        assert SearchPrior.from_dict(p.to_dict()) == p
        d = p.to_dict()
        for field in d["trials"]:
            assert isinstance(d["trials"][field], int)
            assert isinstance(d["wins"][field], float)

    def test_from_dict_rejects_a_bad_shape(self):
        with pytest.raises(ValueError, match="JSON object"):
            SearchPrior.from_dict([1, 2, 3])
        with pytest.raises(ValueError, match="missing"):
            SearchPrior.from_dict({"task": "t"})

    def test_to_bandit_prior_is_the_documented_shape(self, tmp_path):
        src = _prior_run(tmp_path)
        p = read_search_prior(src)
        bp = p.to_bandit_prior()
        assert set(bp) == set(p.trials) | set(p.wins)
        for field, slot in bp.items():
            assert set(slot) == {"trials", "wins"}
            assert isinstance(slot["trials"], int)
            assert isinstance(slot["wins"], float)


# --- 86.2 the bandit prior (A76.2) ------------------------------------------------

class TestBanditPrior:
    """SPEC.md 86.2 (A76.2): ``prior`` seeds the trial/win tables;
    ``prior=None`` is the exact legacy proposal stream (G2)."""

    def test_prior_seeds_the_trial_win_tables(self):
        pol = BanditPolicy(seed=3,
                           prior={"learning_rate": {"trials": 3, "wins": 2.0}})
        assert pol.trials["learning_rate"] == 3
        assert pol.wins["learning_rate"] == 2.0
        # the untouched fields stay at the cold-start zero
        assert pol.trials["batch_size"] == 0 and pol.wins["batch_size"] == 0.0

    def test_prioired_field_is_finite_untried_field_is_cold_start(self):
        pol = BanditPolicy(seed=0,
                           prior={"learning_rate": {"trials": 3, "wins": 2.0}})
        assert math.isfinite(pol._ucb("learning_rate"))
        untried = [f for f in pol.trials if pol.trials[f] == 0]
        assert untried  # the rest of the space is still cold
        assert all(math.isinf(pol._ucb(f)) for f in untried)

    def test_prior_none_is_the_bit_identical_legacy_stream(self):
        """A76.2: ``prior=None`` (and a no-op prior — empty, or fields the
        bandit does not know) leaves the proposal stream bit-identical
        (the A1–A4 pins stay green, G2)."""
        pols = [
            BanditPolicy(seed=5),
            BanditPolicy(seed=5, prior={}),
            BanditPolicy(seed=5, prior={
                "nonexistent_field": {"trials": 4, "wins": 3.0}}),
        ]
        states = [{"best_spec": DEFAULT_SPEC.to_dict(), "last": None}
                  for _ in pols]
        for _ in range(8):
            specs = [p.propose(s) for p, s in zip(pols, states)]
            norm = [ModelSpec.from_dict(d).to_dict() for d in specs]
            assert norm[0] == norm[1] == norm[2]
            states = [{"best_spec": s,
                       "last": {"accepted": False, "fields": []}}
                      for s in specs]
        # the unknown field never leaked into the tables
        assert all(v == 0 for v in pols[2].trials.values())


# --- 86.3 the env prior (A76.3) -----------------------------------------------------

class TestEnvPrior:
    """SPEC.md 86.3 (A76.3): the baseline is the prior's best spec,
    **retrained on the current task**; the legacy path is untouched."""

    def test_prior_run_baseline_is_the_priors_best_spec(self, tmp_path):
        src = _prior_run(tmp_path)
        prior = read_search_prior(src)
        env = AutoRefineEnv(
            task="parity-v1", seed=3,
            budget=Budget(2, 300, 30),
            runs_dir=tmp_path / "new", prior_run=src)
        assert env.prior == prior and env.prior_run_path == str(src)
        _drive(env, SearchPolicy(seed=3))
        assert env.done
        base = _read_rows(env)[0]
        assert base["kind"] == "baseline"
        # the baseline spec is the prior's best spec — retrained, so a
        # fresh holdout score and a real train time
        assert base["spec"] == prior.best_spec
        assert math.isfinite(base["holdout_score"])
        assert base["train_seconds"] > 0.0
        # provenance: the baseline row and the summary name the source
        assert base["from_prior_run"] == src.name
        summary = json.loads(
            (env.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["from_prior_run"] == src.name

    def test_pins_override_the_priors_fields(self, tmp_path):
        src = _prior_run(tmp_path)
        env = AutoRefineEnv(
            task="parity-v1", seed=3,
            budget=Budget(2, 300, 30),
            runs_dir=tmp_path / "pinned",
            steering=SteeringState(pins=[("learning_rate", 0.01)]),
            prior_run=src)
        _drive(env, SearchPolicy(seed=3))
        base = _read_rows(env)[0]
        assert base["spec"]["learning_rate"] == 0.01  # the pin wins
        assert base["from_prior_run"] == src.name     # …over the prior's spec

    def test_prior_and_initial_model_are_mutually_exclusive(self, tmp_path):
        src = _prior_run(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            AutoRefineEnv(
                task="parity-v1", seed=7, runs_dir=tmp_path / "both",
                initial_model="m.npz", prior_run=src)

    def test_bad_prior_dir_is_a_fail_loud_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not a finished run"):
            AutoRefineEnv(
                task="parity-v1", seed=7, runs_dir=tmp_path / "bad",
                prior_run=tmp_path / "nope")

    def test_default_run_keeps_the_legacy_reset(self, tmp_path):
        """A76.3: ``prior_run=None`` is the exact pre-v0.72 reset — the
        baseline spec is ``DEFAULT_SPEC`` and no prior keys appear."""
        env = AutoRefineEnv(
            task="parity-v1", seed=7,
            budget=Budget(2, 300, 30),
            runs_dir=tmp_path / "legacy")
        assert env.prior is None and env.prior_run_path is None
        _drive(env, SearchPolicy(seed=7))
        assert env.done
        base = _read_rows(env)[0]
        assert base["spec"] == DEFAULT_SPEC.to_dict()
        assert "from_prior_run" not in base
        summary = json.loads(
            (env.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert "from_prior_run" not in summary


# --- 86.5 the recipe (A76.4) --------------------------------------------------------

def _base_config(**over) -> RunConfig:
    return RunConfig(task="parity-v1", seed=7, **over)


def test_runconfig_round_trips_prior_run():
    """A76.4 (86.5): ``prior_run`` round-trips; an absent key (a
    pre-v0.72 config) loads as None (the 59.4 back-compat pattern)."""
    cfg = _base_config(prior_run="runs/source")
    d = cfg.to_dict()
    assert d["prior_run"] == "runs/source"
    assert RunConfig.from_dict(d).prior_run == "runs/source"

    legacy = dict(d)
    del legacy["prior_run"]
    assert RunConfig.from_dict(legacy).prior_run is None
    assert _base_config().prior_run is None


def test_recipe_and_flags_emit_prior_run_only_when_set():
    """A76.4 (86.5): ``fit_recipe`` appends ``--prior-run`` and
    ``runconfig_to_flags`` emits the kebab key only when set — a fresh
    run's recipe is unchanged."""
    set_cfg = _base_config(prior_run="runs/source")
    fresh = _base_config()

    recipe = fit_recipe(set_cfg)
    assert "--prior-run" in recipe
    assert "runs/source" in recipe

    assert runconfig_to_flags(set_cfg)["prior-run"] == "runs/source"
    assert "prior-run" not in runconfig_to_flags(fresh)
    assert "--prior-run" not in fit_recipe(fresh)


# --- 86.4 the CLI (A76.5) -------------------------------------------------------------

def _help_text(*argv: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            cli_main(list(argv))
        except SystemExit:
            pass
    return buf.getvalue()


def test_prior_run_flag_is_in_run_and_fit_help():
    """A76.5 (86.4.1): ``--prior-run`` exists on both ``run`` and
    ``fit``."""
    assert "--prior-run" in _help_text("run", "--help")
    assert "--prior-run" in _help_text("fit", "--help")


def test_run_prior_x_from_model_refuses(tmp_path, capsys):
    rc = cli_main(["run", "--task", "parity-v1",
                   "--prior-run", "src", "--from-model", "m.npz"])
    assert rc == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_fit_prior_x_tasks_refuses(tmp_path, capsys):
    rc = cli_main(["fit", "--tasks", "a.csv,b.csv", "--prior-run", "src"])
    assert rc == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_fit_prior_x_from_run_refuses(tmp_path, capsys):
    rc = cli_main(["fit", "--from-run", "old", "--prior-run", "src"])
    assert rc == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_fit_prior_x_from_model_refuses(tmp_path, capsys):
    rc = cli_main(["fit", "--data", "x.csv",
                   "--from-model", "m.npz", "--prior-run", "src"])
    assert rc == 1
    assert "mutually exclusive" in capsys.readouterr().err


# --- 86.6 language / version / index (A76.6) -------------------------------------------

class TestCeremony:
    """SPEC.md 86.6 (A76): the version step, the SPEC section, the index
    row, and the public exports (the app-language scan is covered by
    ``tests/test_app_language.py``)."""

    def test_version_steps_to_v072(self):
        py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert py["project"]["version"] == autorefine.__version__ == "0.74.0"
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '"0.74.0"' in init

    def test_spec_section_and_index_row(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "## 86." in text
        assert "86.6 Acceptance (A76)" in text
        row = next(l for l in text.splitlines() if l.startswith("| M75 |"))
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert cells[1] == "v0.72" and "A76" in cells[3]
        assert "tests/test_transfer_v072.py" in cells[4]

    def test_exports_are_public(self):
        for name in ("SearchPrior", "read_search_prior"):
            assert hasattr(autorefine, name), f"autorefine.{name} missing"
            assert name in autorefine.__all__
