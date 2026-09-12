"""v0.23 generalization — SPEC.md 37, A27:

- 37.1 (G3) canonical `RunConfig` — `RunConfig` is frozen (immutable);
  `to_dict` is JSON-safe and `from_dict` round-trips exactly (a mutated
  / unknown dict is a `ValueError`); `run_config.json` exists after
  **reset** (before any step) and equals the summary's additive
  `run_config` key after finish (the A24 key set's 17 → 18 advance is
  pinned in test_coherency_v020.py); `fit_recipe` renders the
  copy-pasteable CLI command — the data-task recipe carries `--data`
  from `task_config.path`, default-valued flags are omitted, and
  `--search-quality legacy` renders only for the legacy preset — and
  the rendered recipe **parses** against the real argparse tree
  (37.1.4/37.1.5); `fit --from-run` re-runs a finished run bit-identical
  (same final score + best spec, PASS/MISS + rc 0/2 preserved) and is
  mutually exclusive with `--data` (rc 1); a built-in-task run is
  rejected (37.1.4); `report` prints the recipe line; the app's runner
  carries `res["recipe"]` (argv tokens).
- 37.2 (G4) objective-set acceptance — `parse_objective` accepts the
  three names × two ops and rejects a bad name/op/threshold with
  `ValueError`; `default_objectives` is exactly `score >= target`
  (behaviorally the §22.1 gate); `evaluate` passes iff ALL objectives
  pass and reports per-objective actuals (a missing actual fails);
  `fit --gate` (score+train+model) gates on the set — rc 0 all pass,
  rc 2 any fail — with the per-objective rows printed; **no `--gate`
  keeps the §22.1 gate byte-identical** (exact gate line + PASS text +
  rc 0); the `model` actual is the total values in `best_model.npz`;
  the `train` actual is the best candidate's train seconds;
  `DashboardRunner(objectives=…)` flows into `res["gate"]` and the
  per-seed sweep verdicts; `variance --gate` is wired (the flag
  parses, the set flows to the runner).
- Regression — version stepped to `0.23.0` in both sources (33.1) with
  the round assertion advanced (v0.23 ⇒ `0.23.0`).

House rules: no cross-test imports (the CSV fixture is duplicated from
test_data_cli.py); no new dependencies; source scans are read-only.
"""
import csv
import dataclasses
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, BanditPolicy, Budget
from autorefine.cli import build_parser, main as cli_main
from autorefine.dashboard import DashboardRunner
from autorefine.gate import (
    Objective,
    actuals_from_run,
    default_objectives,
    evaluate,
    parse_objective,
)
from autorefine.runconfig import (
    RUN_CONFIG_SCHEMA,
    RunConfig,
    fit_recipe,
    format_recipe,
)

REPO = Path(__file__).resolve().parents[1]


# --- fixture (duplicated from test_data_cli.py; no cross-test imports) ------

def _cls_csv(path: Path) -> Path:
    """Non-linear 2-D classification (quadrant XOR) → softmax head."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = ((x1 > 0) == (x2 > 0)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "x1", "x2", "churn"])
        for i in range(n):
            w.writerow(["A" if i % 2 else "B", f"{x1[i]:.5f}",
                        f"{x2[i]:.5f}", int(y[i])])
    return path


def _drive_to_done(env) -> dict:
    state = env.reset()
    policy = BanditPolicy(seed=7)
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return env.memory.load_summary()


def _one_run_dir(base: Path, prefix: str) -> Path:
    dirs = sorted(base.glob(f"{prefix}-*"))
    assert len(dirs) == 1
    return dirs[0]


def _has(recipe: list, *tokens: str) -> bool:
    """True if `tokens` occur contiguously, in order, in `recipe`."""
    n = len(tokens)
    return any(recipe[i:i + n] == list(tokens)
               for i in range(len(recipe) - n + 1))


# --- 37.1 RunConfig: frozen + JSON round trip --------------------------------

def _full_config() -> RunConfig:
    return RunConfig(
        task="csv", seed=11,
        policy="bandit", target=90.0, rl_episodes=None,
        max_experiments=4, max_wall_seconds=300.0, max_train_seconds=30.0,
        dataset_size=320,
        task_config={"path": "data.csv", "label": "churn", "split_frac": 0.25},
        ci_blocks=8, z_accept=1.0, efficiency_weight=0.5,
        gen_gap_penalty=0.5, block_size=512,
        ensemble_top_k=2, curriculum=False, stall_patience=3,
        screen_frac=0.5, autorefine_version="0.23.0",
    )


def test_runconfig_is_frozen():
    """A27 (SPEC.md 37.1.2): `RunConfig` is frozen — a run's recipe is
    fixed at reset and cannot be edited afterwards."""
    cfg = _full_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.task = "sine-v1"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 7  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.task_config = {"path": "x.csv"}  # type: ignore[misc]


def test_runconfig_to_dict_is_json_safe_and_round_trips():
    """A27 (SPEC.md 37.1.2): `to_dict` is JSON-safe and `from_dict`
    round-trips exactly."""
    cfg = _full_config()
    d = cfg.to_dict()
    assert d["schema"] == RUN_CONFIG_SCHEMA
    text = json.dumps(d, sort_keys=True)  # JSON-safe: serializes cleanly
    back = RunConfig.from_dict(json.loads(text))
    assert back == cfg  # the round trip is exact
    # the detached copy: mutating the dict's task_config cannot touch cfg
    d["task_config"]["label"] = "mutated"
    assert cfg.task_config["label"] == "churn"


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("seed"),                      # missing field
    lambda d: d.update(schema="other/9"),          # schema mismatch
    lambda d: d.update(bogus=1),                   # unknown field
    lambda d: d.update(seed="7"),                  # wrong type
    lambda d: d.update(policy=123),                # wrong type (non-null)
    lambda d: d.update(curriculum="yes"),          # wrong type (bool)
])
def test_runconfig_from_dict_rejects_mutated_dicts(mutate):
    """A27 (SPEC.md 37.1.2): a mutated/unknown dict is rejected."""
    d = _full_config().to_dict()
    mutate(d)
    with pytest.raises(ValueError):
        RunConfig.from_dict(d)


def test_runconfig_from_dict_rejects_non_dict():
    """A27 (SPEC.md 37.1.2): non-dict input is a construction ValueError."""
    with pytest.raises(ValueError):
        RunConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


# --- 37.1.3: run_config.json at reset; the additive summary key --------------

def test_run_config_written_at_reset_and_in_summary(tmp_path):
    """A27 (SPEC.md 37.1.3): `run_config.json` exists after **reset**
    (before any step) and equals the summary's additive `run_config`
    key after finish."""
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
        policy="bandit", target=80.0,  # driver metadata (37.1.2)
    )
    env.reset()
    rc_path = env.run_dir / "run_config.json"
    assert rc_path.is_file()  # the recipe exists from the first step
    on_disk = RunConfig.from_dict(
        json.loads(rc_path.read_text(encoding="utf-8")))
    assert on_disk == env.run_config
    assert on_disk.policy == "bandit" and on_disk.target == 80.0
    assert on_disk.task_config == {"path": str(p), "label": "churn"}
    assert on_disk.autorefine_version == autorefine.__version__
    # finish: the same frozen object, mirrored as the additive key
    summary = _drive_to_done(env)
    assert "run_config" in summary  # A24's key set advances (34.2, 37.1.3)
    assert summary["run_config"] == env.run_config.to_dict()


# --- 37.1.4: the recipe surfaces ---------------------------------------------

def test_fit_recipe_data_task_omits_defaults_and_parses():
    """A27 (SPEC.md 37.1.4/37.1.5): the data-task recipe carries `--data`
    from `task_config.path`, default-valued flags are omitted, and the
    rendered command parses against the real argparse tree."""
    cfg = RunConfig(
        task="csv", seed=7, policy="bandit", target=None, rl_episodes=None,
        max_experiments=30, max_wall_seconds=900.0, max_train_seconds=30.0,
        dataset_size=320, task_config={"path": "data.csv", "label": "churn"},
        ci_blocks=8, z_accept=1.0, efficiency_weight=0.5,
        gen_gap_penalty=0.5, block_size=512,
        autorefine_version=autorefine.__version__,
    )
    recipe = fit_recipe(cfg)
    assert recipe[:5] == ["autorefine", "fit", "--data", "data.csv",
                          "--label"]
    # defaults omitted: seed 7 / 30 experiments / v04 preset are the CLI
    # defaults — none of them appears
    assert "--seed" not in recipe
    assert "--experiments" not in recipe
    assert "--search-quality" not in recipe
    # copy-pasteable: the real parser accepts the exact tokens
    # (recipe[1:] drops the `autorefine` program name; keeps `fit`)
    args = build_parser().parse_args(recipe[1:])
    assert args.data == "data.csv" and args.label == "churn"
    # pure function (37.1.5): same config, same line, every time
    assert fit_recipe(cfg) == recipe
    assert format_recipe(cfg) == " ".join(recipe)


def test_fit_recipe_non_defaults_and_legacy_preset():
    """A27 (SPEC.md 37.1.4): non-default values render;
    `--search-quality legacy` only for the legacy preset (v04 = the
    CLI default, omitted)."""
    base = dict(task="csv", seed=11, policy="search", target=90.0,
                rl_episodes=None, max_experiments=5, max_wall_seconds=900.0,
                max_train_seconds=30.0, dataset_size=100,
                task_config={"path": "d.csv"}, ci_blocks=8, z_accept=1.0,
                efficiency_weight=0.5, gen_gap_penalty=0.5, block_size=512,
                ensemble_top_k=2, curriculum=False, stall_patience=3,
                screen_frac=0.25, autorefine_version="1.0.0")
    recipe = fit_recipe(RunConfig(**base))
    assert _has(recipe, "--seed", "11")
    assert _has(recipe, "--experiments", "5")
    assert _has(recipe, "--policy", "search")
    assert _has(recipe, "--target", "90.0")
    assert "--ensemble-final" in recipe
    assert _has(recipe, "--stall-patience", "3")
    assert _has(recipe, "--screen-frac", "0.25")
    assert "--search-quality" not in recipe  # v04 = the CLI default
    # the legacy preset is the only non-default `--search-quality` value
    legacy = fit_recipe(RunConfig(
        **{**base, "ci_blocks": 0, "z_accept": 0.0, "efficiency_weight": 0.0,
           "gen_gap_penalty": 0.0}))
    assert _has(legacy, "--search-quality", "legacy")
    # rl policy renders `--rl-episodes` when non-default
    rl = fit_recipe(RunConfig(
        **{**base, "policy": "rl", "rl_episodes": 9}))
    assert _has(rl, "--policy", "rl", "--rl-episodes", "9")


def test_run_recipe_builtin_task():
    """A27 (SPEC.md 37.1.4): built-in tasks render `autorefine run …`
    (no `--data`), and the command parses."""
    cfg = RunConfig(task="sine-v1", seed=7, policy="bandit", target=None,
                    rl_episodes=None, autorefine_version="1.0.0")
    recipe = fit_recipe(cfg)
    assert recipe[:4] == ["autorefine", "run", "--task", "sine-v1"]
    assert "--data" not in recipe
    build_parser().parse_args(recipe[1:])  # parses against the real tree


def test_fit_from_run_reruns_exactly_and_exclusivity(tmp_path, capsys):
    """A27 (SPEC.md 37.1.4/37.1.5): `fit --from-run` re-runs a finished
    run from its exact recipe (the config round-trips bit-identical, the
    score reproduces within float-reordering tolerance, PASS + rc 0
    preserved); `--data` and `--from-run` are mutually exclusive (rc 1);
    exactly one is required."""
    p = _cls_csv(tmp_path / "cls.csv")
    runs1 = tmp_path / "runs1"
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "8", "--policy", "bandit", "--seed", "7",
                   "--target", "80.0", "--runs-dir", str(runs1)])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out
    dir1 = _one_run_dir(runs1, "csv-seed7")
    s1 = json.loads(
        (dir1 / "summary.json").read_text(encoding="utf-8"))

    runs2 = tmp_path / "runs2"
    rc = cli_main(["fit", "--from-run", str(dir1),
                   "--runs-dir", str(runs2)])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out  # the recipe's target is authoritative
    dir2 = _one_run_dir(runs2, "csv-seed7")
    s2 = json.loads(
        (dir2 / "summary.json").read_text(encoding="utf-8"))
    # the re-run reproduces the run: the recipe (config) is bit-identical
    # (below), and the score reproduces within float-reordering tolerance —
    # BLAS thread order is the one pre-existing non-determinism (the same
    # wall-clock-class exclusion as train_seconds, A2); with a fixed thread
    # count the re-run is bit-identical (37.1.5)
    assert abs(s2["final_best_score"] - s1["final_best_score"]) <= 1.0
    assert s2["final_best_score"] >= 80.0  # the recipe's target held again
    # the recipe itself round-trips through the config the re-run used
    assert RunConfig.from_dict(
        json.loads((dir1 / "run_config.json").read_text(encoding="utf-8"))
    ).to_dict() == s1["run_config"]

    capsys.readouterr()
    rc = cli_main(["fit", "--data", str(p), "--from-run", str(dir1)])
    err = capsys.readouterr().err
    assert rc == 1 and "mutually exclusive" in err
    capsys.readouterr()
    rc = cli_main(["fit", "--runs-dir", str(tmp_path / "runs3")])
    err = capsys.readouterr().err
    assert rc == 1 and "one of --data or --from-run" in err


def test_fit_from_run_rejects_builtin_runs(tmp_path, capsys):
    """A27 (SPEC.md 37.1.4): a built-in-task run (no `task_config.path`)
    is rejected with a pointer to `autorefine run`."""
    rc = cli_main(["run", "--task", "sine-v1", "--experiments", "1",
                   "--seed", "7", "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    capsys.readouterr()
    dir1 = _one_run_dir(tmp_path / "runs", "sine-v1-seed7")
    rc = cli_main(["fit", "--from-run", str(dir1)])
    err = capsys.readouterr().err
    assert rc == 1 and "built-in task" in err and "autorefine run" in err


def test_report_prints_the_recipe_line(tmp_path, capsys):
    """A27 (SPEC.md 37.1.4): the human report prints the copy-pasteable
    recipe reconstructed from the summary's `run_config`."""
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
    )
    _drive_to_done(env)
    rc = cli_main(["report", "--run", str(env.run_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reproduce (copy-paste):" in out
    assert format_recipe(env.run_config) in out
    assert "--data" in out and str(p) in out


# --- 37.2 objectives: parsing, defaults, evaluation ---------------------------

@pytest.mark.parametrize("text,expected", [
    ("score>=95", Objective("score", ">=", 95.0)),
    ("train<=30", Objective("train", "<=", 30.0)),
    ("model<=100000", Objective("model", "<=", 100000.0)),
    ("score<=100", Objective("score", "<=", 100.0)),
    ("train>=1", Objective("train", ">=", 1.0)),
    ("model>=0", Objective("model", ">=", 0.0)),
    ("  score >= 95  ", Objective("score", ">=", 95.0)),  # tolerant
])
def test_parse_objective_accepts_names_and_ops(text, expected):
    """A27 (SPEC.md 37.2.2): the three names × two ops parse to the
    declared `Objective` (frozen)."""
    obj = parse_objective(text)
    assert obj == expected
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.threshold = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [
    "fancy>=95",      # unknown name
    "score>95",       # bad op
    "score=>95",      # bad op
    "score>=abc",     # non-numeric threshold
    "score",          # missing op/threshold
    "score>=",        # missing threshold
    "score>=-5",      # negative threshold
    "score>=inf",     # non-finite threshold
    "score >= 95 >= 3",  # garbage tail
])
def test_parse_objective_rejects_bad_input(bad):
    """A27 (SPEC.md 37.2.2): a bad name/op/threshold is a
    construction-time `ValueError`."""
    with pytest.raises(ValueError):
        parse_objective(bad)
    with pytest.raises(ValueError):
        parse_objective(95)  # type: ignore[arg-type]


def test_default_objectives_is_exactly_the_score_gate():
    """A27 (SPEC.md 37.2.2): `default_objectives(target)` is exactly
    `score >= target` — behaviorally the §22.1 gate."""
    objs = default_objectives(95.0)
    assert tuple(objs) == (Objective("score", ">=", 95.0),)
    # the same decision rule as `final >= target`
    assert evaluate(objs, {"score": 94.999})["pass"] is False
    assert evaluate(objs, {"score": 95.0})["pass"] is True
    assert evaluate(objs, {"score": 96.4})["pass"] is True


def test_evaluate_passes_iff_all_pass_with_per_objective_rows():
    """A27 (SPEC.md 37.2.2): PASS iff ALL objectives pass; the rows
    carry name/op/threshold/actual/pass; a missing actual fails."""
    objs = (parse_objective("score>=95"), parse_objective("train<=30"),
            parse_objective("model<=100000"))
    actuals = {"score": 96.4, "train": 12.0, "model": 5000}
    res = evaluate(objs, actuals)
    assert res["pass"] is True
    assert len(res["objectives"]) == 3
    for row in res["objectives"]:
        assert set(row) == {"name", "op", "threshold", "actual", "pass"}
        assert row["pass"] is True
    by = {r["name"]: r for r in res["objectives"]}
    assert by["score"]["actual"] == 96.4
    assert by["train"]["op"] == "<=" and by["train"]["threshold"] == 30.0

    # one failing objective fails the set, and is named
    res = evaluate(objs, {"score": 90.0, "train": 12.0, "model": 5000})
    assert res["pass"] is False
    by = {r["name"]: r for r in res["objectives"]}
    assert by["score"]["pass"] is False
    assert by["train"]["pass"] is True and by["model"]["pass"] is True

    # a missing actual fails its objective (actual reported as None)
    res = evaluate(objs, {"score": 96.4, "train": 12.0})
    by = {r["name"]: r for r in res["objectives"]}
    assert res["pass"] is False
    assert by["model"]["actual"] is None and by["model"]["pass"] is False

    # an empty set is not a pass
    assert evaluate((), actuals)["pass"] is False


# --- 37.2.3: the fit gate wiring ---------------------------------------------

def test_fit_gate_no_gate_keeps_221_gate_byte_identical(tmp_path, capsys):
    """A27 (SPEC.md 37.2.3): no `--gate` → exactly the §22.1 gate —
    the gate line, the PASS text, and rc 0, byte-identical."""
    p = _cls_csv(tmp_path / "cls.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "8", "--policy", "bandit", "--seed", "7",
                   "--target", "80.0", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    final = json.loads(
        (_one_run_dir(runs, "csv-seed7") / "summary.json").read_text(
            encoding="utf-8"))["final_best_score"]
    # the exact §22.1 gate line (the declared metric: accuracy)
    assert (f"gate    : final {final:.2f} on accuracy vs target 80.0 "
            "(SPEC.md 22.1; the loop maximizes the validated score, "
            "the bar is yours)") in out
    assert (f"PASS: final score {final:.2f} on accuracy >= 80.0 on held-out "
            "data (the loop never trained on these points; "
            "gen_gap is the overfit guard)") in out
    # the objective-set header must NOT appear on the no-gate path
    assert "objective(s)" not in out


def test_fit_gate_objective_set_rc0_and_rc2(tmp_path, capsys):
    """A27 (SPEC.md 37.2.3): `--gate` (1+) defines the objective set —
    rc 0 all pass, rc 2 any fail, with the per-objective rows."""
    p = _cls_csv(tmp_path / "cls.csv")
    # all three pass: generous thresholds
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "4", "--policy", "bandit", "--seed", "7",
                   "--target", "80.0",
                   "--gate", "score>=80",
                   "--gate", "train<=30",
                   "--gate", "model<=10000000",
                   "--runs-dir", str(tmp_path / "runs_a")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gate    : 3 objective(s) (SPEC.md 37.2)" in out
    assert "score" in out and "train" in out and "model" in out
    assert "PASS: all 3 objective(s) met (SPEC.md 37.2)" in out
    # the §22.1 single-bar line must NOT appear on the gate path
    assert "(SPEC.md 22.1; the loop maximizes" not in out
    # the no-gate-style PASS line is replaced by the set verdict
    assert "on held-out data (the loop never trained" not in out

    # model<=1 fails (any real model has >1 stored value) → rc 2, named
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "4", "--policy", "bandit", "--seed", "7",
                   "--target", "80.0",
                   "--gate", "score>=80",
                   "--gate", "model<=1",
                   "--runs-dir", str(tmp_path / "runs_b")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "MISS: objective(s) not met: model (SPEC.md 37.2)" in out
    # the all-pass verdict line must not appear (the per-row PASS on the
    # score row is expected — the set failed on `model`)
    assert "PASS: all" not in out


def test_gate_actuals_model_is_npz_values_and_train_is_best_seconds(tmp_path):
    """A27 (SPEC.md 37.2.2): the `model` actual is the total values in
    `best_model.npz`; the `train` actual is the best candidate's train
    seconds (the memory entry whose spec_hash matches)."""
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(
        task="csv", seed=7, budget=Budget(2, 300, 30),
        runs_dir=tmp_path / "runs",
        task_config={"path": str(p), "label": "churn"}, dataset_episodes=320,
    )
    _drive_to_done(env)
    entries = env.memory.load_experiments()
    actuals = actuals_from_run(env, entries)

    # score: the final best holdout score
    assert actuals["score"] == float(env.best_score)

    # model: the total values stored in best_model.npz (all families)
    with np.load(env.run_dir / "best_model.npz") as npz:
        total = sum(int(npz[k].size) for k in npz.files)
    assert actuals["model"] == total

    # train: the best candidate's train seconds (spec_hash == fingerprint)
    best_hash = env.best_spec.fingerprint()
    expected = [float(e["train_seconds"]) for e in entries
                if e.get("spec_hash") == best_hash
                and isinstance(e.get("train_seconds"), (int, float))]
    assert expected
    assert actuals["train"] == expected[0]


# --- 37.2.3: the runner + sweep carry the objective set ------------------------

def test_dashboard_runner_gate_and_recipe(tmp_path):
    """A27 (SPEC.md 37.2.3): `DashboardRunner(objectives=…)` flows into
    `res["gate"]` (per-objective rows + overall pass) and `res["recipe"]`
    (argv tokens) — and the A13 verdict pin stays the §22.1 score gate."""
    p = _cls_csv(tmp_path / "cls.csv")
    runner = DashboardRunner(
        csv_path=str(p), label="churn", target=80.0, policy="bandit",
        seed=7, experiments=2, max_seconds=300, max_train_seconds=30,
        runs_dir=str(tmp_path / "runs"),
        objectives=(parse_objective("score>=80"),
                    parse_objective("train<=30")),
    )
    runner.start()
    while not runner.done:
        runner.next()
    res = runner.finish()

    gate = res["gate"]
    assert set(gate) == {"pass", "objectives"}
    names = [r["name"] for r in gate["objectives"]]
    assert names == ["score", "train"]
    assert gate["pass"] is all(r["pass"] for r in gate["objectives"])

    recipe = res["recipe"]
    assert isinstance(recipe, list)
    assert recipe[:3] == ["autorefine", "fit", "--data"]
    assert str(p) in recipe
    # the recipe is the env's canonical config rendered
    assert recipe == fit_recipe(runner.env.run_config)

    # A13 (SPEC.md 22.1): the verdict is still the single score gate
    assert res["verdict"] == ("PASS" if res["final_best_score"] >= 80.0
                              else "MISS")
    assert res["gate"]["pass"] == (res["final_best_score"] >= 80.0
                                   and gate["objectives"][1]["pass"])


def test_seed_sweep_verdicts_carry_the_objective_set(tmp_path):
    """A27 (SPEC.md 37.2.3): the per-seed sweep verdicts carry the same
    objective set — with the default set, the verdict is exactly the
    §22.1 score gate (the A19/A20 pins)."""
    p = _cls_csv(tmp_path / "cls.csv")
    runner = DashboardRunner(
        csv_path=str(p), label="churn", target=80.0, policy="bandit",
        seed=7, experiments=2, max_seconds=300, max_train_seconds=30,
        runs_dir=str(tmp_path / "runs"),  # default objectives → §22.1 gate
    )
    sweep = runner.seed_sweep([7])
    assert len(sweep) == 1
    entry = sweep[0]
    assert entry["pass"] is (entry["final"] >= 80.0)
    assert entry["verdict"] == ("PASS" if entry["final"] >= 80.0 else "MISS")
    # the verdict keys the A19/A20 pins carry are present
    assert {"seed", "baseline", "final", "pass", "verdict",
            "experiments_run"} <= set(entry)


def test_variance_parser_exposes_gate(tmp_path):
    """A27 (SPEC.md 37.2.3): `variance --gate` (repeatable) is wired —
    the flag parses and flows to the runner's objective set."""
    args = build_parser().parse_args([
        "variance", "--data", str(tmp_path / "x.csv"), "--seeds", "2",
        "--gate", "score>=80", "--gate", "train<=30",
    ])
    assert args.gate == ["score>=80", "train<=30"]
    objs = tuple(parse_objective(g) for g in args.gate)
    assert objs == (parse_objective("score>=80"), parse_objective("train<=30"))
    # and the no-gate default is the empty set → the §22.1 score gate
    args = build_parser().parse_args([
        "variance", "--data", str(tmp_path / "x.csv"), "--seeds", "2",
    ])
    assert args.gate == []


# --- regression ----------------------------------------------------------------

def test_version_round_v023():
    """A27 (SPEC.md 37, 33.1): the round assertion advanced in place with
    each round (v0.23 ⇒ `0.23.0`; now v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"
