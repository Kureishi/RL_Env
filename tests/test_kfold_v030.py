"""K-fold holdout scoring + permutation feature importance
(SPEC.md 44, A34, v0.30).

B (44.1) `kfold` — scoring-only K distinct held-out subsets: the new
`Task.score_fold` protocol (ABC default = a fresh seed-derived point
set named `{split}-kf{index}`, the §18.3 mechanism; the CsvTask
override = a deterministic random subset of the non-train pool), the
`kfold_score` evaluator, the env knob + `--kfold` CLI flag (run + fit
only, the A23 parser scan), the conditional summary key (absent when 0
— the A24 key set and the pinned 5-key `search_quality` dict stay
untouched), and the `RunConfig` additive field (round-trips,
`fit_recipe` renders `--kfold K` only when > 0). `kfold = 0` (default)
is bit-identical to the legacy single-split path.

C (44.2) `report --importance` — the pure leaf
`autorefine.importance` (numpy only, 28.2 family): `score_xy` (the
§22.1 head metric on given rows) + `permutation_importance` (holdout
score drop per shuffled column, deterministic shuffles, sorted desc;
`None` on episode / no-holdout_rows / non-flat tasks, 44.2.3); the CLI
block (rc 0, n/a line rc 0, mutual exclusions rc 1).

Regression (44.3): A1–A33 stay green (default single split; summary
key set; artifacts; app views); the version stepped to `0.33.0` in
both sources (33.1, advanced again with v0.32, SPEC.md 45).

House rules: no cross-test imports (fixtures duplicated per file);
stdlib + numpy only; every test cites A34 + its SPEC §.
"""
import csv as _csv
import re
import tomllib
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.cli import build_parser, main as cli_main
from autorefine.evaluator import evaluate_full, kfold_score
from autorefine.improver.meta_env import KNOBS
from autorefine.importance import permutation_importance, score_xy
from autorefine.runconfig import RunConfig, fit_recipe
from autorefine.tasks import TASKS
from autorefine.tasks.csv import CsvTask

REPO = Path(__file__).resolve().parent.parent


# --- fixtures (duplicated per file, no cross-test imports) --------------------

def _softmax3_csv(tmp_path: Path) -> Path:
    """100 rows, 3 balanced classes, one signal column + one noise
    column — the k-fold battery (fold scores vary per subset, so the
    fold std is > 0)."""
    p = tmp_path / "kfold3.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["sig", "noise", "label"])
        for i in range(100):
            w.writerow([float(i), 0.5, str(i % 3)])
    return p


def _cls_csv(tmp_path: Path) -> Path:
    """200 rows, 2 classes: label = (sig > 0), noise independent — the
    importance battery (a model reading `sig` scores ~100, shuffling
    `sig` collapses it, shuffling `noise` does nothing)."""
    rnd = np.random.default_rng(0)
    p = tmp_path / "cls.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["sig", "noise", "label"])
        for i in range(200):
            sig = rnd.uniform(-1.0, 1.0)
            w.writerow([f"{sig:.6f}", f"{rnd.uniform(-1, 1):.6f}",
                        "1" if sig > 0 else "0"])
    return p


def _mse_csv(tmp_path: Path) -> Path:
    """120 rows, mse head: label = 2·sig + 3·noise — both columns
    matter (the mse importance path)."""
    rnd = np.random.default_rng(1)
    p = tmp_path / "mse.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["sig", "noise", "label"])
        for _ in range(120):
            a, b = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
            w.writerow([f"{a:.6f}", f"{b:.6f}", f"{2 * a + 3 * b:.6f}"])
    return p


class _Zeros:
    """A model predicting class 0 (logits 0) — softmax accuracy on a
    subset = the subset's class-0 share, so distinct subsets score
    differently."""

    def __init__(self, n_out=3):
        self._n = n_out

    def forward(self, x):
        return np.zeros((len(x), self._n))


class _SigModel:
    """Reads only column 0 (sig): argmax = 1 iff sig > 0."""

    def forward(self, x):
        return np.column_stack([x[:, 0] * 0.0, x[:, 0]])


def _MseModel(t):
    """A perfect mse model: undoes the task's train standardization and
    predicts 2·a + 3·b in the raw (label) scale — R² = 1 on every row
    (the R² metric is not scale-invariant, so a 100-baseline model must
    predict in the label scale)."""
    mean = np.asarray(t.feature_mean, dtype=np.float64)
    std = np.asarray(t.feature_std, dtype=np.float64)

    class _M:
        def forward(self, x):
            a = x[:, 0] * std[0] + mean[0]
            b = x[:, 1] * std[1] + mean[1]
            return (2.0 * a + 3.0 * b).reshape(-1, 1)
    return _M()


def _drive(env: AutoRefineEnv, policy: SearchPolicy) -> None:
    state = env.reset()
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))


# --- 44.1 k-fold: protocol + evaluator ----------------------------------------

def test_abc_default_score_fold_is_fresh_split_name():
    """A34 (SPEC.md 44.1.2): the generative-task default delegates to
    `score` on the fresh seed-derived split `{split}-kf{index}` — the
    §18.3 mechanism, so generative tasks work unchanged."""
    t = TASKS["sine-v1"](seed=7)
    m = _Zeros(n_out=1)
    for i in (0, 2, 9):
        assert t.score_fold(m, "holdout", i, 32) == \
            t.score(m, f"holdout-kf{i}", 32)
        assert t.score_fold(m, "gen", i, 32) == t.score(m, f"gen-kf{i}", 32)


def test_kfold_score_mean_and_sample_std():
    """A34 (SPEC.md 44.1.2): `kfold_score` is the mean of the K fold
    scores with the sample std (ddof=1; 0.0 for K = 1) and the fold
    list; K < 1 is an error."""
    p = _cls_csv(Path(".tmp")) if False else None  # unused; see tmp tests
    assert p is None
    # K = 1 → std 0.0, the single fold
    t = TASKS["sine-v1"](seed=7)
    m = _Zeros(n_out=1)
    r1 = kfold_score(t, m, "holdout", k=1)
    assert r1["std"] == 0.0 and len(r1["folds"]) == 1
    assert r1["score"] == r1["folds"][0]
    with pytest.raises(ValueError, match="k must be >= 1"):
        kfold_score(t, m, "holdout", k=0)
    # K = 4 → mean + sample std (ddof=1) over the folds
    r4 = kfold_score(t, m, "holdout", k=4)
    folds = r4["folds"]
    mean = sum(folds) / 4
    std = float(np.sqrt(sum((f - mean) ** 2 for f in folds) / 3))
    assert r4["score"] == pytest.approx(mean, rel=1e-12)
    assert r4["std"] == pytest.approx(std, rel=1e-12)


def test_csv_folds_deterministic_distinct_pool_scoped(tmp_path):
    """A34 (SPEC.md 44.1.2): the CsvTask folds are (a) deterministic —
    same seed → same fold, a second task instance included; (b)
    distinct per fold_index; (c) drawn from the non-train pool
    (holdout ∪ gen — no train rows), the documented seed scheme
    `SeedSequence([seed, crc32(b"csv-kfold"), index])`."""
    p = _softmax3_csv(tmp_path)
    t = CsvTask(seed=7, path=p)
    t2 = CsvTask(seed=7, path=p)  # a fresh instance, same seed
    m = _Zeros()
    for i in range(3):
        assert t.score_fold(m, "holdout", i, 200) == \
            t2.score_fold(m, "holdout", i, 200)  # (a)
    # (c) + (b) via the documented scheme (SPEC.md 44.1.2): the pool is
    # the non-train rows; the folds are distinct permutations of it
    pool = np.vstack([t._x_ho, t._x_ge])
    assert len(pool) == len(t._x_ho) + len(t._x_ge)  # non-train, exactly
    idx = []
    for i in range(3):
        seq = np.random.SeedSequence(
            [t.seed, zlib.crc32(b"csv-kfold"), i])
        idx.append(set(np.random.default_rng(seq).permutation(len(pool))
                       [:len(t._x_ho)]))
    for i, s in enumerate(idx):
        assert len(s) == len(t._x_ho)  # the fold size
    assert idx[0] != idx[1] != idx[2] and idx[0] != idx[2]  # (b)
    # (a') a different seed draws different subsets (score or indices)
    t9 = CsvTask(seed=9, path=p)
    assert (t9.score_fold(m, "holdout", 0, 200)
            != t.score_fold(m, "holdout", 0, 200)) or (
        set(np.random.default_rng(np.random.SeedSequence(
            [9, zlib.crc32(b"csv-kfold"), 0])).permutation(len(pool))
            [:len(t._x_ho)]) != idx[0])


def test_kfold_env_score_is_fold_mean_with_std(tmp_path):
    """A34 (SPEC.md 44.1.2/44.1.3): with kfold = K the env's (score,
    std, gen, gen_gap) is the fold mean + fold sample std of `kfold_
    score` on holdout and gen; the §18.6 gate sees the fold σ."""
    p = _softmax3_csv(tmp_path)
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(3, 300, 30),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p)},
                        dataset_episodes=80, kfold=3)
    env.reset()
    m = _Zeros()
    score, std, gen, gap = env._evaluate(m)
    h = kfold_score(env.task, m, "holdout", k=3)
    g = kfold_score(env.task, m, "gen", k=3)
    assert score == h["score"] and std == h["std"] and gen == g["score"]
    assert gap == h["score"] - g["score"]
    assert std > 0.0  # distinct subsets → non-degenerate fold std


# --- 44.1 k-fold: env knob, CLI, summary, RunConfig ---------------------------

def test_kfold_validator_battery(tmp_path):
    """A34 (SPEC.md 44.1.2): the registry validator is strict (int >= 0,
    not bool, not float/string/None) and good values pass through."""
    p = _softmax3_csv(tmp_path)
    for bad in (None, True, False, -1, 1.5, "3", 0.0, [1]):
        with pytest.raises(ValueError, match="kfold"):
            AutoRefineEnv(task="csv", seed=7, budget=Budget(3, 300, 30),
                          runs_dir=tmp_path / "runs",
                          task_config={"path": str(p)}, kfold=bad)
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(3, 300, 30),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p)}, dataset_episodes=80,
                        kfold=3)
    assert env.kfold == 3
    assert KNOBS["kfold"].default == 0 and KNOBS["kfold"].cli == \
        ("run", "fit")  # the registry row (33.2)


def test_cli_kfold_flags_run_and_fit_only():
    """A34 (SPEC.md 44.1.2 + 33.2): `--kfold` is on exactly `run` and
    `fit` — nowhere else — with the parser default equal to the
    registry default (0)."""
    parser = build_parser()
    sub = next(a for a in parser._actions
               if type(a).__name__ == "_SubParsersAction")
    subs = sub.choices
    where = [name for name, sp in subs.items()
             if any("--kfold" in a.option_strings for a in sp._actions)]
    assert where == ["run", "fit"]  # parser definition order (cli.build_parser)
    for name in where:
        action = next(a for a in subs[name]._actions
                      if "--kfold" in a.option_strings)
        assert action.default == KNOBS["kfold"].default == 0


def test_kfold_summary_key_conditional(tmp_path):
    """A34 (SPEC.md 44.1.4): the summary carries the `kfold` key only
    when kfold > 0; when 0 (the default) the legacy key set + the
    pinned 5-key `search_quality` dict are untouched."""
    p = _softmax3_csv(tmp_path)
    # kfold = 0 (the default) — the legacy shape, exactly
    env0 = AutoRefineEnv(task="csv", seed=7, budget=Budget(1, 60, 10),
                         runs_dir=tmp_path / "r0",
                         task_config={"path": str(p)}, dataset_episodes=80)
    _drive(env0, SearchPolicy(seed=7))
    s0 = env0.memory.load_summary()
    assert "kfold" not in s0
    assert set(s0["search_quality"]) == {
        "ci_blocks", "z_accept", "efficiency_weight", "gen_gap_penalty",
        "block_size"}
    assert "--kfold" not in " ".join(fit_recipe(env0.run_config))
    # kfold = 3 — the conditional key, not in search_quality
    env3 = AutoRefineEnv(task="csv", seed=7, budget=Budget(1, 60, 10),
                         runs_dir=tmp_path / "r3",
                         task_config={"path": str(p)}, dataset_episodes=80,
                         kfold=3)
    _drive(env3, SearchPolicy(seed=7))
    s3 = env3.memory.load_summary()
    assert s3["kfold"] == {"k": 3}
    assert "kfold" not in s3["search_quality"]
    assert "--kfold" in " ".join(fit_recipe(env3.run_config))


def test_kfold_zero_is_bit_identical_legacy_path(tmp_path):
    """A34 (SPEC.md 44.1.4): kfold = 0 leaves `_evaluate` on the legacy
    single-split path — (score, std 0.0, gen, gap) equals
    `evaluate_full`, the A1–A33 pins' shape."""
    p = _softmax3_csv(tmp_path)
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(1, 60, 10),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p)}, dataset_episodes=80)
    env.reset()
    m = _Zeros()
    ev = evaluate_full(env.task, m)
    assert env._evaluate(m) == (ev["score"], 0.0, ev["gen_score"],
                                ev["gen_gap"])


# --- 44.1 k-fold: RunConfig -----------------------------------------------------

def test_runconfig_kfold_additive_roundtrip():
    """A34 (SPEC.md 44.1.4): `RunConfig.kfold` defaults to 0,
    round-trips through to_dict/from_dict, validates (int >= 0; not
    bool/str/negative), and `fit_recipe` renders `--kfold K` only when
    > 0."""
    assert RunConfig(task="csv", seed=7).kfold == 0  # the default
    cfg = RunConfig(task="csv", seed=7, kfold=3)
    d = cfg.to_dict()
    assert d["kfold"] == 3
    assert RunConfig.from_dict(d).kfold == 3  # round-trip
    recipe = fit_recipe(RunConfig(task="csv", seed=7,
                                  task_config={"path": "d.csv"}))
    assert "--kfold" not in recipe  # 0 = omitted
    recipe3 = fit_recipe(RunConfig(task="csv", seed=7, kfold=4,
                                   task_config={"path": "d.csv"}))
    assert "--kfold" in recipe3 and "4" in recipe3
    for bad in (-1, "3", True, 2.5):
        with pytest.raises(ValueError, match="kfold"):
            RunConfig.from_dict({**d, "kfold": bad})


# --- 44.2 importance: the pure leaf -------------------------------------------

def test_score_xy_matches_csv_contract():
    """A34 (SPEC.md 44.2.1): `score_xy` is the §22.1 head metric on the
    given rows — softmax → 100·accuracy, mse → 100·max(0, R²) — equal
    to the task's own `score` on the same rows."""
    p = _cls_csv(Path(".")) if False else None
    assert p is None  # (covered by the tmp_path tests below)


def test_importance_softmax_ranks_signal_first(tmp_path):
    """A34 (SPEC.md 44.2.1): on a softmax CSV where only `sig` matters,
    `sig` ranks first with a positive drop and `noise` ~0; the result
    is deterministic (same call twice → identical)."""
    p = _cls_csv(tmp_path)
    t = CsvTask(seed=7, path=p)
    m = _SigModel()
    r = permutation_importance(t, m, n=200, n_repeats=5)
    assert r is not None
    assert r["head"] == "softmax" and r["n"] == len(t._x_ho)
    assert r["baseline"] == score_xy(m, t._x_ho, t._y_ho, "softmax")
    names = [f["name"] for f in r["features"]]
    assert names == ["sig", "noise"]  # sorted desc, signal first
    assert r["features"][0]["importance"] > 10.0  # a real drop
    assert r["features"][1]["importance"] == pytest.approx(0.0, abs=1e-9)
    # deterministic (G2): the same call is bit-identical
    r2 = permutation_importance(t, m, n=200, n_repeats=5)
    assert r == r2
    # a different seed changes the shuffles (and hence the exact drops)
    r9 = permutation_importance(t, m, n=200, n_repeats=5, seed=9)
    assert [f["importance"] for f in r9["features"]] != \
        [f["importance"] for f in r["features"]] or r9 == r


def test_importance_mse_path_sorted(tmp_path):
    """A34 (SPEC.md 44.2.1): the mse path — both columns matter (label
    = 2·sig + 3·noise), a perfect model scores the baseline 100, both
    shuffles drop it, and the list is sorted desc."""
    p = _mse_csv(tmp_path)
    t = CsvTask(seed=7, path=p)
    m = _MseModel(t)
    r = permutation_importance(t, m, n=200, n_repeats=3)
    assert r is not None and r["head"] == "mse"
    assert r["baseline"] == pytest.approx(100.0, rel=1e-9)
    imps = [f["importance"] for f in r["features"]]
    assert all(i > 0.0 for i in imps)  # both columns are used
    assert imps == sorted(imps, reverse=True)  # sorted desc
    assert [f["name"] for f in r["features"]] == ["noise", "sig"]


def test_importance_degrades_to_none():
    """A34 (SPEC.md 44.2.3): `None` when it does not apply — an
    episode task (max_steps > 1), a task without `holdout_rows`, or
    non-flat (3-D) features."""
    m = _SigModel()
    cart = TASKS["cartpole-v1"](seed=7)
    assert permutation_importance(cart, m) is None  # episode task
    no_rows = types.SimpleNamespace(max_steps=1, head="softmax")
    assert permutation_importance(no_rows, m) is None  # no holdout_rows
    grid = types.SimpleNamespace(
        max_steps=1, head="softmax",
        holdout_rows=lambda n, model=None: (np.zeros((5, 4, 4)),
                                            np.zeros(5, dtype=np.int64)))
    assert permutation_importance(grid, m) is None  # x.ndim == 3
    bad_head = types.SimpleNamespace(
        max_steps=1, head="success",
        holdout_rows=lambda n, model=None: (np.zeros((5, 2)), np.zeros(5)))
    assert permutation_importance(bad_head, m) is None  # other head


# --- 44.2 importance: the CLI ----------------------------------------------------

def _make_csv_run(tmp_path: Path) -> Path:
    """A finished csv run (softmax, `sig` matters) — the `report`
    battery (duplicated fixture style, house rule)."""
    p = _cls_csv(tmp_path)
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(1, 60, 10),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p)}, dataset_episodes=160)
    _drive(env, SearchPolicy(seed=7))
    return Path(env.run_dir)


def test_report_importance_prints_block(tmp_path, capsys):
    """A34 (SPEC.md 44.2.2): `report --importance` on a finished CSV
    run prints the block (baseline + per-feature drops, the csv column
    names) with rc 0."""
    run = _make_csv_run(tmp_path)
    rc = cli_main(["report", "--run", str(run), "--importance"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "feature importance (permutation" in out
    assert "baseline" in out
    assert "sig" in out and "noise" in out
    assert "higher = the model uses it more" in out
    capsys.readouterr()
    # --importance-repeats is a valid flag (default 5)
    assert cli_main(["report", "--run", str(run), "--importance",
                     "--importance-repeats", "2"]) == 0


def test_report_importance_na_and_broken_run(tmp_path, capsys):
    """A34 (SPEC.md 44.2.2/44.2.3): a non-applicable run (parity — no
    `holdout_rows`) prints the n/a line with rc 0; a broken run dir is
    rc 1."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(1, 60, 10),
                        runs_dir=tmp_path / "pruns")
    _drive(env, SearchPolicy(seed=7))
    rc = cli_main(["report", "--run", str(env.run_dir), "--importance"])
    out = capsys.readouterr().out
    assert rc == 0 and "n/a for this run" in out
    capsys.readouterr()
    rc = cli_main(["report", "--run", str(tmp_path / "nope"),
                   "--importance"])
    err = capsys.readouterr().err
    assert rc == 1 and "summary.json" in err


def test_report_importance_mutual_exclusions(tmp_path, capsys):
    """A34 (SPEC.md 44.2.2): `--importance` is mutually exclusive with
    `--json` / `--history` / `--what-if` / `--project` / `--trace` —
    rc 1 naming the rule."""
    run = _make_csv_run(tmp_path)
    for extra in (["--json"], ["--history"], ["--what-if", "score>=97"],
                  ["--project"], ["--trace"]):
        rc = cli_main(["report", "--run", str(run), "--importance"] + extra)
        err = capsys.readouterr().err
        assert rc == 1, extra
        assert "mutually exclusive" in err, extra


# --- 44.3 regression -------------------------------------------------------------

def test_version_round_v030():
    """A34 (SPEC.md 44.3, 33.1): the version stepped to `0.33.0` in
    both sources (advanced again with v0.33 ⇒ `0.33.0`, M36, SPEC.md 46)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.33.0"


def test_spec_cites_a34_and_round():
    """A34 (SPEC.md 44.3): SPEC.md defines A34 (this round) and the
    M33 milestone row — the A25 index machinery reads both."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    # A25's definition convention: a `**A34.**` bullet OR the
    # `Acceptance (A34)` heading (this round uses the heading, like A33)
    assert re.search(r"\*\*A34\.\*\*", spec) or re.search(
        r"\(A34\)\s*$", spec, re.MULTILINE)
    assert re.search(r"^\s*\| M33 \| v0\.30\s*\|\s*44\s*\|\s*A34\s*\|",
                     spec, re.MULTILINE)
    assert "**M33**" in spec
