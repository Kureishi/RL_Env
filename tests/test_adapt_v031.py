"""Adapt to more use cases (SPEC.md 45, A35, v0.31).

45.1 `split_mode` — the temporal / walk-forward split: `Task.split_mode`
on the G1 ABC (default "random" everywhere), the `CsvTask` temporal
branch (rows in file order — train = first rows, holdout = next, gen =
last; same sizes as the random rule; the seed is not used for the
partition), the random branch byte-identical to the pre-45.1 seed
permutation (45.1.4), a construction ValueError on a bad value,
`--temporal` on `fit` only (not a KNOBS row — the A23 parser scan is
untouched), and `fit --dry-run --temporal` naming the mode with rc 0.

45.2 `text` — the fourth modality on the same protocol (22.1/24.2):
`ngram_features` (deterministic crc32-hashed word unigram+bigram count
vector, dim 128, empty text → zeros), `TextTask` (subfolders or
index.csv; §22.1 head inference — string names → softmax over sorted
classes, non-integer floats → mse; seed split; train-only
standardization; the holdout-inspection protocol), `detect_modality`
(text-only → "text"; image+text and image+audio → "mixed"), the registry
pins advanced in place (`len(TASKS)` = 8), the `fit --task text`
acceptance path (rc 0 PASS on separable words), `fit --task csv` on a
directory still rc 1, and the preflight media shape (43.1).

45.3 `boost` — status note: `BoostingEnsemble` shipped in v0.5
(SPEC.md 19.2); this round verifies it trains and scores finitely on
parity and that the bandit offers the family (no re-add, no new knobs).

45.4/45.5 regression — A1–A34 stay green (covered by the rest of the
suite); the A25 index advances (defined == set(range(1, 36))) and reads
this file's A35 citation; the version stepped to `0.37.0` in both
sources (33.1).

House rules: no cross-test imports (fixtures duplicated per file);
stdlib + numpy only; every test cites A35 + its SPEC §.
"""
import csv as _csv
import re
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.cli import build_parser, main as cli_main
from autorefine.config import MODEL_FAMILIES
from autorefine.improver.catalog import relevant_families
from autorefine.models.trees import BoostingEnsemble
from autorefine.preflight import data_health
from autorefine.tasks import TASKS
from autorefine.tasks.base import Task
from autorefine.tasks.csv import CsvTask
from autorefine.tasks.media import detect_modality, split_indices
from autorefine.tasks.text import TextTask, ngram_features

REPO = Path(__file__).resolve().parent.parent


# --- fixtures (duplicated per file, no cross-test imports) --------------------

def _txt_dir(base: Path) -> Path:
    """Two word classes, one subfolder each, 5 files per class
    (SPEC.md 45.2) — disjoint vocabularies, so any decent model scores
    100 on the holdout (the `fit` PASS battery)."""
    d = Path(base) / "words"
    for name, words in (("up", "up up up rising higher"),
                        ("down", "down down down falling lower")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            (sub / f"t{i:02d}.txt").write_text(words + " ", encoding="utf-8")
    return d


def _mse_txt_dir(base: Path) -> Path:
    """10 flat .txt files + an index.csv with NON-integer float labels
    (0.5 / 1.5) — the §22.1 rule sends that to the mse head (45.2)."""
    d = Path(base) / "numtxt"
    d.mkdir(parents=True)
    with open(d / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["path", "label"])
        for i in range(10):
            f = d / f"f{i:02d}.txt"
            f.write_text(f"sample number {i} word words here ",
                         encoding="utf-8")
            w.writerow([f.name, "1.5" if i % 2 else "0.5"])
    return d


def _time_csv(tmp_path: Path) -> Path:
    """20 time-ordered rows: x = i, label = 0 for the first 10 rows and
    1 for the last 10 — a pure time trend the temporal split must keep
    out of the train block (SPEC.md 45.1)."""
    p = tmp_path / "time.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x", "label"])
        for i in range(20):
            w.writerow([float(i), 1 if i >= 10 else 0])
    return p


def _cls_csv(tmp_path: Path) -> Path:
    """20 rows, 2 classes, one signal column (the dry-run battery)."""
    rnd = np.random.default_rng(0)
    p = tmp_path / "cls.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["sig", "label"])
        for _ in range(20):
            s = float(rnd.uniform(-1, 1))
            w.writerow([f"{s:.6f}", "1" if s > 0 else "0"])
    return p


class _Const:
    """A constant-logit model (all classes equal) — scores the class-0
    share under softmax; enough to exercise `score` determinism."""

    def __init__(self, n_out=2):
        self._n = n_out

    def forward(self, x):
        return np.zeros((len(x), self._n))


# --- 45.1 split_mode: the ABC + registry ---------------------------------------

def test_abc_split_mode_defaults_random():
    """A35 (SPEC.md 45.1.2): `split_mode` is a declared `Task` ABC
    property defaulting to "random" — every registered task keeps it at
    the class level (only CsvTask gains a temporal instance branch)."""
    assert Task.split_mode == "random"  # the ABC default
    for name, cls in TASKS.items():
        assert cls.split_mode == "random", name  # unchanged per task
    assert TextTask.split_mode == "random"
    assert CsvTask.split_mode == "random"  # class default, not "temporal"


def test_csv_bad_split_mode_raises_citing_451():
    """A35 (SPEC.md 45.1.2): a construction ValueError, naming the rule."""
    with pytest.raises(ValueError, match="45\\.1"):
        CsvTask(seed=7, path="does-not-matter.csv",
                split_mode="shuffled")
    # a valid value passes the validator (the failure, if any, is the
    # missing path — proof the split_mode check did not fire)
    with pytest.raises(ValueError, match="data path"):
        CsvTask(seed=7, path=None, split_mode="temporal")


# --- 45.1 split_mode: temporal = file order ------------------------------------

def test_temporal_split_is_file_order(tmp_path):
    """A35 (SPEC.md 45.1.1): walk-forward — rows in file order: train =
    the first block, holdout = the next, gen = the last; the same sizes
    as the random rule (n=20, split_frac 0.2 → 16/2/2)."""
    p = _time_csv(tmp_path)
    t = CsvTask(seed=7, path=p, split_mode="temporal")
    assert t.split_mode == "temporal"
    n_train, n_hold, n_gen = 16, 2, 2
    assert len(t._y_tr) == n_train and len(t._y_ho) == n_hold \
        and len(t._y_ge) == n_gen
    # raw labels in file order (the time trend: first 10 = 0, last 10 = 1)
    y = np.array([0.0] * 10 + [1.0] * 10)
    assert np.array_equal(t._y_tr, y[:n_train])
    assert np.array_equal(t._y_ho, y[n_train:n_train + n_hold])
    assert np.array_equal(t._y_ge, y[n_train + n_hold:])
    # the train block is the first rows exactly (x = i, standardized)
    assert t._x_tr[-1, 0] > t._x_tr[0, 0]  # order preserved
    assert t.default_dataset_size == n_train


def test_temporal_split_is_seed_independent(tmp_path):
    """A35 (SPEC.md 45.1.4): the temporal partition does not use the
    seed — two instances with different seeds split identically."""
    p = _time_csv(tmp_path)
    a = CsvTask(seed=7, path=p, split_mode="temporal")
    b = CsvTask(seed=9, path=p, split_mode="temporal")
    assert np.array_equal(a._x_tr, b._x_tr)
    assert np.array_equal(a._x_ho, b._x_ho)
    assert np.array_equal(a._x_ge, b._x_ge)


def test_random_split_byte_identical_to_pre_451(tmp_path):
    """A35 (SPEC.md 45.1.4): `split_mode = "random"` (the default) is
    byte-for-byte the pre-45.1 seed permutation — same seed, same salt,
    same blocks — reconstructed from the documented scheme."""
    p = _time_csv(tmp_path)
    t = CsvTask(seed=7, path=p)  # default mode
    assert t.split_mode == "random"
    n = 20
    n_train = int(n * (1.0 - 0.2))
    n_hold = (n - n_train) // 2
    perm = np.random.default_rng(
        np.random.SeedSequence([7, zlib.crc32(b"csv-split")])).permutation(n)
    # _y holds the integer class index (== the raw value here: 0 or 1)
    def _y_of(idx):
        return np.array([1.0 if i >= 10 else 0.0 for i in idx])
    assert np.array_equal(t._y_tr, _y_of(perm[:n_train]))
    assert np.array_equal(t._y_ho, _y_of(perm[n_train:n_train + n_hold]))
    assert np.array_equal(t._y_ge, _y_of(perm[n_train + n_hold:]))


# --- 45.1 split_mode: CLI --------------------------------------------------------

def test_temporal_flag_fit_only():
    """A35 (SPEC.md 45.1.2/45.1.4): `--temporal` is on `fit` only —
    `store_true`, default False — not a KNOBS row (the A23 KNOBS set and
    parser scan are unchanged)."""
    parser = build_parser()
    sub = next(a for a in parser._actions
               if type(a).__name__ == "_SubParsersAction")
    subs = sub.choices
    where = [name for name, sp in subs.items()
             if any("--temporal" in a.option_strings for a in sp._actions)]
    assert where == ["fit"]
    action = next(a for a in subs["fit"]._actions
                  if "--temporal" in a.option_strings)
    assert action.default is False and action.const is True


def test_fit_dry_run_temporal_names_the_mode(tmp_path, capsys):
    """A35 (SPEC.md 45.1.3): `fit --dry-run --temporal` prints the plan
    with rc 0 and names the mode; without it the plan is unchanged (no
    `split mode` line)."""
    p = _cls_csv(tmp_path)
    base = ["fit", "--dry-run", "--data", str(p),
            "--runs-dir", str(tmp_path / "runs")]
    rc = cli_main(base + ["--temporal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "split mode : temporal" in out
    capsys.readouterr()
    rc = cli_main(base)
    out = capsys.readouterr().out
    assert rc == 0
    assert "split mode" not in out


# --- 45.2 text: the n-gram leaf ---------------------------------------------------

def test_ngram_features_shape_values_determinism():
    """A35 (SPEC.md 45.2.1): (dim,) float64, non-negative, nonzero for
    tokened text, zero for empty text, deterministic (crc32, not the
    salted `hash()`), dim = 1 works, dim < 1 is an error."""
    v = ngram_features("The quick brown fox", dim=128)
    assert v.shape == (128,) and v.dtype == np.float64
    assert np.all(v >= 0.0) and v.sum() > 0.0
    # 4 tokens → 4 unigrams + 3 bigrams = 7 counts total
    assert v.sum() == pytest.approx(7.0)
    assert np.array_equal(ngram_features("The quick brown fox", 128), v)
    # case-insensitive lowercasing (the tokens are the same)
    assert np.array_equal(ngram_features("The QUICK brown Fox", 128),
                          ngram_features("the quick brown fox", 128))
    # empty / no-token text → the zero vector
    assert not ngram_features("", 16).any()
    assert not ngram_features("   --- !!!  ", 16).any()
    # dim = 1 collapses every n-gram into the one bucket
    v1 = ngram_features("a b c", dim=1)
    assert v1.shape == (1,) and v1[0] == pytest.approx(5.0)  # 3+2
    for bad in (0, -4):
        with pytest.raises(ValueError, match="dim"):
            ngram_features("hello", dim=bad)


# --- 45.2 text: TextTask ---------------------------------------------------------

def test_text_registered():
    """A35 (SPEC.md 45.2.2): `"text"` is in `TASKS` with `TextTask`; the
    registry shape advanced in place (7 → 8, the v0.8/v0.10 pattern)."""
    assert "text" in TASKS
    assert TASKS["text"] is TextTask
    assert len(TASKS) == 8  # the registry pins advanced in place (7 → 8)
    assert set(TASKS) == {"parity-v1", "sine-v1", "cartpole-v1",
                          "gridnav-v1", "csv", "image", "audio", "text"}


def test_text_task_softmax_protocol(tmp_path):
    """A35 (SPEC.md 45.2.2): string class names → softmax over the
    sorted names; 80/10/10 disjoint splits (10 items → 8/1/1);
    state_dim = the n-gram dim (128); train-only standardization;
    make_dataset / score / holdout_rows deterministic across two
    constructions; item_features round-trips one file."""
    d = _txt_dir(tmp_path)
    t = TextTask(seed=7, path=d)
    t2 = TextTask(seed=7, path=d)  # a fresh instance, same seed
    assert t.name == "text"
    assert t.head == "softmax" and t.n_outputs == 2
    assert t.class_values == ["down", "up"]  # sorted names
    assert t.metric == "accuracy"  # G1 declared metric (36.1)
    assert t.state_dim == 128
    assert t.default_dataset_size == 8  # 80% of 10 items
    # disjoint, covering splits (10 items → train 8, holdout 1, gen 1)
    tr, ho, ge = split_indices(10, 7, 0.2, b"text-split")
    assert set(map(int, ho)) & set(map(int, ge)) == set()
    assert set(map(int, tr)) | set(map(int, ho)) | set(map(int, ge)) \
        == set(range(10))
    assert len(t._x_tr) == 8 and len(t._x_ho) == 1 and len(t._x_ge) == 1
    assert set(map(int, t._ho)) == set(map(int, ho))
    # train-only standardization stats exist (42.1.3 shape)
    assert t.feature_mean.shape == (128,) and t.feature_std.shape == (128,)
    # protocol round-trip, deterministic (G2)
    x1, y1 = t.make_dataset(None)
    x2, y2 = t2.make_dataset(None)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    m = _Const(n_out=2)
    assert t.score(m, "holdout", 100) == t2.score(m, "holdout", 100)
    assert t.score(m, "gen", 100) == t2.score(m, "gen", 100)
    h1 = t.holdout_rows(5)
    h2 = t2.holdout_rows(5)
    assert np.array_equal(h1[0], h2[0]) and np.array_equal(h1[1], h2[1])
    f = d / "up" / "t00.txt"
    assert np.array_equal(t.item_features(f), t2.item_features(f))
    assert np.array_equal(t.item_features(f),
                          ngram_features(f.read_text(encoding="utf-8"), 128))


def test_text_task_mse_variant(tmp_path):
    """A35 (SPEC.md 45.2.2): index.csv with non-integer float labels
    (0.5 / 1.5) → the §22.1 rule sends it to the mse head with the r2
    metric (string names → softmax is the other branch)."""
    d = _mse_txt_dir(tmp_path)
    t = TextTask(seed=7, path=d)
    assert t.head == "mse" and t.n_outputs == 1
    assert t.metric == "r2"
    assert t.class_values is None
    assert t.state_dim == 128
    assert len(t._x_tr) == 8  # the same 80/10/10 rule over 10 items


def test_detect_modality_text_rules(tmp_path):
    """A35 (SPEC.md 45.2.4): text-only directory → "text"; image+text
    and (unchanged, the v0.10 pin) image+audio → "mixed"; a directory
    with none → None."""
    d = _txt_dir(tmp_path)
    assert detect_modality(d) == "text"
    # image + text → mixed
    d2 = tmp_path / "mix"
    (d2 / "a").mkdir(parents=True)
    (d2 / "a" / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (d2 / "up").mkdir()
    (d2 / "up" / "t.txt").write_text("hi", encoding="utf-8")
    assert detect_modality(d2) == "mixed"
    # image + audio → mixed (unchanged)
    d3 = tmp_path / "mix2"
    (d3 / "a").mkdir(parents=True)
    (d3 / "a" / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (d3 / "b").mkdir()
    (d3 / "b" / "x.wav").write_bytes(b"RIFF....WAVEfmt ")
    assert detect_modality(d3) == "mixed"
    empty = tmp_path / "none"
    empty.mkdir()
    assert detect_modality(empty) is None


def test_text_preflight_media_shape(tmp_path):
    """A35 (SPEC.md 45.2 + 43.1): the preflight degrades a TextTask to
    the media shape (no raw rows) — item count + balance, never an
    error."""
    d = _txt_dir(tmp_path)
    h = data_health(TextTask(seed=7, path=d), target=95.0)
    assert h["kind"] == "media"
    assert h["n_rows"] == 10
    assert h["head"] == "softmax"
    assert [b["label"] for b in h["balance"]] == ["down", "up"]
    assert h["majority_share"] == pytest.approx(0.5)


# --- 45.2 text: the fit CLI --------------------------------------------------------

def test_fit_text_dir_passes_gate(tmp_path, capsys):
    """A35 (SPEC.md 45.2.2): `fit --task text --data DIR` on separable
    words runs the loop and passes the §22.1 gate with rc 0 (PASS)."""
    d = _txt_dir(tmp_path)
    rc = cli_main(["fit", "--task", "text", "--data", str(d),
                   "--experiments", "4", "--target", "70",
                   "--seed", "7", "--runs-dir", str(tmp_path / "runs"),
                   "--max-seconds", "60", "--max-train-seconds", "10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PASS" in out
    assert "accuracy" in out  # the gate line names the metric


def test_fit_csv_on_directory_is_error(tmp_path, capsys):
    """A35 (SPEC.md 45.2.2): `fit --task csv --data <directory>` is
    still a resolve failure (rc 1) — a CSV task needs a CSV file."""
    d = _txt_dir(tmp_path)
    rc = cli_main(["fit", "--task", "csv", "--data", str(d),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "CSV file" in err or "--task csv needs a CSV file" in err


def test_fit_auto_detects_text(tmp_path, capsys):
    """A35 (SPEC.md 45.2.2/24.5): `fit --data DIR` (auto) resolves a
    text-only directory to the text task and passes (rc 0)."""
    d = _txt_dir(tmp_path)
    rc = cli_main(["fit", "--data", str(d), "--experiments", "4",
                   "--target", "70", "--seed", "7",
                   "--runs-dir", str(tmp_path / "runs"),
                   "--max-seconds", "60", "--max-train-seconds", "10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    capsys.readouterr()


# --- 45.3 boost: verification (shipped since v0.5, SPEC.md 19.2) -----------------

def test_boost_family_wired():
    """A35 (SPEC.md 45.3): `"boost" in MODEL_FAMILIES`; the bandit
    offered families include "boost" (the flat-task legacy three)."""
    assert "boost" in MODEL_FAMILIES
    assert "boost" in relevant_families("parity-v1")
    assert "boost" in relevant_families(None)


def test_boosting_ensemble_trains_scores_finite_on_parity():
    """A35 (SPEC.md 45.3): `BoostingEnsemble` trains on parity-v1 and
    scores finitely (a genuinely usable member of the model space —
    verified, not re-added)."""
    t = TASKS["parity-v1"](seed=7)
    X, y = t.make_dataset(64)
    m = BoostingEnsemble(n_out=t.n_outputs, n_trees=4, max_depth=2,
                         head=t.head, seed=7)
    m.fit(X, y)
    out = m.forward(X)
    assert out.shape == (len(X), t.n_outputs)
    assert np.all(np.isfinite(out))
    s = t.score(m, "holdout", 256)
    assert np.isfinite(s) and 0.0 <= s <= 100.0


# --- 45.4/45.5 regression -----------------------------------------------------------

def test_version_round_v031():
    """A35 (SPEC.md 45.4, 33.1): the version stepped to `0.37.0` in
    both sources (v0.37 ⇒ `0.37.0`, M40)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.37.0"


def test_spec_cites_a35_and_round():
    """A35 (SPEC.md 45.4/45.5): SPEC.md defines the A35 acceptance block
    and the M34 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 36)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 45.4 Acceptance (A35)" in spec
    assert re.search(r"^\s*\| M34 \| v0\.31\s*\|\s*45\s*\|\s*A35\s*\|",
                     spec, re.MULTILINE)
    assert "**M34**" in spec
