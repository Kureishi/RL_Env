"""Use-the-model tests (SPEC.md 42, A32, v0.28).

A (42.1) `autorefine predict --run DIR` — score NEW rows with a finished
run's best model, using the task's own training-time preprocessing. The
pure leaf `autorefine.predict` is driven with fake tasks/models (no real
task needed): `row_to_features` (name match case-insensitive, the
missing-column and wrong-count errors, positional, the one-key rule for
nameless tasks), `standardize` (train stats when present, identity
otherwise), `predict_features` (softmax → label via `class_values` +
probabilities; mse → float, null probabilities), `csv_rows_to_features`
(header vs positional; extra header columns OK; missing column / wrong
cell count are errors), `media_item_features` (dispatches the task's
`item_features`). The CLI is exercised on finished sine/csv/audio runs:
`--row` array and object, `--csv` with a header + label column,
`--stdin`, `--item`, `--json` the pure array; the cartpole episode
rejection (42.1.2); mode exclusivity; broken run dirs.

B (42.2) `autorefine compare` — the app's run-diff in the terminal:
`diff_two_summaries` lives in core `autorefine.dashboard` (no streamlit
import) and `dashboard_app.diff_two_summaries is
dashboard.diff_two_summaries` (38.4 unchanged); the CLI prints the score
delta + spec-diff table (identical specs → the identical line), `--json`
the pure dict; a missing summary is rc 1.

C (42.3) `autorefine explain` — the one-screen narrative (result / tried
/ why / weak / next): rc 0 informational (the `weak` block degrades to
n/a on an mse run, 28.5), per-class lines + the weakest-class hint on a
classification run, the 41.1 projection verdict in `next` (sub-/
super-target registry history), `--json` exactly the five block keys, a
broken run dir is rc 1.

D (42.4) regression — A1–A31 stay green in their own files (no run
behavior, summary key set, artifact, or app-view change); the version
stepped to `0.30.0` in both sources (33.1).

House rules: no cross-test imports (the sine/csv fixtures are duplicated
from test_simulation_v026.py, the WAV fixture from test_media_tasks.py);
stdlib + numpy only.
"""
import io
import json
import math
import random
import sys
import tomllib
import types
import wave
from pathlib import Path

import numpy as np  # noqa: F401  (house rule: core+tests may use numpy)
import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.cli import main as cli_main
from autorefine.dashboard import diff_two_summaries
from autorefine.predict import (
    csv_rows_to_features,
    media_item_features,
    predict_features,
    row_to_features,
    standardize,
)

REPO = Path(__file__).resolve().parent.parent


# --- SPEC.md 42.1 (A32): the pure `predict` leaf (fake task/model) ------------

def _fake_task(**kw):
    """A §15-protocol *fitting* task (max_steps == 1) with the v0.28
    additive attributes (42.1.3) — drives `autorefine.predict` without a
    real task. `feature_names=None` models the nameless sine-v1 shape."""
    base = dict(
        name="fake-v1", state_dim=2, n_outputs=2, head="softmax",
        max_steps=1,
        class_values=[0.0, 1.0],
        feature_names=["x1", "x2"],
        feature_mean=np.array([1.0, -1.0]),
        feature_std=np.array([2.0, 2.0]),
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class _FakeModel:
    """`forward` returns fixed LOGITS (42.1.4 — `MLP.forward` returns
    logits; `predict_features` softmaxes as a pure rendering)."""

    def __init__(self, logits=(0.0, 3.0)):
        self._logits = np.asarray(logits, dtype=np.float64)

    def forward(self, x):
        return np.broadcast_to(self._logits,
                               (len(x), len(self._logits))).copy()


def test_row_to_features_name_match_case_insensitive():
    """A32 (42.1.1): object rows match `feature_names` case-insensitively;
    extra keys are ignored."""
    t = _fake_task()
    v = row_to_features({"X1": 3.0, "x2": 5.0, "junk": 9.0},
                        t.state_dim, t.feature_names)
    assert v.tolist() == [3.0, 5.0]


def test_row_to_features_missing_column_errors():
    """A32 (42.1.1): a missing feature column is an error naming it."""
    t = _fake_task()
    with pytest.raises(ValueError, match="x1"):
        row_to_features({"x2": 5.0}, t.state_dim, t.feature_names)


def test_row_to_features_positional_and_wrong_count():
    """A32 (42.1.1): a sequence is positional — exactly `state_dim`
    numbers; a wrong count is an error."""
    t = _fake_task()
    assert row_to_features([1.0, 2.0], t.state_dim).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="expected 2"):
        row_to_features([1.0], t.state_dim)


def test_row_to_features_nameless_task_one_key_rule():
    """A32 (42.1.1): a task without `feature_names` (e.g. `sine-v1`)
    accepts a one-key object; more keys are an error."""
    t = _fake_task(feature_names=None, state_dim=1)
    assert row_to_features({"x": 0.5}, t.state_dim, None).tolist() == [0.5]
    with pytest.raises(ValueError):
        row_to_features({"a": 1.0, "b": 2.0}, 1, None)


def test_standardize_applies_train_stats_or_identity():
    """A32 (42.1.3): `(row − feature_mean) / feature_std`; a task without
    the stats (sine-v1) is the identity."""
    t = _fake_task()
    out = standardize(t, np.array([3.0, 3.0]))
    assert out.tolist() == [(3.0 - 1.0) / 2.0, (3.0 - (-1.0)) / 2.0]
    identity = _fake_task(feature_mean=None, feature_std=None)
    assert standardize(identity, np.array([3.0, 3.0])).tolist() == [3.0, 3.0]


def test_standardize_shape_mismatch_errors():
    """A32 (42.1.3): a wrong task/row pairing (shape mismatch) is an error
    instead of a silent broadcast."""
    with pytest.raises(ValueError):
        standardize(_fake_task(), np.array([1.0]))


def test_predict_features_softmax_label_and_probabilities():
    """A32 (42.1.4): the argmax class mapped through `class_values`;
    `probabilities` = the softmax of the forward output (sums to 1)."""
    t = _fake_task(class_values=[0.0, 1.0])
    r = predict_features(t, _FakeModel(logits=(0.0, 3.0)),
                         np.array([0.0, 0.0]))
    assert r["prediction"] == 1.0
    assert r["probabilities"] == pytest.approx(
        [1.0 / (1.0 + math.e ** 3), math.e ** 3 / (1.0 + math.e ** 3)],
        rel=1e-9)
    assert sum(r["probabilities"]) == pytest.approx(1.0)


def test_predict_features_string_class_values():
    """A32 (42.1.4): string class labels (the media tasks) are the
    prediction — the user's labels, not internal indices."""
    t = _fake_task(class_values=["high", "low"])
    r = predict_features(t, _FakeModel(logits=(3.0, 0.0)),
                         np.array([0.0, 0.0]))
    assert r["prediction"] == "high"  # argmax index 0 ↔ class_values[0]


def test_predict_features_mse_scalar_null_probabilities():
    """A32 (42.1.4): the mse head → the scalar output, probabilities null."""
    t = _fake_task(head="mse", n_outputs=1, class_values=None)
    assert predict_features(t, _FakeModel(logits=(2.5,)),
                            np.array([0.0, 0.0])) \
        == {"prediction": 2.5, "probabilities": None}


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_rows_header_match_and_extra_label_column(tmp_path):
    """A32 (42.1.1): a non-numeric first line is a header — cells matched
    by name (feature order, not header order); an extra label column is
    OK."""
    t = _fake_task()
    p = _write(tmp_path / "rows.csv", "x2,label,x1\n2,1,1\n4,0,3\n")
    v = csv_rows_to_features(p, t)
    assert v[0].tolist() == [1.0, 2.0]  # x1=r[2]=1, x2=r[0]=2
    assert v[1].tolist() == [3.0, 4.0]


def test_csv_rows_positional(tmp_path):
    """A32 (42.1.1): an all-numeric first line → positional rows of
    exactly `state_dim` cells."""
    t = _fake_task()
    p = _write(tmp_path / "rows.csv", "1,2\n3,4\n")
    v = csv_rows_to_features(p, t)
    assert v[0].tolist() == [1.0, 2.0] and v[1].tolist() == [3.0, 4.0]


def test_csv_rows_missing_column_and_wrong_count_errors(tmp_path):
    """A32 (42.1.1/42.1.5): a header missing a feature column and a
    positional row with the wrong cell count are errors."""
    t = _fake_task()
    with pytest.raises(ValueError, match="x1"):
        csv_rows_to_features(_write(tmp_path / "bad.csv", "zz,qq\n1,2\n"), t)
    with pytest.raises(ValueError, match="expected 2"):
        csv_rows_to_features(_write(tmp_path / "bad2.csv", "1,2,3\n"), t)


def test_csv_rows_empty_errors(tmp_path):
    """A32 (42.1.5): an empty CSV is an error (zero input rows)."""
    with pytest.raises(ValueError, match="no rows"):
        csv_rows_to_features(_write(tmp_path / "empty.csv", ""),
                             _fake_task())


def test_media_item_features_dispatches_and_rejects_non_media():
    """A32 (42.1.3): dispatches the task's `item_features(path)`; a task
    without it is an error (`predict --item` is media-tasks only)."""
    calls = []

    def item_features(path):
        calls.append(Path(path))
        return np.array([1.0, 2.0])

    t = _fake_task()
    t.item_features = item_features
    assert media_item_features(t, "a.wav").tolist() == [1.0, 2.0]
    assert calls[0].name == "a.wav"
    with pytest.raises(ValueError, match="item_features"):
        media_item_features(_fake_task(), "a.wav")


# --- SPEC.md 42.1 (A32): the `predict` CLI ------------------------------------

def _make_sine_run(tmp_path: Path) -> Path:
    """A finished sine-v1 run (mse, nameless single input, best score
    ≈ 27.9) — duplicated from test_simulation_v026.py (house rule: no
    cross-test imports)."""
    env = AutoRefineEnv(task="sine-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path / "runs")
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def _csv_file(tmp_path: Path) -> Path:
    rows = ["x1,x2,label"]
    rnd = random.Random(0)
    for _ in range(60):
        a, b = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
        rows.append(f"{a:.3f},{b:.3f},{1 if (a + b) > 0 else 0}")
    p = tmp_path / "data.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def _make_csv_run(tmp_path: Path) -> Path:
    """A finished csv run (softmax, 2 features, 0/1 labels)."""
    _csv_file(tmp_path)  # the file lives at tmp_path/data.csv (the test's
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path / "runs2",
                        task_config={"path": str(tmp_path / "data.csv")},
                        dataset_episodes=48)
    state = env.reset()
    policy = SearchPolicy(seed=7)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    return Path(env.run_dir)


def test_predict_sine_row_array_and_object(tmp_path, capsys):
    """A32 (42.1): `--row` array (positional) and one-key object (the
    nameless-task rule) both score with rc 0; the mse row renders
    `index<TAB>value` (42.1.4)."""
    run = _make_sine_run(tmp_path)
    rc = cli_main(["predict", "--run", str(run), "--row", "[0.5]"])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0].startswith("0\t")
    assert math.isfinite(float(out.splitlines()[0].split("\t")[1]))
    rc = cli_main(["predict", "--run", str(run), "--row", '{"x": 0.25}'])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0].startswith("0\t")
    assert math.isfinite(float(out.splitlines()[0].split("\t")[1]))


def test_predict_csv_json_pure_array(tmp_path, capsys):
    """A32 (42.1.4/42.1.5): `--csv` with a header + extra label column is
    rc 0; `--json` is the pure array with exactly the row/prediction/
    probabilities keys, one per input row."""
    run = _make_csv_run(tmp_path)
    rc = cli_main(["predict", "--run", str(run),
                   "--csv", str(tmp_path / "data.csv"), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    arr = json.loads(out)
    assert [e["row"] for e in arr] == list(range(60))
    for e in arr:
        assert set(e) == {"row", "prediction", "probabilities"}
        assert e["prediction"] in (0.0, 1.0)  # the user's 0/1 labels
        assert sum(e["probabilities"]) == pytest.approx(1.0)


def test_predict_csv_row_object_case_insensitive(tmp_path, capsys):
    """A32 (42.1.1): `--row` objects match feature names case-
    insensitively; extra keys are ignored."""
    run = _make_csv_run(tmp_path)
    rc = cli_main(["predict", "--run", str(run),
                   "--row", '{"X1": 0.5, "X2": -0.5, "junk": 9}'])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0].split("\t")[1] in ("0", "1")


def test_predict_csv_unknown_columns_rc1(tmp_path, capsys):
    """A32 (42.1.5): a CSV whose header lacks a feature column is rc 1
    (a wrong task/row pairing, surfaced not silently mis-scored)."""
    run = _make_csv_run(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("zz,qq\n1,2\n", encoding="utf-8")
    rc = cli_main(["predict", "--run", str(run), "--csv", str(bad)])
    err = capsys.readouterr().err
    assert rc == 1 and "x1" in err


def test_predict_stdin_rows_and_empty_rc1(tmp_path, monkeypatch, capsys):
    """A32 (42.1.1/42.1.5): `--stdin` scores JSON lines (object or
    array); zero rows is rc 1."""
    run = _make_csv_run(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("[0.5, 0.5]\n[-0.5, 0.5]\n"))
    rc = cli_main(["predict", "--run", str(run), "--stdin"])
    out = capsys.readouterr().out
    assert rc == 0 and len(out.splitlines()) == 2
    assert out.splitlines()[0].startswith("0\t")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli_main(["predict", "--run", str(run), "--stdin"]) == 1


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 1.0,
               amp: float = 0.6, phase: float = 0.0) -> None:
    """16-bit PCM mono WAV (SPEC.md 24.4) — duplicated from
    test_media_tasks.py (house rule: no cross-test imports)."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = amp * np.sin(2 * math.pi * freq * t + phase)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _tone_dir(tmp_path: Path, n: int = 20, seed: int = 0) -> Path:
    """Two tone classes (220 vs 440 Hz) — duplicated from
    test_media_tasks.py."""
    d = tmp_path / "tones"
    rng = np.random.default_rng(seed)
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _write_wav(sub / f"c{i:02d}.wav", freq,
                       amp=0.5 + 0.3 * rng.random(),
                       phase=rng.random() * 2 * math.pi)
    return d


def test_predict_audio_item_string_label(tmp_path, capsys):
    """A32 (42.1.1/42.1.3): `--item` decodes the media file with the
    task's `item_features` and scores it — the string class label
    (high/low) renders as-is (42.1.4)."""
    d = _tone_dir(tmp_path)
    rc = cli_main(["fit", "--data", str(d), "--seed", "7",
                   "--experiments", "1", "--max-train-seconds", "10",
                   "--target", "50", "--runs-dir", str(tmp_path / "runs")])
    runs = list((tmp_path / "runs").glob("audio-seed7-*"))
    assert rc == 0 and runs
    capsys.readouterr()  # drop the `fit` output; keep only `predict`'s
    wav = next(d.rglob("*.wav"))
    rc = cli_main(["predict", "--run", str(runs[0]), "--item", str(wav)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0].split("\t")[1] in ("high", "low")


def test_predict_episode_task_rc1_with_eval_hint(tmp_path, capsys):
    """A32 (42.1.2): cartpole (`max_steps > 1`) is rejected with the
    message pointing at `eval`."""
    env = AutoRefineEnv(task="cartpole-v1", seed=1, budget=Budget(1, 300, 10),
                        runs_dir=tmp_path / "runs")
    state = env.reset()
    policy = SearchPolicy(seed=1)
    while not env.done:
        state, _r, _d, _info = env.step(policy.propose(state))
    rc = cli_main(["predict", "--run", str(env.run_dir), "--row", "[0, 0, 0, 0]"])
    err = capsys.readouterr().err
    assert rc == 1 and "episode" in err and "eval" in err


def test_predict_mode_exclusivity_and_missing_run(tmp_path, capsys):
    """A32 (42.1.5): none or more than one input mode is rc 1; a run dir
    without `summary.json` is rc 1."""
    run = _make_sine_run(tmp_path)
    assert cli_main(["predict", "--run", str(run)]) == 1
    err = capsys.readouterr().err
    assert "exactly one input mode" in err
    assert cli_main(["predict", "--run", str(run),
                     "--row", "[0.5]", "--stdin"]) == 1
    assert cli_main(["predict", "--run", str(tmp_path / "nope"),
                     "--row", "[0.5]"]) == 1
    err = capsys.readouterr().err
    assert "summary.json" in err


# --- SPEC.md 42.2 (A32): `compare` ---------------------------------------------

def test_diff_two_summaries_core_and_app_identity():
    """A32 (42.2.2): `diff_two_summaries` lives in core `dashboard`
    (no streamlit import); the app's name *is* the core one (38.4)."""
    sa = {"final_best_score": 80.0, "best_spec": {"a": 1}}
    sb = {"final_best_score": 84.5, "best_spec": {"a": 2, "b": 3}}
    d = diff_two_summaries(sa, sb)
    assert d["score_delta"] == pytest.approx(4.5)
    assert [row["field"] for row in d["spec_diff"]] == ["a", "b"]
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    import autorefine.dashboard_app as app
    assert app.diff_two_summaries is diff_two_summaries


def _fake_run_dir(path: Path, score: float, spec: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({"task": "csv", "final_best_score": score,
                    "best_spec": spec}), encoding="utf-8")
    return path


def test_compare_human_and_json(tmp_path, capsys):
    """A32 (42.2.3): the score delta + the per-field spec-diff table
    (only differing fields); `--json` is the pure `diff_two_summaries`
    dict."""
    a = _fake_run_dir(tmp_path / "a", 80.0, {"hidden_dim": 64, "lr": 0.001})
    b = _fake_run_dir(tmp_path / "b", 84.5, {"hidden_dim": 128, "lr": 0.001})
    rc = cli_main(["compare", "--run", str(a), "--run", str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "delta   : 4.5 (B - A)" in out
    assert "hidden_dim" in out and "64 -> 128" in out
    assert "lr" not in out  # identical field → not in the diff
    rc = cli_main(["compare", "--run", str(a), "--run", str(b), "--json"])
    out = capsys.readouterr().out
    assert json.loads(out) == diff_two_summaries(
        json.loads((a / "summary.json").read_text(encoding="utf-8")),
        json.loads((b / "summary.json").read_text(encoding="utf-8")))


def test_compare_identical_specs(tmp_path, capsys):
    """A32 (42.2.3): identical best_specs → the `identical` line."""
    a = _fake_run_dir(tmp_path / "a", 80.0, {"hidden_dim": 64})
    b = _fake_run_dir(tmp_path / "b", 80.0, {"hidden_dim": 64})
    rc = cli_main(["compare", "--run", str(a), "--run", str(b)])
    out = capsys.readouterr().out
    assert rc == 0 and "best_spec: identical" in out


def test_compare_errors(tmp_path, capsys):
    """A32 (42.2.3): not exactly two run dirs is rc 1; a missing summary
    is rc 1."""
    a = _fake_run_dir(tmp_path / "a", 80.0, {})
    assert cli_main(["compare", "--run", str(a)]) == 1
    capsys.readouterr()
    assert cli_main(["compare", "--run", str(a),
                     "--run", str(tmp_path / "nope")]) == 1
    err = capsys.readouterr().err
    assert "summary.json" in err


# --- SPEC.md 42.3 (A32): `explain` ---------------------------------------------

def test_explain_sine_blocks_and_weak_na(tmp_path, capsys):
    """A32 (42.3): rc 0 with all five blocks; an mse run's `weak` degrades
    to n/a (28.5) instead of failing; a fresh runs dir → insufficient
    history (41.1.4)."""
    run = _make_sine_run(tmp_path)
    rc = cli_main(["explain", "--run", str(run)])
    out = capsys.readouterr().out
    assert rc == 0
    for block in ("result", "tried", "why", "weak", "next"):
        assert block in out
    assert "n/a (not a classification task)" in out
    assert "insufficient history" in out


def test_explain_json_exactly_five_keys(tmp_path, capsys):
    """A32 (42.3.3): `--json` is exactly {result, tried, why, weak,
    next}; the `why` chain is the baseline champion first, then the
    accepted experiments (log order)."""
    run = _make_sine_run(tmp_path)
    rc = cli_main(["explain", "--run", str(run), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    blocks = json.loads(out)
    assert set(blocks) == {"result", "tried", "why", "weak", "next"}
    assert blocks["why"] and blocks["why"][0]["step"] == "baseline"
    assert all(w["step"] == "accepted" for w in blocks["why"][1:])
    assert blocks["weak"]["note"] == "n/a (not a classification task)"
    assert blocks["next"]["verdict"] == "insufficient"


def test_explain_csv_weak_per_class(tmp_path, capsys):
    """A32 (42.3.2): a classification run's `weak` shows the per-class
    holdout lines + the weakest-class hint (28.2)."""
    run = _make_csv_run(tmp_path)
    rc = cli_main(["explain", "--run", str(run)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "class 0" in out and "class 1" in out
    assert ("fails on class" in out) or ("all classes at 100%" in out)


def test_explain_next_verdicts_from_registry(tmp_path, capsys):
    """A32 (42.3.2): the `next` block renders the 41.1 projection — a
    sub-target registry history → CEILING; a super-target one → MORE."""
    run = _make_sine_run(tmp_path)
    reg_path = run.parent / "registry.json"
    base = json.loads(reg_path.read_text(encoding="utf-8"))  # the real run
    # sub-target: the curve saturates below 95 → CEILING
    reg_path.write_text(json.dumps(base + [
        {"task": "sine-v1", "run_id": "fake-ceiling-1",
         "experiments_run": 10, "final_score": 28.0},
        {"task": "sine-v1", "run_id": "fake-ceiling-2",
         "experiments_run": 20, "final_score": 29.0},
    ]), encoding="utf-8")
    rc = cli_main(["explain", "--run", str(run)])
    out = capsys.readouterr().out
    assert rc == 0 and "CEILING" in out
    # super-target: the asymptote clears 95 → MORE (~N more)
    reg_path.write_text(json.dumps(base + [
        {"task": "sine-v1", "run_id": "fake-more-1",
         "experiments_run": 10, "final_score": 70.0},
        {"task": "sine-v1", "run_id": "fake-more-2",
         "experiments_run": 20, "final_score": 85.0},
        {"task": "sine-v1", "run_id": "fake-more-3",
         "experiments_run": 30, "final_score": 92.0},
    ]), encoding="utf-8")
    rc = cli_main(["explain", "--run", str(run)])
    out = capsys.readouterr().out
    assert rc == 0 and "MORE" in out


def test_explain_missing_run_rc1(tmp_path, capsys):
    """A32 (42.3.3): a run dir without `summary.json` is rc 1."""
    rc = cli_main(["explain", "--run", str(tmp_path / "nope")])
    err = capsys.readouterr().err
    assert rc == 1 and "summary.json" in err


# --- SPEC.md 33.1 / 42.4 (A32): version ----------------------------------------

def test_version_028_both_sources():
    """A32 (42.4, 33.1): the version stepped to `0.30.0` in both sources
    (v0.30 ⇒ `0.30.0`, both together)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.30.0"
