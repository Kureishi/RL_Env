"""v0.14 learning views (SPEC.md 28, A18, M17): the dashboard views that
explain *what the model is actually learning* — per-experiment training
curves (C1), final-model diagnostics (C2), the media error gallery (C3),
and the spec → architecture diagram (C4).

C1 is the only core change (SPEC.md 28.1): `train()` returns a bounded
`loss_history`, logged per experiment. C2-C4 are zero-new-data views.
Optional dependencies follow the §3/§21.1 pattern: PIL for image fixtures
and the thumbnail assertion, streamlit for the app render test — skipped
when absent. Fixtures are duplicated, not imported, from the other test
modules (no cross-test imports in this repo).
"""
import csv
import html
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autorefine import (
    AutoRefineEnv,
    BanditPolicy,
    Budget,
    CsvTask,
    DashboardRunner,
    ParityCurriculum,
)
from autorefine.cli import _report_extras, main as cli_main
from autorefine.config import DEFAULT_SPEC, ModelSpec
from autorefine.diagnostics import holdout_diagnostics
from autorefine.plotting import (
    html_report,
    svg_architecture,
    svg_audio_waveform,
    svg_loss_curves,
)
from autorefine.tasks import AudioTask, ImageTask, SineRegressionV1
from autorefine.tasks.cartpole import CartPoleV1
from autorefine.trainer import LOSS_HISTORY_MAX, train

# --- fixtures (duplicated per the no-cross-test-imports house rule) ---------


def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    Same shape as tests/test_dashboard.py's fixture."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _cls_csv(path: Path) -> Path:
    """400-row non-linear 2-D classification (quadrant XOR). Same shape as
    tests/test_data_cli.py's fixture (a fuller holdout for CLI runs)."""
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = ((x1 > 0) == (x2 > 0)).astype(int)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "x1", "x2", "churn"])
        for i in range(n):
            w.writerow(["A" if i % 2 else "B", f"{x1[i]:.5f}", f"{x2[i]:.5f}",
                        int(y[i])])
    return path


def _drive_to_done(env, seed: int = 7) -> dict:
    state = env.reset()
    policy = BanditPolicy(seed=seed)
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    return env.memory.load_summary()


class _AlwaysClass:
    """A deterministic fake model: `forward`'s argmax is always `idx`."""

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros((x.shape[0], 2), dtype=np.float64)
        out[:, self.idx] = 1.0
        return out


class _AllClass0Task:
    """A minimal softmax task whose holdout is all class 0 (class 1 empty)
    — the C2 empty-class invariance case (SPEC.md 28.2)."""

    head = "softmax"
    class_values = [0.0, 1.0]

    def holdout_rows(self, n: int, model=None):
        x = np.zeros((5, 2))
        y = np.zeros(5, dtype=np.int64)
        n = min(int(n), len(y))
        return x[:n], y[:n]


# --- C1: per-experiment training curves (SPEC.md 28.1) ----------------------


def test_loss_history_sampling_matches_spec_formula():
    """t_i = 1 + round(i·(T−1)/(M−1)), M = min(48, T) (SPEC.md 28.1), for
    T = 200 (the min of TRAIN_STEPS_RANGE)."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    T = 200
    spec = ModelSpec(architecture=(16, 8), optimizer="momentum",
                     learning_rate=1e-3, batch_size=32, weight_decay=1e-4,
                     train_steps=T, input_noise=0.02, activation="tanh")
    r = train(ds, spec, seed=7, n_out=task.n_outputs, head=task.head)
    hist = r.loss_history
    M = min(LOSS_HISTORY_MAX, T)
    expected = sorted(1 + round(i * (T - 1) / (M - 1)) for i in range(M))
    steps = [rec["step"] for rec in hist]
    assert steps == expected
    assert steps[0] == 1 and steps[-1] == T  # hand-computed endpoints
    assert len(steps) == M
    assert all(b > a for a, b in zip(steps, steps[1:]))  # strictly increasing
    for rec in hist:
        assert set(rec) == {"step", "train", "holdout"}
        assert math.isfinite(rec["train"]) and math.isfinite(rec["holdout"])


def test_loss_history_single_record_tree_boost_knn():
    """tree / boost / knn each yield exactly one record (SPEC.md 28.1);
    knn memorizes → train == holdout."""
    task = CartPoleV1(seed=7)
    ds = task.make_dataset()
    base = dict(optimizer="sgd", learning_rate=1e-3, batch_size=32,
                weight_decay=0.0, input_noise=0.0, activation="tanh")
    for family, arch in (("tree", (2,)), ("boost", (2,)), ("knn", ())):
        spec = ModelSpec(architecture=arch, model_family=family,
                         train_steps=1000, **base)
        r = train(ds, spec, seed=7, n_out=task.n_outputs, head=task.head)
        assert len(r.loss_history) == 1
        rec = r.loss_history[0]
        assert rec["step"] == 1
        assert math.isfinite(rec["train"]) and math.isfinite(rec["holdout"])
        if family == "knn":
            assert rec["train"] == rec["holdout"]


def test_env_log_rows_carry_loss_history(tmp_path):
    """The env's baseline / experiment rows carry `loss_history`; invalid
    rows carry none (SPEC.md 28.1, A18)."""
    p = _write_csv(tmp_path)
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p)}, dataset_episodes=48)
    summary = _drive_to_done(env)
    assert summary["task"] == "csv"
    entries = env.memory.load_experiments()
    scored = [e for e in entries if e.get("kind") in ("baseline", "experiment")]
    assert scored
    for e in scored:
        hist = e.get("loss_history")
        assert isinstance(hist, list) and hist
        for rec in hist:
            assert set(rec) == {"step", "train", "holdout"}
            assert all(math.isfinite(rec[k]) for k in ("step", "train",
                                                        "holdout"))


def test_invalid_spec_row_has_no_loss_history(tmp_path):
    """A convnet on a flat task is a train-time SpecError → the logged
    invalid_spec row never carries `loss_history` (SPEC.md 25.4/28.1)."""
    env = AutoRefineEnv(task="cartpole-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path / "runs")
    env.reset()
    bad = DEFAULT_SPEC.to_dict()
    bad.update({"model_family": "convnet", "architecture": [8, 16]})
    _state, _r, _d, info = env.step(bad)
    assert info["reason"] == "invalid_spec"
    entries = env.memory.load_experiments()
    invalid = [e for e in entries if e.get("kind") == "invalid_spec"]
    assert len(invalid) == 1
    assert "loss_history" not in invalid[0]


def test_curriculum_rows_carry_loss_history_and_labels(tmp_path):
    """Curriculum step-up rows carry `loss_history` and the loss_curves
    labels are correct per kind (SPEC.md 28.1, A18)."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(3, 300, 30),
                        runs_dir=tmp_path / "cur",
                        curriculum=ParityCurriculum(seed=7,
                                                     trigger_fraction=0.001))
    _drive_to_done(env)
    entries = env.memory.load_experiments()
    cur_rows = [e for e in entries if e.get("kind") == "curriculum"]
    assert cur_rows  # trigger_fraction=0.001 forces a step-up
    for e in cur_rows:
        assert isinstance(e.get("loss_history"), list) and e["loss_history"]
    curves = DashboardRunner._loss_curves(entries)
    assert "baseline" in curves
    assert "experiment 1" in curves
    assert "curriculum 1" in curves


def test_runner_updates_carry_loss_history_json_safe(tmp_path):
    """The runner's per-step updates carry `loss_history` and stay
    JSON-safe (SPEC.md 23.1/28.1, A18)."""
    r = DashboardRunner(csv_path=str(_write_csv(tmp_path)), seed=7,
                        experiments=3, policy="bandit",
                        runs_dir=str(tmp_path / "runs"))
    r.start()
    n = 0
    while not r.done:
        u = r.next()
        json.dumps(u, sort_keys=True)  # every row JSON-safe
        assert isinstance(u["loss_history"], list)
        n += 1
    assert n >= 1


def test_svg_loss_curves_valid_deterministic_and_hand_computed():
    hist = [{"step": 1, "train": 1.0, "holdout": 1.0},
            {"step": 2, "train": 0.5, "holdout": 0.75}]
    svg = svg_loss_curves(hist)
    ET.fromstring(svg)  # valid XML (G2)
    # w=640, h=360: L=72, T=28, R=24, B=44 -> pw=544, ph=288; lo=0.5, hi=1.0
    # x(1)=72, x(2)=616; y(1.0)=28, y(0.75)=172, y(0.5)=316
    assert "points=\"72.0,28.0 616.0,316.0\"" in svg  # train line
    assert "points=\"72.0,28.0 616.0,172.0\"" in svg  # holdout line
    assert "loss over training steps" in svg
    assert svg_loss_curves(hist) == svg  # deterministic (G2)
    empty = svg_loss_curves([])
    ET.fromstring(empty)
    assert "no loss history recorded" in empty


def test_html_report_per_row_curve_column():
    """report.html gains the per-row curve column only when some entry has
    a non-empty `loss_history` (SPEC.md 28.1, A18)."""
    summary = {"task": "csv", "seed": 7}
    row = {"kind": "baseline", "accepted": True, "holdout_score": 50.0,
           "gen_gap": 0.0, "train_seconds": 1.0}
    plain = html_report(summary, [dict(row)])
    assert "<th>curve</th>" not in plain  # old-run shape stays byte-identical
    with_hist = dict(row, loss_history=[{"step": 1, "train": 0.9,
                                         "holdout": 1.0}])
    html = html_report(summary, [with_hist])
    assert "<th>curve</th>" in html
    assert "loss over training steps" in html


# --- C2: final-model diagnostics (SPEC.md 28.2) ------------------------------


def test_holdout_diagnostics_invariants_on_fixed_csv(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    task = CsvTask(seed=7, path=str(p), label="churn")
    y = np.asarray(task.holdout_rows(200, None)[1])
    c0, c1 = int((y == 0).sum()), int((y == 1).sum())
    for idx in (0, 1):
        d = holdout_diagnostics(task, _AlwaysClass(idx))
        assert d is not None
        assert d["n"] == c0 + c1
        assert d["class_counts"] == [c0, c1]
        assert d["class_labels"] == ["0", "1"]
        for i in range(2):  # row sums = class_counts
            assert sum(d["confusion"][i]) == d["class_counts"][i]
        assert d["correct"] == d["confusion"][0][0] + d["confusion"][1][1]
        for i in range(2):  # the per-class formula
            cnt = d["class_counts"][i]
            want = 100.0 * d["confusion"][i][i] / cnt if cnt else 0.0
            assert d["per_class"][i] == pytest.approx(want)
        # an always-class model puts all mass in column `idx`
        want_conf = [[c0 if idx == 0 else 0, c0 if idx == 1 else 0],
                     [c1 if idx == 0 else 0, c1 if idx == 1 else 0]]
        assert d["confusion"] == want_conf


def test_holdout_diagnostics_empty_class_is_zero():
    """A class absent from the holdout reads 0.0, never a division by zero
    (SPEC.md 28.2)."""
    d = holdout_diagnostics(_AllClass0Task(), _AlwaysClass(0))
    assert d["confusion"] == [[5, 0], [0, 0]]
    assert d["class_counts"] == [5, 0]
    assert d["per_class"] == [100.0, 0.0]
    assert d["correct"] == 5


def test_holdout_diagnostics_none_for_mse_and_episode_tasks():
    """C2 is classification-only (SPEC.md 28.5): an mse task and a task
    without `holdout_rows` both return None."""
    assert holdout_diagnostics(SineRegressionV1(seed=7),
                               _AlwaysClass(0)) is None
    assert holdout_diagnostics(CartPoleV1(seed=7), _AlwaysClass(0)) is None


def test_fit_cli_prints_diagnostics_and_weakest_hint(tmp_path, capsys):
    p = _cls_csv(tmp_path / "cls.csv")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--label", "churn",
                   "--experiments", "2", "--policy", "bandit", "--seed", "7",
                   "--target", "1.0", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out
    assert "holdout diagnostics (SPEC.md 28.2):" in out
    assert "class 0" in out and "class 1" in out
    # the weakest-class hint appears iff some class is below 100%
    dirs = sorted(runs.glob("csv-seed7-*"))
    summary = json.loads((dirs[0] / "summary.json").read_text(encoding="utf-8"))
    extras = _report_extras(summary, dirs[0])
    any_below = bool(extras["diagnostics"]) and any(
        v < 100.0 for v in extras["diagnostics"]["per_class"])
    assert ("the model fails on class" in out) == any_below


def test_report_plot_writes_learning_view_svgs(tmp_path):
    p = _cls_csv(tmp_path / "cls.csv")
    env = AutoRefineEnv(task="csv", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path / "runs",
                        task_config={"path": str(p), "label": "churn"},
                        dataset_episodes=320)
    _drive_to_done(env)
    assert cli_main(["report", "--run", str(env.run_dir), "--plot"]) == 0
    for name in ("score_curve.svg", "pareto_frontier.svg",
                 "per_class.svg", "confusion.svg", "architecture.svg"):
        f = Path(env.run_dir) / name
        assert f.is_file()
        ET.fromstring(f.read_text(encoding="utf-8"))  # valid XML (G2)


def test_html_report_extras_sections():
    diag = {"n": 9, "correct": 8, "per_class": [100.0, 60.0],
            "confusion": [[5, 0], [1, 3]], "class_counts": [5, 4],
            "class_labels": ["0", "1"]}
    arch = svg_architecture(DEFAULT_SPEC.to_dict(), 4, 2)
    html = html_report({"task": "csv", "seed": 7}, [],
                       diagnostics=diag, arch_svg=arch)
    assert "<h2>Holdout diagnostics</h2>" in html
    assert "per-class accuracy (holdout)" in html
    assert "confusion matrix (rows true, columns predicted)" in html
    assert "<h2>Architecture</h2>" in html
    assert html == html_report({"task": "csv", "seed": 7}, [],
                               diagnostics=diag, gallery=None, arch_svg=arch)


def test_html_report_none_kwargs_byte_identical():
    """`html_report(summary, entries)` with the v0.14 kwargs at None is
    byte-identical (SPEC.md 22.2/28, A18)."""
    summary = {"task": "csv", "seed": 7, "baseline_score": 50.0,
               "final_best_score": 70.0, "best_spec": DEFAULT_SPEC.to_dict()}
    entries = [{"kind": "baseline", "accepted": True, "holdout_score": 50.0,
                "gen_gap": 0.0, "train_seconds": 1.0}]
    assert html_report(summary, entries) == \
        html_report(summary, entries, diagnostics=None, gallery=None,
                    arch_svg=None)


# --- C3: error gallery (SPEC.md 28.3) ----------------------------------------


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 1.0,
               amp: float = 0.6, phase: float = 0.0) -> None:
    """16-bit PCM mono WAV — stdlib only (tests/test_media_tasks.py shape)."""
    import wave
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = amp * np.sin(2 * math.pi * freq * t + phase)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _tone_dir(tmp_path: Path, n: int = 20) -> Path:
    """Two tone classes (220 vs 440 Hz), one subfolder per class (24.2)."""
    d = tmp_path / "tones"
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _write_wav(sub / f"c{i:02d}.wav", freq)
    return d


def _make_image(path: Path, kind: str, size: int = 64) -> None:
    pytest.importorskip("PIL",
                        reason="image fixtures need autorefine[image] (SPEC.md 24.1)")
    from PIL import Image
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    if kind == "ring":
        a = (np.abs(np.hypot(xx - cx, yy - cy) - size * 0.30)
             < size * 0.07).astype(np.uint8) * 255
    else:
        a = (np.minimum(np.abs(xx - cx), np.abs(yy - cy))
             < size * 0.07).astype(np.uint8) * 255
    Image.fromarray(a, mode="L").save(str(path))


def _shape_dir(tmp_path: Path, n: int = 100) -> Path:
    d = tmp_path / "shapes"
    for name, kind in (("ring", "ring"), ("cross", "cross")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _make_image(sub / f"s{i:02d}.png", kind)
    return d


def test_image_holdout_errors_from_canonical_labels(tmp_path):
    d = _shape_dir(tmp_path)
    task = ImageTask(seed=7, path=str(d))
    errors = task.holdout_errors(_AlwaysClass(0))
    assert 0 < len(errors) <= 8  # n_max cap (SPEC.md 28.3)
    for it in errors:
        assert set(it) >= {"file", "path", "label", "predicted"}
        # the model always predicts class 0 ("cross") → errors are "ring" items
        assert it["label"] == "ring" and it["predicted"] == "cross"
        assert Path(it["path"]).is_file()
    # the n_max cap is exact relative to the (deterministic, seed=7) number of
    # ring holdout items; the fixture is large enough that the cap kicks in
    n_ring = len(task.holdout_errors(_AlwaysClass(0), n_max=10**9))
    assert n_ring >= 3
    assert len(task.holdout_errors(_AlwaysClass(0), n_max=3)) == 3
    assert len(task.holdout_errors(_AlwaysClass(0), n_max=10**9)) == n_ring
    assert task.holdout_errors(_AlwaysClass(0), n_max=0) == []


def test_audio_holdout_errors_carry_waveform(tmp_path):
    d = _tone_dir(tmp_path)
    task = AudioTask(seed=7, path=str(d))
    errors = task.holdout_errors(_AlwaysClass(0))
    assert 0 < len(errors) <= 8
    for it in errors:
        assert it["label"] == "low" and it["predicted"] == "high"
        wf = it["waveform"]
        assert 1 <= len(wf) <= 512
        assert all(math.isfinite(v) for v in wf)
        assert it["sample_rate"] == 16000


def test_svg_audio_waveform_valid_deterministic_empty():
    vals = [0.1, -0.2, 0.3, 0.5]
    svg = svg_audio_waveform(vals, sample_rate=16000, title="c00.wav")
    ET.fromstring(svg)  # valid XML (G2)
    assert "sample rate: 16000 Hz" in svg
    assert svg_audio_waveform(vals, sample_rate=16000,
                              title="c00.wav") == svg  # deterministic
    empty = svg_audio_waveform([])
    ET.fromstring(empty)
    assert "no waveform data" in empty


def test_html_report_carries_gallery(tmp_path):
    img = tmp_path / "s0.png"
    _make_image(img, "ring")  # importorskip PIL
    gallery = [
        {"file": "s0.png", "path": str(img), "label": "ring",
         "predicted": "cross"},
        {"file": "c0.wav", "path": str(tmp_path / "c0.wav"), "label": "low",
         "predicted": "high", "waveform": [0.1, -0.2, 0.3],
         "sample_rate": 16000},
    ]
    html = html_report({"task": "image", "seed": 7}, [], gallery=gallery)
    assert "<h2>Error gallery</h2>" in html
    assert "true ring -> predicted cross" in html
    assert "true low -> predicted high" in html
    assert "data:image/png;base64," in html  # PIL present: thumbnail (28.3)
    assert "sample rate: 16000 Hz" in html
    assert html_report({"task": "image", "seed": 7}, [], gallery=None) == \
        html_report({"task": "image", "seed": 7}, [])


# --- C4: spec → architecture diagram (SPEC.md 28.4) ---------------------------


def _spec_dict(**kw) -> dict:
    base = DEFAULT_SPEC.to_dict()
    base.update(kw)
    return base


def test_svg_architecture_mlp_blocks_and_annotations():
    svg = svg_architecture(_spec_dict(), 4, 2)
    ET.fromstring(svg)
    assert "best spec architecture (mlp)" in svg
    assert "hidden 16" in svg and "hidden 8" in svg
    assert "activation tanh" in svg
    assert "input (4)" in svg and "output (2)" in svg
    assert "optimizer: momentum" in svg
    # defaults → no extra annotation rows (SPEC.md 28.4)
    for note in ("lr schedule:", "early stopping:", "gradient clip:",
                 "label smoothing:"):
        assert note not in svg
    # depth 0 → a linear block
    assert "linear" in svg_architecture(_spec_dict(architecture=[]), 4, 2)


def test_svg_architecture_convnet_pipeline():
    svg = svg_architecture(_spec_dict(model_family="convnet",
                                      architecture=[8, 16]), 1, 2)
    ET.fromstring(svg)
    # `->` is html.escape'd to `-&gt;` in the valid-XML text (G2); decode
    # before asserting the human labels the browser will render.
    decoded = html.unescape(svg)
    assert "best spec architecture (convnet)" in decoded
    assert svg.count("conv 3x3") == 2
    assert svg.count("pool 2x2") == 2
    assert "1 -> 8 filters" in decoded
    assert "8 -> 16 filters" in decoded
    assert "FC 32" in decoded


def test_svg_architecture_tree_boost_knn():
    tree = svg_architecture(_spec_dict(model_family="tree", architecture=[2],
                                       train_steps=1000), 4, 2)
    ET.fromstring(tree)
    assert "10 trees (depth 2)" in tree  # train_steps // 100, clamped 2..50
    assert "bagged" in tree
    boost = svg_architecture(_spec_dict(model_family="boost",
                                        architecture=[2], train_steps=200),
                             4, 2)
    ET.fromstring(boost)
    assert "2 rounds (depth 2)" in boost  # 200 // 100 = 2 (the clamp floor)
    assert "boosted" in boost
    knn = svg_architecture(_spec_dict(model_family="knn", knn_k=11), 4, 2)
    ET.fromstring(knn)
    assert "k nearest neighbors" in knn
    assert "k = 11" in knn


def test_svg_architecture_non_default_annotations():
    svg = svg_architecture(_spec_dict(lr_schedule="warmup_cosine",
                                      early_stopping_patience=10,
                                      gradient_clipping=1.0,
                                      label_smoothing=0.1), 4, 2)
    ET.fromstring(svg)
    assert "lr schedule: warmup_cosine" in svg
    assert "early stopping: patience 10" in svg
    assert "gradient clip: 1" in svg
    assert "label smoothing: 0.1" in svg


def test_svg_architecture_accepts_dict_and_modelspec_and_none():
    d = _spec_dict()
    assert svg_architecture(d, 4, 2) == \
        svg_architecture(ModelSpec.from_dict(d), 4, 2)
    empty = svg_architecture(None, 4, 2)
    ET.fromstring(empty)
    assert "no spec to render" in empty


# --- runner/app (SPEC.md 28.6, A18) -------------------------------------------


def test_finish_carries_all_learning_view_keys(tmp_path):
    r = DashboardRunner(csv_path=str(_write_csv(tmp_path)), seed=7,
                        experiments=4, policy="bandit",
                        runs_dir=str(tmp_path / "runs"))
    r.start()
    res = r.run_all()
    for key in ("loss_curves", "diagnostics", "per_class_svg",
                "confusion_svg", "error_gallery", "arch_svg"):
        assert key in res
    assert res["diagnostics"] is not None
    assert res["per_class_svg"].startswith("<svg")
    assert res["confusion_svg"].startswith("<svg")
    assert res["error_gallery"] is None  # csv: no gallery (SPEC.md 28.5)
    assert res["arch_svg"].startswith("<svg")
    # loss_curves labels: baseline + one per experiment row with a history
    expected = set()
    n_exp = 0
    for e in r.env.memory.load_experiments():
        if not e.get("loss_history"):
            continue
        if e.get("kind") == "baseline":
            expected.add("baseline")
        elif e.get("kind") == "experiment":
            n_exp += 1
            expected.add(f"experiment {n_exp}")
    assert set(res["loss_curves"]) == expected


def test_app_renders_learning_views(tmp_path):
    """A18: the app renders the "Learning views" subheader (streamlit
    optional — the §3/§21.1 pattern)."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "src" / "autorefine" / \
        "dashboard_app.py"
    csv_path = _write_csv(tmp_path)
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()  # first render materializes the sidebar widgets
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception
    subs = [h.value for h in at.subheader]
    assert "Learning views" in subs
    md = " ".join(m.value for m in at.markdown)
    assert "loss over training steps" in md  # C1 curve
    assert "per-class accuracy (holdout)" in md  # C2
    assert "confusion matrix (rows true, columns predicted)" in md  # C2
    assert "best spec architecture" in md  # C4


def test_import_autorefine_never_pulls_in_learning_view_deps():
    code = (
        "import sys, autorefine; "
        "bad = [m for m in ('PIL', 'streamlit', 'soundfile') if m in sys.modules]; "
        "assert not bad, bad"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
