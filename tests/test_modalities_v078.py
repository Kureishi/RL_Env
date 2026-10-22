"""v0.78 seven new data modalities + upload→utilization flows — SPEC.md 92, A82.

SPEC.md 92 (v0.78): the data space widens from CSV + image/audio/text to
ten modalities, all on the shared flat protocol (22.1) — so the improver,
the RL catalog, and every reporting view work on them unchanged — plus the
five flows (B1–B5) that close the last mile of "data in → model out":

  * A1 (92.1)  ``SignalTask``    — .sig/.npy/.dat 1-D signal clips (38-d);
  * A2 (92.2)  ``JsonlTask``     — .jsonl/.json log records, flattened (92.2.1);
  * A3 (92.3)  ``MidiTask``      — .mid/.midi music (35-d, pure-stdlib parser);
  * A4 (92.4)  ``PointCloudTask`` — .ply/.obj 3-D clouds (83-d);
  * A5 (92.5)  ``GraphTask``     — nodes.csv + edges.csv (1-hop aggregation);
  * A6 (92.6)  ``VideoTask``     — .mp4/.mov/.avi clips (8199-d, imageio optional);
  * A7 (92.7)  ``MixedTask``     — multi-modality labelled dirs (1031-d);
  * B1 (92.8)  ``discover_data`` finds the new modalities + .jsonl files;
  * B2 (92.9)  ``fit``/``go --data -`` reads a piped CSV table (stdin → temp);
  * B3 (92.10) ``autorefine export`` → a portable numpy-only zip + predict.py;
  * B4 (92.11) the dashboard upload surface accepts all of it;
  * B5 (92.12) ``predict --data`` is an alias for ``--csv``.

House rules: stdlib + numpy only, no cross-test imports (every fixture is
synthesized here), deterministic (G2), tiny budgets (200 steps, a 50 bar
where a PASS exit code is wanted), and one module-scoped real-run fixture
where a finished run is needed.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.cli import _resolve_fit_task, main as cli_main
from autorefine.config import DEFAULT_SPEC, ModelSpec
from autorefine.discover import CANDIDATE_KEYS, discover_data
from autorefine.exporter import PORTABLE_FAMILIES, export_run
from autorefine.predict import (csv_rows_to_features, predict_features,
                                standardize)
from autorefine.tasks import TASKS
from autorefine.tasks.csv import CsvTask
from autorefine.tasks.graph import (GraphTask, aggregate_neighbors,
                                    load_graph)
from autorefine.tasks.jsonl import JsonlTask, flatten_record
from autorefine.tasks.media import ROW_MODALITIES, detect_modality
from autorefine.tasks.midi import MidiTask, midi_features, parse_midi
from autorefine.tasks.mixed import MixedTask, mixed_row
from autorefine.tasks.pointcloud import (PointCloudTask, load_pointcloud,
                                         pointcloud_features)
from autorefine.tasks.signals import (SignalTask, load_signal,
                                      signal_features)
from autorefine.tasks.video import (VideoTask, _grid_resize, _to_gray,
                                    video_features)
from autorefine.trainer import train_from_task

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"
APP = REPO / "src" / "autorefine" / "dashboard_app.py"

# v0.78 (SPEC.md 92, A82.8): the round's new top-level exports.
NEW_TASK_EXPORTS = (
    "SignalTask", "JsonlTask", "MidiTask", "PointCloudTask",
    "GraphTask", "VideoTask", "MixedTask",
)


# --- shared fixtures (test-local; the house rules forbid cross-test imports) ---

def _spec(family: str) -> ModelSpec:
    """A valid, tiny spec for `family` (200 steps; tree/boost need (depth,))."""
    d = {**DEFAULT_SPEC.to_dict(), "model_family": family, "train_steps": 200}
    if family in ("tree", "boost"):
        d["architecture"] = [2]  # tree/boost: architecture = (depth,) in 1..3
    return ModelSpec.from_dict(d)


def _write_csv(tmp_path: Path, name: str = "data.csv",
               rows: int = 60) -> Path:
    """Deterministic quadrant-XOR classification CSV (2 classes, 60 rows)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _csv_text(rows: int = 60) -> str:
    """The same quadrant-XOR CSV as text (for the B2 stdin fixtures)."""
    lines = ["a,b,churn"]
    for i in range(rows):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    return "\n".join(lines) + "\n"


def _sine(freq: float, n: int = 128) -> np.ndarray:
    """A deterministic 1-D signal (fundamental + one harmonic)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 3 * freq * t)


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 0.5) -> None:
    """16-bit PCM mono WAV (stdlib `wave` — zero dependencies)."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = 0.6 * np.sin(2 * np.pi * freq * t)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _write_signal_dir(tmp_path: Path) -> Path:
    """``low/`` + ``high/`` subfolders, 3 signal clips each (.sig / .npy)."""
    root = tmp_path / "signals"
    for cls, freq in (("low", 2.0), ("high", 14.0)):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(3):
            arr = _sine(freq)
            if i == 2:
                np.save(str(d / f"{cls}{i}.npy"), arr.astype(np.float32))
            else:
                (d / f"{cls}{i}.sig").write_text(
                    " ".join(f"{v:.6f}" for v in arr) + "\n", encoding="utf-8")
    return root


def _write_jsonl(tmp_path: Path, name: str = "logs.jsonl") -> Path:
    """8 records exercising the 92.2.1 leaf shapes (nested / string /
    list / bool / null)."""
    records = []
    for i in range(8):
        cls = i % 2
        records.append({
            "label": cls,
            "a": float(cls),
            "nested": {"b": cls * 1.5, "c": "cat" if cls else "other"},
            "list": [cls, cls * 2, cls * 3],
            "flag": bool(cls),
            "none": None,
        })
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                 encoding="utf-8")
    return p


def _vlq(v: int) -> bytes:
    out = bytes([v & 0x7F])
    v >>= 7
    while v:
        out = bytes([(v & 0x7F) | 0x80]) + out
        v >>= 7
    return out


def _write_mid(path: Path, pitches, division: int = 960,
               tempo_us: int = 500000) -> None:
    """One tiny format-0 MIDI file: set-tempo + note-on events.

    The parser's channel-voice convention (midi.py) is low-nibble 0x1 =
    note-on (velocity in byte 2) — a self-consistent simplification, not
    real-MIDI status bytes.
    """
    trk = b""
    trk += bytes([0x00, 0xFF, 0x51, 0x03]) + int(tempo_us).to_bytes(3, "big")
    for delta, pitch in enumerate(pitches):
        trk += _vlq(delta * 24) + bytes([0x91, pitch, 100])
    trk += bytes([0x00, 0xFF, 0x2F, 0x00])  # end of track
    header = (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
              + (1).to_bytes(2, "big") + int(division).to_bytes(2, "big"))
    path.write_bytes(header + b"MTrk" + len(trk).to_bytes(4, "big") + trk)


def _write_midi_dir(tmp_path: Path) -> Path:
    root = tmp_path / "midi"
    classes = {"jazz": [60, 62, 67, 72, 76], "rock": [40, 45, 50, 55, 59]}
    for cls, pitches in classes.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(3):
            _write_mid(d / f"{cls}{i}.mid", pitches)
    return root


def _write_obj(path: Path, points) -> None:
    lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for (x, y, z) in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pc_dir(tmp_path: Path) -> Path:
    """``round/`` (a thin shell) vs ``flat/`` (a disk) point clouds."""
    root = tmp_path / "pointclouds"
    ang = np.linspace(0.0, 2 * np.pi, 24)
    shell = [(np.cos(a), np.sin(a), 0.5) for a in ang]
    disk = [(np.cos(a), np.sin(a), 0.02) for a in ang]
    for cls, pts in (("round", shell), ("flat", disk)):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(3):
            _write_obj(d / f"{cls}{i}.obj", pts)
    return root


def _write_graph_dir(tmp_path: Path) -> Path:
    """A tiny two-community graph (nodes.csv + edges.csv, 6 nodes)."""
    root = tmp_path / "graph"
    root.mkdir(parents=True)
    nodes = ["id,f1,f2,label"]
    data = [(0, 0.1, 0.2, 0), (1, 0.2, 0.1, 0), (2, 0.3, 0.2, 0),
            (3, 0.8, 0.9, 1), (4, 0.9, 0.8, 1), (5, 0.7, 0.9, 1)]
    for i, f1, f2, lab in data:
        nodes.append(f"{i},{f1},{f2},{lab}")
    (root / "nodes.csv").write_text("\n".join(nodes) + "\n", encoding="utf-8")
    edges = ["source,target", "0,1", "1,2", "0,2", "3,4", "4,5", "3,5"]
    (root / "edges.csv").write_text("\n".join(edges) + "\n", encoding="utf-8")
    return root


def _write_mixed_dir(tmp_path: Path) -> Path:
    """``low/`` + ``high/`` mixing signal (.sig) and audio (.wav) rows."""
    root = tmp_path / "mixed"
    layout = {
        "low": [("s1", "sig"), ("a1", "wav"), ("s9", "sig")],
        "high": [("s2", "sig"), ("a2", "wav"), ("a9", "wav")],
    }
    for cls, items in layout.items():
        d = root / cls
        d.mkdir(parents=True)
        for name, ext in items:
            if ext == "sig":
                arr = _sine(2.0 if cls == "low" else 14.0)
                (d / f"{name}.sig").write_text(
                    " ".join(f"{v:.6f}" for v in arr) + "\n", encoding="utf-8")
            else:
                _write_wav(d / f"{name}.wav", 220.0 if cls == "low" else 880.0)
    return root


def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run one CLI command, capturing stdout + stderr (the house pattern)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def portable_runs(tmp_path_factory):
    """One synthetic finished run dir per portable family (A82.5, B3)."""
    root = tmp_path_factory.mktemp("export")
    csv = _write_csv(root)
    out: dict[str, dict] = {}
    for fam in PORTABLE_FAMILIES:
        task = CsvTask(seed=7, path=str(csv))
        result = train_from_task(task, _spec(fam), seed=11)
        rd = root / f"run_{fam}"
        rd.mkdir(parents=True)
        result.model.save(str(rd / "best_model.npz"))
        summary = {
            "task": "csv", "seed": 7,
            "task_config": {"path": str(csv)},
            "best_spec": {"model_family": fam},
            "final_best_score": 90.0,
        }
        (rd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        out[fam] = {"run_dir": rd, "model": result.model, "task": task,
                    "csv": csv, "family": fam}
    return out


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    """One real, tiny, finished ``go`` run (the A82.7 predict alias)."""
    root = tmp_path_factory.mktemp("modalities")
    csv = _write_csv(root)
    runs = root / "runs"
    rc, out, err = _run_cmd([
        "go", "--data", str(csv),
        "--runs-dir", str(runs),
        "--experiments", "3", "--target", "50", "--quiet",
    ])
    assert rc == 0, out + err
    run_dirs = sorted(runs.glob("csv-*"))
    assert run_dirs and (run_dirs[0] / "summary.json").is_file()
    return {"root": root, "csv": csv, "runs": runs, "run_dir": run_dirs[0]}


# --- A82.1 / A1 — SignalTask (92.1) -----------------------------------------

class TestSignal:
    """SPEC.md 92.1 (A82.1, A1): ``SignalTask`` — 1-D signal clips (38-d)."""

    def test_protocol_and_feature_dim(self, tmp_path):
        d = _write_signal_dir(tmp_path)
        t = SignalTask(seed=7, path=str(d))
        assert t.name == "signal"
        assert t.head == "softmax" and t.n_outputs == 2
        assert t.metric == "accuracy"
        assert t.state_dim == 38  # 6 stats + 16 lags + 16 bands
        assert t.class_values == ["high", "low"]

    def test_signal_features_shape(self):
        f = signal_features(_sine(3.0))
        assert f.shape == (38,)
        assert np.all(np.isfinite(f))

    def test_load_signal_round_trip(self, tmp_path):
        arr = _sine(5.0)
        (tmp_path / "a.sig").write_text(
            " ".join(f"{v:.6f}" for v in arr) + "\n", encoding="utf-8")
        (tmp_path / "b.npy").write_bytes(b"")
        np.save(str(tmp_path / "b.npy"), arr.astype(np.float32))
        assert np.allclose(load_signal(tmp_path / "a.sig"), arr, atol=1e-5)
        assert np.allclose(load_signal(tmp_path / "b.npy"), arr, atol=1e-3)

    def test_dataset_score_and_train_round_trip(self, tmp_path):
        d = _write_signal_dir(tmp_path)
        t = SignalTask(seed=7, path=str(d))
        X, y = t.make_dataset()
        assert X.shape == (4, 38) and y.shape == (4,)
        assert np.all(np.isfinite(t.feature_mean))
        assert np.all(t.feature_std > 0)
        score = t.score(_dummy_logit_model(), "holdout", 4)
        assert 0.0 <= score <= 100.0
        result = train_from_task(t, _spec("mlp"), seed=11)
        s = t.score(result.model, "holdout", 4)
        assert 0.0 <= s <= 100.0

    def test_constant_signal_fails_loud(self):
        with pytest.raises(ValueError):
            signal_features(np.zeros(64))


# --- A82.1 / A2 — JsonlTask (92.2) ------------------------------------------

class TestJsonl:
    """SPEC.md 92.2 (A82.1, A2): ``JsonlTask`` — flattened log records."""

    def test_flatten_record_leaf_shapes(self):
        rec = {"a": 1.0, "nested": {"b": 2.0, "c": "cat"},
               "list": [1, 2, 3], "flag": True, "none": None}
        flat = flatten_record(rec)
        assert flat["a"] == 1.0
        assert flat["nested.b"] == 2.0          # dotted key path
        assert flat["flag"] == 1.0              # bool → 0/1
        assert flat["list"] == 2.0              # list → mean of numerics
        assert "none" not in flat               # null dropped
        assert any(k.startswith("s") for k in flat)  # string → s<B> bucket
        for k, v in flat.items():
            if k.startswith("s"):
                assert v == 1.0  # one "cat" string in the record

    def test_protocol_and_score(self, tmp_path):
        p = _write_jsonl(tmp_path)
        t = JsonlTask(seed=7, path=str(p))
        assert t.name == "jsonl" and t.head == "softmax" and t.n_outputs == 2
        assert t.state_dim >= 3  # a, nested.b, the string bucket, list, flag
        assert t.class_values == [0.0, 1.0]
        assert np.all(np.isfinite(t.feature_mean))
        X, y = t.make_dataset()
        assert X.shape[1] == t.state_dim and X.shape[0] == 6  # 8 * 0.8
        score = t.score(_dummy_logit_model(), "holdout", 1)
        assert 0.0 <= score <= 100.0

    def test_missing_label_fails_loud(self, tmp_path):
        p = tmp_path / "nolabel.jsonl"
        p.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            JsonlTask(seed=7, path=str(p))


# --- A82.1 / A3 — MidiTask (92.3) -------------------------------------------

class TestMidi:
    """SPEC.md 92.3 (A82.1, A3): ``MidiTask`` — pure-stdlib MIDI parsing."""

    def test_parse_midi(self, tmp_path):
        p = tmp_path / "m.mid"
        pitches = [60, 62, 67, 72, 76]
        _write_mid(p, pitches)
        division, tempo, onsets = parse_midi(p)
        assert division == 960
        assert tempo == pytest.approx(60_000_000 / 500_000)  # 120 bps
        assert len(onsets) == 5
        assert [p_ for (_, p_) in onsets] == pitches

    def test_midi_features_shape(self, tmp_path):
        p = tmp_path / "m.mid"
        _write_mid(p, [60, 62, 67])
        division, tempo, onsets = parse_midi(p)
        f = midi_features(division, tempo, onsets)
        assert f.shape == (35,)
        assert np.all(np.isfinite(f))

    def test_protocol_and_score(self, tmp_path):
        d = _write_midi_dir(tmp_path)
        t = MidiTask(seed=7, path=str(d))
        assert t.state_dim == 35 and t.head == "softmax" and t.n_outputs == 2
        assert t.class_values == ["jazz", "rock"]
        X, y = t.make_dataset()
        assert X.shape == (4, 35)
        score = t.score(_dummy_logit_model(), "holdout", 4)
        assert 0.0 <= score <= 100.0


# --- A82.1 / A4 — PointCloudTask (92.4) -------------------------------------

class TestPointCloud:
    """SPEC.md 92.4 (A82.1, A4): ``PointCloudTask`` — 3-D clouds (83-d)."""

    def test_load_and_features(self, tmp_path):
        d = _write_pc_dir(tmp_path)
        pts = load_pointcloud(next(iter((d / "round").iterdir())))
        assert pts.shape == (24, 3)
        f = pointcloud_features(pts)
        assert f.shape == (83,)  # 19 scalars + 4^3 voxels
        assert np.all(np.isfinite(f))

    def test_protocol_and_score(self, tmp_path):
        d = _write_pc_dir(tmp_path)
        t = PointCloudTask(seed=7, path=str(d))
        assert t.state_dim == 83 and t.head == "softmax" and t.n_outputs == 2
        assert t.class_values == ["flat", "round"]
        X, y = t.make_dataset()
        assert X.shape == (4, 83)
        score = t.score(_dummy_logit_model(), "holdout", 4)
        assert 0.0 <= score <= 100.0


# --- A82.1 / A5 — GraphTask (92.5) ------------------------------------------

class TestGraph:
    """SPEC.md 92.5 (A82.1, A5): ``GraphTask`` — nodes + edges."""

    def test_load_and_aggregate(self, tmp_path):
        d = _write_graph_dir(tmp_path)
        X, labels, edges = load_graph(str(d))
        assert X.shape == (6, 2) and len(labels) == 6  # labels is a raw list
        assert len(edges) == 6
        agg = aggregate_neighbors(X, edges)
        assert agg.shape == X.shape
        assert np.all(np.isfinite(agg))

    def test_protocol_and_no_holdout_errors(self, tmp_path):
        d = _write_graph_dir(tmp_path)
        t = GraphTask(seed=7, path=str(d))
        assert t.state_dim == 2 and t.head == "softmax" and t.n_outputs == 2
        assert t.class_values == [0.0, 1.0]
        assert not hasattr(t, "holdout_errors")  # 92.5.2: no displayable media
        X, y = t.make_dataset()
        assert X.shape == (4, 2)
        score = t.score(_dummy_logit_model(), "holdout", 4)
        assert 0.0 <= score <= 100.0


# --- A82.1 / A6 — VideoTask (92.6) ------------------------------------------

class TestVideo:
    """SPEC.md 92.6 (A82.1, A6): ``VideoTask`` — clips (8199-d).

    The pure-numpy feature pipeline is exercised without ``imageio``; the
    full task build is exercised only when the optional ``[video]`` extra
    is present (monkeypatching its reader).
    """

    def test_pure_numpy_pipeline(self):
        g = np.arange(100, dtype=np.float64).reshape(10, 10)
        assert _grid_resize(g, 4).shape == (4, 4)
        rgb = np.zeros((8, 32, 32, 3), dtype=np.float64)
        gray = _to_gray(rgb[0])
        assert gray.shape == (32, 32)
        f = video_features(np.random.default_rng(0).random((8, 32, 32)))
        assert f.shape == (8199,)  # 8 * 32 * 32 + 7 motion deltas

    def test_task_builds(self, tmp_path, monkeypatch):
        iio = pytest.importorskip("imageio.v3",
                                  reason="video decoding needs [video] extra")
        rng = np.random.default_rng(3)
        frames = rng.integers(0, 256, size=(10, 48, 64, 3), dtype=np.uint8)
        monkeypatch.setattr(iio, "imread", lambda p: frames)
        root = tmp_path / "video"
        for cls in ("slow", "fast"):
            (root / cls).mkdir(parents=True)
            for i in range(3):
                (root / cls / f"v{i}.mp4").write_bytes(b"fake")
        t = VideoTask(seed=7, path=str(root))
        assert t.state_dim == 8199 and t.n_outputs == 2
        assert t.class_values == ["fast", "slow"]
        X, y = t.make_dataset()
        assert X.shape == (4, 8199)


# --- A82.1 / A7 — MixedTask (92.7) ------------------------------------------

class TestMixed:
    """SPEC.md 92.7 (A82.1, A7): ``MixedTask`` — multi-modality dirs (1031-d)."""

    def test_mixed_row_onehot_position(self, tmp_path):
        sig = tmp_path / "s.sig"
        sig.write_text(" ".join(f"{v:.6f}" for v in _sine(2.0)) + "\n",
                       encoding="utf-8")
        wav = tmp_path / "a.wav"
        _write_wav(wav, 440.0)
        for modality, path in (("signal", sig), ("audio", wav)):
            row = mixed_row(modality, path)
            assert row.shape == (1031,)  # 7 one-hot + 1024 row feature
            idx = ROW_MODALITIES.index(modality)
            assert row[idx] == 1.0 and sum(row[:7]) == 1.0  # exactly one-hot

    def test_protocol_and_score(self, tmp_path):
        d = _write_mixed_dir(tmp_path)
        t = MixedTask(seed=7, path=str(d))
        assert t.state_dim == 1031 and t.head == "softmax" and t.n_outputs == 2
        assert t.class_values == ["high", "low"]
        X, y = t.make_dataset()
        assert X.shape == (4, 1031)
        score = t.score(_dummy_logit_model(), "holdout", 4)
        assert 0.0 <= score <= 100.0

    def test_single_modality_dir_fails_loud(self, tmp_path):
        # one subfolder, one modality only → not a mixed task
        d = _write_signal_dir(tmp_path)  # signal clips only
        with pytest.raises(ValueError):
            MixedTask(seed=7, path=str(d))


# --- A82.2 — data → task resolvers ------------------------------------------

class TestResolvers:
    """SPEC.md 92.2/92.9 (A82.2): ``_resolve_fit_task`` over the new data."""

    def test_jsonl_file_auto_and_explicit(self, tmp_path):
        p = _write_jsonl(tmp_path)
        assert _resolve_fit_task(p, "auto") == "jsonl"
        assert _resolve_fit_task(p, "jsonl") == "jsonl"

    def test_csv_file_rejects_jsonl_and_vice_versa(self, tmp_path):
        csv = _write_csv(tmp_path)
        jl = _write_jsonl(tmp_path)
        with pytest.raises(ValueError):
            _resolve_fit_task(csv, "jsonl")   # a CSV file is not jsonl
        with pytest.raises(ValueError):
            _resolve_fit_task(jl, "csv")      # a .jsonl file is not csv

    def test_mixed_and_image_audio_dirs_auto_detect(self, tmp_path):
        mixed = _write_mixed_dir(tmp_path)
        assert detect_modality(mixed) == "mixed"
        assert _resolve_fit_task(mixed, "auto") == "mixed"
        # legacy pin: an image + audio dir is also mixed (SPEC.md 45.2.4)
        ia = tmp_path / "imgaud"
        (ia / "img").mkdir(parents=True)
        (ia / "aud").mkdir(parents=True)
        (ia / "img" / "a.png").write_bytes(b"png")
        (ia / "aud" / "a.wav").write_bytes(b"wav")
        assert _resolve_fit_task(ia, "auto") == "mixed"

    def test_dir_rejects_csv_task(self, tmp_path):
        d = _write_signal_dir(tmp_path)
        with pytest.raises(ValueError):
            _resolve_fit_task(d, "csv")


# --- A82.3 — B1: discover_data over the new modalities (92.8) ---------------

class TestB1:
    """SPEC.md 92.8 (A82.3, B1): ``discover_data`` finds the new data."""

    def test_candidates_cover_the_new_modalities(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        _write_csv(root, "data.csv")
        _write_jsonl(root, "logs.jsonl")
        sig = root / "signals"
        for cls, freq in (("low", 2.0), ("high", 14.0)):
            (sig / cls).mkdir(parents=True)
            for i in range(3):
                (sig / cls / f"{cls}{i}.sig").write_text(
                    " ".join(f"{v:.6f}" for v in _sine(freq)) + "\n",
                    encoding="utf-8")
        g = root / "graph"
        g.mkdir()
        (g / "nodes.csv").write_text("id,f1,f2,label\n0,0.1,0.2,0\n"
                                     "1,0.2,0.1,0\n2,0.3,0.2,0\n"
                                     "3,0.8,0.9,1\n4,0.9,0.8,1\n5,0.7,0.9,1\n",
                                     encoding="utf-8")
        (g / "edges.csv").write_text("source,target\n0,1\n3,4\n",
                                     encoding="utf-8")
        cands = discover_data(root)
        assert cands, "no data candidates found"
        for c in cands:
            assert set(c.keys()) == set(CANDIDATE_KEYS)
        by_task = {c["task"]: c for c in cands}
        assert {"csv", "jsonl", "signal", "graph"} <= set(by_task)
        assert by_task["csv"]["reason"] == "a CSV table"
        assert by_task["jsonl"]["reason"] == "a JSONL/JSON log"
        assert by_task["signal"]["reason"] == "a folder of signal items"
        assert by_task["graph"]["reason"] == "a folder of graph items"
        # files are listed before directories (the documented order)
        order = [c["task"] for c in cands]
        assert order.index("csv") < order.index("signal")


# --- A82.4 — B2: fit/go --data - from stdin (92.9) --------------------------

class TestB2:
    """SPEC.md 92.9 (A82.4, B2): ``fit``/``go --data -`` reads stdin."""

    def test_fit_reads_a_piped_csv(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        monkeypatch.setattr(sys, "stdin", io.StringIO(_csv_text()))
        rc, out, err = _run_cmd([
            "fit", "--data", "-", "--runs-dir", str(runs),
            "--experiments", "3", "--target", "50", "--quiet",
        ])
        assert rc == 0, out + err
        assert any(runs.glob("csv-*"))

    def test_go_reads_a_piped_csv(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        monkeypatch.setattr(sys, "stdin", io.StringIO(_csv_text()))
        rc, out, err = _run_cmd([
            "go", "--data", "-", "--runs-dir", str(runs),
            "--experiments", "3", "--target", "50", "--quiet",
        ])
        assert rc == 0, out + err
        assert any(runs.glob("csv-*"))

    def test_empty_stdin_is_an_error(self, tmp_path, monkeypatch):
        for verb in ("fit", "go"):
            monkeypatch.setattr(sys, "stdin", io.StringIO(""))
            rc, out, err = _run_cmd([
                verb, "--data", "-", "--runs-dir", str(tmp_path / "runs"),
                "--experiments", "2", "--target", "50", "--quiet",
            ])
            assert rc == 1, (verb, out, err)


# --- A82.5 — B3: autorefine export (92.10) ----------------------------------

class TestB3:
    """SPEC.md 92.10 (A82.5, B3): a portable numpy-only zip + predict.py."""

    def test_export_is_portable_and_complete(self, portable_runs, tmp_path):
        for fam in PORTABLE_FAMILIES:
            info = portable_runs[fam]
            out = tmp_path / f"{fam}.zip"
            res = export_run(info["run_dir"], out)
            assert res["portable"] is True and res["family"] == fam
            assert "predict.py" in res["files"]
            with zipfile.ZipFile(out) as zf:
                names = zf.namelist()
                assert set(names) == {"best_model.npz", "model_config.json",
                                      "predict.py", "README.md"}
                # best_model.npz is copied byte-for-byte
                assert zf.read("best_model.npz") == \
                    (info["run_dir"] / "best_model.npz").read_bytes()
                cfg = json.loads(zf.read("model_config.json"))
                assert cfg["model_family"] == fam
                assert cfg["state_dim"] == info["task"].state_dim
                assert len(cfg["feature_mean"]) == info["task"].state_dim
                assert len(cfg["feature_std"]) == info["task"].state_dim
                assert cfg["class_values"] == [
                    float(v) for v in info["task"].class_values]

    def test_export_is_deterministic(self, portable_runs, tmp_path):
        info = portable_runs["mlp"]
        a = export_run(info["run_dir"], tmp_path / "a.zip")
        b = export_run(info["run_dir"], tmp_path / "b.zip")
        assert a["out"] != b["out"]
        assert (tmp_path / "a.zip").read_bytes() == \
            (tmp_path / "b.zip").read_bytes()

    def test_standalone_predict_matches_the_library(self, portable_runs,
                                                    tmp_path):
        info = portable_runs["mlp"]
        out = export_run(info["run_dir"], tmp_path / "portable.zip")
        clean = tmp_path / "clean"
        clean.mkdir()
        with zipfile.ZipFile(out["out"]) as zf:
            zf.extractall(clean)
        new_csv = clean / "new_data.csv"
        new_csv.write_text("a,b\n0.1,0.2\n0.8,0.9\n0.5,0.5\n",
                           encoding="utf-8")
        task, model = info["task"], info["model"]
        raws = csv_rows_to_features(new_csv, task)
        lib = [predict_features(task, model, standardize(task, r))
               for r in raws]
        proc = subprocess.run(
            [sys.executable, "predict.py", "new_data.csv"],
            cwd=str(clean), capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        stand = [json.loads(line) for line in proc.stdout.splitlines()
                 if line.strip()]
        assert len(stand) == len(lib) == 3
        for l, s in zip(lib, stand):
            assert l["prediction"] == s["prediction"]
            for a, b in zip(l["probabilities"], s["probabilities"]):
                assert abs(a - b) < 1e-6

    def test_non_portable_family_omits_predict(self, tmp_path):
        # a gp run (non-portable): the zip says so and has no predict.py
        fam = "gp"
        task = CsvTask(seed=7, path=str(_write_csv(tmp_path)))
        result = train_from_task(task, _spec(fam), seed=11)
        rd = tmp_path / "run_gp"
        rd.mkdir()
        result.model.save(str(rd / "best_model.npz"))
        (rd / "summary.json").write_text(json.dumps({
            "task": "csv", "seed": 7, "task_config": {"path": str(task.path)},
            "best_spec": {"model_family": fam}, "final_best_score": 90.0,
        }), encoding="utf-8")
        out = tmp_path / "gp.zip"
        res = export_run(rd, out)
        assert res["portable"] is False and "predict.py" not in res["files"]
        with zipfile.ZipFile(out) as zf:
            readme = zf.read("README.md").decode("utf-8")
            assert "NOT included" in readme

    def test_broken_run_dir_raises(self, portable_runs, tmp_path):
        broken = tmp_path / "broken"
        broken.mkdir()
        with pytest.raises(ValueError):
            export_run(broken, tmp_path / "x.zip")

    def test_export_cli_broken_is_rc1(self, portable_runs, tmp_path):
        broken = tmp_path / "broken2"
        broken.mkdir()
        rc, out, err = _run_cmd([
            "export", "--run", str(broken), "--out", str(tmp_path / "y.zip")])
        assert rc == 1, out + err


# --- A82.6 — B4: the dashboard upload surface (92.11) -----------------------

class TestB4:
    """SPEC.md 92.11 (A82.6, B4): the app's upload surface accepts all."""

    def test_uploader_type_list_has_the_new_extensions(self):
        text = APP.read_text(encoding="utf-8")
        assert '"npy", "dat", "sig", "mid", "midi", "ply", "obj",' in text
        assert '"mp4", "mov", "avi", "jsonl", "json"' in text

    def test_resolve_index_and_preview(self, tmp_path):
        pytest.importorskip("streamlit", reason="dashboard app is optional")
        from autorefine.dashboard_app import (_preview_data,
                                              _resolve_index_to_dir)
        # an index.csv spanning two modalities resolves to its parent dir
        d = tmp_path / "mixed_idx"
        d.mkdir()
        (d / "s1.sig").write_text(
            " ".join(f"{v:.4f}" for v in _sine(2.0)) + "\n", encoding="utf-8")
        (d / "a1.wav").write_bytes(b"")
        _write_wav(d / "a1.wav", 440.0)
        (d / "index.csv").write_text(
            "file,label\ns1.sig,0\na1.wav,1\n", encoding="utf-8")
        assert _resolve_index_to_dir(d / "index.csv") == str(d)
        # a plain tabular CSV is not an index (resolves to None)
        assert _resolve_index_to_dir(_write_csv(tmp_path, "plain.csv")) is None
        # the preview accepts a mixed directory and a .jsonl file
        pv = _preview_data(str(_write_mixed_dir(tmp_path)), 0, 0)
        assert "error" not in pv
        pv2 = _preview_data(str(_write_jsonl(tmp_path)), 0, 0)
        assert pv2.get("kind") == "jsonl"


# --- A82.7 — B5: predict --data alias (92.12) --------------------------------

class TestB5:
    """SPEC.md 92.12 (A82.7, B5): ``predict --data`` aliases ``--csv``."""

    def test_data_is_an_alias_for_csv(self, champion, tmp_path):
        new_csv = _write_csv(tmp_path, "new.csv", rows=40)
        run = str(champion["run_dir"])
        rc_csv, out_csv, err_csv = _run_cmd(
            ["predict", "--run", run, "--csv", str(new_csv), "--json"])
        rc_data, out_data, err_data = _run_cmd(
            ["predict", "--run", run, "--data", str(new_csv), "--json"])
        assert rc_csv == 0 and rc_data == 0, (out_csv, err_csv, out_data, err_data)
        assert out_csv == out_data  # byte-identical JSON output

    def test_both_flags_together_is_an_error(self, champion, tmp_path):
        new_csv = _write_csv(tmp_path, "new.csv", rows=40)
        rc, out, err = _run_cmd([
            "predict", "--run", str(champion["run_dir"]),
            "--csv", str(new_csv), "--data", str(new_csv), "--json"])
        assert rc == 1, out + err


# --- A82.8 — ceremony ---------------------------------------------------------

class TestCeremony:
    """SPEC.md 92.13 (A82.8): the version step + index row + the exports."""

    def test_version_steps_to_078_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.84.0"' in init
        assert 'version = "0.84.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.84.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 92." in spec
        assert "### 92.13 Acceptance (A82)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M81 |"))
        assert "92" in row and "A82" in row
        assert "tests/test_modalities_v078.py" in row

    def test_new_task_classes_in_all(self):
        for name in NEW_TASK_EXPORTS:
            assert name in autorefine.__all__
            assert hasattr(autorefine, name)

    def test_registry_maps_the_new_tasks(self):
        assert TASKS["signal"] is SignalTask
        assert TASKS["jsonl"] is JsonlTask
        assert TASKS["midi"] is MidiTask
        assert TASKS["pointcloud"] is PointCloudTask
        assert TASKS["graph"] is GraphTask
        assert TASKS["video"] is VideoTask
        assert TASKS["mixed"] is MixedTask

    def test_dashboard_user_facing_strings_are_app_language(self):
        # The authoritative gate is tests/test_app_language.py (AST scan of
        # non-docstring strings). Here we only assert the module's user-facing
        # *uploader* strings stay clean; docstrings legitimately keep SPEC refs.
        import re as _re
        text = APP.read_text(encoding="utf-8")
        banned = _re.compile(r"SPEC\.md|SPEC_FIELDS|§|v0\.\d+")
        # Every st.file_uploader(...) help string must be app-language.
        for m in _re.finditer(r'help=\s*[\'"]([^\'"]*)[\'"]', text):
            assert not banned.search(m.group(1)), m.group(1)


# --- shared helper (test-local) ---------------------------------------------

def _dummy_logit_model():
    """A minimal two-output model for the score-in-range protocol checks."""
    class _Dummy:
        def forward(self, x):
            x = np.atleast_2d(np.asarray(x, dtype=np.float64))
            return np.zeros((x.shape[0], 2))
    return _Dummy()
