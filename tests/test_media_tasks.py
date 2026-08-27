"""v0.10 input modalities: ImageTask + AudioTask (SPEC.md 24, M13, A14).

Both modalities are tasks on the §22.1 protocol (SPEC.md 24.2); the
improver loop, gate, and artifacts are exercised through `fit` (24.5) and
the registry (15). Optional dependencies follow the §3/§21.1 pattern:
Pillow for images (24.1), soundfile for MP3 (24.1) — core stays clean.
"""
import builtins
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from autorefine import AutoRefineEnv
from autorefine.cli import main as cli_main
from autorefine.tasks import (
    AudioTask,
    ImageTask,
    TASKS,
    detect_modality,
    log_mel_features,
)


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 1.0,
               amp: float = 0.6, phase: float = 0.0) -> None:
    """16-bit PCM mono WAV (SPEC.md 24.4) — stdlib only."""
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
    """Two tone classes (220 vs 440 Hz), one subfolder per class (24.2)."""
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


def _make_image(path: Path, kind: str, size: int = 64) -> None:
    pytest.importorskip("PIL", reason="image fixtures need autorefine[image] (SPEC.md 24.1)")
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


def _shape_dir(tmp_path: Path, n: int = 20) -> Path:
    d = tmp_path / "shapes"
    for name, kind in (("ring", "ring"), ("cross", "cross")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _make_image(sub / f"s{i:02d}.png", kind)
    return d


# --- registry (SPEC.md 15, 24) -----------------------------------------------

def test_media_tasks_registered():
    assert TASKS["image"] is ImageTask
    assert TASKS["audio"] is AudioTask


# --- AudioTask (SPEC.md 24.4) -------------------------------------------------

def test_audio_task_protocol_and_determinism(tmp_path):
    d = _tone_dir(tmp_path)
    a = AudioTask(seed=7, path=d)
    assert a.head == "softmax" and a.n_outputs == 2
    assert a.class_values == ["high", "low"]  # sorted subfolder names (24.2)
    assert a.state_dim == 26  # default mel bands (SPEC.md 24.4)
    assert a.default_dataset_size == 32  # 80% of 40 (SPEC.md 20.3)
    x, y = a.make_dataset()
    assert x.shape == (32, 26) and set(np.unique(y)) == {0, 1}
    # G2: same seed → identical features/splits
    b = AudioTask(seed=7, path=d)
    assert np.array_equal(a._x_tr, b._x_tr)
    assert np.array_equal(a._y_tr, b._y_tr)
    # the §22.1 score protocol on the holdout
    class _Argmax:
        def forward(self, x):
            p = np.zeros((len(x), 2))
            p[np.arange(len(x)), (x[:, 0] > 0).astype(int)] = 1.0
            return p
    # 220 Hz < 440 Hz → band energy ordering is a usable signal
    assert a.score(_Argmax(), "holdout", 6) in (0.0, 100.0)


def test_audio_task_index_csv_regression_head(tmp_path):
    d = _tone_dir(tmp_path, n=10)
    # index.csv: file (relative) + numeric label (SPEC.md 24.2)
    rows = ["path,label"]
    for sub in sorted(d.iterdir()):
        for f in sorted(sub.iterdir()):
            rows.append(f"{sub.name}/{f.name},{0.0 if sub.name == 'low' else 1.0}")
    (d / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    a = AudioTask(seed=3, path=d)
    # labels 0.0/1.0 are integer-valued → softmax per the §22.1 rule
    assert a.head == "softmax"
    # and a genuinely non-integer label set → mse
    rows = ["path,label"]
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():  # index.csv from the first pass
            continue
        lab = 0.25 if sub.name == "low" else 0.75
        for f in sorted(sub.iterdir()):
            rows.append(f"{sub.name}/{f.name},{lab}")
    (d / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    b = AudioTask(seed=3, path=d)
    assert b.head == "mse" and b.n_outputs == 1
    assert b.default_dataset_size == 16


def test_audio_mp3_routes_through_soundfile(tmp_path, monkeypatch):
    """MP3 decode is delegated to soundfile (SPEC.md 24.1): the task's
    contract is the routing + downstream feature pipeline, which we drive
    with a stubbed decoder (libsndfile itself decodes the MP3)."""
    sf = pytest.importorskip("soundfile")
    d = tmp_path / "clips"
    (d / "a").mkdir(parents=True)
    (d / "b").mkdir()
    for k in range(10):  # 20 items → 16/2/2 splits (SPEC.md 24.2)
        (d / "a" / f"a{k:02d}.mp3").write_bytes(b"stub")
        (d / "b" / f"b{k:02d}.mp3").write_bytes(b"stub")
    calls = []

    def fake_read(path, dtype=None, always_2d=False):
        calls.append(str(path))
        t = np.arange(16000) / 16000.0
        sig = 0.5 * np.sin(2 * math.pi * (220.0 if "a" in str(path) else 440.0)
                           * t)
        return sig[:, None], 16000

    monkeypatch.setattr(sf, "read", fake_read)
    a = AudioTask(seed=1, path=d)
    assert a.head == "softmax"
    assert len(calls) == 20  # every item decoded through soundfile
    x, _ = a.make_dataset()
    assert x.shape == (16, 26)


def test_audio_mp3_without_soundfile_is_a_clean_hint(tmp_path, monkeypatch):
    d = tmp_path / "clips"
    (d / "a").mkdir(parents=True)
    (d / "a" / "one.mp3").write_bytes(b"stub")
    real_import = builtins.__import__

    def blocked(name, *a_, **k_):
        if name == "soundfile" or name.startswith("soundfile."):
            raise ImportError("blocked for test")
        return real_import(name, *a_, **k_)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="autorefine\\[audio\\]"):
        AudioTask(seed=1, path=d)


def test_audio_task_error_cases(tmp_path):
    # no items at all (SPEC.md 24.4 errors)
    empty = tmp_path / "empty"
    (empty / "x").mkdir(parents=True)
    with pytest.raises(ValueError, match="no audio files"):
        AudioTask(seed=1, path=empty)
    # non-16-bit WAV is rejected with a hint
    d = tmp_path / "bits"
    (d / "c").mkdir(parents=True)
    with wave.open(str(d / "c" / "x.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit
        w.setframerate(8000)
        w.writeframes(b"\x80" * 800)
    with pytest.raises(ValueError, match="16-bit PCM"):
        AudioTask(seed=1, path=d)
    # too-short signal
    short = tmp_path / "short"
    (short / "c").mkdir(parents=True)
    with wave.open(str(short / "c" / "x.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<4h", 0, 1, 2, 3))
    with pytest.raises(ValueError, match="too short"):
        AudioTask(seed=1, path=short)


def test_log_mel_features_deterministic_and_sane():
    sr = 16000
    t = np.arange(sr) / sr
    lo = 0.5 * np.sin(2 * math.pi * 220.0 * t)
    hi = 0.5 * np.sin(2 * math.pi * 440.0 * t)
    f_lo = log_mel_features(lo, sr)
    f_hi = log_mel_features(hi, sr)
    assert f_lo.shape == (26,)
    assert np.all(np.isfinite(f_lo))
    assert log_mel_features(lo, sr) is not None
    # deterministic (G2) and the two tones differ (a real feature gradient)
    assert np.array_equal(log_mel_features(lo, sr), f_lo)
    assert np.abs(f_hi - f_lo).max() > 1e-3


# --- ImageTask (SPEC.md 24.3) -------------------------------------------------

def test_image_task_protocol_and_determinism(tmp_path):
    d = _shape_dir(tmp_path)
    a = ImageTask(seed=7, path=d)
    assert a.head == "softmax" and a.n_outputs == 2
    assert a.state_dim == 32 * 32  # default grid (SPEC.md 24.3)
    assert a.feature_names == ["32x32 grayscale"]
    assert a.default_dataset_size == 32  # 80% of 40 (SPEC.md 20.3)
    x, y = a.make_dataset()
    assert x.shape == (32, 32 * 32)
    b = ImageTask(seed=7, path=d)
    assert np.array_equal(a._x_tr, b._x_tr)  # G2


def test_image_task_missing_pillow_is_a_clean_hint(tmp_path, monkeypatch):
    d = _shape_dir(tmp_path)
    real_import = builtins.__import__

    def blocked(name, *a_, **k_):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("blocked for test")
        return real_import(name, *a_, **k_)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="autorefine\\[image\\]"):
        ImageTask(seed=1, path=d)


# --- CLI fit on directories (SPEC.md 24.5, A14) -------------------------------

def test_fit_cli_audio_dir_passes(tmp_path, capsys):
    import json
    d = _tone_dir(tmp_path)
    rc = cli_main(["fit", "--data", str(d), "--seed", "7",
                   "--experiments", "4", "--max-train-seconds", "10",
                   "--target", "90", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0  # the §22.1 gate: PASS (SPEC.md 24.6 acceptance)
    assert "PASS" in out
    # auto-detection picked the audio task (SPEC.md 24.5)
    runs = list((tmp_path / "runs").glob("audio-seed7-*"))
    assert runs
    assert json.loads((runs[0] / "summary.json").read_text())["task"] == "audio"
    # the run round-trips through the registry for `eval` (SPEC.md 15)
    assert cli_main(["eval", "--run", str(runs[0])]) == 0


def test_fit_cli_image_dir_passes(tmp_path, capsys):
    d = _shape_dir(tmp_path)
    rc = cli_main(["fit", "--data", str(d), "--seed", "7",
                   "--experiments", "4", "--max-train-seconds", "10",
                   "--target", "90", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0  # the §22.1 gate: PASS
    assert "PASS" in out
    runs = list((tmp_path / "runs").glob("image-seed7-*"))
    assert runs
    assert cli_main(["eval", "--run", str(runs[0])]) == 0


def test_fit_cli_mixed_dir_needs_explicit_task(tmp_path, capsys):
    d = tmp_path / "mixed"
    (d / "low").mkdir(parents=True)
    _write_wav(d / "low" / "a.wav", 220.0)
    _make_image(d / "low" / "b.png", "ring")
    rc = cli_main(["fit", "--data", str(d), "--experiments", "1"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "--task image or --task audio" in err
    # explicit --task resolves it (probe-level: the task builds)
    assert detect_modality(d) == "mixed"


def test_fit_cli_missing_path_still_errors(tmp_path, capsys):
    rc = cli_main(["fit", "--data", "/definitely/not/here.csv"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no such file or directory" in err


# --- core import cleanliness (SPEC.md 24.1, the §23 pattern) ------------------

def test_import_autorefine_never_pulls_in_pil_or_soundfile():
    code = ("import sys; import autorefine; "
            "assert 'PIL' not in sys.modules; "
            "assert 'soundfile' not in sys.modules")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))


# --- dashboard runner accepts a media directory (SPEC.md 24.5) ----------------

def test_dashboard_runner_audio_dir(tmp_path):
    from autorefine import DashboardRunner
    d = _tone_dir(tmp_path)
    r = DashboardRunner(csv_path=d, seed=7, experiments=3, max_train_seconds=5.0,
                        runs_dir=str(tmp_path / "runs"))
    info = r.start()
    assert info["label"] == "class"
    assert info["head"] == "softmax"
    assert info["rows"]["train"] == 32  # 80% of 40 (SPEC.md 24.2)
    assert r.env.task.name == "audio"
    # forced modality on a csv file is rejected (SPEC.md 24.5)
    csv = tmp_path / "x.csv"
    csv.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n", encoding="utf-8")
    r2 = DashboardRunner(csv_path=csv, modality="audio")
    with pytest.raises(ValueError, match="modality must be 'csv' or 'auto'"):
        r2.start()
