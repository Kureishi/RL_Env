"""Extension-task tests: SineRegressionV1, GridNavV1 (SPEC.md 15), Parity4V1 (SPEC.md 17);
acceptance A7 (SPEC.md 12: the v0.3 parity-task extension coverage)."""
from pathlib import Path

import numpy as np
import pytest

from autorefine.config import ModelSpec
from autorefine.evaluator import run_episodes
from autorefine.tasks import TASKS
from autorefine.tasks.csv import CsvTask
from autorefine.tasks.gridnav import GOAL, GridNavV1, HEIGHT, WIDTH
from autorefine.tasks.parity import N_BITS, P_FLIP, Parity4V1
from autorefine.tasks.sine import SineRegressionV1, target_function


def _fixture_csv(path):
    """Minimal CSV for the data-driven registry check (SPEC.md 22.1)."""
    import csv as _csv
    import numpy as _np
    rng = _np.random.default_rng(11)
    n = 60
    x = rng.uniform(-1, 1, n)
    y = (x > 0).astype(int) ^ (2 * x > 0)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x", "y"])
        for i in range(n):
            w.writerow([f"{x[i]:.5f}", int(y[i])])
    return path


def _fixture_wav_dir(base) -> "Path":
    """Two tone classes, one subfolder each (SPEC.md 24.4, stdlib wave)."""
    import math
    import wave
    d = Path(base) / "tones"
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            sr = 8000
            sig = 0.8 * np.sin(2 * math.pi * freq * np.arange(sr) / sr)
            pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            with wave.open(str(sub / f"c{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
    return d


def _fixture_text_dir(base) -> "Path":
    """Two word classes, one subfolder each (SPEC.md 45.2, v0.31)."""
    d = Path(base) / "words"
    for name, words in (("up", "up up up rising higher"),
                        ("down", "down down down falling lower")):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            (sub / f"t{i:02d}.txt").write_text(words + " ", encoding="utf-8")
    return d


def _fixture_image_dir(base) -> "Path":
    """Two bar-orientation classes, one subfolder each (SPEC.md 24.3)."""
    pytest.importorskip("PIL", reason="image fixture needs autorefine[image] (SPEC.md 24.1)")
    from PIL import Image
    d = Path(base) / "shapes"
    for name, vertical in (("h", False), ("v", True)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(5):
            a = np.zeros((16, 16), dtype=np.uint8)
            if vertical:
                a[:, 4:12] = 255
            else:
                a[4:12, :] = 255
            Image.fromarray(a, mode="L").save(str(sub / f"s{i:02d}.png"))
    return d


def _sine(freq: float, n: int = 64) -> np.ndarray:
    """A deterministic 1-D signal (fundamental + one harmonic)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 3 * freq * t)


def _fixture_signal_dir(base) -> "Path":
    """Two frequency classes, one subfolder each (SPEC.md 92.1, v0.78)."""
    d = Path(base) / "signals"
    for name, freq in (("low", 2.0), ("high", 14.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(3):
            (sub / f"c{i:02d}.sig").write_text(
                " ".join(f"{v:.6f}" for v in _sine(freq)) + "\n",
                encoding="utf-8")
    return d


def _fixture_jsonl(base) -> "Path":
    """Ten JSONL records, two classes (SPEC.md 92.2, v0.78)."""
    import json as _json
    p = Path(base) / "logs.jsonl"
    lines = []
    for i in range(10):
        cls = i % 2
        lines.append(_json.dumps({
            "label": cls,
            "a": float(cls),
            "nested": {"b": cls * 1.5},
            "flag": bool(cls),
        }))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _fixture_midi_dir(base) -> "Path":
    """Two pitch-class sets, one subfolder each (SPEC.md 92.3, v0.78).

    Minimal format-0 MIDI files: set-tempo meta + note-on events under
    this codebase's channel-voice convention (low nibble 0x1 = note-on,
    velocity in byte 2) — the parser's own contract (midi.py, 92.3.1).
    """
    def _vlq(v: int) -> bytes:
        out = bytes([v & 0x7F])
        v >>= 7
        while v:
            out = bytes([(v & 0x7F) | 0x80]) + out
            v >>= 7
        return out
    d = Path(base) / "midi"
    for name, pitches in (("jazz", (60, 62, 67, 72, 76)),
                          ("rock", (40, 45, 50, 55, 59))):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(3):
            trk = bytes([0x00, 0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")
            for delta, pitch in enumerate(pitches):
                trk += _vlq(delta * 24) + bytes([0x91, pitch, 100])
            trk += bytes([0x00, 0xFF, 0x2F, 0x00])  # end of track
            header = (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
                      + (1).to_bytes(2, "big") + (960).to_bytes(2, "big"))
            (sub / f"c{i:02d}.mid").write_bytes(
                header + b"MTrk" + len(trk).to_bytes(4, "big") + trk)
    return d


def _fixture_pc_dir(base) -> "Path":
    """Round shell vs flat disk point clouds (SPEC.md 92.4, v0.78)."""
    d = Path(base) / "pointclouds"
    ang = np.linspace(0.0, 2 * np.pi, 20)
    for name, z in (("round", 0.5), ("flat", 0.02)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(3):
            pts = [(np.cos(a), np.sin(a), z) for a in ang]
            (sub / f"c{i:02d}.obj").write_text(
                "\n".join(f"v {x:.4f} {y:.4f} {z2:.4f}" for x, y, z2 in pts) + "\n",
                encoding="utf-8")
    return d


def _fixture_graph_dir(base) -> "Path":
    """Two-community graph: nodes.csv + edges.csv (SPEC.md 92.5, v0.78)."""
    d = Path(base) / "graph"
    d.mkdir(parents=True)
    nodes = ["id,f1,f2,label"]
    for i in range(10):
        cls = i % 2
        nodes.append(f"{i},{0.1 * cls:.2f},{0.2 * cls:.2f},{cls}")
    (d / "nodes.csv").write_text("\n".join(nodes) + "\n", encoding="utf-8")
    edges = ["source,target"] + [f"{i},{i + 1}" for i in range(0, 9, 2)]
    (d / "edges.csv").write_text("\n".join(edges) + "\n", encoding="utf-8")
    return d


def _fixture_fusion_dir(base) -> "Path":
    """Six (image, text) rows, two classes, via index.csv (SPEC.md 99,
    v0.85 C1) — ≥6 items so the train/holdout/gen split is legal."""
    import csv as _csv
    from PIL import Image
    d = Path(base) / "fusion"
    d.mkdir(parents=True)
    rows = []
    for i, (name, v) in enumerate(
            (("dark", 40), ("dark", 60), ("dark", 90),
             ("light", 165), ("light", 195), ("light", 225))):
        Image.new("L", (8, 8), v).save(d / f"a{i}.png")
        words = ("dark dark heavy heavy shadow" if v < 128
                 else "light light bright bright glow")
        (d / f"n{i}.txt").write_text(words, encoding="utf-8")
        rows.append((f"a{i}.png", f"n{i}.txt", name))
    with (d / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["image", "text", "label"])
        w.writerows(rows)
    return d


def _fixture_mixed_dir(base) -> "Path":
    """low/high rows mixing signal (.sig) and audio (.wav) (SPEC.md 92.7, v0.78)."""
    import wave
    d = Path(base) / "mixed"
    for name, freq in (("low", 2.0), ("high", 14.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(3):
            (sub / f"c{i:02d}.sig").write_text(
                " ".join(f"{v:.6f}" for v in _sine(freq)) + "\n",
                encoding="utf-8")
            sr = 16000
            t = np.arange(sr // 2) / sr
            sig = 0.6 * np.sin(2 * np.pi * (220.0 if name == "low" else 880.0) * t)
            pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            with wave.open(str(sub / f"a{i:02d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
    return d


def _fixture_video_dir(base) -> "Path":
    """slow/fast clip folders (SPEC.md 92.6, v0.78); the bytes are dummies —
    decoding is monkeypatched at `imageio.v3.imread` by the caller."""
    d = Path(base) / "video"
    for name in ("slow", "fast"):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(3):
            (sub / f"v{i:02d}.mp4").write_bytes(b"fake")
    return d


class _TargetModel:
    """'Model' that knows the sine target exactly: score must be ~100."""

    def forward(self, x):
        return target_function(np.asarray(x, dtype=np.float64).reshape(-1))[:, None]


def test_task_registry_covers_all_tasks(tmp_path, monkeypatch):
    # SPEC.md 22.1/24/45.2/83.1/92/99: csv/image/audio/text and the v0.78
    # signal/jsonl/midi/pointcloud/graph/video/mixed tasks plus the v0.85
    # fusion task are data-driven (need a file/dir), the rest are
    # seed-constructable (fusion-v1 is the synthetic FusionV1 pair)
    assert set(TASKS) == {"cartpole-v1", "sine-v1", "gridnav-v1", "parity-v1",
                          "csv", "image", "audio", "text",
                          "medical-v1", "finance-v1", "sequence-motif-v1",
                          "signal", "jsonl", "midi", "pointcloud",
                          "graph", "video", "mixed",
                          "fusion", "fusion-v1"}
    try:
        import imageio.v3 as _iio  # optional [video] extra (SPEC.md 92.6.1)
    except ImportError:
        _iio = None
    for name, cls in TASKS.items():
        if name == "csv":
            task = CsvTask(seed=7, path=str(_fixture_csv(tmp_path / "csv_fixture.csv")))
        elif name == "audio":
            task = cls(seed=7, path=str(_fixture_wav_dir(tmp_path)))
        elif name == "image":
            task = cls(seed=7, path=str(_fixture_image_dir(tmp_path)))
        elif name == "text":
            task = cls(seed=7, path=str(_fixture_text_dir(tmp_path)))
        elif name == "signal":
            task = cls(seed=7, path=str(_fixture_signal_dir(tmp_path)))
        elif name == "jsonl":
            task = cls(seed=7, path=str(_fixture_jsonl(tmp_path)))
        elif name == "midi":
            task = cls(seed=7, path=str(_fixture_midi_dir(tmp_path)))
        elif name == "pointcloud":
            task = cls(seed=7, path=str(_fixture_pc_dir(tmp_path)))
        elif name == "graph":
            task = cls(seed=7, path=str(_fixture_graph_dir(tmp_path)))
        elif name == "mixed":
            task = cls(seed=7, path=str(_fixture_mixed_dir(tmp_path)))
        elif name == "fusion":
            task = cls(seed=7, path=str(_fixture_fusion_dir(tmp_path)))
        elif name == "video":
            root = _fixture_video_dir(tmp_path)
            if _iio is None:
                # fail-loud contract (92.6.1): no imageio -> install hint
                with pytest.raises(ValueError, match="imageio"):
                    cls(seed=7, path=str(root))
                continue
            frames = np.random.default_rng(5).integers(
                0, 256, size=(10, 48, 64, 3), dtype=np.uint8)
            monkeypatch.setattr(_iio, "imread", lambda p: frames)
            task = cls(seed=7, path=str(root))
        else:
            task = cls(seed=7)
        assert task.name == name
        assert task.state_dim > 0 and task.n_outputs >= 1
        assert task.head in ("softmax", "mse")
        assert callable(task.make_dataset) and callable(task.score)


# --- SineRegressionV1 --------------------------------------------------------
def test_sine_dataset_deterministic_and_invariants():
    task = SineRegressionV1(seed=7)
    a = task.make_dataset()
    b = task.make_dataset()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    X, y = a
    assert X.ndim == 2 and X.shape[0] == y.shape[0]
    assert (X >= 0.0).all() and (X <= 1.0).all()  # unit-scaled inputs
    # split determinism and distinctness
    for split in ("train", "holdout", "gen"):
        assert np.array_equal(task.initial_conditions(split, 33),
                              task.initial_conditions(split, 33))
    assert not np.allclose(task.initial_conditions("holdout", 33),
                           task.initial_conditions("gen", 33))


def test_sine_target_is_nontrivial():
    u = np.linspace(0.0, 1.0, 1000)
    y = target_function(u)
    assert y.std() > 0.5  # real variance to explain
    assert np.array_equal(target_function(u), target_function(u))  # deterministic


def test_sine_scoring_sanity():
    """Perfect model ~100; constant (mean) model clamps to 0."""
    task = SineRegressionV1(seed=7)
    assert task.score(_TargetModel(), "holdout", 300) == pytest.approx(100.0, abs=1e-6)

    class _MeanModel:
        def __init__(self, task):
            X = task.initial_conditions("holdout", 2000)
            self.mean = float(target_function(X[:, 0]).mean())

        def forward(self, x):
            return np.full((np.asarray(x).shape[0], 1), self.mean)

    assert task.score(_MeanModel(task), "holdout", 300) == 0.0  # R^2 ~ 0, clamped


def test_sine_split_independence_of_n():
    """Score of the perfect model does not depend on the eval sample size."""
    task = SineRegressionV1(seed=7)
    assert task.score(_TargetModel(), "holdout", 100) == pytest.approx(100.0, abs=1e-6)
    assert task.score(_TargetModel(), "gen", 500) == pytest.approx(100.0, abs=1e-6)


# --- GridNavV1 ---------------------------------------------------------------
class _ReferenceModel:
    """Plays the task's own reference policy."""

    def __init__(self, task: GridNavV1):
        self.task = task

    def forward(self, x):
        a = self.task.reference_action(np.asarray(x, dtype=np.float64))
        out = np.full((len(a), 4), 0.0)
        out[np.arange(len(a)), a] = 1.0
        return out


def test_gridnav_dataset_invariants():
    task = GridNavV1(seed=7)
    X, y = task.make_dataset()
    assert X.shape[1] == task.state_dim == WIDTH * HEIGHT
    assert np.allclose(X.sum(axis=1), 1.0)  # one-hot rows
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
    # start cells never include the goal
    ics = task.initial_conditions("train", 500)
    assert GOAL not in set(np.argmax(ics, axis=1))


def test_gridnav_dataset_deterministic():
    a = GridNavV1(seed=7).make_dataset()
    b = GridNavV1(seed=7).make_dataset()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_gridnav_torus_wraparound():
    task = GridNavV1(seed=7)
    # top-left cell, action 0 (up) wraps to bottom-left
    st = np.zeros((1, WIDTH * HEIGHT)); st[0, 0] = 1.0
    nxt, term = task.step_vec(st, np.array([0]))
    assert np.argmax(nxt[0]) == (HEIGHT - 1) * WIDTH + 0
    assert not term[0]
    # stepping into the goal => terminal
    one_above = (HEIGHT - 2) * WIDTH + (WIDTH - 1)
    g = np.zeros((1, WIDTH * HEIGHT)); g[0, one_above] = 1.0
    nxt, term = task.step_vec(g, np.array([1]))  # down into the goal
    assert term[0]
    assert np.argmax(nxt[0]) == GOAL


def test_gridnav_reference_policy_is_strong():
    """The reference policy reaches the goal from every cell (success rate 1)."""
    task = GridNavV1(seed=7)
    model = _ReferenceModel(task)
    stats = run_episodes(task, model, "holdout", 120)
    assert stats["success_rate"] == 1.0
    assert stats["mean_steps"] <= HEIGHT + WIDTH  # shortest torus paths
    score = task.score(model, "holdout", 120)
    assert score >= 100.0  # 100 * success_rate floor


def test_gridnav_scoring_penalizes_slow_policies():
    """A policy that always spins in place scores far below the reference."""

    class _Spin:
        def forward(self, x):
            out = np.zeros((np.asarray(x).shape[0], 4))
            out[:, 3] = 1.0  # always right; only succeeds by chance
            return out

    task = GridNavV1(seed=7)
    spin = task.score(_Spin(), "holdout", 120)
    ref = task.score(_ReferenceModel(task), "holdout", 120)
    assert spin < ref


# --- Parity4V1 (SPEC.md 17) --------------------------------------------------
class _NoisyParityModel:
    """Predicts the parity of the (noisy) observed bits."""

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        p = (x.sum(axis=1) % 2).astype(np.float64)
        out = np.zeros((len(p), 2))
        out[np.arange(len(p)), p.astype(int)] = 1.0
        return out


def test_parity_dataset_invariants():
    task = Parity4V1(seed=7)
    X, y = task.make_dataset()
    assert X.ndim == 2 and X.shape[1] == task.state_dim == N_BITS
    assert set(np.unique(X)) <= {0, 1}  # binary features
    assert set(np.unique(y)) <= {0, 1}
    assert y.mean() == pytest.approx(0.5, abs=0.05)  # balanced by construction
    # the label is the parity of the CLEAN bits, so it is NOT the parity of X
    assert not np.array_equal(y, X.sum(axis=1) % 2)


def test_parity_dataset_deterministic():
    a = Parity4V1(seed=7).make_dataset()
    b = Parity4V1(seed=7).make_dataset()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # different seed => different points
    c = Parity4V1(seed=8).make_dataset()
    assert not np.array_equal(a[0], c[0])


def test_parity_split_determinism_and_independence():
    task = Parity4V1(seed=7)
    for split in ("train", "holdout", "gen"):
        assert np.array_equal(task.initial_conditions(split, 17),
                             task.initial_conditions(split, 17))
    assert not np.allclose(task.initial_conditions("holdout", 17),
                           task.initial_conditions("gen", 17))


def test_parity_scoring_sanity():
    task = Parity4V1(seed=7)
    # parity of the noisy bits beats random (only flips hurt): well above 50
    assert task.score(_NoisyParityModel(), "holdout", 4096) > 55.0

    class _Constant:
        def forward(self, x):
            n = np.asarray(x).shape[0]
            out = np.zeros((n, 2)); out[:, 0] = 1.0
            return out

    # labels are balanced, so a constant model is ~coin flip
    assert task.score(_Constant(), "holdout", 4096) == pytest.approx(50.0, abs=3.0)


def test_parity_has_a_family_gradient():
    """SPEC.md 17: linear can't represent parity; an MLP approaches the
    Bayes ceiling; the improver therefore has real signal in architecture,
    capacity, optimizer and training budget."""
    task = Parity4V1(seed=7)
    ds = task.make_dataset()
    # Bayes ceiling: P(correct) = 1 - P(odd # flips) = (1 + (1-2p)^n) / 2
    ceiling = 100.0 * (1.0 + (1.0 - 2.0 * P_FLIP) ** N_BITS) / 2.0

    def score(spec: ModelSpec) -> float:
        from autorefine.trainer import train
        r = train(ds, spec, seed=7, n_out=2, head="softmax")
        return task.score(r.model, "gen", 4096)

    linear = ModelSpec(architecture=(), optimizer="sgd", learning_rate=3e-3,
                       batch_size=32, weight_decay=0.0, train_steps=400,
                       input_noise=0.02, activation="tanh")
    baseline = ModelSpec(architecture=(16, 8), optimizer="momentum",
                         learning_rate=1e-3, batch_size=32, weight_decay=1e-4,
                         train_steps=400, input_noise=0.02, activation="tanh")
    big = ModelSpec(architecture=(64, 32), optimizer="adam", learning_rate=3e-3,
                    batch_size=32, weight_decay=1e-4, train_steps=3000,
                    input_noise=0.02, activation="tanh")
    s_linear, s_base, s_big = score(linear), score(baseline), score(big)
    assert s_linear <= 60.0          # linear can't represent XOR
    assert s_base > 55.0             # small MLP learns it
    assert s_big > s_base + 10.0     # capacity/optimizer/budget signal
    assert s_big >= ceiling - 10.0   # approaches the Bayes ceiling
    assert s_big <= ceiling + 3.0    # scoring stays honest (not over-claiming)
