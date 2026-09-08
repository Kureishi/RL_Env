"""AudioTask: a directory of labelled audio clips as a fitting task
(SPEC.md 24.4).

Formats (SPEC.md 24.4): `.wav` (16-bit PCM, stdlib `wave` — zero
dependencies; mono or stereo → mean downmix) and `.mp3` (soundfile via the
optional `autorefine[audio]` extra, SPEC.md 24.1 — imported lazily, only
when an MP3 is decoded, so `import autorefine` stays soundfile-free).

Features: a hand-rolled NumPy-only log-mel spectrogram (S1): STFT (Hann,
25 ms window / 10 ms hop), power spectrum, 26-band mel filterbank
(0 Hz → min(sr/2, 8 kHz)), log(1+x), mean over time → (bands,) vector.
Deterministic given the file bytes (G2).

Head / splits / standardization / score are the §22.1 rule verbatim
(shared via `media.py`, SPEC.md 24.2).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .base import Task
from .media import (_AUDIO_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)

_SF_HINT = (
    "decoding MP3 needs soundfile (libsndfile >= 1.2): "
    "pip install autorefine[audio] (SPEC.md 24.1); WAV (16-bit PCM) needs "
    "no extra"
)


def _hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def log_mel_frames(sig: np.ndarray, sr: int, n_bands: int = 26,
                   fmin: float = 0.0, fmax_cap: float = 8000.0,
                   win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    """NumPy-only log-mel spectrogram frames → (T_i, n_bands) (SPEC.md 24.4/25.3).

    The per-frame (time × mel) spectrogram in log1p space. Deterministic for a
    given signal + sample rate (G2). `log_mel_features` (the flat v0.10
    feature) is the time-mean of this — so the flat path stays bit-identical
    to v0.10 (SPEC.md 25.3).
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    win = max(4, int(round(win_s * sr)))
    hop = max(2, int(round(hop_s * sr)))
    if len(sig) < 8:
        raise ValueError("audio signal too short to feature "
                         "(fewer than 8 samples)")
    if len(sig) < win:
        sig = np.pad(sig, (0, win - len(sig)))
    frames = 1 + (len(sig) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    frame = sig[idx] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frame, axis=1)) ** 2

    fmax = min(sr / 2.0, fmax_cap)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax),
                                   n_bands + 1))
    bins = np.fft.rfftfreq(win, d=1.0 / sr)
    fb = np.zeros((n_bands, bins.size), dtype=np.float64)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b] + (edges[b + 1] - edges[b]) / 2.0, edges[b + 1]
        # triangular mel filter over the FFT bin axis
        rising = hi - lo
        if rising > 0:
            fb[b, bins < mid] = np.clip((bins[bins < mid] - lo) / (mid - lo),
                                        0.0, 1.0) if mid > lo else 0.0
            fb[b, bins >= mid] = np.clip((hi - bins[bins >= mid]) / rising,
                                         0.0, 1.0)
    feats = spec @ fb.T
    if not np.all(np.isfinite(feats)) or feats.sum() <= 0.0:
        raise ValueError("audio feature extraction produced no signal "
                         "(silence or unsupported sample rate)")
    return np.log1p(feats)


def log_mel_features(sig: np.ndarray, sr: int, n_bands: int = 26,
                     fmin: float = 0.0, fmax_cap: float = 8000.0,
                     win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    """NumPy-only log-mel spectrogram, time-mean → (n_bands,) (SPEC.md 24.4).

    Bit-identical to the v0.10 single-shot feature: `log_mel_frames` (the
    (T_i, bands) spectrogram) time-meaned over the time axis (SPEC.md 25.3).
    """
    return log_mel_frames(sig, sr, n_bands, fmin, fmax_cap, win_s, hop_s).mean(axis=0)


class AudioTask(Task):
    name = "audio"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    # per-data-dir attributes; __init__ overrides (defaults keep the class
    # usable for protocol introspection before a directory is given)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    # SPEC.md 25.3: the audio modality is the log-mel spectrogram (time x mel)
    grid_capable = True
    feature_grid = None  # instance value: (1, T, bands)
    # SPEC.md 36.1 (v0.22, G1): declared metric + capabilities. Class-level
    # default matches the `head = "softmax"` introspection default; __init__
    # sets the instance metric alongside `head` (r2 for mse-labelled data).
    metric = "accuracy"
    capabilities = frozenset({"grid", "media"})
    # SPEC.md 42.1.3 (v0.28): the train-only standardization stats —
    # instance values; None keeps class introspection usable.
    feature_mean = None
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 bands: int = 26) -> None:
        if path is None:
            raise ValueError(
                "AudioTask needs a directory of labelled audio clips: use "
                "`autorefine fit --data DIR` or "
                "AutoRefineEnv(task='audio', task_config={'path': DIR}) "
                "(SPEC.md 24.4)"
            )
        self.seed = int(seed)
        self.path = str(path)
        self.bands = int(bands)
        self.split_frac = float(split_frac)
        self.label_name = "class"  # subfolder name / index.csv label column

        items = collect_items(self.path, _AUDIO_EXTS, "audio")
        # SPEC.md 25.3: decode each clip once; the flat feature is the
        # time-mean of the per-item spectrogram frames (bit-identical to the
        # v0.10 flat path) and the grid layout reuses the same frames.
        frames_list = [self._frames(p) for p, _ in items]
        x = np.array([f.mean(axis=0) for f in frames_list], dtype=np.float64)

        # head / labels / splits / standardization — the §22.1 rule
        # (SPEC.md 24.2); raw labels resolve to §22.1 targets or string
        # class names (SPEC.md 24.2)
        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = self.bands
        self.feature_names = [f"log-mel {self.bands} bands"]
        # SPEC.md 28.3 (C3): keep the items + holdout split indices so the
        # error gallery can map holdout row i -> items[ho[i]].
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"audio-split")
        self._ho = ho
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean  # SPEC.md 42.1.3 (v0.28): expose the stats
        self.feature_std = std  # so `predict` standardizes new clips like train
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))  # SPEC.md 20.3 (items)
        # --- SPEC.md 25.3: grid (spectrogram) layout for the convnet family -
        # T = min(128, max frame count); items shorter than T are zero-padded
        # on the time axis (a padded frame reads as silence). Standardization
        # reuses the flat train per-band mean/std (exactly consistent).
        t_max = min(128, max(int(f.shape[0]) for f in frames_list))
        # (n, 1, T, bands): the grid protocol is one (C, H, W) sample per
        # item (SPEC.md 25.3) — C=1, H=T (time), W=bands (mel)
        frames = np.zeros((len(items), 1, t_max, self.bands), dtype=np.float64)
        for i, f in enumerate(frames_list):
            k = min(int(f.shape[0]), t_max)
            frames[i, 0, :k, :] = f[:k]
        self._T = int(t_max)
        self.feature_grid = (1, int(t_max), self.bands)
        g = (frames - mean[None, None, None, :]) / std[None, None, None, :]
        self._g_tr = g[tr]
        self._g_ho = g[ho]
        self._g_ge = g[ge]

    # --- decoding / features ----------------------------------------------------
    @staticmethod
    def _load_signal(p: Path) -> tuple[np.ndarray, int]:
        """(mono float signal in [-1, 1], sample rate) for one clip."""
        ext = p.suffix.lower()
        if ext == ".mp3":
            try:
                import soundfile as sf
            except ImportError as exc:
                raise ValueError(_SF_HINT) from exc
            data, sr = sf.read(str(p), dtype="float64", always_2d=True)
            sig = data.mean(axis=1) if data.ndim == 2 else data
            return np.asarray(sig, dtype=np.float64), int(sr)
        if ext == ".wav":
            with wave.open(str(p), "rb") as w:
                if w.getsampwidth() != 2:
                    raise ValueError(
                        f"{p}: expected 16-bit PCM WAV, got "
                        f"{w.getsampwidth() * 8}-bit (SPEC.md 24.4); convert "
                        "with any audio tool or use MP3 (autorefine[audio])"
                    )
                sr = int(w.getframerate())
                ch = int(w.getnchannels())
                raw = w.readframes(int(w.getnframes()))
            sig = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            if ch > 1:
                sig = sig.reshape(-1, ch).mean(axis=1)
            return sig, sr
        raise ValueError(
            f"{p}: unsupported audio extension {ext!r} — accepted: "
            f"{', '.join(_AUDIO_EXTS)} (SPEC.md 24.4)"
        )

    def _frames(self, p: Path) -> np.ndarray:
        """Per-item log-mel spectrogram frames (T_i, bands) (SPEC.md 25.3)."""
        try:
            sig, sr = self._load_signal(p)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read audio clip {p}: {exc}") from exc
        return log_mel_frames(sig, sr, self.bands)

    def _features(self, p: Path) -> np.ndarray:
        """Flat log-mel feature (bands,) — the time-mean of `_frames` (25.3)."""
        return self._frames(p).mean(axis=0)

    def item_features(self, path: str | Path) -> np.ndarray:
        """SPEC.md 42.1.3 (v0.28): decode one audio clip to the raw
        (unstandardized) flat log-mel (bands,) feature — the public
        wrapper over `_features` that `predict --item` (42.1.1) and
        `autorefine.predict` call; the exact feature the model was
        trained on (the flat path, SPEC.md 25.3)."""
        return self._features(Path(path))

    # --- protocol (SPEC.md 15, 24.2 — mirrors CsvTask) ------------------------
    def _rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._x_ge, self._y_ge
        if s.startswith("train"):
            return self._x_tr, self._y_tr
        return self._x_ho, self._y_ho  # "holdout", "holdout-b3", ...

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._rows_for(split)
        return x[: max(1, int(n))] if len(x) else x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # already standardized in __init__

    def make_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Train split: what the improver is allowed to see (SPEC.md 15)."""
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    # --- grid protocol (SPEC.md 25.3) ---------------------------------------
    def grid_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Train split in grid layout (n, 1, T, bands) — the log-mel
        spectrogram (SPEC.md 25.3), the audio temporal model's input."""
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._g_tr[:n], self._y_tr[:n]

    def _grid_rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._g_ge, self._y_ge
        if s.startswith("train"):
            return self._g_tr, self._y_tr
        return self._g_ho, self._y_ho

    def score(self, model, split: str, n: int) -> float:
        # SPEC.md 25.3: a wants_grid model (convnet) gets grid-layout rows;
        # flat models get the flat rows (today's behavior, unchanged).
        if getattr(model, "wants_grid", False):
            x, y = self._grid_rows_for(split)
        else:
            x, y = self._rows_for(split)
        if len(x) == 0:
            return 0.0
        n = min(int(n), len(x))
        pred = np.asarray(model.forward(x[:n]), dtype=np.float64)
        if self.head == "softmax":
            acc = (pred.argmax(axis=1) == y[:n]).mean()
            return float(100.0 * acc)
        pred = pred.reshape(-1)
        ss_res = float(((pred - y[:n]) ** 2).sum())
        ss_tot = float(((y[:n] - y[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)

    # --- holdout inspection (SPEC.md 28.2/28.3) ------------------------------
    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 28.2 (C2): the holdout split `(x, y)`, clamped like
        `score()`; a `wants_grid` model (convnet) gets grid rows (25.3)."""
        if getattr(model, "wants_grid", False):
            x, y = self._grid_rows_for("holdout")
        else:
            x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    def holdout_errors(self, model, n: int = 200,
                       n_max: int = 8) -> list[dict]:
        """SPEC.md 28.3 (C3): the holdout items the model misclassifies
        (argmax mismatch), in holdout order, <= `n_max` items — the image
        entry shape `{"file", "path", "label", "predicted"}` plus
        `{"waveform": [<=512 floats], "sample_rate": int}` (a deterministic
        block-mean downsample of the decoded signal; omitted if the clip
        cannot be re-decoded). Labels come from the canonical
        `class_values`, not the raw item labels."""
        if self.head != "softmax" or int(n_max) <= 0:
            return []
        x, y = self.holdout_rows(n, model)
        if len(x) == 0:
            return []
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        out: list[dict] = []
        for i in range(len(x)):
            if pred[i] == y[i]:
                continue
            p, _ = self._items[int(self._ho[i])]  # holdout row i -> item
            entry = {
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            }
            try:  # best-effort: a decode failure skips the waveform only
                sig, sr = self._load_signal(p)
                sig = np.asarray(sig, dtype=np.float64).ravel()
                if sig.size > 512:
                    block = int(sig.size) // 512
                    wf = (sig[:block * 512].reshape(512, block)
                          .mean(axis=1).tolist())
                else:
                    wf = [float(v) for v in sig]
                entry["waveform"] = [float(v) for v in wf][:512]
                entry["sample_rate"] = int(sr)
            except Exception:
                pass
            out.append(entry)
            if len(out) >= int(n_max):
                break
        return out
