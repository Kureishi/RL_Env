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

from .media import _AUDIO_EXTS, collect_items, resolve_labels, split_indices

_SF_HINT = (
    "decoding MP3 needs soundfile (libsndfile >= 1.2): "
    "pip install autorefine[audio] (SPEC.md 24.1); WAV (16-bit PCM) needs "
    "no extra"
)


def _hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def log_mel_features(sig: np.ndarray, sr: int, n_bands: int = 26,
                     fmin: float = 0.0, fmax_cap: float = 8000.0,
                     win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    """NumPy-only log-mel spectrogram, time-mean → (n_bands,) (SPEC.md 24.4).

    Deterministic for a given signal + sample rate (G2).
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
    return np.log1p(feats).mean(axis=0)


class AudioTask:
    name = "audio"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    # per-data-dir attributes; __init__ overrides (defaults keep the class
    # usable for protocol introspection before a directory is given)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)

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
        x = np.array([self._features(p) for p, _ in items], dtype=np.float64)

        # head / labels / splits / standardization — the §22.1 rule
        # (SPEC.md 24.2); raw labels resolve to §22.1 targets or string
        # class names (SPEC.md 24.2)
        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.state_dim = self.bands
        self.feature_names = [f"log-mel {self.bands} bands"]
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"audio-split")
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))  # SPEC.md 20.3 (items)

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

    def _features(self, p: Path) -> np.ndarray:
        try:
            sig, sr = self._load_signal(p)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read audio clip {p}: {exc}") from exc
        return log_mel_features(sig, sr, self.bands)

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

    def score(self, model, split: str, n: int) -> float:
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
