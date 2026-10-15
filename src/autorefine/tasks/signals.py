"""SignalTask: a directory of labelled 1-D signal clips as a fitting task
(SPEC.md 92.1, v0.78, A82).

The generic sensor/signal modality — ECG, accelerometer, telemetry — as a
thin sibling of AudioTask (24.4): the same labelled-directory protocol
(24.2), the §22.1 head inference, the seed split, train-only
standardization, and the §22.1 score. The feature path makes **no mel /
audio assumption**: resample → detrend → statistics + autocorrelation +
log-magnitude spectrum (all pure NumPy, deterministic given the file bytes,
G2).

Formats (92.1): `.npy` (`np.load`, a 1-D float array), `.dat` (raw
float32), `.sig` (whitespace-separated text).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task
from .media import (_SIGNAL_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)


def _resample(sig: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1-D signal to exactly `n` points (G2)."""
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size == n:
        return sig
    if sig.size < 2:
        return np.full(n, float(sig[0])) if sig.size else np.zeros(n)
    x0 = np.linspace(0.0, 1.0, sig.size)
    x1 = np.linspace(0.0, 1.0, n)
    return np.interp(x1, x0, sig)


def signal_features(sig: np.ndarray, n_resample: int = 128,
                    n_lags: int = 16, n_bands: int = 16) -> np.ndarray:
    """SPEC.md 92.1 (v0.78): the raw (6 + n_lags + n_bands,) feature of one
    1-D signal clip.

    - 6 shape statistics (crest factor, peak-to-peak, zero-crossing rate,
      skewness, excess kurtosis, mean absolute value) of the detrended,
      resampled signal (unit-std, so every statistic is data-driven);
    - `n_lags` normalized autocorrelation coefficients (lags 1..n_lags);
    - `n_bands` log1p rfft magnitudes binned over the positive-frequency
      axis (a coarse, resolution-free spectrum).

    Deterministic given the samples (G2). A constant or all-NaN/Inf clip
    is an error (the task has no silent fallback — the caller sees it).
    """
    s = _resample(sig, n_resample)
    if not np.all(np.isfinite(s)):
        raise ValueError("signal contains non-finite samples")
    s = s - s.mean()  # detrend
    sd = s.std()
    if sd == 0.0:
        raise ValueError("signal is constant (zero variance) — no features")
    z = s / sd
    crest = float(np.abs(z).max())
    p2p = float((z.max() - z.min()))
    zc = float(np.mean(np.abs(np.diff(np.signbit(s))) > 0)) if n_resample > 1 else 0.0
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean()) - 3.0
    mad = float(np.abs(z).mean())
    stats = np.array([crest, p2p, zc, skew, kurt, mad], dtype=np.float64)

    denom = float((z * z).sum())
    ac = np.zeros(n_lags, dtype=np.float64)
    if denom > 0.0:
        for lag in range(1, n_lags + 1):
            ac[lag - 1] = float((z[:-lag] * z[lag:]).sum() / denom)

    mag = np.abs(np.fft.rfft(s))  # n_resample//2 + 1 bins
    nb = int(n_bands)
    edges = np.linspace(0, mag.size, nb + 1)
    spec = np.zeros(nb, dtype=np.float64)
    for b in range(nb):
        lo, hi = int(edges[b]), max(int(edges[b + 1]), int(edges[b]) + 1)
        spec[b] = float(np.log1p(mag[lo:hi].mean()))
    return np.concatenate([stats, ac, spec])


def load_signal(p: Path) -> np.ndarray:
    """SPEC.md 92.1 (v0.78): one signal file → a 1-D float64 array."""
    ext = p.suffix.lower()
    if ext == ".npy":
        a = np.load(str(p))
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.size == 0:
            raise ValueError(f"{p}: empty .npy signal")
        return a
    if ext == ".dat":
        a = np.frombuffer(p.read_bytes(), dtype="<f4").astype(np.float64)
        if a.size == 0:
            raise ValueError(f"{p}: empty .dat signal")
        return a
    if ext == ".sig":
        a = np.array([float(v) for v in
                      p.read_text(encoding="utf-8-sig", errors="replace")
                      .split() if v.strip()], dtype=np.float64)
        if a.size == 0:
            raise ValueError(f"{p}: no samples in .sig")
        return a
    raise ValueError(
        f"{p}: unsupported signal extension {ext!r} — accepted: "
        f"{', '.join(_SIGNAL_EXTS)} (SPEC.md 92.1)")


class SignalTask(Task):
    name = "signal"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    feature_mean = None  # instance values (92.1)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2,
                 n_resample: int = 128, n_lags: int = 16,
                 n_bands: int = 16) -> None:
        if path is None:
            raise ValueError(
                "SignalTask needs a directory of labelled signal clips: "
                "use `autorefine fit --data DIR` or "
                "AutoRefineEnv(task='signal', task_config={'path': DIR}) "
                "(SPEC.md 92.1)")
        self.seed = int(seed)
        self.path = str(path)
        self.n_resample = int(n_resample)
        self.n_lags = int(n_lags)
        self.n_bands = int(n_bands)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        items = collect_items(self.path, _SIGNAL_EXTS, "signal")
        x = np.array([signal_features(load_signal(p), self.n_resample,
                                      self.n_lags, self.n_bands)
                      for p, _ in items], dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = x.shape[1]
        self.feature_names = (["shape stats (6)"]
                             + [f"autocorr lag {i}" for i in range(1, self.n_lags + 1)]
                             + [f"spectrum band {i}" for i in range(self.n_bands)])
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"signal-split")
        self._ho = ho
        mean = x[tr].mean(axis=0)
        std = x[tr].std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std
        self.feature_mean = mean
        self.feature_std = std
        self._x_tr, self._y_tr = x[tr], self._y[tr]
        self._x_ho, self._y_ho = x[ho], self._y[ho]
        self._x_ge, self._y_ge = x[ge], self._y[ge]
        self.default_dataset_size = int(len(tr))

    # --- decoding -------------------------------------------------------------
    def item_features(self, path: str | Path) -> np.ndarray:
        """SPEC.md 92.1 (v0.78): decode one signal file to the raw
        (unstandardized) feature — the public wrapper `predict --item`
        calls."""
        return signal_features(load_signal(Path(path)), self.n_resample,
                               self.n_lags, self.n_bands)

    # --- protocol (SPEC.md 15, 24.2 — mirrors AudioTask) ---------------------
    def _rows_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        s = (split or "").lower()
        if s.startswith("gen"):
            return self._x_ge, self._y_ge
        if s.startswith("train"):
            return self._x_tr, self._y_tr
        return self._x_ho, self._y_ho

    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        x, _ = self._rows_for(split)
        return x[: max(1, int(n))] if len(x) else x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states

    def make_dataset(self, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._x_tr) if n_points is None else min(int(n_points), len(self._x_tr))
        return self._x_tr[:n], self._y_tr[:n]

    def score(self, model, split: str, n: int) -> float:
        x, y = self._rows_for(split)
        if len(x) == 0:
            return 0.0
        n = min(int(n), len(x))
        pred = np.asarray(model.forward(x[:n]), dtype=np.float64)
        if self.head == "softmax":
            return float(100.0 * (pred.argmax(axis=1) == y[:n]).mean())
        pred = pred.reshape(-1)
        ss_res = float(((pred - y[:n]) ** 2).sum())
        ss_tot = float(((y[:n] - y[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return max(0.0, 100.0 * r2)

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._rows_for("holdout")
        if len(x) == 0:
            return x, y
        n = min(int(n), len(x))
        return x[:n], y[:n]

    def holdout_errors(self, model, n: int = 200, n_max: int = 8) -> list[dict]:
        """SPEC.md 92.1 (v0.78): the misclassified holdout items, with a
        best-effort downsampled `waveform` (the app's error gallery)."""
        if self.head != "softmax" or int(n_max) <= 0:
            return []
        x, y = self.holdout_rows(n)
        if len(x) == 0:
            return []
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        out: list[dict] = []
        for i in range(len(x)):
            if pred[i] == y[i]:
                continue
            p, _ = self._items[int(self._ho[i])]
            entry = {
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            }
            try:  # best-effort: a decode failure skips the waveform only
                sig = load_signal(p)
                if sig.size > 512:
                    block = sig.size // 512
                    sig = sig[:block * 512].reshape(512, block).mean(axis=1)
                entry["waveform"] = [float(v) for v in np.asarray(sig)[:512]]
            except Exception:
                pass
            out.append(entry)
            if len(out) >= int(n_max):
                break
        return out
