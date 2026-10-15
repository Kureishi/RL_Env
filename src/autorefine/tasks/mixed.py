"""MixedTask: a directory of labelled media spanning several modalities
as a fitting task (SPEC.md 92.7, v0.78, A82).

The "which modality is this row" modality: one labelled directory (an
`index.csv` or one-subfolder-per-class, 24.2) whose rows mix two or more
of the row modalities (92.6's `ROW_MODALITIES`: image, audio, text,
signal, midi, pointcloud, video). Each row's feature is a fixed
7 + 1024 = 1031 vector (92.7.2): the 7-dim modality one-hot (in
`ROW_MODALITIES` order — the model can branch on the modality) followed
by the row's own modality feature (each modality's own decoder, 92.7.1)
zero-padded / truncated to 1024 (a constant width for every row, so the
flat protocol of 22.1 applies unchanged).

The decoders are the tasks' own — `ImageTask._decode` (24.3),
`AudioTask._load_signal` + `log_mel_features` (24.4), `TextTask`'s
n-grams (45.2.1), `SignalTask` (92.1), `MidiTask` (92.3),
`PointCloudTask` (92.4), `VideoTask` at a smaller 4×16×16 sampling
(92.6, kept under the 1024 row budget) — so a mixed row is exactly what
the single-modality task would have seen, with the one-hot prefix.
Optional-decoder rules propagate (no imageio → the video row fails with
the install hint, like the video task). Head inference, the seed split,
train-only standardization, and the §22.1 score are the shared protocol.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task
from .media import (_AUDIO_EXTS, _IMAGE_EXTS, _MIDI_EXTS, _PC_EXTS,
                    _SIGNAL_EXTS, _TEXT_EXTS, _VIDEO_EXTS,
                    ROW_MODALITIES, class_label_str, collect_items,
                    modality_for_ext, resolve_labels, split_indices)

# 92.7.1: every row extension, in one tuple (the union of the per-
# modality sets, so `collect_items` keeps exactly the row modalities)
_ALL_ROW_EXTS = (_IMAGE_EXTS + _AUDIO_EXTS + _TEXT_EXTS
                 + _SIGNAL_EXTS + _MIDI_EXTS + _PC_EXTS + _VIDEO_EXTS)

_ROW_DIM = 1024  # 92.7.2: the constant per-row feature width
MIXED_FEATURE_DIM = len(ROW_MODALITIES) + _ROW_DIM  # 1031 (92.7.2)


def _pad_or_cut(v: np.ndarray) -> np.ndarray:
    """A modality feature → exactly `_ROW_DIM` values (92.7.2):
    truncated when longer (the video 4×16×16 + 3 case), zero-padded
    when shorter. Deterministic (G2)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size >= _ROW_DIM:
        return v[:_ROW_DIM].copy()
    out = np.zeros(_ROW_DIM, dtype=np.float64)
    out[:v.size] = v
    return out


def _row_feature(modality: str, p: Path) -> np.ndarray:
    """SPEC.md 92.7.1 (v0.78): one row's modality feature, via that
    modality's own decoder (the single-modality task's feature,
    unstandardized)."""
    from .images import ImageTask
    from .audio import AudioTask, log_mel_features
    from .text import ngram_features
    from .signals import load_signal, signal_features
    from .midi import midi_features, parse_midi
    from .pointcloud import load_pointcloud, pointcloud_features
    from .video import load_video_frames, video_features
    if modality == "image":
        return ImageTask._decode(p, 32)  # 1024 (24.3)
    if modality == "audio":
        sig, sr = AudioTask._load_signal(p)  # (24.4)
        return log_mel_features(sig, sr, 26)  # 26
    if modality == "text":
        return ngram_features(p.read_text(encoding="utf-8-sig",
                                          errors="replace"), 128)  # 128
    if modality == "signal":
        return signal_features(load_signal(p))  # 38 (92.1)
    if modality == "midi":
        return midi_features(*parse_midi(p))  # 35 (92.3)
    if modality == "pointcloud":
        return pointcloud_features(load_pointcloud(p))  # 83 (92.4)
    if modality == "video":
        grids = load_video_frames(p, 4, 16)  # (4, 16, 16), 92.6
        return video_features(grids)[:_ROW_DIM]  # 1027 -> 1024
    raise ValueError(
        f"{p}: unknown row modality {modality!r} — expected one of "
        f"{', '.join(ROW_MODALITIES)} (SPEC.md 92.7.1)")


def mixed_row(modality: str, p: Path) -> np.ndarray:
    """SPEC.md 92.7.2 (v0.78): one labelled row → the raw (1031,)
    feature — the 7-dim modality one-hot (`ROW_MODALITIES` order) + the
    row's modality feature padded/cut to 1024 (92.7.1). Deterministic
    given the file bytes (G2)."""
    if modality not in ROW_MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r} — expected one of "
            f"{', '.join(ROW_MODALITIES)} (SPEC.md 92.7.2)")
    one = np.zeros(len(ROW_MODALITIES), dtype=np.float64)
    one[ROW_MODALITIES.index(modality)] = 1.0
    return np.concatenate([one, _pad_or_cut(_row_feature(modality, p))])


class MixedTask(Task):
    name = "mixed"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train rows)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    feature_mean = None  # instance values (92.7)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2) -> None:
        if path is None:
            raise ValueError(
                "MixedTask needs a directory of labelled multi-modality "
                "media: use `autorefine fit --data DIR` or "
                "AutoRefineEnv(task='mixed', task_config={'path': DIR}) "
                "(SPEC.md 92.7)")
        p = Path(path)
        if not p.is_dir():
            raise ValueError(f"no such mixed directory: {path!r} "
                             "(SPEC.md 92.7)")
        self.seed = int(seed)
        self.path = str(path)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        items = collect_items(self.path, _ALL_ROW_EXTS, "media")
        mods = [modality_for_ext(Path(f).suffix.lower()) for f, _ in items]
        if any(m is None for m in mods):
            raise ValueError(
                f"{self.path}: an item extension is not a row modality "
                f"({', '.join(_ALL_ROW_EXTS)}; SPEC.md 92.7)")
        distinct = sorted(set(mods))
        if len(distinct) < 2:
            raise ValueError(
                f"{self.path}: a mixed task needs rows of at least 2 "
                f"modalities, got only {distinct} — use the "
                f"{distinct[0] if distinct else 'single'} task instead "
                f"(SPEC.md 92.7)")
        x = np.array([mixed_row(m, f) for (f, _), m in zip(items, mods)],
                     dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(x.shape[1])
        self.feature_names = ([f"modality {m}" for m in ROW_MODALITIES]
                              + [f"row feature {i}" for i in range(_ROW_DIM)])
        self._items = list(items)
        self._mods = list(mods)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"mixed-split")
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
        """SPEC.md 92.7 (v0.78): decode one media file to the raw
        (unstandardized) (1031,) row feature — the public wrapper
        `predict --item` calls."""
        p = Path(path)
        mod = modality_for_ext(p.suffix.lower())
        if mod is None:
            raise ValueError(
                f"{p}: not a recognised row-modality file "
                f"({', '.join(_ALL_ROW_EXTS)}) (SPEC.md 92.7)")
        return mixed_row(mod, p)

    # --- protocol (SPEC.md 15, 24.2 — mirrors PointCloudTask) ----------------
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
        """SPEC.md 92.7 (v0.78): the misclassified holdout rows (file
        entries; the mixed row has no single inline-renderable preview)."""
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
            out.append({
                "file": p.name,
                "path": str(p),
                "label": class_label_str(self.class_values[int(y[i])]),
                "predicted": class_label_str(self.class_values[int(pred[i])]),
            })
            if len(out) >= int(n_max):
                break
        return out
