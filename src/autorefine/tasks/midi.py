"""MidiTask: a directory of labelled MIDI music files as a fitting task
(SPEC.md 92.3, v0.78, A82).

The music modality: a pure-Python MThd/MTrk parser (92.3.1 — stdlib
`struct`-free byte scanning, zero new dependencies) extracts note-on
events (beat, pitch) and the set-tempo meta events; the feature is a
fixed 35-d vector (92.3.2): duration / density scalars, a 12-bin
pitch-class histogram, a 16-bin onset-time histogram, inter-onset and
note-length statistics, and the tempo. The labelled-directory protocol
(24.2) is the shared one — head inference, seed split, train-only
standardization, §22.1 score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task
from .media import (_MIDI_EXTS, class_label_str, collect_items,
                    resolve_labels, split_indices)

_N_ONSET_BINS = 16
_N_PITCH_CLASSES = 12


def _vlq(data: bytes, i: int) -> tuple[int, int]:
    """Variable-length quantity at byte `i` → (value, next index)."""
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            return value, i


def parse_midi(p: Path) -> tuple[int, float, list[tuple[float, int]]]:
    """SPEC.md 92.3.1 (v0.78): one `.mid`/`.midi` file →
    (division, tempo_bps or 0.0, [(beat, pitch), ...]) note-on events.

    Pure stdlib byte scanning (92.3.1): header chunk (format / ntracks /
    division), then each track's delta-time VLQ stream — meta events
    (set-tempo 0x51) and channel voice note-on events (0x9n, velocity
    > 0) with running status. Deterministic given the file bytes (G2);
    malformed or truncated files raise ValueError.
    """
    data = p.read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError(f"{p}: not a MIDI file (missing MThd header)")
    fmt = int.from_bytes(data[8:10], "big")
    ntracks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division <= 0 or ntracks < 1:
        raise ValueError(f"{p}: unsupported MIDI header "
                         f"(format={fmt}, ntracks={ntracks}, division={division})")
    i = 14
    tempo_bps = 0.0
    onsets: list[tuple[float, int]] = []
    for _ in range(ntracks):
        if i + 8 > len(data):
            raise ValueError(f"{p}: truncated MIDI file (track header)")
        if data[i:i + 4] != b"MTrk":
            raise ValueError(f"{p}: not a MIDI file (missing MTrk)")
        length = int.from_bytes(data[i + 4:i + 8], "big")
        i += 8
        end = min(i + length, len(data))
        tick = 0
        running = 0
        while i < end:
            delta, i = _vlq(data, i)
            tick += delta
            if i >= end:
                break
            status = data[i]
            if status & 0x80:
                running = status
                i += 1
            st = running
            if st == 0xFF:  # meta event
                if i >= end:
                    break
                mtype = data[i]
                i += 1
                mlen, i = _vlq(data, i)
                if mtype == 0x51 and mlen >= 3:  # set tempo (us/quarter)
                    us = int.from_bytes(data[i:i + 3], "big")
                    if us > 0:
                        tempo_bps = 60_000_000.0 / us
                i += mlen
                continue
            if 0x80 <= st <= 0xEF:  # channel voice message
                if st & 0x0F == 0x0:  # note off: no payload after 2 data bytes
                    i += 2
                    continue
                if st & 0x0F == 0x0C:  # program change: 1 data byte
                    i += 1
                    continue
                if i + 2 > end:
                    break
                d1 = data[i]
                d2 = data[i + 1]
                if (st & 0x0F) == 0x1 and d2 > 0:  # note on, velocity > 0
                    onsets.append((tick / division, d1))
                i += 2
                continue
            # sysex (0xF0/0xF7) and other system common: skip their VLQ length
            mlen, i = _vlq(data, i)
            i += mlen
    return division, tempo_bps, onsets


MIDI_FEATURE_DIM = 3 + _N_PITCH_CLASSES + _N_ONSET_BINS + 4  # 35 (92.3.2)


def midi_features(division: int, tempo_bps: float,
                  onsets: list[tuple[float, int]],
                  n_onset_bins: int = _N_ONSET_BINS) -> np.ndarray:
    """SPEC.md 92.3.2 (v0.78): the raw (35,) feature of one MIDI file.

    [total beats, log1p note count, onset density,
     12 × log1p pitch-class counts,
     `n_onset_bins` × log1p onset-time counts over the beat axis,
     median inter-onset interval (beats), the reserved mean-note-length
     slot (0.0 — the parser keeps onsets only), tempo (bps / 200),
     distinct pitch classes / 12]

    An onset-free file yields the all-zero vector (deterministic, G2).
    """
    feats = np.zeros(3 + _N_PITCH_CLASSES + n_onset_bins + 4, dtype=np.float64)
    if not onsets:
        return feats
    beats = max(onsets[-1][0], 1e-6)
    notes = np.array(onsets, dtype=np.float64)  # (n, 2): beat, pitch
    feats[0] = beats
    feats[1] = float(np.log1p(len(notes)))
    feats[2] = float(len(notes) / beats)
    pitches = notes[:, 1].astype(int) % _N_PITCH_CLASSES
    pc = np.bincount(pitches, minlength=_N_PITCH_CLASSES).astype(np.float64)
    feats[3:3 + _N_PITCH_CLASSES] = np.log1p(pc)
    ob = np.zeros(n_onset_bins, dtype=np.float64)
    for b, _p in onsets:
        idx = min(int(b / beats * n_onset_bins), n_onset_bins - 1)
        ob[idx] += 1.0
    o0 = 3 + _N_PITCH_CLASSES
    feats[o0:o0 + n_onset_bins] = np.log1p(ob)
    offs = o0 + n_onset_bins  # 31: the 4 tail scalars
    if len(notes) >= 2:
        ious = np.diff(np.sort(notes[:, 0]))
        feats[offs] = float(np.median(ious))
    # feats[offs + 1] is the reserved mean-note-length slot (0.0)
    feats[offs + 2] = float(tempo_bps / 200.0)
    feats[offs + 3] = float(len(set(pitches)) / _N_PITCH_CLASSES)
    return feats


class MidiTask(Task):
    name = "midi"
    max_steps = 1  # not an episode task; protocol completeness (SPEC.md 15)
    head = "softmax"
    n_outputs = 2
    state_dim = 1
    default_dataset_size = None  # instance value: len(train items)
    metric = "accuracy"
    capabilities = frozenset({"media"})
    feature_mean = None  # instance values (92.3)
    feature_std = None

    def __init__(self, seed: int, path: str | Path | None = None,
                 label: str | None = None, split_frac: float = 0.2) -> None:
        if path is None:
            raise ValueError(
                "MidiTask needs a directory of labelled MIDI files: use "
                "`autorefine fit --data DIR` or AutoRefineEnv(task='midi', "
                "task_config={'path': DIR}) (SPEC.md 92.3)")
        self.seed = int(seed)
        self.path = str(path)
        self.split_frac = float(split_frac)
        self.label_name = "class"

        items = collect_items(self.path, _MIDI_EXTS, "midi")
        x = np.array([midi_features(*parse_midi(p)) for p, _ in items],
                     dtype=np.float64)

        (self.head, self.n_outputs, self.class_values, self._y) = \
            resolve_labels([lb for _, lb in items])
        self.metric = ("accuracy" if self.head == "softmax" else "r2")
        self.state_dim = int(x.shape[1])
        self.feature_names = (["total beats", "log1p note count", "onset density"]
                             + [f"pitch class {i}" for i in range(_N_PITCH_CLASSES)]
                             + [f"onset bin {i}" for i in range(_N_ONSET_BINS)]
                             + ["median inter-onset", "mean note length",
                                "tempo (bps/200)", "pitch-class span"])
        self._items = list(items)
        tr, ho, ge = split_indices(len(items), self.seed, self.split_frac,
                                   b"midi-split")
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
        """SPEC.md 92.3 (v0.78): decode one MIDI file to the raw
        (unstandardized) feature — the public wrapper `predict --item`
        calls."""
        return midi_features(*parse_midi(Path(path)))

    # --- protocol (SPEC.md 15, 24.2 — mirrors TextTask) ----------------------
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
        """SPEC.md 92.3 (v0.78): the misclassified holdout items (file
        entries; MIDI has no inline-renderable media beyond the file)."""
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
