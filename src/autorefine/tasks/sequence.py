"""SequenceMotifV1: a native *sequence* task (SPEC.md 83.1, v0.69).

The model reads a (T, C) binary sequence and classifies whether a fixed
short *motif* (a specific run of bits, e.g. ``1 0 1``) appears anywhere in
it. It exists to exercise the two native temporal families (``conv1d`` /
``rnn``, SPEC.md 83.2 / 83.3): motif detection is the textbook problem both
temporal aggregators solve naturally — a causal Conv1D is a literal local
pattern detector, and an Elman RNN recognizes the motif as a finite-state
machine — while a flat MLP on the flattened bits must re-recognize the motif
at every position (a strictly harder, position-entangled problem). On the
built-in T = 10 level the temporal families reach high accuracy (the causal
conv ~100, the RNN ~94) while an un-tuned flat MLP trails (~87): the clearest
built-in demonstration that the temporal families earn their place (83.4).

Layout (SPEC.md 83.1):
  * ``capabilities = frozenset({"sequence"})`` — the bandit offers the
    temporal families for this task (catalog.relevant_families, 25.5);
  * ``make_dataset`` / ``initial_conditions`` yield **3-D** ``X`` of shape
    ``(n, T, C)`` — the sequence layout the conv1d / rnn trainer branch reads
    directly (the 83.1 guard: a conv1d/rnn spec on non-3-D data is a clean
    SpecError, never a silent fallback; a flat family reads the flattened
    (n, T*C) rows — the trainer's and ``score``'s shared layout rule);
  * ``score`` = 100 * accuracy of ``argmax(model.forward(X))`` on fresh
    seed-derived sequences (declared ``metric`` = "accuracy", G1).

Deterministic given the seed (G2, S3): the bits are drawn from a seed-derived
RNG (the §22.1 seed-permutation pattern), and the motif label is a pure
search over the bits (no RNG).
"""
from __future__ import annotations

import zlib

import numpy as np

from .base import Task

# SPEC.md 83.1 defaults: a moderate-length binary sequence with a 3-bit motif.
# T = 10 gives the motif (length 3) 8 possible start positions — enough for a
# flat MLP to struggle to recognize it everywhere, short enough that the
# causal Conv1D / Elman RNN converge in a few hundred steps (83.4).
DEFAULT_TIMESTEPS = 10
DEFAULT_CHANNELS = 1
DEFAULT_MOTIF = (1, 0, 1)   # the pattern the task classifies "present / absent"


def motif_present(bits: np.ndarray, motif: tuple[int, ...] = DEFAULT_MOTIF
                  ) -> np.ndarray:
    """Whether the fixed `motif` appears anywhere in each (T, C) / (T,) row.

    A pure search over the bits (deterministic, no RNG): the label of a
    sequence-motif sample. Works on a single (T,) / (T, C) row or a batch
    (n, T, C) (reduces over the trailing axes; C must be 1 — the motif is a
    run of bits along the time axis of a single channel)."""
    b = np.asarray(bits).astype(np.int64) % 2
    m = np.asarray(motif, dtype=np.int64)
    L = len(m)
    if b.ndim == 1:
        return int(any((b[t:t + L] == m).all() for t in range(b.shape[0] - L + 1)))
    if b.ndim != 3 or b.shape[2] != 1:
        raise ValueError(f"motif_present needs (T,) or (n, T, 1); got {b.shape}")
    col = b[:, :, 0]
    n, T = col.shape
    hit = np.zeros(n, dtype=np.int64)
    for t in range(T - L + 1):
        hit = np.bitwise_or(hit, (col[:, t:t + L] == m).all(axis=1))
    return hit


class SequenceMotifV1(Task):
    name = "sequence-motif-v1"
    n_outputs = 2          # motif absent (0) / present (1)
    head = "softmax"
    max_steps = 1          # not an episode task; present for protocol completeness
    default_dataset_size = 512   # SPEC.md 20.3 (number of sequences)
    # SPEC.md 36.1.4 / 83.1: a sequence-layout task — the temporal families
    # (conv1d / rnn) are offered for it (catalog.relevant_families, 25.5).
    capabilities = frozenset({"sequence"})
    metric = "accuracy"    # score() = 100 * accuracy (G1: declared, not inferred)

    # The flattened feature width a flat model would see (T * C); the
    # temporal families read the native (T, C) layout instead (SPEC.md 83.1).
    state_dim: int = DEFAULT_TIMESTEPS * DEFAULT_CHANNELS

    def __init__(self, seed: int,
                 timesteps: int = DEFAULT_TIMESTEPS,
                 channels: int = DEFAULT_CHANNELS,
                 motif: tuple[int, ...] = DEFAULT_MOTIF) -> None:
        if not (1 <= timesteps <= 64):
            raise ValueError(f"timesteps must be in 1..64, got {timesteps!r}")
        if not (1 <= channels <= 16):
            raise ValueError(f"channels must be in 1..16, got {channels!r}")
        if len(motif) < 1 or len(motif) > timesteps:
            raise ValueError(f"motif length {len(motif)} outside 1..T={timesteps}")
        self.seed = int(seed)
        self.T = int(timesteps)
        self.C = int(channels)
        self.motif = tuple(int(v) % 2 for v in motif)
        # the protocol's state_dim reflects the flattened layout (T * C)
        self.state_dim = self.T * self.C

    def _split_rng(self, split: str) -> np.random.Generator:
        seq = np.random.SeedSequence([self.seed, zlib.crc32(split.encode("utf-8"))])
        return np.random.default_rng(seq)

    # --- data ----------------------------------------------------------------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived binary sequences (n, T, C) on one of train/holdout/gen."""
        rng = self._split_rng(split)
        bits = rng.integers(0, 2, (n, self.T, self.C))
        return bits.astype(np.float64)

    def make_dataset(self, n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """(X (n, T, C), y (n,)) on the train split: the bits and the label."""
        X = self.initial_conditions("train", n_points)
        y = motif_present(X, self.motif).astype(np.int64)
        return X, y

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # binary bits need no transform

    # --- scoring ---------------------------------------------------------------
    def score(self, model, split: str, n: int) -> float:
        """100 * accuracy of ``argmax(model.forward(X))`` on fresh sequences.

        The native temporal families (conv1d / rnn) declare ``wants_sequence``
        and read the (n, T, C) layout directly; a flat model reads the
        flattened (n, T*C) rows (the same 83.1 layout rule the trainer uses).
        Higher is better; this is the official holdout/gen scoring (SPEC.md 15).
        """
        X = self.initial_conditions(split, n)
        y = motif_present(X, self.motif).astype(np.int64)
        if not getattr(model, "wants_sequence", False):
            X = X.reshape(n, -1)
        logits = np.asarray(model.forward(X), dtype=np.float64)
        pred = np.argmax(logits, axis=1)
        return float((pred == y).mean()) * 100.0
