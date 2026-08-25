"""Parity tasks: noisy XOR classification (README "Extending" / SPEC.md 17).

A deliberately non-linear classification task: the label is the XOR (parity)
of n hidden bits, and the model only observes each bit after seeded bit-flip
noise. This creates a clean family gradient — linear models cannot represent
parity (~random), an MLP approaches the Bayes ceiling, and shallow tree
ensembles are weak — so the improver has real signal in architecture,
capacity, optimizer and training budget.

`ParityTask` is the parameterized family (n_bits, p_flip) used by the v0.6
curriculum (SPEC.md 20.1); `Parity4V1` is the v0.3 registry entry
(4 bits, 8% flip) with its literal identity kept for back-compat.

Score (holdout/gen): 100 * accuracy on fresh seed-derived points.
Deterministic given the seed (SPEC.md G2, S3).
"""
from __future__ import annotations

import zlib

import numpy as np

N_BITS = 4
P_FLIP = 0.08  # v0.3 seeded observation noise per bit


def _split_rng(seed: int, split: str) -> np.random.Generator:
    seq = np.random.SeedSequence([seed, zlib.crc32(split.encode("utf-8"))])
    return np.random.default_rng(seq)


def _parity_points(seed: int, split: str, n: int,
                   n_bits: int = N_BITS, p_flip: float = P_FLIP
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(noisy bits x (n, n_bits), clean parity y (n,)) — balanced by construction."""
    rng = _split_rng(seed, split)
    bits = rng.integers(0, 2, (n, n_bits))
    y = bits.sum(axis=1) % 2
    x = np.where(rng.random((n, n_bits)) < p_flip, 1 - bits, bits)
    return x, y


def parity_ceiling(n_bits: int = N_BITS, p_flip: float = P_FLIP) -> float:
    """Bayes accuracy ceiling in score points (SPEC.md 20.1).

    P(correct parity) = P(even # flips) = (1 + (1 - 2p)^n) / 2."""
    return 100.0 * (1.0 + (1.0 - 2.0 * float(p_flip)) ** int(n_bits)) / 2.0


class ParityTask:
    """Parameterized noisy-parity task (SPEC.md 20.1): n_bits in 1..8,
    p_flip in [0.0, 0.5). Identity derives from (n_bits, p_flip);
    Parity4V1 below pins the v0.3 identity."""

    head = "softmax"
    n_outputs = 2
    max_steps = 1  # not an episode task; present for protocol completeness
    default_dataset_size = 4096  # SPEC.md 20.3 (points)

    def __init__(self, seed: int, n_bits: int = N_BITS, p_flip: float = P_FLIP) -> None:
        if not 1 <= int(n_bits) <= 8:
            raise ValueError(f"n_bits must be in 1..8, got {n_bits!r}")
        if not 0.0 <= float(p_flip) < 0.5:
            raise ValueError(f"p_flip must be in [0.0, 0.5), got {p_flip!r}")
        self.seed = int(seed)
        self.n_bits = int(n_bits)
        self.p_flip = float(p_flip)

    @property
    def name(self) -> str:
        return f"parity-{self.n_bits}-p{self.p_flip:.2f}"

    @property
    def state_dim(self) -> int:
        return self.n_bits

    # --- data ----------------------------------------------------------------
    def initial_conditions(self, split: str, n: int) -> np.ndarray:
        """Seed-derived noisy bit observations (n, n_bits)."""
        x, _ = _parity_points(self.seed, split, n, self.n_bits, self.p_flip)
        return x

    def prepare(self, states: np.ndarray) -> np.ndarray:
        return states  # binary features need no transform

    def make_dataset(self, n_points: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        """(noisy bits (n, n_bits), parity labels (n,)) on the train split."""
        return _parity_points(self.seed, "train", n_points, self.n_bits, self.p_flip)

    # --- scoring ---------------------------------------------------------------
    def score(self, model, split: str, n: int) -> float:
        """100 * accuracy on fresh points of `split`; higher is better."""
        x, y = _parity_points(self.seed, split, n, self.n_bits, self.p_flip)
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        return float(100.0 * (pred == y).mean())


class Parity4V1(ParityTask):
    """v0.3 registry task: 4 bits, 8% flip (SPEC.md 17). Literal name and
    state_dim pin the v0.3 identity (back-compat, A7)."""

    name = "parity-v1"
    state_dim = N_BITS

    def __init__(self, seed: int) -> None:
        super().__init__(seed, n_bits=N_BITS, p_flip=P_FLIP)
