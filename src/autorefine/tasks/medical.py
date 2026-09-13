"""MedicalTabularTask: imbalanced, cost-sensitive classification (SPEC.md 61.2.1, A2).

A reference *domain task pack*: a binary classification with a ~5% positive
rate (imbalanced by construction) and a **cost-sensitive** score
``100 · (TPR − β·FPR)`` (β the false-positive penalty, default 1.0, clamped
at 0) — a single number the improver can climb while the *per-class*
constraint (A1, ``per_class_f1``) keeps the minority class from being
sacrificed. Positives are a shifted-Gaussian blob (2σ off the negative
mean), so the task is learnable by a small net and the score is
meaningful, not a coin flip.

Deterministic given the seed (G2); stdlib + numpy only. Delivered through
the §21 plugin loader (the in-repo ``TASKS`` registry is the same
mechanism an external entry-point uses).
"""
from __future__ import annotations

import zlib

import numpy as np

from ..config import DEFAULT_SPEC
from ..gate import Objective
from .base import Task

POS_RATE = 0.05      # ~5% positive by construction (imbalanced)
BETA = 1.0           # false-positive penalty in 100·(TPR − β·FPR)
POS_SHIFT = 2.0      # positives = a 2σ-shifted blob (learnable)
N_FEATURES = 4


def _medical_points(seed: int, split: str, n: int,
                    pos_rate: float = POS_RATE,
                    n_features: int = N_FEATURES
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(features (n, d), binary labels (n,)) on one split — ~pos_rate
    positive, positives a shifted blob. Deterministic (G2)."""
    seq = np.random.SeedSequence([seed, zlib.crc32(f"medical-{split}".encode("utf-8"))])
    rng = np.random.default_rng(seq)
    y = (rng.random(n) < float(pos_rate)).astype(np.int64)
    x = rng.normal(0.0, 1.0, (n, n_features))
    x[y == 1] += float(POS_SHIFT)  # the positive blob (learnable, no label leak)
    return x, y


class MedicalTabularTask(Task):
    """SPEC.md 61.2.1 (A2): imbalanced cost-sensitive binary task,
    ``medical-v1``. ``score = 100·(TPR − β·FPR)`` clamped at 0; exposes
    ``holdout_rows`` (A1's ``per_class_f1`` actual) + a ``starter_spec`` /
    ``default_objectives`` (the domain's bar)."""

    head = "softmax"
    n_outputs = 2
    state_dim = N_FEATURES
    max_steps = 1  # not an episode task; protocol completeness
    default_dataset_size = 4096  # points
    metric = "cost_sensitive"    # score = 100·(TPR − β·FPR)
    capabilities = frozenset()
    split_mode = "random"
    class_values = [0.0, 1.0]    # class 0 = negative, class 1 = positive

    def __init__(self, seed: int, pos_rate: float = POS_RATE,
                 beta: float = BETA, n_features: int = N_FEATURES) -> None:
        if not 0.0 < float(pos_rate) < 1.0:
            raise ValueError(f"pos_rate must be in (0, 1), got {pos_rate!r}")
        if not float(beta) >= 0.0:
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        if int(n_features) < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features!r}")
        self.seed = int(seed)
        self.pos_rate = float(pos_rate)
        self.beta = float(beta)
        self.n_features = int(n_features)
        self.state_dim = int(n_features)

    @property
    def name(self) -> str:
        return "medical-v1"

    def make_dataset(self, n_points: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """(features (n, d), binary labels (n,)) on the train split."""
        return _medical_points(self.seed, "train", n_points,
                               self.pos_rate, self.n_features)

    def _split(self, split: str) -> str:
        s = (split or "").lower()
        if s.startswith("gen"):
            return "gen"
        if s.startswith("train"):
            return "train"
        return "holdout"

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 61.2.1 (A1): the holdout ``(x, y)``, clamped to ``n`` —
        the rows A1's ``per_class_f1`` / ``ece`` actuals are computed over.
        ``model`` is accepted for a uniform protocol but ignored."""
        x, y = _medical_points(self.seed, "holdout", n,
                               self.pos_rate, self.n_features)
        return x[:n], y[:n]

    def score(self, model, split: str, n: int) -> float:
        """``100 · (TPR − β·FPR)`` on `split`, clamped at 0 (higher =
        better). A model that predicts all-negative scores 0 (TPR=0);
        all-positive scores 0 too (FPR=1, β=1)."""
        x, y = _medical_points(self.seed, self._split(split), n,
                               self.pos_rate, self.n_features)
        n = min(int(n), len(x))
        x, y = x[:n], y[:n]
        pred = np.asarray(model.forward(x), dtype=np.float64).argmax(axis=1)
        pos = (y == 1)
        neg = (y == 0)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        tp = int(((pred == 1) & pos).sum())
        fp = int(((pred == 1) & neg).sum())
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        return float(max(0.0, 100.0 * (tpr - self.beta * fpr)))

    # --- the domain's bar (61.2) ----------------------------------------------
    @property
    def starter_spec(self) -> dict:
        """The domain-appropriate starting ModelSpec (a ``DEFAULT_SPEC``-
        shaped dict; 61.2.1)."""
        return DEFAULT_SPEC.to_dict()

    def default_objectives(self, target: float) -> tuple[Objective, ...]:
        """The domain's acceptance bar (61.2.1): the score gate plus a
        per-class constraint so the minority class is not sacrificed."""
        return (
            Objective(name="score", op=">=", threshold=float(target)),
            Objective(name="per_class_f1", op=">=", threshold=0.6),
        )
