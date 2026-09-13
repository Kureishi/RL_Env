"""FinanceForecastTask: time-ordered horizon forecasting (SPEC.md 61.2.2, A2).

A reference *domain task pack* for time-ordered data: a seeded random-walk
plus sinusoid price series, **temporal** (walk-forward — the §45.1 rule, no
leakage), predicting the **next-``h``** change from a ``lookback``-length
window. The score is a **directional** metric (``100 · directional_accuracy``)
where a flat (zero) prediction scores exactly 50 (chance level) — the
intuitive "no signal → coin flip" bar.

Deterministic given the seed (G2); stdlib + numpy only. Exposes
``holdout_rows`` (A1's ``cost`` actual) + ``starter_spec`` /
``default_objectives`` (the domain's bar).
"""
from __future__ import annotations

import zlib

import numpy as np

from ..config import DEFAULT_SPEC
from ..gate import Objective
from .base import Task

LOOKBACK = 5      # feature window length (state_dim)
HORIZON = 1       # predict the change over the next `h` steps
BASE_N = 1024     # rows per split (default_dataset_size)


def _price_series(seed: int, length: int) -> np.ndarray:
    """A deterministic random-walk plus a slow sinusoid price series of
    ``length`` points (G2) — the single source every split draws from, so
    train strictly precedes holdout (temporal, no leakage)."""
    seq = np.random.SeedSequence([seed, zlib.crc32(b"finance-series")])
    rng = np.random.default_rng(seq)
    steps = rng.normal(0.0, 1.0, length)
    walk = np.cumsum(steps)
    t = np.arange(length, dtype=np.float64)
    season = 2.0 * np.sin(0.05 * t) + 1.0 * np.cos(0.013 * t)
    return walk + season


def _directional_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """SPEC.md 61.2.2: per-row direction credit in ``[0, 1]`` —
    * ``true == 0`` → 0.5 (no direction to match);
    * ``pred == 0`` (flat) → 0.5 (a coin flip);
    * else → 1.0 when ``sign(pred) == sign(true)``, else 0.0.
    A fully-flat model therefore scores exactly 0.5 (chance)."""
    credit = np.zeros(len(true), dtype=np.float64)
    zero_true = (true == 0)
    credit[zero_true] = 0.5
    nz = ~zero_true
    zero_pred = (pred == 0)
    credit[nz & zero_pred] = 0.5
    both_nz = nz & ~zero_pred
    same = (pred[both_nz] > 0) == (true[both_nz] > 0)
    credit[both_nz] = np.where(same, 1.0, 0.0)
    return float(credit.mean()) if len(true) else 0.0


class FinanceForecastTask(Task):
    """SPEC.md 61.2.2 (A2): time-ordered horizon forecasting,
    ``finance-v1``. ``split_mode = "temporal"``; ``score = 100 ·
    directional_accuracy`` (flat → 50); exposes ``holdout_rows`` + a
    ``cost`` (mean absolute scaled error, 61.5)."""

    head = "mse"
    n_outputs = 1
    state_dim = LOOKBACK
    max_steps = 1  # not an episode task; protocol completeness
    default_dataset_size = BASE_N
    metric = "directional"   # score = 100·directional_accuracy
    capabilities = frozenset()
    split_mode = "temporal"  # walk-forward (SPEC.md 45.1)

    def __init__(self, seed: int, lookback: int = LOOKBACK,
                 horizon: int = HORIZON) -> None:
        if int(lookback) < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback!r}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon!r}")
        self.seed = int(seed)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.state_dim = int(lookback)
        # one canonical series long enough for 3 splits of BASE_N rows
        # (each row i needs s[i : i+lookback+horizon]); cached (G2).
        _len = 3 * BASE_N + self.lookback + self.horizon + 1
        self._series = _price_series(self.seed, _len)

    @property
    def name(self) -> str:
        return "finance-v1"

    def _rows(self, block: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The (x, y) rows for one time block, clamped to ``n`` — block is
        'train' / 'holdout' / 'gen', at the corresponding time indices."""
        x = self._series
        lb, h = self.lookback, self.horizon
        start = {"train": 0, "holdout": BASE_N, "gen": 2 * BASE_N}[block]
        x_rows = np.stack([x[start + i:start + i + lb] for i in range(n)],
                          axis=0) if n else np.zeros((0, lb))
        # row i: target = s[start+i+lb+h] - s[start+i+lb] (the next-`h` change)
        y = x[start + lb + h:start + lb + h + n] - x[start + lb:start + lb + n]
        return x_rows[:n], y[:n]

    def make_dataset(self, n_points: int = BASE_N) -> tuple[np.ndarray, np.ndarray]:
        """Train-split (window features (n, lookback), next-``h`` change)."""
        n = min(int(n_points), BASE_N)
        return self._rows("train", n)

    def _block_for(self, split: str) -> str:
        s = (split or "").lower()
        if s.startswith("gen"):
            return "gen"
        if s.startswith("train"):
            return "train"
        return "holdout"

    def holdout_rows(self, n: int, model=None) -> tuple[np.ndarray, np.ndarray]:
        """SPEC.md 61.2.2 (A1): the holdout ``(x, y)``, clamped to ``n`` —
        the rows A1's ``cost`` actual is computed over."""
        return self._rows("holdout", min(int(n), BASE_N))

    def score(self, model, split: str, n: int) -> float:
        """``100 · directional_accuracy`` on `split` (higher = better;
        a flat prediction → 50)."""
        x, y = self._rows(self._block_for(split), min(int(n), BASE_N))
        pred = np.asarray(model.forward(x), dtype=np.float64).ravel()
        pred = pred[:len(y)]
        return float(100.0 * _directional_accuracy(pred, y))

    def cost(self, model, split: str, n: int) -> float:
        """SPEC.md 61.5 (A5): the domain-specific per-prediction badness —
        the mean absolute scaled error ``mean(|pred − true|) / scale``
        (lower = better), where ``scale = max(1, std(true))``."""
        x, y = self._rows(self._block_for(split), min(int(n), BASE_N))
        pred = np.asarray(model.forward(x), dtype=np.float64).ravel()[:len(y)]
        mae = float(np.abs(pred - y).mean())
        scale = max(1.0, float(np.std(y)))
        return mae / scale

    # --- the domain's bar (61.2) ----------------------------------------------
    @property
    def starter_spec(self) -> dict:
        """The domain-appropriate starting ModelSpec (61.2.2)."""
        return DEFAULT_SPEC.to_dict()

    def default_objectives(self, target: float) -> tuple[Objective, ...]:
        """The domain's acceptance bar (61.2.2): the directional score gate
        plus a cost (badness) upper bound (61.5)."""
        return (
            Objective(name="score", op=">=", threshold=float(target)),
            Objective(name="cost", op="<=", threshold=1.0),
        )
