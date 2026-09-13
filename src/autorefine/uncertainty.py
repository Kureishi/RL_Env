"""SPEC.md 64.2 (v0.50, B6): uncertainty on headline numbers — one home.

"96.4% ± 0.8, 5 seeds" on every headline figure, not just the gate: the
existing same-task seed spread (the 41.1 registry machinery) re-sinned as
a compact ``value ± half-spread (N runs)`` read. This leaf is the *single
home* for the statistics so the audience views (``audience``), the
decision / user-guide views (``reporting``), and the CLI all agree — the
``accounting`` / ``memory`` one-constant pattern.

House rules: stdlib only; pure and deterministic (G2) — same inputs, same
dict / string; no reads, no RNG, no timestamps. ``bool`` is excluded from
the numeric population (the 62.5 / 63 ``_num`` convention); non-finite
values are skipped, never propagated.
"""
from __future__ import annotations

import math


def _finite(v) -> bool:
    """A finite int/float that is not a bool (the 62.5 guard, one rule)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def seed_spread_stats(scores) -> dict | None:
    """64.2.1 (B6): the descriptive statistics over a population of final
    scores (usually the registry's same-task finals).

    ``{n_runs, min, max, mean, spread}`` where ``mean`` is the arithmetic
    mean of the finite values and ``spread = max - min``. ``None`` when the
    population is empty (``None`` input, no finite values) — a single run
    *is* a valid one-point population (spread 0), the caller decides
    whether to show it (``headline_uncertainty`` requires >= 2).
    """
    vals = [float(v) for v in (scores or []) if _finite(v)]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    return {
        "n_runs": len(vals),
        "min": lo,
        "max": hi,
        "mean": sum(vals) / len(vals),
        "spread": hi - lo,
    }


def format_pm(value: float, plus_minus: float) -> str:
    """64.2.1 (B6): the compact ``value ± plus_minus`` read (``%g`` — no
    trailing zeros; a zero half-spread renders as ``x ± 0``)."""
    return f"{value:g} \u00b1 {plus_minus:g}"


def headline_uncertainty(best_score, seed_spread) -> dict | None:
    """64.2.1 (B6): attach the seed spread to a headline number.

    ``{value, plus_minus, n_runs, text}`` where ``plus_minus`` is the
    *half-spread* (``(max - min) / 2``) and ``text`` is the ready-to-print
    ``"<value> ± <half-spread> (N runs)"`` line. ``None`` when there is no
    finite headline value, no spread population, or fewer than 2 runs —
    a single run has no variance to display (the views render the absent
    block as ``—`` and the guides omit the line; the A52/A53 pins keep
    asserting the pre-v0.50 output).
    """
    if not _finite(best_score):
        return None
    stats = seed_spread_stats(seed_spread)
    if stats is None or stats["n_runs"] < 2:
        return None
    value = float(best_score)
    half = stats["spread"] / 2.0
    return {
        "value": value,
        "plus_minus": half,
        "n_runs": stats["n_runs"],
        "text": f"{format_pm(value, half)} ({stats['n_runs']} runs)",
    }


__all__ = [
    "seed_spread_stats",
    "format_pm",
    "headline_uncertainty",
]
