"""SPEC.md 64.2 (v0.50, B6): uncertainty on headline numbers — one home.

"96.4% ± 0.8, 5 seeds" on every headline figure, not just the gate: the
existing same-task seed spread (the 41.1 registry machinery) re-sinned as
a compact ``value ± half-spread (N runs)`` read. This leaf is the *single
home* for the statistics so the audience views (``audience``), the
decision / user-guide views (``reporting``), and the CLI all agree — the
``accounting`` / ``memory`` one-constant pattern.

66 (v0.52, B7) adds ``verdict_robustness`` — the same home also answers
the decision-maker's trust question: does the GO/NO-GO *survive* the
seed band, or is it a near-miss — and ``summary_target``, the one home
for resolving a run's acceptance target (synthetic top-level first,
then the canonical ``run_config.target`` of real runs, 66.1.1).

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


def verdict_robustness(target, best, seed_spread) -> dict | None:
    """66.1 (v0.52, B7): is the GO/NO-GO verdict *robust* to run-to-run
    variance — the decision-maker's trust question, answered against the
    same seed band ``headline_uncertainty`` already displays.

    Classification (hand-computable, from ``half = (max − min) / 2``,
    ``band_low = best − half``, ``band_high = best + half``):

    - ``robust_go``    — ``band_low >= target`` (the whole band clears the line)
    - ``marginal_go``  — ``best >= target`` but ``band_low < target``
      (the point clears; a different seed could dip below)
    - ``marginal_no``  — ``best < target`` but ``band_high >= target``
      (the point misses; a different seed could clear it)
    - ``robust_no``    — ``band_high < target`` (the whole band is below)
    - ``unassessable`` — no finite ``target`` set, or fewer than 2 seed
      runs: the point-estimate verdict stands and the read says so
      ("run a seed sweep to bound the risk")

    ``None`` when there is no finite ``best`` — there is no verdict to
    robustify (the views render the block as ``—``, the 64.2.3 rule).

    Result ``{status, band, one_liner, flip}``: ``band`` is
    ``{low, high, half, n_runs}`` (``None`` in the unassessable case),
    ``one_liner`` is the quoteable plain-language decision, and ``flip``
    is the "what would flip this" note (the band edge's distance to the
    target; ``None`` in the unassessable case). Pure and deterministic
    (G2); ``bool`` is excluded from the population (the 62.5 guard); the
    status is a function of ``(target, best, seed_spread)`` only.
    """
    b = float(best) if _finite(best) else None
    if b is None:
        return None
    t = float(target) if _finite(target) else None
    stats = seed_spread_stats(seed_spread)
    n = stats["n_runs"] if stats is not None else 0
    if t is None or n < 2:
        if t is None:
            one_liner = ("no acceptance target was set — nothing to "
                         "robustify against; the point-estimate verdict "
                         "stands")
        else:
            one_liner = ("the point-estimate verdict stands — fewer than "
                         "2 seed runs, so the variance band is unmeasured; "
                         "run a seed sweep to bound the risk")
        return {"status": "unassessable", "band": None,
                "one_liner": one_liner, "flip": None}
    half = stats["spread"] / 2.0
    low, high = b - half, b + half
    band = {"low": low, "high": high, "half": half, "n_runs": n}
    if b >= t:
        if low >= t:
            status = "robust_go"
            one_liner = (f"GO is robust: the whole seed band "
                         f"({low:g}–{high:g}) clears the {t:g} target "
                         f"({n} seed runs)")
            flip = (f"safe by {low - t:g} pt(s): the band's low end "
                    f"({low:g}) still clears the target ({t:g})")
        else:
            status = "marginal_go"
            one_liner = (f"GO, but marginal: the point {b:g} clears the "
                         f"{t:g} target, yet the seed band dips to {low:g} "
                         f"— a different seed could miss ({n} seed runs)")
            flip = (f"at risk by {t - low:g} pt(s): a seed run this low "
                    f"({low:g}) would miss the target ({t:g})")
    else:
        if high < t:
            status = "robust_no"
            one_liner = (f"NO-GO is robust: the whole seed band "
                         f"({low:g}–{high:g}) sits below the {t:g} target "
                         f"({n} seed runs)")
            flip = (f"short by {t - high:g} pt(s): even the band's high "
                    f"end ({high:g}) sits below the target ({t:g})")
        else:
            status = "marginal_no"
            one_liner = (f"NO-GO, but marginal: the point {b:g} misses "
                         f"the {t:g} target, yet the seed band reaches "
                         f"{high:g} — a different seed could clear it "
                         f"({n} seed runs)")
            flip = (f"within reach by {high - t:g} pt(s): a seed run this "
                    f"high ({high:g}) would clear the target ({t:g})")
    return {"status": status, "band": band,
            "one_liner": one_liner, "flip": flip}


def summary_target(summary) -> float | None:
    """66.1.1 (v0.52, B7): the acceptance target of a run's summary — one
    home so the two decision surfaces (and the tests that recompute them)
    cannot drift from the run's own artifacts.

    The top-level ``target`` (the synthesized-summary convention the A52–
    A54 views are pinned on) wins when finite; else the canonical
    ``run_config.target`` (real-run summaries keep the RunConfig, 47.1 —
    the same source ``research.run_verdict`` reads). ``None`` when
    neither carries a finite target (an exploratory run: nothing to
    robustify against). Pure; ``bool`` never qualifies (the 62.5 guard).
    """
    if not isinstance(summary, dict):
        return None
    t = summary.get("target")
    if _finite(t):
        return float(t)
    rc = summary.get("run_config")
    if isinstance(rc, dict):
        t = rc.get("target")
        if _finite(t):
            return float(t)
    return None


__all__ = [
    "seed_spread_stats",
    "format_pm",
    "headline_uncertainty",
    "verdict_robustness",
]
