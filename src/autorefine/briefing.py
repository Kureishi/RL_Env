"""SPEC.md 87.5/87.6 (v0.73, B2/B3) - the "So what?" card + the
confidence chip, in the reader's words.

The decision artifact (63.3) answers the *analyst*; the briefing answers
the person who asked "did it work?". One home (87.5.3, the 64.2 pattern):
``so_what(summary, entries, seed_spread, diag)`` builds the four plain
lines - did we hit the bar? / how sure are we? / where does it fail? /
what next? - and ``confidence_tier`` / ``plain_verdict`` give the chip
(87.6) and the one-sentence CLI verdict (``go``, 87.3) the same words.

House rules: stdlib only; pure and deterministic (G2); ASCII-safe
strings (the 48.5 console rule); no jargon (87.4) - the reader never
meets a knob's name.
"""
from __future__ import annotations

import math

from . import uncertainty


def _is_num(v) -> bool:
    """A finite int/float that is not a bool (the 62.5 guard, one rule)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def confidence_tier(target, best, seed_spread) -> str | None:
    """SPEC.md 87.6 (B3): the "trust" chip for a headline number.

    The 66.1 robustness read re-skinned to a tier (87.6.1):

    - ``robust_go`` / ``robust_no`` -> ``"high"`` - the whole seed band
      is on one side of the bar;
    - ``marginal_go`` / ``marginal_no`` -> ``"medium"`` - the point
      clears (or misses) but a different seed could flip it;
    - ``unassessable`` (no target, or a single seed) -> ``"low"`` - the
      verdict stands on one point estimate;
    - ``None`` - no finite best score: there is no number to chip.
    """
    if not _is_num(best):
        return None
    robust = uncertainty.verdict_robustness(target, best, seed_spread)
    status = robust["status"] if robust else "unassessable"
    if status in ("robust_go", "robust_no"):
        return "high"
    if status in ("marginal_go", "marginal_no"):
        return "medium"
    return "low"


def when_to_distrust(tier: str | None) -> str:
    """SPEC.md 87.6 (B3): the one-line "when to distrust" for a tier -
    the reader's caution, in words (87.6.2)."""
    if tier == "high":
        return ("trust: high - the verdict holds across the seed band; "
                "the remaining risk is new, unseen data")
    if tier == "medium":
        return ("trust: medium - a different random seed could flip this "
                "verdict; run a seed sweep before you bet on it")
    if tier == "low":
        return ("trust: low - one seed was run; a different one could "
                "change the answer")
    return "trust: none - there is no final score yet"


def so_what(summary: dict, entries=None, seed_spread=None,
            diag=None) -> dict | None:
    """SPEC.md 87.5 (B2): the "So what?" card - four plain lines.

    ``None`` when the summary carries no finite final best score (there
    is nothing to say). Otherwise:

    - ``headline`` - "We hit your bar." / "We missed your bar by X." /
      "Final score X (no bar was set)." (87.5.1);
    - ``met`` / ``margin`` - vs the target (``None`` when no target;
      the target resolves through the 66.1.1 ``summary_target`` home,
      so a run's canonical ``run_config.target`` counts);
    - ``confidence`` - the 87.6 tier + the ``when_to_distrust`` line;
    - ``where_it_fails`` - the weakest holdout class from a 28.2
      ``diag`` ("weakest: class X - 54% correct"), else the honest
      "no per-class data was logged for this run" (87.5.2);
    - ``next_step`` - the plain next move (87.5.3).

    Pure over the summary / diag / spread (G2); ``entries`` is accepted
    for the 87.5.3 call shape but only a precomputed ``diag`` is read -
    the holdout pass stays with the 28.2 owner.
    """
    s = summary if isinstance(summary, dict) else {}
    best = s.get("final_best_score")
    if not _is_num(best):
        return None
    target = s.get("target")
    if target is None:
        target = uncertainty.summary_target(s)
    if target is not None and not _is_num(target):
        target = None

    best_f = float(best)
    met = None if target is None else bool(best_f >= float(target))
    margin = None if target is None else best_f - float(target)
    if met is True:
        headline = (f"We hit your bar - final {best_f:g} vs target "
                    f"{float(target):g}.")
    elif met is False:
        headline = (f"We missed your bar by {abs(margin):g} - "
                    f"final {best_f:g} vs target {float(target):g}.")
    else:
        headline = f"Final score {best_f:g} (no bar was set)."

    tier = confidence_tier(target, best_f, seed_spread)
    if isinstance(diag, dict) and diag.get("per_class") is not None:
        per = diag["per_class"]
        idx = min(range(len(per)), key=lambda i: float(per[i]))
        labels = diag.get("class_labels") or []
        name = str(labels[idx]) if idx < len(labels) else str(idx)
        fails = f"weakest: class {name} - {float(per[idx]):.0f}% correct"
    else:
        fails = "no per-class data was logged for this run"

    if met is True:
        nxt = ("put it to work - export the model and its model card "
               "(the Export tab, or the share bundle)")
    elif met is False:
        nxt = ("give it more to learn from - raise the experiment "
               "budget and run it again, or lower the bar")
    else:
        nxt = "set a target score so the loop knows when it is done"

    return {
        "headline": headline,
        "met": met,
        "margin": margin,
        "target": None if target is None else float(target),
        "confidence": tier,
        "when_to_distrust": when_to_distrust(tier),
        "where_it_fails": fails,
        "next_step": nxt,
    }


def plain_verdict(summary: dict, bar: float | None = None) -> str:
    """SPEC.md 87.3.3 (A3): the one-sentence plain verdict the ``go``
    command prints after the gate (the 48.5 narration style, one line).

    ``bar`` is the acceptance bar that was actually used (wins over the
    summary's target when given). Plain words only (87.4) - asserted
    jargon-free by A77.3."""
    s = summary if isinstance(summary, dict) else {}
    best = s.get("final_best_score")
    if not _is_num(best):
        return "The loop finished without a final score - check the run dir."
    best_f = float(best)
    target = bar if _is_num(bar) else s.get("target")
    if target is None:
        target = uncertainty.summary_target(s)
    if target is not None and _is_num(target):
        t = float(target)
        if best_f >= t:
            return (f"You asked for {t:g}. We got {best_f:g} - that beats "
                    f"the bar, and the model is ready to use.")
        return (f"You asked for {t:g}. We got {best_f:g} - just short of "
                f"the bar; a bigger budget (more experiments) is the "
                f"next step.")
    return f"We finished at {best_f:g} (no bar was set)."


# 87.4.4 (A77): the jargon guard - the fixed reader-facing templates this
# module emits (the dynamic numbers carry no words).
_JARGON_FREE_TEMPLATES: tuple[str, ...] = (
    "trust: high - the verdict holds across the seed band; "
    "the remaining risk is new, unseen data",
    "trust: medium - a different random seed could flip this "
    "verdict; run a seed sweep before you bet on it",
    "trust: low - one seed was run; a different one could "
    "change the answer",
    "trust: none - there is no final score yet",
    "weakest: class X - 54% correct",
    "no per-class data was logged for this run",
    "put it to work - export the model and its model card "
    "(the Export tab, or the share bundle)",
    "give it more to learn from - raise the experiment "
    "budget and run it again, or lower the bar",
    "set a target score so the loop knows when it is done",
    "The loop finished without a final score - check the run dir.",
)


def jargon_free_check() -> list[str]:
    """SPEC.md 87.4.4: the 87.5/87.6 reader-facing templates that still
    carry an 87.4 jargon key (expected ``[]`` - the A77 invariant)."""
    from .vocab import plain_free
    return [t for t in _JARGON_FREE_TEMPLATES if not plain_free(t)]
