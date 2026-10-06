"""SPEC.md 87.8 (v0.73, C2) - the model card one-pager.

The end-user's page: what the model predicts, a few real
input -> output examples, how sure to be, the known limits, and how to
read the number. Built on the 63.2 ``user_guide_view`` (the one holder
of the bounded holdout sample) + the 87.5/87.6 briefing tier, so the
card never re-derives a holdout pass (the 28.2 protocol stays with its
owner). One home (87.8.3): ``model_card`` builds the view dict,
``render_model_card`` / ``render_model_card_md`` render it - the CLI
(``autorefine card``), the app's download button, and the tests all
read the same words (G2: a re-render is byte-identical).

House rules: stdlib only (+ the core's existing readers); pure over the
summary/entries/task/model inputs; episode / synthetic tasks degrade to
the 63.2 explanatory note (never invented rows).
"""
from __future__ import annotations

from . import briefing, reporting

# 87.8.1: the card's example budget - the "a few real examples" the
# one-pager promises (the 63.2 default is 5; the card is tighter).
DEFAULT_EXAMPLES = 3


def model_card(summary: dict, entries=None, task=None, model=None,
               n_examples: int = DEFAULT_EXAMPLES,
               headline=None) -> dict:
    """SPEC.md 87.8 (C2): the model card view.

    Fields (stable order, 87.8.2):

    - ``what_it_predicts`` - the 63.2 block (input / output / metric /
      direction), or ``None`` when no task was supplied;
    - ``examples`` / ``examples_note`` - the first ``n_examples``
      holdout rows (63.2), or the honest note for a task with no split;
    - ``confidence`` / ``when_to_distrust`` / ``known_limits`` - the 63.2
      reads, verbatim (one holder, no drift);
    - ``trust`` - the 87.6 chip tier for the headline number (``None``
      when there is no final score);
    - ``how_to_read`` - the 63.2 "how to read the number" line, with the
      64.2.2 seed-spread read when ``headline`` carries one;
    - ``verdict`` - the 87.5 "So what?" headline (``None`` when the
      summary has no final score).

    Pure (G2); ``entries`` is passed through for the 63.2 shape (the
    card reads only the summary for its own blocks).
    """
    s = summary if isinstance(summary, dict) else {}
    guide = reporting.user_guide_view(
        s, entries, task=task, model=model,
        n_examples=n_examples, headline=headline)
    best = s.get("final_best_score")
    spread = None
    if isinstance(headline, dict) and headline.get("n_runs"):
        # the 64.2.2 read already bounds the band - re-derive the spread
        # population the tier needs (one home, 87.6)
        half = float(headline.get("plus_minus") or 0.0)
        mid = float(headline.get("value") or 0.0)
        n = int(headline.get("n_runs") or 0)
        spread = [mid - half, mid + half] if n >= 2 else None
    sw = briefing.so_what(s, entries, seed_spread=spread)
    return {
        "title": "Model card - what this model does",
        "task": s.get("task"),
        "what_it_predicts": guide["what_it_predicts"],
        "examples": guide["examples"],
        "examples_note": guide["examples_note"],
        "confidence": guide["confidence"],
        "when_to_distrust": guide["when_to_distrust"],
        "known_limits": guide["known_limitations"],
        "trust": sw["confidence"] if sw else None,
        "verdict": sw["headline"] if sw else None,
        "how_to_read": guide["how_to_read"],
    }


def _fmt(v) -> str:
    """A cell value (None -> an em dash, the 62.6 rule)."""
    if v is None:
        return "-"
    return str(v)


def render_model_card(view: dict) -> str:
    """SPEC.md 87.8 (C2): the stable text rendering (G2 - a re-render of
    the same view is byte-identical; the 63.2 ``render_user_guide``
    shape, tighter)."""
    v = view if isinstance(view, dict) else {}
    lines = [f"=== {v.get('title', 'Model card')} ==="]
    pred = v.get("what_it_predicts")
    if isinstance(pred, dict):
        lines.append(f"predicts : {_fmt(pred.get('input'))} -> "
                     f"{_fmt(pred.get('output'))}")
        lines.append(f"metric   : {_fmt(pred.get('metric'))} "
                     f"({_fmt(pred.get('direction'))})")
        lines.append(f"task     : {_fmt(v.get('task'))}")
    else:
        lines.append("predicts : (no task was supplied - a built-in run)")
    examples = v.get("examples")
    if examples:
        lines.append("examples :")
        for e in examples:
            conf = e.get("confidence")
            conf_s = f" (confidence {conf:.2f})" if conf is not None else ""
            mark = "ok" if e.get("correct") else "MISS"
            lines.append(
                f"  - {_fmt(e.get('input'))} -> true {_fmt(e.get('true'))}, "
                f"predicted {_fmt(e.get('predicted'))}{conf_s}  [{mark}]")
    elif v.get("examples_note"):
        lines.append(f"examples : {v['examples_note']}")
    distrust = v.get("when_to_distrust") or []
    lines.append("when to distrust : "
                 + ("; ".join(distrust) if distrust else "(none logged)"))
    limits = v.get("known_limits") or []
    lines.append("known limits   : "
                 + ("; ".join(limits) if limits else "(none logged)"))
    trust = v.get("trust")
    if trust:
        lines.append(f"trust        : {trust} "
                     f"({briefing.when_to_distrust(trust).split(' - ', 1)[1]})")
    if v.get("verdict"):
        lines.append(f"verdict      : {v['verdict']}")
    if v.get("how_to_read"):
        lines.append(f"how to read  : {v['how_to_read']}")
    return "\n".join(lines)


def render_model_card_md(view: dict) -> str:
    """SPEC.md 87.8 (C2): the Markdown rendering (wikis / PRs / Slack -
    the 63.1 ``render_report_md`` sibling; G2: byte-identical)."""
    v = view if isinstance(view, dict) else {}
    out = [f"# {v.get('title', 'Model card')}", ""]
    pred = v.get("what_it_predicts")
    if isinstance(pred, dict):
        out += [
            f"- **Predicts**: {_fmt(pred.get('input'))} -> "
            f"{_fmt(pred.get('output'))}",
            f"- **Metric**: {_fmt(pred.get('metric'))} "
            f"({_fmt(pred.get('direction'))})",
            f"- **Task**: {_fmt(v.get('task'))}",
            f"- **Trust**: {_fmt(v.get('trust'))}",
            f"- **Verdict**: {_fmt(v.get('verdict'))}",
            "",
        ]
    else:
        out += ["(no task was supplied - a built-in run)", ""]
    examples = v.get("examples")
    if examples:
        out += ["## Examples (real holdout rows)", "",
                "| input | true | predicted | confidence | result |",
                "| --- | --- | --- | --- | --- |"]
        for e in examples:
            conf = e.get("confidence")
            out.append(
                f"| {_fmt(e.get('input'))} | {_fmt(e.get('true'))} | "
                f"{_fmt(e.get('predicted'))} | "
                f"{_fmt('%.2f' % conf) if conf is not None else '-'} | "
                f"{'ok' if e.get('correct') else 'MISS'} |")
        out.append("")
    elif v.get("examples_note"):
        out += [f"*examples: {v['examples_note']}*", ""]
    distrust = v.get("when_to_distrust") or []
    if distrust:
        out += ["## When to distrust it"] + [f"- {d}" for d in distrust] + [""]
    limits = v.get("known_limits") or []
    if limits:
        out += ["## Known limits"] + [f"- {l}" for l in limits] + [""]
    if v.get("how_to_read"):
        out += ["## How to read the number", "", v["how_to_read"], ""]
    return "\n".join(out)
