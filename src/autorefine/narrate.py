"""SPEC.md 48 (v0.34): the app's plain-English narration — pure,
stdlib + one core import (``dashboard.field_stats``, the same single source
of truth the D1 view and ``cli._explain_blocks`` use).

- **48.5 narrate the loop** — ``narrate_baseline`` / ``narrate_step``: one
  friendly line per step, from the runner's ``start()`` info and each
  ``next()`` update dict. The terminal form of this is ``simulate.trace_lines``
  (41.2/41.3, used by ``report --trace`` and ``run --demo``); this is the
  beginner-facing app form of the same per-step story.
- **48.4 the "what happened" narrative** — ``narrate_run``: the finished-run
  plain-English summary, derived from the runner's ``finish()`` result (the
  ``res`` dict: headline, what it tried, how it improved, the winning
  recipe). The same accepted-chain walk ``cli._explain_blocks`` uses (42.3).

House rules: pure and deterministic (G2); no ``st.*`` calls (the app is the
only renderer); every line is ASCII so the Windows console (cp1252) is
never asked to print a non-ASCII glyph.
"""
from __future__ import annotations

from autorefine.dashboard import field_stats


def _num(x) -> float | None:
    """A finite real number, or ``None`` — bools are not numbers here, and
    NaN/inf are not renderable."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if v == v and v not in (float("inf"), float("-inf")):
            return v
    return None


def narrate_baseline(info: dict) -> str:
    """48.5.2: the friendly first line, from ``DashboardRunner.start()``'s
    info (baseline_score, target). Pure."""
    base = _num(info.get("baseline_score"))
    tgt = _num(info.get("target"))
    parts = ["Started the search"]
    if base is not None:
        parts.append(f"from a baseline of {base:.2f}")
    if tgt is not None:
        parts.append(f"(you wanted at least {tgt:g})")
    return " ".join(parts) + "."


def narrate_step(u: dict) -> str:
    """48.5.1: one plain-English line for a single experiment step, from the
    runner's ``next()`` update dict (``index``, ``mutation``,
    ``candidate_score``, ``accepted``, ``best_score``). A duplicate (no
    numeric ``candidate_score``) is named as such; an accepted step reports
    the new best; a rejected step reports the miss. Pure and deterministic."""
    n = u.get("index")
    head = f"Experiment {n}" if isinstance(n, int) else "Step"
    mut = ", ".join(u.get("mutation") or [])
    tried = f"tried {mut}" if mut else "tried a small change"
    score = _num(u.get("candidate_score"))
    if score is None:
        return f"{head}: {tried} — a duplicate, so it was skipped (no retrain)."
    if u.get("accepted"):
        best = _num(u.get("best_score"))
        tail = f"new best {best:.2f}" if best is not None else "new best"
        return (f"{head}: {tried} — scored {score:.2f}, better than the best "
                f"so far (kept it, {tail}).")
    return (f"{head}: {tried} — scored {score:.2f}, not better than the best "
            f"so far (skipped it).")


def _accepted_chain(res: dict) -> list[float]:
    """48.4.2: the score chain — the baseline first, then each accepted
    candidate's holdout score, in stream order (the same walk
    ``cli._explain_blocks`` uses for its ``why`` block)."""
    chain: list[float] = []
    base = _num(res.get("baseline_score"))
    if base is not None:
        chain.append(base)
    for u in (res.get("updates") or []):
        if u.get("accepted"):
            s = _num(u.get("candidate_score"))
            if s is not None:
                chain.append(s)
    return chain


def _spec_bits(spec) -> list[str]:
    """48.4.2: the best_spec rendered as ``key=value`` chips (None fields
    dropped, list values joined — the app's ``_spec_v`` convention)."""
    if not isinstance(spec, dict):
        return []
    bits = []
    for k in sorted(spec):
        v = spec[k]
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        bits.append(f"{k}={v}")
    return bits


def narrate_run(res: dict) -> str:
    """48.4.1: the finished-run "what happened" narrative — plain English,
    derived from the runner's ``finish()`` result (48.4.2: reuses
    ``dashboard.field_stats`` for the per-field summary and the accepted
    chain for the improvement story; the winning recipe reads
    ``best_spec``). Pure and deterministic; the app renders it as its
    "What happened" block (48.4.3). Missing pieces degrade gracefully to an
    honest "no ..." line (28.5-style)."""
    tgt = _num(res.get("target"))
    fin = _num(res.get("final_best_score"))
    base = _num(res.get("baseline_score"))
    n_exp = res.get("experiments_run")
    verdict = res.get("verdict", "—")

    # headline (48.4.1)
    detail = []
    if tgt is not None:
        detail.append(f"you wanted at least {tgt:g}")
    if fin is not None:
        detail.append(f"the loop finished at {fin:.2f}")
    if base is not None:
        detail.append(f"up from a {base:.2f} baseline")
    if isinstance(n_exp, int) and not isinstance(n_exp, bool):
        detail.append(f"after {n_exp} experiments")
    head = f"Verdict: {verdict}."
    if detail:
        head += " " + ", ".join(detail) + "."

    # what it tried (48.4.2) — per-field trial/win summary, single source
    fs = field_stats(res.get("updates"))
    if fs:
        top_name, top = sorted(fs.items(), key=lambda kv: (-kv[1]["trials"],
                                                           kv[0]))[0]
        tried = (f"It explored {len(fs)} model knob(s), most often {top_name} "
                 f"({top['trials']}×, improving {top['wins']:.0f}×).")
    else:
        tried = "It did not log any model-knob changes to try."

    # how it improved (48.4.2) — the accepted chain
    chain = _accepted_chain(res)
    if len(chain) >= 2:
        improved = " -> ".join(f"{x:.2f}" for x in chain)
        how = (f"It made {len(chain) - 1} improvement(s), each a new best: "
               f"{improved}.")
    elif fin is not None and base is not None and fin < base:
        how = ("It never beat its starting baseline — every candidate "
               "scored lower, so the baseline is the best model.")
    elif fin is not None:
        how = "It made no accepted improvements over the baseline."
    else:
        how = "It recorded no usable scores."

    # the winning recipe (48.4.2) — best_spec as chips
    bits = _spec_bits(res.get("best_spec"))
    final = ("The winning recipe: " + ", ".join(bits) + ".") if bits \
        else "No final recipe was recorded."

    return "\n\n".join([head, tried, how, final])
