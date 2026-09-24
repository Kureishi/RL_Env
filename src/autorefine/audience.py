"""The audience axis for `report` (SPEC.md 62, v0.48, B1).

Same derivations, different vocabulary: every number an audience view shows
is already in ``summary.json`` / ``experiments.jsonl`` (or a single extra
forward pass over the holdout, the 28.2 protocol) — the only new bit is the
reader-facing *re-skinning* plus ``verify_run``, which independently
re-derives the reported numbers and asserts them.

Four audiences (``AUDIENCES``):

- **exec**      — one screen: what we built · target met? by how much ·
  cost · risk · go/no-go. No hyperparameters, no mutation SVGs.
- **domain**    — the clinician/analyst's words: per-class performance,
  the weakest class, calibration (ECE), and worked examples (the re-framed
  error gallery / confusion / difficulty data).
- **technical** — today's report (already done); the CLI's default path is
  byte-identical and does NOT route through this module.
- **regulator** — chain of custody: config hash, seed, environment facts,
  a data fingerprint, and the decision trace.

House rules: stdlib + numpy (a core dependency) + the existing sibling leaf
modules (``memory`` / ``calibration`` / ``diagnostics`` / ``provenance`` /
``simulate`` / ``accounting`` — none of which import ``cli`` or
``dashboard_app``), so there is no import cycle and no new dependency (23.1
streamlit-free). Every renderer is a pure, deterministic function of its
inputs (G2): a re-render of the same view is byte-identical; no timestamps
and no RNG. ``verify_run`` is pure over the run dir's two artifacts.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from .memory import (  # 35.1 (C4): the kind registry — the value source
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
    LOG_KINDS,
)
from .uncertainty import (  # 64.2 (B6) + 66 (v0.52, B7): one home for the reads
    headline_uncertainty,
    summary_target,
    verdict_robustness,
)


def _uncertainty_block(summary: dict, seed_spread) -> dict | None:
    """64.2.2 (B6): the audience views' shared uncertainty block — the
    ``uncertainty.headline_uncertainty`` read of the summary's
    ``final_best_score`` against the same-task seed spread (``None`` →
    rendered as ``—``; the A52 pins keep asserting the pre-v0.50 views).
    One home so the four views cannot drift."""
    s = summary if isinstance(summary, dict) else {}
    return headline_uncertainty(s.get("final_best_score"), seed_spread)


def _robustness_block(summary: dict, seed_spread) -> dict | None:
    """66.2 (A56): the exec view's robustness read — the
    ``uncertainty.verdict_robustness`` classification of the summary's
    go/no-go verdict (its target via ``uncertainty.summary_target``, 66.1.1;
    its ``final_best_score``) against the same-task seed spread (``None`` →
    rendered as ``—``, the 64.2.3 rule). One home so the two decision
    surfaces (the exec view and the decision artifact) cannot drift."""
    s = summary if isinstance(summary, dict) else {}
    return verdict_robustness(summary_target(s), s.get("final_best_score"),
                              seed_spread)


# SPEC.md 62.1: the four reader audiences, in the documented order.
AUDIENCES = ("exec", "domain", "technical", "regulator")

# 62.5: the finite-number guard (rows may carry non-numeric / bool values).
_REL_TOL = 1e-9


def _num(value) -> float | None:
    """A finite number, else ``None`` (mirrors ``accounting._num``)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


def _fmt_scalar(v) -> str:
    """The human rendering of a scalar value (None → an em-dash)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return f"{v:g}"
        return str(v)
    return str(v)


def _render_dict(d: dict, lines: list, indent: int = 0) -> None:
    """A stable, nested text rendering of a view dict (62.6).

    Dict values recurse; lists render as ``- `` bullets (dict items collapse
    to one ``k=v, k=v`` line); scalars render inline. Pure and deterministic.
    """
    pad = "  " * indent
    for key, value in (d or {}).items():
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{key}:")
                _render_dict(value, lines, indent + 1)
            else:
                lines.append(f"{pad}{key}: (none)")
        elif isinstance(value, (list, tuple)):
            if not value:
                lines.append(f"{pad}{key}: (none)")
            else:
                lines.append(f"{pad}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        parts = ", ".join(f"{k}={_fmt_scalar(v)}"
                                          for k, v in item.items())
                        lines.append(f"{pad}  - {parts}")
                    else:
                        lines.append(f"{pad}  - {_fmt_scalar(item)}")
        else:
            lines.append(f"{pad}{key}: {_fmt_scalar(value)}")


# --- 62.2 exec ---------------------------------------------------------------

def _spec_in_words(spec) -> str:
    """62.2 (A52): the best spec in plain English — family + shape, no
    hyperparameters (the exec view deliberately hides the knob values)."""
    spec = spec if isinstance(spec, dict) else {}
    family = spec.get("model_family", "mlp")
    if family == "knn":
        return f"a k-nearest-neighbours model (k={spec.get('knn_k')})"
    if family == "tree":
        return "a random forest of small decision trees"
    if family == "boost":
        return "a gradient-boosted tree ensemble"
    if family == "convnet":
        return "a small convolutional network"
    if family == "gp":  # SPEC.md 75 (v0.61)
        return ("a gaussian-process model "
                "(random-feature kernel approximation)")
    if family == "gam":  # SPEC.md 75 (v0.61)
        return ("a generalized additive model "
                "(interpretable per-feature curves)")
    arch = spec.get("architecture") or []
    act = spec.get("activation", "tanh")
    if arch:
        shape = " x ".join(str(a) for a in arch)
        return (f"a {len(arch)}-layer neural net ({shape} hidden units, "
                f"{act} activation)")
    return f"a small neural net ({act} activation)"


def _exec_risk(summary: dict, entries, diag=None) -> list:
    """62.2 (A52): the exec risk lines — derived from the logged rejections
    (``accounting.account_run``) + the weakest class (the 28.2 diagnostics).
    Pure over already-loaded data; degrades to a single all-clear line."""
    notes: list[str] = []
    try:
        from .accounting import account_run  # local: keep the top imports light
        acc = account_run(summary, entries or [])
        rej = acc.get("rejections", {})
        if rej.get("overfit"):
            notes.append(f"{rej['overfit']} candidate(s) rejected on the "
                         f"overfit (gen-gap) gate")
        if rej.get("ci"):
            notes.append(f"{rej['ci']} candidate(s) rejected on the CI gate")
    except Exception:
        pass
    if isinstance(diag, dict) and diag.get("class_labels"):
        per = diag["per_class"]
        worst = min(range(len(per)), key=lambda i: per[i])
        if per[worst] < 100.0:
            notes.append(f"weakest class '{diag['class_labels'][worst]}' "
                         f"at {per[worst]:.1f}% holdout accuracy")
    if not notes:
        notes.append("no red flags in the logged history")
    return notes


def _exec_go_no_go(target, best, baseline) -> dict:
    """62.2 (A52): the go/no-go verdict — a fixed, hand-computable rule.

    - no target set            → REVIEW (an exploratory run; nothing to gate on)
    - no final score           → NO-GO
    - final >= target          → GO
    - final < target, improved → REVIEW (progress; needs more budget)
    - final < target, not      → NO-GO
      improved on baseline
    """
    best = _num(best)
    baseline = _num(baseline)
    tgt = _num(target)
    if tgt is None:
        return {"verdict": "REVIEW",
                "rationale": "no acceptance target was set (exploratory run)"}
    if best is None:
        return {"verdict": "NO-GO", "rationale": "no final score to gate on"}
    if best >= tgt:
        return {"verdict": "GO",
                "rationale": f"final score {best:g} meets the target {tgt:g}"}
    if baseline is not None and best > baseline:
        return {"verdict": "REVIEW",
                "rationale": (f"improved ({baseline:g} -> {best:g}) but below "
                              f"the target {tgt:g}; more budget may close it")}
    return {"verdict": "NO-GO",
            "rationale": (f"final score {best:g} is below the target {tgt:g} "
                          f"and did not clear the baseline")}


def exec_view(summary, entries, diag=None, seed_spread=None) -> dict:
    """62.2 (A52): the exec view — one screen (what we built · target met? by
    how much · cost · risk · go/no-go). Pure over the loaded summary/entries;
    ``diag`` (the 28.2 diagnostics, when the task is classification) feeds the
    risk block. No hyperparameters and no mutation data appear.
    ``seed_spread`` (64.2.2, B6) attaches the "± half-spread (N runs)" read
    to the headline score; absent → the block renders as an em-dash.
    ``robustness`` (66.2, A56) answers the trust question — does the
    verdict survive the seed band — from the same spread; ``None`` (no
    final score) renders as an em-dash."""
    s = summary if isinstance(summary, dict) else {}
    best = s.get("final_best_score")
    baseline = s.get("baseline_score")
    target = s.get("target")
    spec = s.get("best_spec") or {}
    task = s.get("task", "a task")
    what = _spec_in_words(spec)
    what = (f"A {task} model: {what}; final score {_fmt_scalar(best)}."
            if best is not None else f"A {task} model: {what}.")
    if target is None:
        target_met = {"target": None, "met": None, "margin": None,
                      "note": "exploratory — no acceptance target was set"}
    else:
        met = best is not None and float(best) >= float(target)
        margin = (float(best) - float(target)) if best is not None else None
        target_met = {"target": _num(target), "met": bool(met),
                      "margin": margin}
    return {
        "what_we_built": what,
        "target_met": target_met,
        # 64.2.2 (B6): how sure to be — the seed-spread read on the headline
        "uncertainty": _uncertainty_block(s, seed_spread),
        # 66.2 (A56): is the verdict robust to the seed band (None → —)
        "robustness": _robustness_block(s, seed_spread),
        "cost": {"wall_seconds": s.get("wall_seconds"),
                 "experiments_run": s.get("experiments_run"),
                 "finished_reason": s.get("finished_reason")},
        "risk": _exec_risk(s, entries, diag),
        "go_no_go": _exec_go_no_go(target, best, baseline),
    }


# --- 62.3 domain -------------------------------------------------------------

def domain_view(summary, entries, diag=None, extras=None,
                seed_spread=None) -> dict:
    """62.3 (A52): the domain view — per-class performance, the weakest
    class, calibration (ECE), and worked examples, in plain analyst words.

    ``diag`` is the 28.2 ``holdout_diagnostics`` dict (per_class /
    class_labels / confusion / class_counts); ``extras`` may carry ``ece``
    (``calibration.ece`` over the holdout softmax) and ``gallery`` (the 28.3
    error items). Regression / non-classification runs degrade gracefully:
    the absent fields read ``None`` (rendered as an em-dash / ``(none)``).
    """
    s = summary if isinstance(summary, dict) else {}
    extras = extras if isinstance(extras, dict) else {}
    if not isinstance(diag, dict):
        diag = extras.get("diagnostics")
    view = {
        "task": s.get("task"),
        "final_score": s.get("final_best_score"),
        # 64.2.2 (B6): the ± read on the headline number (None → "—")
        "uncertainty": _uncertainty_block(s, seed_spread),
        "per_class": None,
        "weakest_class": None,
        "calibration": None,
        "worked_examples": None,
    }
    if isinstance(diag, dict) and diag.get("class_labels"):
        rows = []
        per = diag["per_class"]
        for i, lbl in enumerate(diag["class_labels"]):
            rows.append({"class": lbl,
                         "accuracy_pct": round(float(per[i]), 1),
                         "count": diag["class_counts"][i],
                         "correct": diag["confusion"][i][i]})
        view["per_class"] = rows
        worst = min(range(len(per)), key=lambda i: per[i])
        view["weakest_class"] = {
            "class": diag["class_labels"][worst],
            "accuracy_pct": round(float(per[worst]), 1),
        }
    ece = extras.get("ece")
    if ece is not None:
        view["calibration"] = {
            "ece": round(float(ece), 4),
            "note": "expected calibration error (0 = perfectly calibrated)",
        }
    gallery = extras.get("gallery")
    if gallery:
        view["worked_examples"] = [
            {"item": it.get("file", "?"),
             "true": it.get("label"),
             "predicted": it.get("predicted")}
            for it in list(gallery)[:10]
        ]
    return view


# --- 62.4 regulator ----------------------------------------------------------

def regulator_view(summary, entries, provenance=None, trace=None,
                   data=None, seed_spread=None) -> dict:
    """62.4 (A52): the regulator / auditor view — chain of custody.

    ``provenance`` is the 56.1 certificate payload (tool / version / seed /
    target / config_hash / environment facts); ``data`` is the
    ``data_fingerprint`` of the run's dataset; ``trace`` is the 41.2 decision
    trace. All fields render even when a source is absent (they read
    ``None``), so a partial run still produces a legible audit block.
    """
    s = summary if isinstance(summary, dict) else {}
    prov = provenance if isinstance(provenance, dict) else {}
    return {
        "tool": prov.get("tool", "autorefine"),
        "version": prov.get("version"),
        "task": s.get("task"),
        "seed": s.get("seed"),
        "target": prov.get("target"),
        "config_hash": prov.get("config_hash"),
        "environment": prov,
        "data_fingerprint": data,
        "chain_of_custody": {
            "experiments_run": s.get("experiments_run"),
            "final_best_score": s.get("final_best_score"),
            "finished_reason": s.get("finished_reason"),
        },
        # 64.2.2 (B6): the ± read on the headline number (None → "—")
        "uncertainty": _uncertainty_block(s, seed_spread),
        "decision_trace": list(trace or []),
    }


def technical_view(summary, entries, seed_spread=None) -> dict:
    """62.1 (A52): the technical view — a structured mirror of today's
    report (the summary's headline keys + the per-entry decision log). The
    CLI's *default* path does not render through this; it exists so the
    audience axis is complete and testable in isolation. ``seed_spread``
    (64.2.2, B6) adds the "± half-spread (N runs)" read on the headline
    score; absent → ``uncertainty`` renders as ``—``."""
    s = summary if isinstance(summary, dict) else {}
    keys = ("task", "seed", "finished_reason", "baseline_score",
            "final_best_score", "improvement_factor", "experiments_run",
            "wall_seconds", "best_spec")
    view = {k: s.get(k) for k in keys}
    view["uncertainty"] = _uncertainty_block(s, seed_spread)  # 64.2.2 (B6)
    view["experiments"] = [
        {"kind": e.get("kind"), "accepted": e.get("accepted"),
         "score": e.get("holdout_score"), "gen_gap": e.get("gen_gap"),
         "mutation": e.get("mutation") or []}
        for e in (entries or []) if isinstance(e, dict)
    ]
    return view


def build_view(audience: str, summary, entries, diag=None, extras=None,
               provenance=None, trace=None, data=None,
               seed_spread=None) -> dict:
    """62.1 (A52): the audience dispatcher — the right view for the requested
    audience. ``technical`` is the documented default; an unknown audience
    raises ``ValueError`` (the CLI's ``--audience`` ``choices`` guard this).
    ``seed_spread`` (64.2.2, B6) forwards the same-task seed spread to every
    view's ``uncertainty`` block; absent → every block renders as ``—``
    (the A52 pins keep asserting the pre-v0.50 views)."""
    if audience == "exec":
        return exec_view(summary, entries, diag=diag, seed_spread=seed_spread)
    if audience == "domain":
        return domain_view(summary, entries, diag=diag, extras=extras,
                           seed_spread=seed_spread)
    if audience == "technical":
        return technical_view(summary, entries, seed_spread=seed_spread)
    if audience == "regulator":
        return regulator_view(summary, entries, provenance=provenance,
                              trace=trace, data=data,
                              seed_spread=seed_spread)
    raise ValueError(f"unknown audience {audience!r}; expected one of "
                     f"{AUDIENCES}")


# --- rendering ---------------------------------------------------------------

def render_view(view, audience: str | None = None) -> str:
    """62.6 (A52): the stable text rendering of a view dict (G2 — a
    re-render of the same view is byte-identical). With an ``audience`` the
    block is headed by it."""
    lines: list[str] = []
    if audience:
        lines.append(f"=== AutoRefine report ({audience} view) ===")
    _render_dict(view if isinstance(view, dict) else {}, lines)
    return "\n".join(lines)


def html_view(audience: str, view, title: str | None = None) -> str:
    """62.6 (A52): a self-contained HTML rendering of a view (for the
    optional ``--html`` → ``report_<audience>.html``). Deterministic (G2);
    the view text is HTML-escaped inside a ``<pre>`` block."""
    title = title or f"AutoRefine report — {audience}"
    body = "<pre>" + html.escape(render_view(view, audience)) + "</pre>"
    return ("<!DOCTYPE html>\n"
            "<html><head><meta charset=\"utf-8\"/>"
            f"<title>{html.escape(title)}</title></head>\n"
            "<body>\n"
            f"<h1>{html.escape(title)}</h1>\n"
            f"{body}\n"
            "</body></html>\n")


# --- 62.4 data fingerprint ---------------------------------------------------

def data_fingerprint(task_config) -> dict | None:
    """62.4 (A52): a bounded, pure fingerprint of the run's dataset.

    A single **file** → its content ``sha256`` (``kind='file'``). A
    **directory** (image / audio tasks) → a manifest ``sha256`` over the
    sorted ``(relpath, size)`` pairs — no byte reads, so a large corpus never
    blocks (``kind='dir'``). Degrades to ``None`` on a missing / unreadable
    path or a non-dict config (pure and bounded, G2).
    """
    if not isinstance(task_config, dict):
        return None
    path = task_config.get("path")
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    try:
        if p.is_file():
            return {"kind": "file", "path": p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "bytes": int(p.stat().st_size)}
        if p.is_dir():
            manifest = [(f.relative_to(p).as_posix(), int(f.stat().st_size))
                        for f in sorted(p.rglob("*")) if f.is_file()]
            blob = json.dumps(manifest, sort_keys=True,
                              separators=(",", ":"))
            return {"kind": "dir", "path": p.name, "files": len(manifest),
                    "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}
    except (OSError, ValueError):
        return None
    return None


# --- 62.5 verify -------------------------------------------------------------

def _check(name: str, expected, actual, exact: bool = False) -> dict:
    """62.5 (A52): one re-derivation check. Numeric values compare with a
    relative tolerance (the summary rounds); ``exact`` compares for identity
    (integer counts / dict equality). Both sides are always reported."""
    if exact:
        ok = (expected == actual)
    else:
        e, a = _num(expected), _num(actual)
        if e is None or a is None:
            ok = (expected == actual)
        else:
            ok = abs(e - a) <= _REL_TOL * max(1.0, abs(e))
    return {"name": name, "ok": bool(ok), "expected": expected,
            "actual": actual}


def _first_baseline_score(entries) -> float | None:
    for e in entries:
        if isinstance(e, dict) and e.get("kind") == KIND_BASELINE:
            return _num(e.get("holdout_score"))
    return None


def _reconstructed_best(entries) -> float | None:
    """62.5 (A52): the sequential running-best reconstruction — the baseline
    seeds it, an accepted experiment raises it if higher, a curriculum step-up
    re-pins it to ``new_baseline_score`` (the 39.2.2 rule that
    ``accounting._rejections`` and ``simulate.trace_lines`` both use)."""
    best: float | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind == KIND_BASELINE:
            s = _num(e.get("holdout_score"))
            if s is not None:
                best = s
        elif kind == KIND_CURRICULUM:
            nb = _num(e.get("new_baseline_score"))
            if nb is not None:
                best = nb
        elif kind == KIND_EXPERIMENT and e.get("accepted"):
            s = _num(e.get("holdout_score"))
            if s is not None and (best is None or s > best):
                best = s
    return best


def _derived_win_rate(entries) -> dict:
    """62.5 (A52): the per-field mutation win-rate, re-derived exactly as
    ``improver.meta_env._write_artifacts`` builds it (trials = # experiment
    rows mutating a field; wins = # of those accepted)."""
    win_rate: dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("kind") != KIND_EXPERIMENT:
            continue
        for fld in e.get("mutation") or []:
            slot = win_rate.setdefault(fld, {"trials": 0, "wins": 0})
            slot["trials"] += 1
            if e.get("accepted"):
                slot["wins"] += 1
    return win_rate


def verify_run(run_dir) -> dict:
    """62.5 (A52): ``autorefine verify --run DIR`` — independently re-derive
    every reported number from ``experiments.jsonl`` + ``summary.json`` and
    assert them.

    Each check is ``{name, ok, expected, actual}`` and is **only included
    when both sides are present** (we never assert an unreported number).
    Checks: ``experiments_run`` (count of experiment rows), ``baseline_score``
    (the first baseline row), ``final_best_score`` (the running-best
    reconstruction), ``improvement_factor`` (final/baseline, when baseline>0),
    ``mutation_win_rate`` (per-field trials/wins, exact), ``kinds_valid``
    (every row's kind is in ``LOG_KINDS``), and ``wall_seconds_consistency``
    (sum of per-row train_seconds <= the reported wall_seconds).

    Returns ``{"passed": bool, "checks": [...], "error": None}``, or
    ``{"error": str}`` (with ``passed: False``) when a required artifact is
    missing. Pure over the two artifacts; no training, no side effects (G2).
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    exp_path = run_dir / "experiments.jsonl"
    if not summary_path.is_file():
        return {"passed": False, "checks": [],
                "error": "no summary.json in run dir"}
    if not exp_path.is_file():
        return {"passed": False, "checks": [],
                "error": "no experiments.jsonl in run dir"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entries = [json.loads(line)
                   for line in exp_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "checks": [],
                "error": f"unreadable run artifacts: {exc}"}

    checks: list[dict] = []
    baseline = _first_baseline_score(entries)
    final = _reconstructed_best(entries)

    if "experiments_run" in summary:
        exp_count = sum(1 for e in entries
                        if isinstance(e, dict) and e.get("kind") == KIND_EXPERIMENT)
        checks.append(_check("experiments_run", summary["experiments_run"],
                             exp_count, exact=True))
    if baseline is not None and "baseline_score" in summary:
        checks.append(_check("baseline_score", summary["baseline_score"],
                             baseline))
    if final is not None and "final_best_score" in summary:
        checks.append(_check("final_best_score", summary["final_best_score"],
                             final))
    if (summary.get("improvement_factor") is not None
            and baseline is not None and baseline > 0
            and final is not None):
        checks.append(_check("improvement_factor",
                             summary["improvement_factor"], final / baseline))
    if "mutation_win_rate" in summary:
        checks.append(_check("mutation_win_rate", summary["mutation_win_rate"],
                             _derived_win_rate(entries), exact=True))
    kinds_ok = all(isinstance(e, dict) and e.get("kind") in LOG_KINDS
                   for e in entries)
    checks.append({"name": "kinds_valid", "ok": bool(kinds_ok),
                   "expected": "all kinds in LOG_KINDS",
                   "actual": sorted({e.get("kind") for e in entries
                                     if isinstance(e, dict)})})
    if "wall_seconds" in summary:
        train = sum(_num(e.get("train_seconds")) or 0.0
                    for e in entries if isinstance(e, dict)
                    and e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT))
        wall = _num(summary["wall_seconds"]) or 0.0
        checks.append({"name": "wall_seconds_consistency",
                       "ok": bool(train <= wall + 1e-6),
                       "expected": f"sum(train_seconds) <= {wall:g}",
                       "actual": f"{train:.6f}"})

    return {"passed": all(c["ok"] for c in checks), "checks": checks,
            "error": None}


def render_verify(result, run_dir) -> str:
    """62.5 (A52): the human rendering of a ``verify_run`` result — a PASS /
    FAIL headline plus one line per check (an error short-circuits)."""
    if result.get("error"):
        return f"verify {run_dir}: ERROR — {result['error']}"
    headline = "PASS" if result["passed"] else "FAIL"
    lines = [f"verify {run_dir}: {headline} "
             f"({sum(c['ok'] for c in result['checks'])}/{len(result['checks'])} "
             f"checks passed)"]
    for c in result["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        if c["expected"] != c["actual"]:
            lines.append(f"  [{mark}] {c['name']}: expected {c['expected']!r} "
                         f"got {c['actual']!r}")
        else:
            lines.append(f"  [{mark}] {c['name']}: {c['actual']!r}")
    return "\n".join(lines)


__all__ = [
    "AUDIENCES",
    "exec_view",
    "domain_view",
    "technical_view",
    "regulator_view",
    "build_view",
    "render_view",
    "html_view",
    "data_fingerprint",
    "verify_run",
    "render_verify",
]
