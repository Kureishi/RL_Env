"""SPEC.md 63 (v0.49, B2/B3/B4): reporting for a wider range of audiences.

The v0.48 round (SPEC.md 62) added the *reader axis* (exec / domain /
technical / regulator) and ``verify``. This round adds three more
audiences of the **same already-logged data**, each as a pure view +
renderer in this leaf module, plus three output *formats*:

- **B2 — format breadth** (``--format md|txt|pdf``): the technical report
  re-rendered as Markdown, plain text, or a self-contained PDF
  (``reportlab`` is a lazy optional dependency; the core stays
  stdlib+numpy). All three render one shared ``doc`` model, so a
  re-render of the same doc is byte-identical (G2).
- **B3 — the end-user guide** (``--user``): "what this model does" for the
  person who *consumes* the predictions, not the one who trained it —
  what it predicts, a few real input->output examples, how confident it
  is and when to distrust it, known limitations, and how to read the
  number. Episode / non-row tasks (parity / sine / cartpole / gridnav)
  have no per-row holdout split and degrade to a note rather than invent
  examples.
- **B4 — the decision artifact** (``--decision``): a go/no-go verdict
  (reusing the 62.2.2 rule, one home), the target margin, a confidence
  read (seed variance + the logged CI / overfit rejections), the top-3
  failure modes, and one concrete next step (ship / more budget /
  relax-or-expand / collect history) with targeted hints.

House rules: stdlib + numpy + the existing sibling leaves
(``audience`` / ``accounting`` / ``dashboard.field_stats`` /
``tasks.media.class_label_str``) — none of which import ``cli`` or
``dashboard_app`` (23.1 streamlit-free core, no import cycle). Every view
and renderer is a **pure, deterministic** function of its inputs (G2): a
re-render of the same view / doc is byte-identical; no timestamps, no RNG.
``reportlab`` is imported only inside ``render_report_pdf`` (a clean
``ImportError`` message if it is absent), so the module itself loads with
just stdlib + numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .accounting import account_run          # 39.2 (T4): the rejection buckets
from .audience import (  # 62 (v0.48, B1): one home for the 62.2.2 rule + render
    _exec_go_no_go,
    _fmt_scalar,
    _render_dict,
)
from .dashboard import field_stats           # 26.1 (D1): the per-field win rollup
from .tasks.media import class_label_str     # 28.3: the class display string
from .uncertainty import headline_uncertainty  # 64.2 (B6): one home for the ± read

# 63.1 (B2): the three output formats, in documented order (the CLI's
# ``--format`` ``choices`` share this tuple — one source).
REPORT_FORMATS = ("md", "txt", "pdf")

# 63.1.3 (B2): the reportlab trailer ``/ID [<hex32><hex32>]`` — matched so
# the (otherwise random per build) ID can be replaced with a
# doc-deterministic one, making a re-render byte-identical (G2 pin).
_PDF_ID_RE = re.compile(
    rb"/ID\s*\[\s*<[0-9a-fA-F]{32}>\s*<[0-9a-fA-F]{32}>\s*\]")


# --- shared pure helpers ------------------------------------------------------

def _num(value) -> float | None:
    """A finite int/float (not a bool), else ``None`` (mirrors
    ``audience._num``); rows may carry non-numeric values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


def _softmax(z) -> np.ndarray:
    """A numerically-stable row softmax (a 5-line math op, kept local so
    this leaf never imports ``cli`` / a private calibration helper)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _escape(s) -> str:
    """HTML-escape for the reportlab ``<font>`` / ``<pre>`` wrappers."""
    return (str(s).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _md_cell(v) -> str:
    """A Markdown table cell (None -> an em-dash; a list -> ``a, b``)."""
    if v is None:
        return "\u2014"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _txt_cell(v) -> str:
    """A plain-text value (None -> ``n/a``; a list -> ``a, b``)."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _num_fmt(v) -> str:
    """A numeric cell for the plain-text log (None / non-numeric ->
    ``n/a``; a number -> two decimals)."""
    n = _num(v)
    if n is None:
        return "n/a"
    return f"{n:.2f}"


# --- B2: the shared doc model + the three renderers ---------------------------

def build_report_doc(summary, entries) -> dict:
    """63.1 (B2): the single technical-report document model — the source
    for all three formats. A pure function of the already-loaded
    ``summary`` + ``entries`` (no training, no reads).

    Keys: ``title``; ``headline`` (the run's key/value pairs, in the
    documented order); ``win_rate`` (the 26.1 per-field trials/wins/win_rate
    rollup, one home via ``dashboard.field_stats``); ``log`` (one row per
    ``experiments.jsonl`` entry: kind / accepted / score / gen_gap /
    mutation); ``best_spec``.
    """
    s = summary if isinstance(summary, dict) else {}
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    headline = [
        ("task", s.get("task")),
        ("seed", s.get("seed")),
        ("finished_reason", s.get("finished_reason")),
        ("baseline_score", s.get("baseline_score")),
        ("final_best_score", s.get("final_best_score")),
        ("improvement_factor", s.get("improvement_factor")),
        ("experiments_run", s.get("experiments_run")),
        ("wall_seconds", s.get("wall_seconds")),
    ]
    win_rate = [
        {"field": f, "trials": int(stats["trials"]),
         "wins": int(stats["wins"]),
         "win_rate": float(stats["win_rate"])}
        for f, stats in field_stats(entries).items()
    ]
    log = [
        {"kind": e.get("kind"), "accepted": e.get("accepted"),
         "score": e.get("holdout_score"), "gen_gap": e.get("gen_gap"),
         "mutation": list(e.get("mutation") or [])}
        for e in entries
    ]
    title = (f"AutoRefine report \u2014 {s.get('task', '?')} "
             f"(seed {s.get('seed', '?')})")
    return {
        "title": title,
        "headline": headline,
        "win_rate": win_rate,
        "log": log,
        "best_spec": s.get("best_spec"),
    }


def render_report_md(doc) -> str:
    """63.1 (B2): the Markdown rendering — ``#`` headings, ``|`` tables,
    and a fenced ``json`` block for the best spec. Pure and byte-stable
    (G2): a re-render of the same doc is identical."""
    d = doc if isinstance(doc, dict) else {}
    out = [f"# {d.get('title', 'AutoRefine report')}", ""]

    out += ["## Run", ""]
    pairs = list(d.get("headline") or [])
    if pairs:
        out += ["| key | value |", "|-----|-------|"]
        out += [f"| {k} | {_md_cell(v)} |" for k, v in pairs]
    else:
        out.append("(no run metadata)")
    out.append("")

    out += ["## Win rate (per-field)", ""]
    wr = list(d.get("win_rate") or [])
    if wr:
        out += ["| field | trials | wins | win rate |",
                "|-------|--------|------|----------|"]
        out += [f"| {r['field']} | {r['trials']} | {r['wins']} | "
                f"{100.0 * r['win_rate']:.0f}% |" for r in wr]
    else:
        out.append("(no logged mutations)")
    out.append("")

    out += ["## Experiment log", ""]
    log = list(d.get("log") or [])
    if log:
        out += ["| kind | accepted | score | gen_gap | mutation |",
                "|------|----------|-------|---------|----------|"]
        out += [f"| {r.get('kind', '?')} | {_md_cell(r.get('accepted'))} | "
                f"{_md_cell(r.get('score'))} | {_md_cell(r.get('gen_gap'))} | "
                f"{_md_cell(r.get('mutation'))} |" for r in log]
    else:
        out.append("(no log entries)")
    out.append("")

    out += ["## Best spec", ""]
    bs = d.get("best_spec")
    if bs is not None:
        out.append("```json")
        out.append(json.dumps(bs, indent=2, sort_keys=True))
        out.append("```")
    else:
        out.append("(none)")
    out.append("")
    return "\n".join(out)


def render_report_txt(doc) -> str:
    """63.1 (B2): the plain-text one-pager — aligned columns, **no**
    ``|`` table pipes and **no** code fences (email / terminal friendly).
    Pure and byte-stable (G2)."""
    d = doc if isinstance(doc, dict) else {}
    title = str(d.get("title", "AutoRefine report"))
    out = [title, "=" * max(3, len(title)), ""]

    pairs = list(d.get("headline") or [])
    if pairs:
        key_w = max(len(str(k)) for k, _ in pairs)
        out += [f"  {str(k).ljust(key_w)}  {_txt_cell(v)}" for k, v in pairs]
        out.append("")

    out.append("win rate (per-field)")
    wr = list(d.get("win_rate") or [])
    if wr:
        out.append(f"  {'field':<24s} {'trials':>6s} {'wins':>5s} {'win%':>5s}")
        out += [f"  {r['field']:<24s} {r['trials']:>6d} {r['wins']:>5d} "
                f"{100.0 * r['win_rate']:>4.0f}%" for r in wr]
    else:
        out.append("  (no logged mutations)")
    out.append("")

    out.append("experiment log")
    log = list(d.get("log") or [])
    if log:
        out.append(f"  {'kind':<12s} {'accepted':<9s} {'score':>8s} "
                   f"{'gen_gap':>9s}  mutation")
        out += [f"  {str(r.get('kind', '?')):<12s} "
                f"{str(r.get('accepted')):<9s} {_num_fmt(r.get('score')):>8s} "
                f"{_num_fmt(r.get('gen_gap')):>9s}  "
                f"{','.join(r.get('mutation') or []) or '-'}" for r in log]
    else:
        out.append("  (no log entries)")
    out.append("")

    out.append("best spec")
    bs = d.get("best_spec")
    if bs is not None:
        out += ["  " + ln for ln in
                json.dumps(bs, indent=2, sort_keys=True).splitlines()]
    else:
        out.append("  (none)")
    out.append("")
    return "\n".join(out)


def render_report_pdf(doc, out) -> Path:
    """63.1 (B2): a self-contained PDF rendering (``reportlab`` platypus).

    ``reportlab`` is imported **lazily** here (the core stays stdlib+numpy;
    a clean ``ImportError`` message points at ``pip install reportlab``).
    Determinism (G2): the build runs with ``invariant=1`` (suppresses the
    random creation/modification dates) and the trailer ``/ID`` — which
    reportlab still randomizes per build — is replaced with a
    doc-deterministic ``sha256`` of the canonical doc, so a re-render of
    the same doc is **byte-identical**. Returns the written path.
    """
    out = Path(out)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfgen.canvas import Canvas as _rl_canvas
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reportlab is not installed \u2014 the PDF format is an optional "
            "extra (pip install reportlab)") from exc

    class _DetCanvas(_rl_canvas):
        def __init__(self, *a, **k):
            k["invariant"] = 1
            super().__init__(*a, **k)

    d = doc if isinstance(doc, dict) else {}
    styles = getSampleStyleSheet()
    grid = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]

    story = [Paragraph(_escape(d.get("title", "AutoRefine report")),
                       styles["Title"]), Spacer(1, 14)]

    pairs = list(d.get("headline") or [])
    if pairs:
        story.append(Paragraph("Run", styles["Heading2"]))
        story.append(Table([[k, _txt_cell(v)] for k, v in pairs],
                           colWidths=[150, 300]), )
        story[-1].setStyle(TableStyle(grid))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Win rate (per-field)", styles["Heading2"]))
    wr = list(d.get("win_rate") or [])
    if wr:
        data = [["field", "trials", "wins", "win rate"]]
        data += [[r["field"], str(r["trials"]), str(r["wins"]),
                  f"{100.0 * r['win_rate']:.0f}%"] for r in wr]
        t = Table(data, colWidths=[150, 60, 60, 70])
        t.setStyle(TableStyle(grid))
        story.append(t)
    else:
        story.append(Paragraph("(no logged mutations)", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Experiment log", styles["Heading2"]))
    log = list(d.get("log") or [])
    if log:
        data = [["kind", "accepted", "score", "gen_gap", "mutation"]]
        data += [[str(r.get("kind", "?")), _txt_cell(r.get("accepted")),
                  _num_fmt(r.get("score")), _num_fmt(r.get("gen_gap")),
                  ", ".join(r.get("mutation") or []) or "-"] for r in log]
        t = Table(data, colWidths=[70, 70, 60, 70, 150])
        t.setStyle(TableStyle(grid))
        story.append(t)
    else:
        story.append(Paragraph("(no log entries)", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Best spec", styles["Heading2"]))
    bs = d.get("best_spec")
    if bs is not None:
        spec_html = "<br/>".join(_escape(ln) for ln in
                                 json.dumps(bs, indent=2,
                                            sort_keys=True).splitlines())
        story.append(Paragraph(f'<font face="Courier" size="8">{spec_html}'
                               f"</font>", styles["BodyText"]))
    else:
        story.append(Paragraph("(none)", styles["BodyText"]))

    import io as _io
    buf = _io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=letter, canvasmaker=_DetCanvas,
                            title=str(d.get("title", "AutoRefine report")),
                            author="autorefine")
    pdf.build(story)
    data = buf.getvalue()

    # G2: replace the per-build random /ID with a doc-deterministic one.
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"),
                           default=str)
    id_hex = hashlib.sha256(canonical.encode("utf-8")).digest().hex()[:32]
    data, _n = _PDF_ID_RE.subn(
        f"/ID [<{id_hex}><{id_hex}>]".encode("ascii"), data, count=1)
    out.write_bytes(data)
    return out


# --- B5: the benchmark / longitudinal report (v0.50) ---------------------------

def spec_fingerprint(spec) -> str | None:
    """64.1.2 (B5): the spec's identity — first 12 hex chars of the SHA-256
    of the canonical (``sort_keys``) JSON of a spec dict (the 38.1.3
    ``config_fingerprint`` rule applied to a *model spec* rather than a run
    config). Same spec → same fp regardless of key order; a non-dict or
    empty spec → ``None`` (rendered as an em-dash, never invented)."""
    if not isinstance(spec, dict) or not spec:
        return None
    canon = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def benchmark_view(entries, spec_map=None) -> dict:
    """64.1 (B5): the benchmark / longitudinal report — "one run" becomes
    "a reportable program of runs". A pure function of the run registry
    (``registry.load_registry`` rows, append order = chronological) plus an
    optional ``spec_map`` (``run_id -> best_spec`` dict, loaded None-safe by
    the CLI; absent → every fingerprint reads ``None``).

    ``leaderboard`` (64.1.3) — per task (sorted by name): ``n_runs``,
    ``best_score`` (max finite final), ``best_run`` (the run id achieving
    it; ties keep the *first* in append order — chronological),
    ``spec_fp`` (the best run's ``spec_fingerprint``), ``met_target`` (the
    best run's gate, PASS/MISS/— via ``registry.gate_label``).

    ``generalization`` (64.1.3) — the matrix "same spec across tasks":
    fingerprint → ``{spec_fp, tasks: {task: best score}, runs, n_tasks}``,
    **only** fingerprints that appear on >= 2 distinct tasks (a spec on a
    single task is not a generalization signal); sorted by ``spec_fp``.
    Runs with an unknown spec fall into one shared ``"?"`` bucket so the
    matrix still groups them honestly.

    ``trend`` (64.1.3) — per task, in append order: ``run_id`` / ``seed`` /
    ``score`` / ``best_so_far`` (the running max of finite scores; a
    non-finite score carries the previous best forward); plus a ``global``
    best-so-far chain across all tasks in the same order. Pure and
    deterministic (G2).
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    spec_map = spec_map if isinstance(spec_map, dict) else {}
    from .registry import gate_label  # local: keep the top imports light

    tasks: dict[str, list[dict]] = {}
    for e in entries:
        t = e.get("task")
        if not isinstance(t, str) or not t:
            continue
        tasks.setdefault(t, []).append(e)

    leaderboard = []
    for t in sorted(tasks):
        rows = tasks[t]
        best_i, best_s = None, None
        for i, e in enumerate(rows):  # first max in append order (ties: first)
            s = e.get("final_score")
            if _num(s) is not None and (best_s is None or _num(s) > best_s):
                best_i, best_s = i, _num(s)
        if best_i is None:
            leaderboard.append({"task": t, "n_runs": len(rows),
                                "best_score": None, "best_run": None,
                                "spec_fp": None, "met_target": "\u2014"})
            continue
        br = rows[best_i]
        fp = spec_fingerprint(spec_map.get(br.get("run_id")))
        leaderboard.append({
            "task": t, "n_runs": len(rows), "best_score": best_s,
            "best_run": br.get("run_id"), "spec_fp": fp,
            "met_target": gate_label(br.get("met_target")),
        })

    # generalization: spec_fp -> task -> best score (64.1.3)
    by_fp: dict[str, dict] = {}
    for e in entries:
        t = e.get("task")
        s = e.get("final_score")
        if not isinstance(t, str) or _num(s) is None:
            continue
        fp = spec_fingerprint(spec_map.get(e.get("run_id"))) or "?"
        cell = by_fp.setdefault(fp, {"tasks": {}, "runs": []})
        cell["runs"].append(e.get("run_id"))
        prev = cell["tasks"].get(t)
        if prev is None or _num(s) > prev:
            cell["tasks"][t] = _num(s)
    generalization = [
        {"spec_fp": fp, "tasks": dict(sorted(cell["tasks"].items())),
         "runs": list(cell["runs"]), "n_tasks": len(cell["tasks"])}
        for fp, cell in sorted(by_fp.items()) if len(cell["tasks"]) >= 2
    ]

    def _chain(rows) -> list[dict]:
        out, best = [], None
        for e in rows:
            s = _num(e.get("final_score"))
            if s is not None and (best is None or s > best):
                best = s
            out.append({"run_id": e.get("run_id"), "seed": e.get("seed"),
                        "score": s, "best_so_far": best})
        return out

    trend = {t: _chain(tasks[t]) for t in sorted(tasks)}
    trend["global"] = _chain(entries)

    return {
        "n_runs": len(entries),
        "tasks": sorted(tasks),
        "leaderboard": leaderboard,
        "generalization": generalization,
        "trend": trend,
    }


def _bench_cell(v) -> str:
    """64.1.4 (B5): a benchmark table cell (None → ``—``; a number → ``%g``)."""
    if v is None:
        return "\u2014"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:g}"
    return str(v)


def render_benchmark(view) -> str:
    """64.1.4 (B5): the aligned-columns text rendering of a
    ``benchmark_view`` (no table pipes — terminal friendly). Pure and
    byte-stable (G2): a re-render of the same view is byte-identical."""
    v = view if isinstance(view, dict) else {}
    out = ["=== AutoRefine benchmark (longitudinal) ===",
           f"runs : {v.get('n_runs', 0)}    "
           f"tasks : {', '.join(v.get('tasks') or []) or '—'}", ""]

    out.append("leaderboard (per task)")
    lb = list(v.get("leaderboard") or [])
    if lb:
        out.append(f"  {'task':16s} {'runs':>5s} {'best':>8s} {'gate':>5s} "
                   f"{'spec_fp':>12s}  best_run")
        out += [f"  {r['task']:16s} {r['n_runs']:5d} "
                f"{_bench_cell(r.get('best_score')):>8s} "
                f"{r.get('met_target', '—'):>5s} "
                f"{str(r.get('spec_fp') or '—'):>12s}  "
                f"{r.get('best_run') or '—'}" for r in lb]
    else:
        out.append("  (no runs)")
    out.append("")

    out.append("generalization (same spec across tasks)")
    gen = list(v.get("generalization") or [])
    if gen:
        for g in gen:
            cells = "  ".join(f"{t} {_bench_cell(s)}" for t, s in
                              g["tasks"].items())
            out.append(f"  {g['spec_fp']}  ({g['n_tasks']} tasks, "
                       f"{len(g['runs'])} run(s))  {cells}")
    else:
        out.append("  (no spec shared by >= 2 tasks)")
    out.append("")

    out.append("trend (best score across runs)")
    trend = v.get("trend") or {}
    for t in list(trend):
        if t == "global":
            continue
        rows = trend.get(t) or []
        if rows:
            chain = "  ".join(
                f"{r.get('run_id', '?')}: {_bench_cell(r.get('score'))} "
                f"(best {_bench_cell(r.get('best_so_far'))})" for r in rows)
            out.append(f"  {t}: {chain}")
    g = trend.get("global") or []
    if g:
        chain = "  ".join(
            f"{r.get('run_id', '?')}: {_bench_cell(r.get('score'))} "
            f"(best {_bench_cell(r.get('best_so_far'))})" for r in g)
        out.append(f"  global: {chain}")
    if not g and not [1 for t in trend if t != 'global' and trend.get(t)]:
        out.append("  (no runs)")
    out.append("")
    return "\n".join(out)


def render_benchmark_md(view) -> str:
    """64.1.4 (B5): the Markdown rendering (``#`` headings + ``|`` tables) —
    the leaderboard as a table, the generalization matrix as a table, the
    trend as one row per task. Pure and byte-stable (G2)."""
    v = view if isinstance(view, dict) else {}
    out = ["# AutoRefine benchmark (longitudinal)", "",
           f"runs: {v.get('n_runs', 0)}    "
           f"tasks: {', '.join(v.get('tasks') or []) or '—'}", ""]

    out += ["## Leaderboard (per task)", ""]
    lb = list(v.get("leaderboard") or [])
    if lb:
        out += ["| task | runs | best | gate | spec_fp | best_run |",
                "|------|------|------|------|---------|----------|"]
        out += [f"| {r['task']} | {r['n_runs']} | "
                f"{_md_cell(r.get('best_score'))} | "
                f"{r.get('met_target', '—')} | "
                f"{_md_cell(r.get('spec_fp'))} | "
                f"{_md_cell(r.get('best_run'))} |" for r in lb]
    else:
        out.append("(no runs)")
    out.append("")

    out += ["## Generalization (same spec across tasks)", ""]
    gen = list(v.get("generalization") or [])
    if gen:
        cols = sorted({t for g in gen for t in g["tasks"]})
        out += ["| spec_fp | " + " | ".join(cols) + " | runs |",
                "|" + "------|" * (len(cols) + 2)]
        for g in gen:
            cells = " | ".join(_md_cell(g["tasks"].get(c)) for c in cols)
            out.append(f"| {g['spec_fp']} | {cells} | {len(g['runs'])} |")
    else:
        out.append("(no spec shared by >= 2 tasks)")
    out.append("")

    out += ["## Trend (best score across runs)", ""]
    trend = v.get("trend") or {}
    any_row = False
    for t in list(trend):
        if t == "global":
            continue
        for r in trend.get(t) or []:
            any_row = True
            out.append(f"- **{t}** — `{r.get('run_id', '?')}`: "
                       f"{_md_cell(r.get('score'))} "
                       f"(best so far {_md_cell(r.get('best_so_far'))})")
    for r in trend.get("global") or []:
        any_row = True
        out.append(f"- **global** — `{r.get('run_id', '?')}`: "
                   f"{_md_cell(r.get('score'))} "
                   f"(best so far {_md_cell(r.get('best_so_far'))})")
    if not any_row:
        out.append("(no runs)")
    out.append("")
    return "\n".join(out)


# --- B3: the end-user guide ----------------------------------------------------

def _what_it_predicts(task) -> dict:
    """63.2 (B3): the plain-English "what it predicts" block, derived from
    the task's ``head`` / ``n_outputs`` / ``state_dim`` / ``class_values``
    (all optional attributes; absent -> a graceful generic phrase)."""
    head = getattr(task, "head", None)
    n_out = getattr(task, "n_outputs", None)
    sdim = getattr(task, "state_dim", None)
    classes = getattr(task, "class_values", None)
    inp = f"{sdim} numeric feature(s)" if sdim else "tabular input"
    if head == "softmax":
        metric, direction = "accuracy", "higher is better"
        output = (f"one of {len(classes)} classes" if classes
                  else "a class label")
        n_classes = len(classes) if classes else None
    elif head == "mse":
        metric, direction = "mean squared error (MSE)", "lower is better"
        output = (f"{n_out} numeric value(s)" if n_out else "numeric value(s)")
        n_classes = None
    else:
        metric, direction = "score", "higher is better"
        output = (f"{n_out} output(s)" if n_out else "model output")
        n_classes = None
    return {"input": inp, "output": output, "metric": metric,
            "direction": direction, "n_classes": n_classes}


def _row_str(row) -> str:
    """63.2 (B3): a one-line rendering of an input row (3 sig figs)."""
    vals = np.atleast_1d(np.asarray(row, dtype=np.float64)).ravel()
    return ", ".join(f"{float(v):.3g}" for v in vals)


def _val_str(arr) -> object:
    """63.2 (B3): a true/predicted value for the guide — integer-valued
    floats without a trailing ``.0``; a single value stays scalar."""
    a = np.atleast_1d(np.asarray(arr, dtype=np.float64)).ravel()

    def _one(v: float) -> object:
        v = float(v)
        return int(v) if v.is_integer() else round(v, 4)

    return _one(a[0]) if a.size == 1 else [_one(float(x)) for x in a]


def _class_str(task, idx) -> object:
    """63.2 (B3): a class index -> its display label (28.3), falling back
    to the integer index when the task exposes no ``class_values``."""
    i = int(idx)
    values = getattr(task, "class_values", None)
    if values and 0 <= i < len(values):
        return class_label_str(values[i])
    return i


def _examples_and_confidence(task, model, n_examples):
    """63.2 (B3): one bounded holdout sample -> ``(examples, note,
    confidence)``. A task with no ``holdout_rows`` (episode / synthetic
    tasks) or an empty split degrades to ``(None, note, None)`` — the
    guide explains the absence instead of inventing rows. ``confidence``
    is the softmax max-probability read (None for an mse head)."""
    if not callable(getattr(task, "holdout_rows", None)):
        return None, ("examples not available: this task has no per-row "
                      "holdout split (episode / synthetic tasks)"), None
    if model is None:
        return None, "examples not available: no best model to score with", None
    try:
        x, y = task.holdout_rows(int(n_examples), model)
    except Exception as exc:  # a broken run dir must not crash the guide
        return None, f"examples not available: {exc}", None
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    if n == 0:
        return None, "examples not available: the holdout split is empty", None
    try:
        logits = np.asarray(model.forward(x), dtype=np.float64)
    except Exception as exc:  # a broken model must not crash the guide either
        return None, f"examples not available: {exc}", None
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    head = getattr(task, "head", None)
    if head == "softmax":
        y_true = np.clip(np.asarray(y, dtype=np.int64).ravel(),
                         0, logits.shape[1] - 1)
        probs = _softmax(logits)
        examples = []
        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(logits[i].argmax())
            examples.append({
                "input": _row_str(x[i]),
                "true": _class_str(task, t),
                "predicted": _class_str(task, p),
                "confidence": round(float(probs[i].max()), 4),
                "correct": bool(p == t),
            })
        maxp = probs.max(axis=1)
        confidence = {
            "mean_max_prob": round(float(maxp.mean()), 4),
            "frac_at_least_0.8": round(float((maxp >= 0.8).mean()), 4),
            "n": int(n),
            "note": "mean of the model's max softmax probability over the "
                    "sample; frac_at_least_0.8 = the share of rows it is "
                    "confident about (>= 0.80)",
        }
        return examples, None, confidence
    # mse (or any non-softmax head with holdout rows)
    target = np.asarray(y, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    pred = logits
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    examples = []
    for i in range(len(target)):
        tol = 0.05 * max(1.0, float(np.abs(target[i]).max()))
        d = float(np.abs(pred[i] - target[i]).max())
        examples.append({
            "input": _row_str(x[i]),
            "true": _val_str(target[i]),
            "predicted": _val_str(pred[i]),
            "confidence": None,
            "correct": bool(d <= tol),
        })
    return examples, None, None


def user_guide_view(summary, entries, task=None, model=None,
                    n_examples: int = 5, headline=None) -> dict:
    """63.2 (B3): the "what this model does" view for the person who will
    *consume* the predictions.

    ``what_it_predicts`` (input / output / metric / direction, from
    ``task.head``); ``examples`` (the first ``n_examples`` holdout rows:
    input / true / predicted / confidence / correct, or ``None`` + an
    explanatory ``examples_note`` for episode tasks); ``confidence``
    (softmax max-probability read; ``None`` for an mse head);
    ``when_to_distrust``; ``known_limitations``; and ``how_to_read``
    (the metric in plain words + the target margin + the seed-variance
    caveat). Pure over the summary/entries + one bounded holdout pass
    (28.2 protocol); never invents rows for a task with no split.
    """
    s = summary if isinstance(summary, dict) else {}
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    view = {
        "what_it_predicts": None,
        "examples": None,
        "examples_note": None,
        "confidence": None,
        "when_to_distrust": [],
        "known_limitations": [],
        "how_to_read": None,
    }
    if task is not None:
        view["what_it_predicts"] = _what_it_predicts(task)
        examples, note, confidence = _examples_and_confidence(
            task, model, n_examples)
        view["examples"] = examples
        view["examples_note"] = note
        view["confidence"] = confidence

        distrust = []
        if confidence is not None:
            frac = confidence["frac_at_least_0.8"]
            if frac is not None and frac < 1.0:
                distrust.append(
                    f"the model is <80% confident on "
                    f"{100.0 * (1.0 - frac):.0f}% of the sampled rows")
        if examples:
            wrong = sum(1 for e in examples if not e.get("correct"))
            if wrong:
                distrust.append(f"{wrong} of the first {len(examples)} "
                                f"sampled row(s) are mispredicted")
        view["when_to_distrust"] = distrust

        limits = []
        if confidence is not None and confidence.get("n") is not None:
            limits.append(f"illustrated on a sample of {confidence['n']} "
                          f"holdout row(s), not the full holdout")
        fr = s.get("finished_reason")
        if fr:
            limits.append(f"the search stopped for: {fr}")
        try:
            rej = account_run(s, entries)["rejections"]
            if rej.get("overfit"):
                limits.append(f"{rej['overfit']} candidate(s) were rejected "
                              f"for overfitting (gen-gap)")
        except Exception:
            pass
        view["known_limitations"] = limits
    else:
        view["examples_note"] = (
            "examples not available: no task was supplied (a built-in / "
            "episode task, or a run whose task could not be reconstructed)")

    best_n = _num(s.get("final_best_score"))
    tgt_n = _num(s.get("target"))
    if best_n is not None and tgt_n is not None:
        read = (f"final score {best_n:g} vs target {tgt_n:g} "
                f"(margin {best_n - tgt_n:+g})")
    elif best_n is not None:
        read = f"final score {best_n:g} (no acceptance target was set)"
    else:
        read = "no final score recorded"
    how_to_read = (
        read + "; a higher score is better. The number is a single-run "
        "point estimate \u2014 it can vary across seeds, so treat a one-off "
        "score as a range, not a guarantee")
    if isinstance(headline, dict) and headline.get("text"):
        # 64.2.2 (B6): the seed-spread read on the headline number
        how_to_read += (f" Across same-task runs the final score reads "
                        f"{headline['text']}")
    view["how_to_read"] = how_to_read
    return view


def render_user_guide(view) -> str:
    """63.2 (B3): the stable text rendering of a user-guide view (G2 — a
    re-render of the same view is byte-identical)."""
    lines = ["=== AutoRefine user guide (what this model does) ==="]
    _render_dict(view if isinstance(view, dict) else {}, lines)
    return "\n".join(lines)


# --- B4: the decision artifact -------------------------------------------------

def _seed_variance(seed_spread) -> dict | None:
    """63.3 (B4): the seed-variance block from a list of final scores
    (``{n_runs, min, max, spread}``), or ``None`` when there is nothing to
    report (empty / non-numeric)."""
    if not seed_spread:
        return None
    nums = [float(v) for v in seed_spread
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v))]
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    return {"n_runs": len(nums), "min": lo, "max": hi, "spread": hi - lo}


def _weakest_class(diag) -> dict | None:
    """63.3 (B4): the weakest holdout class from a 28.2 ``diag``
    (``{class, accuracy}``), or ``None`` when it does not apply."""
    if not isinstance(diag, dict) or not diag.get("class_labels"):
        return None
    per = diag.get("per_class") or []
    if not per:
        return None
    worst = min(range(len(per)), key=lambda i: per[i])
    return {"class": diag["class_labels"][worst],
            "accuracy": float(per[worst])}


def decision_view(summary, entries, proj=None, seed_spread=None,
                  diag=None) -> dict:
    """63.3 (B4): the go/no-go decision artifact — target met? by how
    much · confidence (seed variance + the logged CI / overfit
    rejections) · the top-3 failure modes · one concrete next step.

    ``verdict`` reuses the 62.2.2 rule (``audience._exec_go_no_go`` — one
    home). ``proj`` is a ``simulate.project_budget`` result (``verdict``
    in ``more`` / ``ceiling`` / ``insufficient``); ``seed_spread`` is a
    list of same-task final scores; ``diag`` is a 28.2 diagnostics dict.
    All pure over the already-loaded data (G2).
    """
    s = summary if isinstance(summary, dict) else {}
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    target = s.get("target")
    best = s.get("final_best_score")
    baseline = s.get("baseline_score")
    best_n = _num(best)
    tgt_n = _num(target)

    verdict = _exec_go_no_go(target, best, baseline)  # 62.2.2 rule, one home

    if tgt_n is None:
        target_met = {"target": None, "met": None, "margin": None}
    else:
        met = best_n is not None and best_n >= tgt_n
        margin = (best_n - tgt_n) if best_n is not None else None
        target_met = {"target": tgt_n, "met": bool(met), "margin": margin}

    try:
        rej = account_run(s, entries)["rejections"]
        ci_rej = int(rej.get("ci", 0))
        overfit_rej = int(rej.get("overfit", 0))
    except Exception:
        ci_rej = overfit_rej = 0

    sv = _seed_variance(seed_spread)
    parts = []
    if sv is not None:
        parts.append(f"across {sv['n_runs']} seeds the final score spans "
                     f"{sv['min']:g}-{sv['max']:g} (spread {sv['spread']:g})")
    else:
        parts.append("single run \u2014 no seed-variance data")
    if overfit_rej > 0:
        parts.append(f"{overfit_rej} candidate(s) rejected for overfitting "
                     f"(gen-gap)")
    if ci_rej > 0:
        parts.append(f"{ci_rej} candidate(s) rejected by the CI gate")
    confidence = {
        "seed_variance": sv,
        "ci_rejections": ci_rej,
        "overfit_rejections": overfit_rej,
        "assessment": "; ".join(parts),
    }

    weakest = _weakest_class(diag)
    modes: list[dict] = []
    if weakest is not None and weakest["accuracy"] < 100.0:
        modes.append({"kind": "weakest_class",
                      "detail": f"class '{weakest['class']}' at "
                                f"{weakest['accuracy']:.1f}% holdout accuracy"})
    if overfit_rej > 0:
        modes.append({"kind": "overfit",
                      "detail": f"{overfit_rej} candidate(s) rejected for "
                                f"overfitting (gen-gap above tolerance)"})
    if ci_rej > 0:
        modes.append({"kind": "ci",
                      "detail": f"{ci_rej} candidate(s) rejected by the "
                                f"CI gate"})
    if (s.get("finished_reason") == "budget" and tgt_n is not None
            and best_n is not None and best_n < tgt_n):
        modes.append({"kind": "budget",
                      "detail": f"stopped at the budget with final "
                                f"{best_n:g} below the target {tgt_n:g}"})
    failure_modes = modes[:3]

    hints: list[str] = []
    if overfit_rej > 0:
        hints.append("relax the gen-gap (overfit) gate to admit stronger "
                     "candidates")
    if weakest is not None and weakest["accuracy"] < 80.0:
        hints.append(f"add features / data targeting class "
                     f"'{weakest['class']}'")

    if verdict.get("verdict") == "GO":
        next_step = {"action": "ship",
                     "detail": "confirm with a seed-variance sweep before "
                               "deployment",
                     "hints": []}
    else:
        p = proj if isinstance(proj, dict) else {}
        v = p.get("verdict")
        if v == "more":
            more = p.get("more")
            vmax = _num(p.get("vmax"))
            detail = (f"run ~{more} more experiment(s) to close the gap "
                      f"(curve asymptote {vmax:g})" if vmax is not None
                      else f"run ~{more} more experiment(s) to close the gap")
            action = "more_budget"
        elif v == "ceiling":
            vmax = _num(p.get("vmax"))
            detail = (f"relax the target to <= {vmax:g}, or expand the model "
                      f"space" if vmax is not None
                      else "relax the target, or expand the model space")
            action = "relax_or_expand"
        elif v == "insufficient":
            action = "collect_history"
            detail = "gather >= 2 same-task runs so the budget can be " \
                     "projected"
        else:
            action = "investigate"
            detail = "no budget projection available \u2014 inspect the " \
                     "decision trace and the rejection buckets"
        next_step = {"action": action, "detail": detail, "hints": hints}

    return {
        "verdict": verdict,
        "target_met": target_met,
        "confidence": confidence,
        "failure_modes": failure_modes,
        "next_step": next_step,
        # 64.2.2 (B6): the ± read on the headline number (None-safe — a
        # single run has no variance to display; the A53 pin is untouched)
        "headline_uncertainty": headline_uncertainty(best, seed_spread),
    }


def render_decision(view) -> str:
    """63.3 (B4): the stable text rendering of a decision view (G2 — a
    re-render of the same view is byte-identical)."""
    lines = ["=== AutoRefine decision (go/no-go + next step) ==="]
    _render_dict(view if isinstance(view, dict) else {}, lines)
    return "\n".join(lines)


# --- convenience: one call -> a rendered artifact ------------------------------

def render_report(fmt: str, doc) -> str:
    """63.1 (B2): render a ``build_report_doc`` doc to a *text* format
    (``md`` / ``txt``). ``pdf`` is a file output and is served by
    ``render_report_pdf`` instead; requesting it here is a ``ValueError``
    (one path per output kind)."""
    if fmt == "md":
        return render_report_md(doc)
    if fmt == "txt":
        return render_report_txt(doc)
    if fmt == "pdf":
        raise ValueError("pdf is a file output \u2014 use render_report_pdf")
    raise ValueError(f"unknown report format {fmt!r}; expected one of "
                     f"{REPORT_FORMATS}")


__all__ = [
    "spec_fingerprint",
    "benchmark_view",
    "render_benchmark",
    "render_benchmark_md",
    "REPORT_FORMATS",
    "build_report_doc",
    "render_report_md",
    "render_report_txt",
    "render_report_pdf",
    "render_report",
    "user_guide_view",
    "render_user_guide",
    "decision_view",
    "render_decision",
]
