"""Hand-written ASCII/SVG charts for `report --plot` (SPEC.md 21.2).

The core stays NumPy-only (SPEC.md 3), so charts are generated directly as
text/XML — no matplotlib. Every function is a pure, deterministic function of
its input rows/points (G2), ASCII-safe, and escapes SVG text.

Inputs are plain dicts exactly as logged by `RunMemory` (experiments.jsonl
rows and the `summary["pareto_frontier"]` points, SPEC.md 9/15).
"""
from __future__ import annotations

import base64
import html
import json
import math

_AXIS = "#333333"
_LINE = "#1a66c2"
_BASELINE = "#c2410c"
_ACCEPTED = "#15803d"
_REJECTED = "#6b7280"
_SCORED_REJ = "#b91c1c"  # timeline: rejected with a score (SPEC.md 26.3)
_LADDER = "#7c3aed"  # curriculum step-up marker (SPEC.md 27.3)


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _score_rows(rows) -> list[tuple[str, float, bool]]:
    """Scored entries of an experiment log, in log order (SPEC.md 21.2)."""
    out = []
    for r in rows or []:
        kind = r.get("kind")
        if kind not in ("baseline", "experiment") or not _finite(r.get("holdout_score")):
            continue
        out.append((kind, float(r["holdout_score"]), bool(r.get("accepted"))))
    return out


def _pareto_points(points) -> list[tuple[float, float]]:
    """(score, seconds) pairs from frontier/log points (SPEC.md 21.2)."""
    out = []
    for p in points or []:
        if not isinstance(p, dict) or not _finite(p.get("score")):
            continue
        t = p.get("train_seconds", p.get("seconds"))
        if _finite(t) and float(t) >= 0.0:
            out.append((float(p["score"]), float(t)))
    return out


def _axis_frame(width: int, height: int) -> list[list[str]]:
    return [[" "] * width for _ in range(height)]


def _mark(grid: list[list[str]], width: int, height: int, n: int, i: int,
          x01: float, y01: float, ch: str) -> None:
    x = int(round(x01 * (width - 1))) if n > 1 else 0
    y = int(round(y01 * (height - 1)))
    grid[height - 1 - y][x] = ch


def ascii_score_curve(rows, width: int = 48, height: int = 12) -> str:
    """Score vs experiment index (baseline `B`, accepted `+`, rejected `o`)."""
    pts = _score_rows(rows)
    if not pts:
        return "no scored experiments in log"
    scores = [p[1] for p in pts]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(pts)
    grid = _axis_frame(width, height)
    for i, (kind, s, accepted) in enumerate(pts):
        ch = "B" if kind == "baseline" else ("+" if accepted else "o")
        _mark(grid, width, height, n, i, i / (n - 1) if n > 1 else 0.5,
              (s - lo) / (hi - lo), ch)
    lines = []
    for row in range(height):
        label = hi - (hi - lo) * row / (height - 1)
        lines.append(f"{label:8.2f} |{''.join(grid[row])}")
    lines.append(" " * 8 + "+" + "-" * width)
    lines.append(" " * 9 + "0" + " " * (width - 1) + str(n - 1))
    lines.append("score vs experiment index:  B baseline  + accepted  o rejected")
    return "\n".join(lines)


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="14" '
        f'fill="{_AXIS}">{html.escape(title)}</text>',
    ]


def _svg_axes(L: float, T: float, pw: float, ph: float, y_lo: float, y_hi: float,
              x_lo: float, x_hi: float, x_unit: str) -> list[str]:
    parts = [
        f'<line x1="{L:.1f}" y1="{T:.1f}" x2="{L:.1f}" y2="{T + ph:.1f}" stroke="{_AXIS}"/>',
        f'<line x1="{L:.1f}" y1="{T + ph:.1f}" x2="{L + pw:.1f}" y2="{T + ph:.1f}" '
        f'stroke="{_AXIS}"/>',
    ]
    for k in range(5):  # 5 y-ticks, evenly spaced
        val = y_hi - (y_hi - y_lo) * k / 4
        y = T + ph * k / 4
        parts.append(f'<line x1="{L - 4:.1f}" y1="{y:.1f}" x2="{L:.1f}" y2="{y:.1f}" '
                     f'stroke="{_AXIS}"/>')
        parts.append(f'<text x="{L - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_AXIS}">{val:.2f}</text>')
    for x, lab in ((L, f"{x_lo:.2f}"), (L + pw, f"{x_hi:.2f}")):
        anchor = "start" if x == L else "end"
        parts.append(f'<text x="{x:.1f}" y="{T + ph + 18:.1f}" text-anchor="{anchor}" '
                     f'font-size="11" fill="{_AXIS}">{lab}</text>')
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{T + ph + 34:.1f}" text-anchor="middle" '
                 f'font-size="11" fill="{_AXIS}">{html.escape(x_unit)}</text>')
    return parts


def _svg_plot(parts: list[str], xs: list[float], ys: list[float], colors: list[str],
              titles: list[str]) -> None:
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    parts.append(f'<polyline fill="none" stroke="{_LINE}" stroke-width="2" '
                 f'points="{poly}"/>')
    for x, y, c, t in zip(xs, ys, colors, titles):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}">'
                     f'<title>{html.escape(t)}</title></circle>')


def svg_score_curve(rows, width: int = 640, height: int = 360) -> str:
    """Score vs experiment index as a valid-XML SVG (SPEC.md 21.2)."""
    pts = _score_rows(rows)
    if not pts:
        parts = _svg_header(width, height, "score vs experiment (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no scored experiments in log</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    scores = [p[1] for p in pts]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(pts)
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    xs = [L + pw * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    ys = [T + ph * (1.0 - (s - lo) / (hi - lo)) for s in scores]
    colors = [_BASELINE if k == "baseline" else (_ACCEPTED if a else _REJECTED)
              for k, s, a in pts]
    titles = [f"{k} #{i}: {s:.2f}" for i, (k, s, a) in enumerate(pts)]
    parts = _svg_header(width, height, "score vs experiment")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(n - 1), "experiment index")
    _svg_plot(parts, xs, ys, colors, titles)
    parts.append("</svg>")
    return "\n".join(parts)


def ascii_pareto(points, width: int = 48, height: int = 12) -> str:
    """Pareto frontier, score vs training seconds (`*` per point)."""
    pts = _pareto_points(points)
    if not pts:
        return "no pareto points in log"
    lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    tlo, thi = min(p[1] for p in pts), max(p[1] for p in pts)
    if thi - tlo < 1e-9:
        thi = tlo + 1e-3
    n = len(pts)
    grid = _axis_frame(width, height)
    for i, (s, t) in enumerate(pts):
        _mark(grid, width, height, n, i, i / (n - 1) if n > 1 else 0.5,
              (s - lo) / (hi - lo), "*")
    lines = []
    for row in range(height):
        label = hi - (hi - lo) * row / (height - 1)
        lines.append(f"{label:8.2f} |{''.join(grid[row])}")
    lines.append(" " * 8 + "+" + "-" * width)
    lines.append(" " * 9 + f"{tlo:.2f}s" + " " * (width - 1) + f"{thi:.2f}s")
    lines.append("pareto frontier (score vs training seconds):  * frontier point")
    return "\n".join(lines)


def _fmt(v) -> str:
    """Human cell for a summary value (None-safe, SPEC.md 22.2)."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def html_report(summary: dict, entries, diagnostics: dict | None = None,
                gallery: list | None = None, arch_svg: str | None = None) -> str:
    """One self-contained HTML5 run report (SPEC.md 22.2).

    Hand-written, no new dependency, no JS, no external assets: the summary
    table, the best spec as JSON in a <pre>, the two §21.2 SVGs embedded
    inline, and the experiment table. Pure/deterministic in (summary,
    entries) — no timestamps, byte-identical on rerun (G2). All dynamic
    text is HTML-escaped.

    SPEC.md 28 (v0.14) optional extras, each rendering a section only when
    present (calls without them stay byte-identical):
    * `arch_svg` (C4) — the best spec's architecture diagram, after the
      Best spec section;
    * `diagnostics` (C2) — `holdout_diagnostics` output, a per-class bar
      chart + confusion matrix after the Experiments table;
    * `gallery` (C3) — `holdout_errors` items: image data-URI thumbnails
      (a text-list fallback when PIL is absent or a decode fails) and
      inline waveform SVGs, each captioned true -> predicted;
    * any entry with a non-empty `loss_history` (C1) — the Experiments
      table gains a per-row curve SVG column (old runs: no column).
    """
    summary = summary or {}
    entries = list(entries or [])
    esc = html.escape

    pareto_pts = summary.get("pareto_frontier") or [
        {"score": e["holdout_score"], "train_seconds": e.get("train_seconds", 0.0)}
        for e in entries
        if e.get("kind") in ("baseline", "experiment")
        and isinstance(e.get("holdout_score"), (int, float))
    ]

    L: list[str] = []
    a = L.append
    task = summary.get("task", "?")
    a("<!doctype html>")
    a('<html lang="en"><head><meta charset="utf-8">')
    a(f"<title>AutoRefine report — {esc(task)}</title>")
    a("<style>")
    a("body{font-family:ui-monospace,Consolas,monospace;margin:24px;color:#1f2933;}")
    a("h1{font-size:20px;} h2{font-size:15px;margin-top:28px;}")
    a("table{border-collapse:collapse;margin:8px 0 4px;}")
    a("td,th{border:1px solid #cbd2d9;padding:4px 10px;text-align:left;font-size:13px;}")
    a("th{background:#f0f4f8;}")
    a("pre{background:#f0f4f8;border:1px solid #cbd2d9;padding:10px;overflow:auto;}")
    a(".foot{color:#6b7280;font-size:12px;margin-top:32px;}")
    a("</style></head><body>")
    a(f"<h1>AutoRefine report — {esc(task)} (seed {esc(_fmt(summary.get('seed')))})</h1>")

    a("<h2>Summary</h2>")
    a("<table>")
    for key in ("finished_reason", "baseline_score", "final_best_score",
                "improvement_factor", "experiments_run", "wall_seconds"):
        a(f"<tr><th>{key}</th><td>{esc(_fmt(summary.get(key)))}</td></tr>")
    a("</table>")

    if summary.get("task_config"):
        a("<h2>Task config</h2>")
        a(f"<pre>{esc(json.dumps(summary['task_config'], indent=2, sort_keys=True))}</pre>")

    a("<h2>Best spec</h2>")
    spec = summary.get("best_spec")
    body = json.dumps(spec, indent=2, sort_keys=True) if spec is not None else "-"
    a(f"<pre>{esc(body)}</pre>")

    # C4 (SPEC.md 28.4): the architecture diagram — only when supplied
    if arch_svg:
        a("<h2>Architecture</h2>")
        a(arch_svg)

    a("<h2>Score vs experiment</h2>")
    a(svg_score_curve(entries))
    a("<h2>Pareto frontier</h2>")
    a(svg_pareto(pareto_pts))

    # G3 (SPEC.md 27.3): the ladder section only when the run has one;
    # non-curriculum reports are byte-identical to v0.12
    if summary.get("curriculum"):
        a("<h2>Curriculum ladder</h2>")
        a(svg_ladder_curve(entries, summary["curriculum"].get("levels")))

    a("<h2>Experiments</h2>")
    # C1 (SPEC.md 28.1): a per-row curve column ONLY when some entry carries
    # a non-empty loss_history — old-run reports stay byte-identical.
    has_curves = any(isinstance(e.get("loss_history"), (list, tuple))
                     and e.get("loss_history") for e in entries)
    curve_th = "<th>curve</th>" if has_curves else ""
    a("<table><tr><th>kind</th><th>accepted</th><th>score</th><th>gen_gap</th>"
      f"<th>train_seconds</th><th>mutation</th>{curve_th}</tr>")
    for e in entries:
        cells = (
            e.get("kind", "?"),
            e.get("accepted"),
            _fmt(e.get("holdout_score")),
            _fmt(e.get("gen_gap")),
            _fmt(e.get("train_seconds")),
            ",".join(e.get("mutation") or []) or "-",
        )
        row = "<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in cells)
        if has_curves:
            hist = e.get("loss_history")
            if isinstance(hist, (list, tuple)) and hist:
                row += f"<td>{svg_loss_curves(hist, width=360, height=200)}</td>"
            else:
                row += "<td>-</td>"
        row += "</tr>"
        a(row)
    a("</table>")

    # C2 (SPEC.md 28.2): holdout diagnostics — only when the data is present
    if diagnostics:
        a("<h2>Holdout diagnostics</h2>")
        a(svg_per_class_bars(diagnostics))
        a(svg_confusion_matrix(diagnostics))

    # C3 (SPEC.md 28.3): the error gallery — only for non-empty media runs
    if gallery:
        a("<h2>Error gallery</h2>")
        a("<ul>")
        for it in gallery:
            it = it if isinstance(it, dict) else {}
            uri = _image_data_uri(it.get("path"))
            cap = (f"{esc(str(it.get('file', '?')))} — true "
                   f"{esc(str(it.get('label', '?')))} -> predicted "
                   f"{esc(str(it.get('predicted', '?')))}")
            if uri:  # image thumbnail (PIL lazy; the data URI is self-contained)
                a(f'<li><img src="{uri}" alt="{esc(str(it.get("file", "?")))}" '
                  f'style="max-width:180px;max-height:180px;">')
            a(f"<div>{cap}</div>")
            if isinstance(it.get("waveform"), (list, tuple)):
                a(svg_audio_waveform(it["waveform"], it.get("sample_rate"),
                                     title=str(it.get("file", "audio"))))
            a("</li>")
        a("</ul>")

    a("<p class=\"foot\">Generated by autorefine — self-contained, no external "
      "assets (SPEC.md 22.2).</p>")
    a("</body></html>")
    return "\n".join(L) + "\n"


def _strip_rows(rows) -> list[tuple[int, bool, float | None, str | None]]:
    """(step, accepted, score-or-None, reason) from update rows (SPEC.md 27.1).

    `rows` are the runner's update dicts (SPEC.md 23.1): one tuple per
    update — the G1 strip has exactly one dot per update, scored or not.
    """
    out = []
    for i, r in enumerate(rows or [], start=1):
        s = r.get("candidate_score")
        out.append((i, bool(r.get("accepted")),
                    float(s) if _finite(s) else None, r.get("reason")))
    return out


def svg_score_strip(rows, width: int = 640) -> str:
    """G1 (SPEC.md 27.1): candidate score strip — one lane per outcome class,
    one dot per update. Scored candidates sit at their exact score on a fixed
    0-100 axis; unscored rejections (duplicate / invalid_spec, SPEC.md 6 R3 /
    25.4) stack in the labelled no-score zone at the left edge. Pure, valid
    XML like the 21.2/26.3 charts; deterministic in `rows` (G2).
    """
    pts = _strip_rows(rows)
    if not pts:
        height = 120
        parts = _svg_header(width, height, "candidate score strip (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no updates in the update stream</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L = 72.0
    R = 24.0
    pw = width - L - R
    lane_h = 26.0
    T = 34.0
    lanes = (  # (class label, fill) in lane order (SPEC.md 27.1)
        ("accepted", _ACCEPTED),
        ("scored-rejected", _SCORED_REJ),
        ("unscored", _REJECTED),
    )
    height = 176
    parts = _svg_header(width, height, "candidate score strip")
    counts = {label: 0 for label, _ in lanes}
    for _, accepted, s, _reason in pts:
        label = "accepted" if accepted else ("scored-rejected" if s is not None
                                             else "unscored")
        counts[label] += 1
    lane_top = {label: T + i * lane_h for i, (label, _f) in enumerate(lanes)}
    lane_cy = {label: top + lane_h / 2 for label, top in lane_top.items()}
    lane_fill = dict(lanes)
    bottom = T + len(lanes) * lane_h
    # lane backgrounds + left-gutter labels
    for label, fill in lanes:
        y = lane_top[label]
        parts.append(f'<rect x="{L:.1f}" y="{y:.1f}" width="{pw:.1f}" '
                     f'height="{lane_h:.1f}" fill="#f6f8fa"/>')
        parts.append(f'<text x="{L - 8:.1f}" y="{lane_cy[label] - 3:.1f}" '
                     f'text-anchor="end" font-size="11" fill="{fill}">{label}</text>')
    # the labelled no-score zone at the left edge (SPEC.md 27.1)
    ny = lane_top["unscored"]
    parts.append(f'<rect x="{L:.1f}" y="{ny:.1f}" width="72" height="{lane_h:.1f}" '
                 f'fill="#e8edf3"/>')
    parts.append(f'<text x="{L - 8:.1f}" y="{lane_cy["unscored"] + 11:.1f}" '
                 f'text-anchor="end" font-size="10" fill="{_REJECTED}">no score</text>')
    # one dot per update (SPEC.md 27.1)
    no_score_k = 0
    for step, accepted, s, reason in pts:
        if accepted:
            label, fill = "accepted", _ACCEPTED
        elif s is not None:
            label, fill = "scored-rejected", _SCORED_REJ
        else:
            label, fill = "unscored", _REJECTED
        cy = lane_cy[label]
        if s is not None:
            cx = L + pw * (min(max(s, 0.0), 100.0) / 100.0)
            title = f"step {step}: {label} @ {s:.2f}"
        else:
            cx = L + 14 + 18 * no_score_k  # stacked at the left edge
            no_score_k += 1
            why = html.escape(str(reason) if reason is not None else "unscored")
            title = f"step {step}: unscored ({why})"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}">'
                     f'<title>{title}</title></circle>')
    # score axis, fixed 0-100 (SPEC.md 27.1)
    parts.append(f'<line x1="{L:.1f}" y1="{bottom:.1f}" x2="{L + pw:.1f}" '
                 f'y2="{bottom:.1f}" stroke="{_AXIS}"/>')
    for s in (0, 25, 50, 75, 100):
        x = L + pw * s / 100.0
        parts.append(f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" '
                     f'y2="{bottom + 4:.1f}" stroke="{_AXIS}"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="{_AXIS}">{s}</text>')
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{bottom + 32:.1f}" '
                 f'text-anchor="middle" font-size="11" fill="{_AXIS}">'
                 f'candidate score (fixed 0-100)</text>')
    # legend with the per-class counts (SPEC.md 27.1)
    ly = bottom + 54
    lx = L
    for label, _fill in lanes:
        text = f"{label} ({counts[label]})"
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="10" height="10" '
                     f'fill="{lane_fill[label]}"/>')
        parts.append(f'<text x="{lx + 14:.1f}" y="{ly:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{text}</text>')
        lx += 14 + 7 * len(text) + 20
    parts.append("</svg>")
    return "\n".join(parts)


def _gap_rows(rows) -> list[tuple[int, float, float, bool]]:
    """(step, score, gen_gap, accepted) for scored updates (SPEC.md 27.2).

    Unscored updates (duplicate / invalid_spec rejections) carry no
    `candidate_score` and contribute no dots (SPEC.md 27.2).
    """
    out = []
    for i, r in enumerate(rows or [], start=1):
        s = r.get("candidate_score")
        g = r.get("gen_gap")
        if _finite(s) and _finite(g):
            out.append((i, float(s), float(g), bool(r.get("accepted"))))
    return out


def svg_score_gap_scatter(rows, width: int = 640, height: int = 360) -> str:
    """G2 (SPEC.md 27.2): score vs gen-gap scatter, colored by outcome, with
    the 18.5 tolerance boundary `gen_gap = 0.05 * score` as a dashed line —
    the region above it is where the overfit penalty bites. Fixed 0-100
    score axis, data-driven gap axis. Pure, valid XML, deterministic (G2).
    """
    pts = _gap_rows(rows)
    if not pts:
        parts = _svg_header(width, height, "score vs gen-gap scatter (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no scored updates in the update stream</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    gaps = [g for _i, _s, g, _a in pts]
    y_lo = min(0.0, min(gaps))
    y_hi = max(0.05 * max(s for _i, s, _g, _a in pts), max(gaps)) * 1.15
    if y_hi - y_lo < 1e-9:  # degenerate guard, like the other charts
        y_hi = y_lo + 1.0

    def x(score: float) -> float:
        return L + pw * (min(max(score, 0.0), 100.0) / 100.0)

    def y(gap: float) -> float:
        return T + ph * (1.0 - (gap - y_lo) / (y_hi - y_lo))

    parts = _svg_header(width, height, "score vs gen-gap scatter")
    parts += _svg_axes(L, T, pw, ph, y_lo, y_hi, 0.0, 100.0,
                       "candidate score (fixed 0-100)")
    # the 5%-of-score tolerance boundary (SPEC.md 18.5) — dashed reference
    a = (L, y(0.0))
    s_at_top = y_hi / 0.05
    b = (x(s_at_top), T) if s_at_top <= 100.0 else (L + pw, y(5.0))
    parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                 f'y2="{b[1]:.1f}" stroke="{_REJECTED}" stroke-width="1.5" '
                 f'stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{(a[0] + b[0]) / 2:.1f}" y="{(a[1] + b[1]) / 2 - 8:.1f}" '
                 f'text-anchor="middle" font-size="11" fill="{_REJECTED}">'
                 f'gen_gap = 0.05 * score (18.5 tolerance)</text>')
    for step, s, g, accepted in pts:
        fill = _ACCEPTED if accepted else _SCORED_REJ
        label = "accepted" if accepted else "scored-rejected"
        parts.append(f'<circle cx="{x(s):.1f}" cy="{y(g):.1f}" r="5" fill="{fill}">'
                     f'<title>step {step}: {label}, score {s:.2f}, '
                     f'gen_gap {g:.3f}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_ladder_curve(entries, levels, width: int = 640, height: int = 360) -> str:
    """G3 (SPEC.md 27.3): the running-best curve over a curriculum run's
    experiment log — baseline score, a candidate's score when accepted,
    unchanged on rejection, `new_baseline_score` at each step-up; one
    vertical marker per level row, labelled with its `n_bits`/`p_flip`
    (SPEC.md 20.1). Rendered only for runs that actually have a ladder
    (SPEC.md 27.3/27.4).
    """
    rows = [e for e in (entries or [])
            if e.get("kind") in ("baseline", "experiment", "curriculum")]
    cur_idx = [i for i, e in enumerate(rows) if e.get("kind") == "curriculum"]
    if not rows:
        parts = _svg_header(width, height, "curriculum ladder (best score) (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no rows in the experiment log</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    if not cur_idx:
        parts = _svg_header(width, height, "curriculum ladder (best score) (no step-ups)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no step-ups in this run</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    levels = list(levels or [])
    # running best (SPEC.md 27.3): baseline -> accepted bumps -> rejected
    # keeps -> new_baseline_score at each step-up
    best: float | None = None
    pts: list[tuple[int, str, float]] = []
    for i, e in enumerate(rows):
        kind = e["kind"]
        if kind == "baseline":
            if _finite(e.get("holdout_score")):
                best = float(e["holdout_score"])
        elif kind == "experiment":
            if e.get("accepted") and _finite(e.get("holdout_score")):
                best = float(e["holdout_score"])
        elif kind == "curriculum":
            if _finite(e.get("new_baseline_score")):
                best = float(e["new_baseline_score"])
        if best is not None:
            pts.append((i, kind, best))
    if not pts:
        parts = _svg_header(width, height, "curriculum ladder (best score) (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no rows in the experiment log</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    vals = [v for _i, _k, v in pts]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    n = len(pts)
    pos = {i: L + pw * (i / (n - 1) if n > 1 else 0.5) for i, _k, _v in pts}
    xs = [pos[i] for i, _k, _v in pts]
    ys = [T + ph * (1.0 - (v - lo) / (hi - lo)) for _i, _k, v in pts]
    parts = _svg_header(width, height, "curriculum ladder (best score)")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(n - 1), "experiment log index")
    # one marker per level row, matched by order (SPEC.md 27.3)
    for k, i in enumerate(cur_idx):
        x = pos.get(i)
        if x is None:
            continue
        parts.append(f'<line x1="{x:.1f}" y1="{T:.1f}" x2="{x:.1f}" y2="{T + ph:.1f}" '
                     f'stroke="{_LADDER}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        lv = levels[k] if k < len(levels) else None
        if isinstance(lv, dict) and _finite(lv.get("p_flip")):
            parts.append(f'<text x="{x:.1f}" y="{T - 8:.1f}" text-anchor="middle" '
                         f'font-size="10" fill="{_LADDER}">'
                         f'{lv.get("n_bits")} bits, p_flip {lv["p_flip"]:.2f}</text>')
    colors = []
    for i, k, _v in pts:
        if k == "curriculum":
            colors.append(_LADDER)
        elif k == "baseline":
            colors.append(_BASELINE)
        else:
            colors.append(_ACCEPTED if rows[i].get("accepted") else _REJECTED)
    titles = []
    for i, k, v in pts:
        if k == "baseline":
            titles.append(f"baseline: best {v:.2f}")
        elif k == "curriculum":
            titles.append(f"step-up: re-baselined, best {v:.2f}")
        else:
            ok = bool(rows[i].get("accepted"))
            titles.append(f"experiment: {'accepted' if ok else 'rejected'}, "
                          f"best {v:.2f}")
    _svg_plot(parts, xs, ys, colors, titles)
    parts.append("</svg>")
    return "\n".join(parts)


def _timeline_cells(rows) -> list[tuple[int, str, bool, bool]]:
    """(step, field, accepted, scored) cells from update rows (SPEC.md 26.3).

    `rows` are the runner's update dicts (SPEC.md 23.1): `mutation` fields,
    `accepted` outcome, `candidate_score` present iff the step was scored.
    """
    cells = []
    for i, r in enumerate(rows or [], start=1):
        scored = _finite(r.get("candidate_score"))
        accepted = bool(r.get("accepted"))
        for f in (r.get("mutation") or []):
            if isinstance(f, str):
                cells.append((i, f, accepted, scored))
    return cells


def svg_mutation_timeline(rows, width: int = 640, row_h: int = 18) -> str:
    """Mutation timeline: field x update-index strip (SPEC.md 26.3).

    One cell per (step, field) mutation — green accepted, red
    scored-rejected, grey unscored (duplicate / invalid-spec). Pure,
    deterministic in `rows` (G2), valid XML like the other §21.2 charts.
    """
    cells = _timeline_cells(rows)
    if not cells:
        height = 120
        parts = _svg_header(width, height, "mutation timeline (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no mutations in the update stream</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    fields = sorted({f for _, f, _, _ in cells})
    n_steps = max(i for i, _, _, _ in cells)
    row = {f: j for j, f in enumerate(fields)}
    L = 130  # left label gutter
    T = 30
    col_w = (width - L - 16) / n_steps
    bottom = T + len(fields) * row_h
    height = bottom + 46  # step ticks + legend
    parts = _svg_header(width, height, "mutation timeline (field x step)")
    parts.append(f'<text x="8" y="{T - 10}" font-size="11" '
                 f'fill="{_AXIS}">field</text>')
    for f in fields:
        y = T + row[f] * row_h
        parts.append(f'<text x="{L - 8}" y="{y + row_h - 5}" text-anchor="end" '
                     f'font-size="11" fill="{_AXIS}">{html.escape(f)}</text>')
    for i, f, accepted, scored in cells:
        if accepted:
            fill, label = _ACCEPTED, "accepted"
        elif scored:
            fill, label = _SCORED_REJ, "scored-rejected"
        else:
            fill, label = _REJECTED, "unscored"
        x = L + (i - 1) * col_w
        y = T + row[f] * row_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 1:.1f}" width="{max(1.0, col_w - 1):.1f}" '
            f'height="{row_h - 2:.1f}" fill="{fill}">'
            f'<title>step {i}: {html.escape(f)} ({label})</title></rect>')
    for i in {1, n_steps // 2 + 1, n_steps}:  # 1, mid, n step ticks
        x = L + (i - 1) * col_w + col_w / 2
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16}" text-anchor="middle" '
                     f'font-size="11" fill="{_AXIS}">{i}</text>')
    ly = bottom + 34
    lx = float(L)
    for fill, label in ((_ACCEPTED, "accepted"),
                        (_SCORED_REJ, "scored-rejected"),
                        (_REJECTED, "unscored")):
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="10" height="10" '
                     f'fill="{fill}"/>')
        parts.append(f'<text x="{lx + 14:.1f}" y="{ly}" font-size="11" '
                     f'fill="{_AXIS}">{label}</text>')
        lx += 14 + 7 * len(label) + 20
    parts.append("</svg>")
    return "\n".join(parts)


def svg_pareto(points, width: int = 640, height: int = 360) -> str:
    """Pareto frontier as a valid-XML SVG (SPEC.md 21.2)."""
    pts = _pareto_points(points)
    if not pts:
        parts = _svg_header(width, height, "pareto frontier (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no pareto points in log</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    tlo, thi = min(p[1] for p in pts), max(p[1] for p in pts)
    if thi - tlo < 1e-9:
        thi = tlo + 1e-3
    n = len(pts)
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    xs = [L + pw * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    ys = [T + ph * (1.0 - (s - lo) / (hi - lo)) for s, t in pts]
    colors = [_LINE] * n
    titles = [f"score {s:.2f} @ {t:.3f}s" for s, t in pts]
    parts = _svg_header(width, height, "pareto frontier")
    parts += _svg_axes(L, T, pw, ph, lo, hi, tlo, thi, "training seconds")
    _svg_plot(parts, xs, ys, colors, titles)
    parts.append("</svg>")
    return "\n".join(parts)


# --- SPEC.md 28 (v0.14): learning views (C1-C4) ----------------------------

def _curve_records(history) -> list[tuple[float, float, float]]:
    """(step, train, holdout) finite records, sorted by step (SPEC.md 28.1)."""
    out = []
    for r in history or []:
        if not isinstance(r, dict):
            continue
        s, tr, ho = r.get("step"), r.get("train"), r.get("holdout")
        if _finite(s) and _finite(tr) and _finite(ho):
            out.append((float(s), float(tr), float(ho)))
    out.sort(key=lambda t: t[0])
    return out


def svg_loss_curves(history, width: int = 640, height: int = 360) -> str:
    """C1 (SPEC.md 28.1): train/holdout loss-over-step from a
    `TrainResult.loss_history` (SPEC.md 28.1) — two polylines (train = the
    blue line, holdout probe = the orange baseline color), a data-driven
    loss axis, a step axis, a legend. Pure, valid XML, deterministic (G2);
    the empty case renders a header + message like the other charts."""
    pts = _curve_records(history)
    if not pts:
        parts = _svg_header(width, height, "loss curves (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no loss history recorded</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    s_lo = min(p[0] for p in pts)
    s_hi = max(p[0] for p in pts)
    if s_hi - s_lo < 1e-9:  # degenerate guard, like the other charts
        s_hi = s_lo + 1.0
    vals = [v for p in pts for v in (p[1], p[2])]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1e-3

    def x(step: float) -> float:
        return L + pw * (step - s_lo) / (s_hi - s_lo)

    def y(v: float) -> float:
        return T + ph * (1.0 - (v - lo) / (hi - lo))

    parts = _svg_header(width, height, "loss over training steps")
    parts += _svg_axes(L, T, pw, ph, lo, hi, s_lo, s_hi, "training step")
    parts.append(f'<polyline fill="none" stroke="{_LINE}" stroke-width="2" '
                 f'points="{" ".join(f"{x(p[0]):.1f},{y(p[1]):.1f}" for p in pts)}"/>')
    parts.append(f'<polyline fill="none" stroke="{_BASELINE}" stroke-width="2" '
                 f'points="{" ".join(f"{x(p[0]):.1f},{y(p[2]):.1f}" for p in pts)}"/>')
    for s, tr, ho in pts:
        parts.append(f'<circle cx="{x(s):.1f}" cy="{y(tr):.1f}" r="3" fill="{_LINE}">'
                     f'<title>step {int(s)}: train {tr:.4f}</title></circle>')
        parts.append(f'<circle cx="{x(s):.1f}" cy="{y(ho):.1f}" r="3" fill="{_BASELINE}">'
                     f'<title>step {int(s)}: holdout probe {ho:.4f}</title></circle>')
    # legend, top-right (the title sits top-center)
    legend = (("train", _LINE), ("holdout (probe)", _BASELINE))
    total = sum(14 + 7 * len(_label) + 18 for _label, _c in legend)
    lx = width - R - total
    for label, color in legend:
        parts.append(f'<rect x="{lx:.1f}" y="8" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14:.1f}" y="17" font-size="11" '
                     f'fill="{_AXIS}">{label}</text>')
        lx += 14 + 7 * len(label) + 18
    parts.append("</svg>")
    return "\n".join(parts)


def svg_per_class_bars(diag: dict | None, width: int = 640) -> str:
    """C2 (SPEC.md 28.2): per-class holdout accuracy bars (0-100) from a
    `holdout_diagnostics` dict — one bar per class, its holdout count in the
    left gutter, the percentage on the right. Pure, valid XML, empty-safe."""
    d = diag if isinstance(diag, dict) else None
    per = d.get("per_class") if d else None
    labels = d.get("class_labels") if d else None
    counts = d.get("class_counts") if d else None
    if not per or not labels or len(per) != len(labels):
        height = 160
        parts = _svg_header(width, height, "per-class accuracy (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no holdout diagnostics</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    n = len(per)
    L, R = 160.0, 72.0
    pw = width - L - R
    bar_h = 22.0
    T = 34.0
    bottom = T + n * bar_h
    height = int(bottom + 34)
    parts = _svg_header(width, height, "per-class accuracy (holdout)")
    for s in (0, 25, 50, 75, 100):
        x = L + pw * s / 100.0
        parts.append(f'<line x1="{x:.1f}" y1="{T:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                     f'stroke="#d9e0e7"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="{_AXIS}">{s}%</text>')
    for i in range(n):
        y = T + i * bar_h
        frac = min(max(float(per[i]), 0.0), 100.0) / 100.0
        cnt = counts[i] if counts and i < len(counts) else "?"
        parts.append(f'<rect x="{L:.1f}" y="{y + 2:.1f}" width="{pw:.1f}" '
                     f'height="{bar_h - 6:.1f}" fill="#f6f8fa"/>')
        parts.append(f'<rect x="{L:.1f}" y="{y + 2:.1f}" width="{pw * frac:.1f}" '
                     f'height="{bar_h - 6:.1f}" fill="{_LINE}">'
                     f'<title>{html.escape(str(labels[i]))}: '
                     f'{float(per[i]):.1f}%</title></rect>')
        parts.append(f'<text x="{L - 8:.1f}" y="{y + 14:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_AXIS}">'
                     f'{html.escape(str(labels[i]))} (n={html.escape(str(cnt))})</text>')
        parts.append(f'<text x="{L + pw + 8:.1f}" y="{y + 14:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{float(per[i]):.1f}%</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_confusion_matrix(diag: dict | None, cell: int = 56) -> str:
    """C2 (SPEC.md 28.2): the k x k confusion matrix (rows = true class,
    columns = predicted class) from a `holdout_diagnostics` dict — the
    diagonal highlighted, counts in every cell (an empty class reads 0).
    Pure, valid XML, empty-safe."""
    d = diag if isinstance(diag, dict) else None
    conf = d.get("confusion") if d else None
    labels = d.get("class_labels") if d else None
    if not conf or not labels or len(conf) != len(labels):
        height = 200
        parts = _svg_header(420, height, "confusion matrix (empty)")
        parts.append(f'<text x="210" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no holdout diagnostics</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    k = len(labels)
    L = 150
    T = 34
    width = L + k * cell + 16
    bottom = T + k * cell
    height = bottom + 46
    parts = _svg_header(width, height, "confusion matrix (rows true, columns predicted)")
    parts.append(f'<text x="8" y="{T - 8}" font-size="11" fill="{_AXIS}">'
                 f'true class (rows)</text>')
    parts.append(f'<text x="{L}" y="{T - 8}" font-size="11" fill="{_AXIS}">'
                 f'predicted class (columns)</text>')
    for j in range(k):  # column labels
        parts.append(f'<text x="{L + j * cell + cell // 2}" y="{T - 8}" '
                     f'text-anchor="middle" font-size="11" fill="{_AXIS}">'
                     f'{html.escape(str(labels[j]))}</text>')
    for i in range(k):
        parts.append(f'<text x="{L - 8}" y="{T + i * cell + cell // 2 + 4}" '
                     f'text-anchor="end" font-size="11" fill="{_AXIS}">'
                     f'{html.escape(str(labels[i]))}</text>')
        for j in range(k):
            x = L + j * cell
            y = T + i * cell
            diag_cell = (i == j)
            fill = "#d1fae5" if diag_cell else "#ffffff"
            tcol = _ACCEPTED if diag_cell else _AXIS
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{fill}" stroke="{_AXIS}"/>')
            parts.append(f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" '
                         f'text-anchor="middle" font-size="12" fill="{tcol}">'
                         f'{int(conf[i][j])}</text>')
    parts.append(f'<text x="{L + k * cell // 2}" y="{bottom + 18}" '
                 f'text-anchor="middle" font-size="11" fill="{_AXIS}">'
                 f'green diagonal = correct predictions</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_audio_waveform(values, sample_rate: int | None = None,
                       title: str = "") -> str:
    """C3 (SPEC.md 28.3): an audio waveform as a hand-rolled SVG from the
    raw signal (the `holdout_errors` block-mean downsample, <= 512 floats).
    Data-driven y-axis, a dashed zero line when 0 is in range; the empty
    case renders a header + message. Pure, valid XML, deterministic (G2)."""
    vals = [float(v) for v in (values or []) if _finite(v)]
    name = title or "audio waveform"
    if not vals:
        width, height = 480, 120
        parts = _svg_header(width, height, name + " (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no waveform data</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    width, height = 480, 200
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 1e-3, hi + 1e-3

    def x(i: int) -> float:
        n = len(vals)
        return L + pw * (i / (n - 1) if n > 1 else 0.5)

    def y(v: float) -> float:
        return T + ph * (1.0 - (v - lo) / (hi - lo))

    parts = _svg_header(width, height, name)
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(len(vals) - 1), "sample index")
    zero_in = lo <= 0.0 <= hi
    if zero_in:  # the dashed zero reference (SPEC.md 28.3)
        z = y(0.0)
        parts.append(f'<line x1="{L:.1f}" y1="{z:.1f}" x2="{L + pw:.1f}" '
                     f'y2="{z:.1f}" stroke="{_REJECTED}" stroke-width="1" '
                     f'stroke-dasharray="4 3"/>')
    # one vertical trace per sample, from the zero line (or the axis midline
    # when 0 is out of range) to the value — the classic waveform look
    base = y(0.0) if zero_in else y((lo + hi) / 2.0)
    step = max(1, len(vals) // 512)  # bound the element count
    for i in range(0, len(vals), step):
        parts.append(f'<line x1="{x(i):.1f}" y1="{base:.1f}" x2="{x(i):.1f}" '
                     f'y2="{y(vals[i]):.1f}" stroke="{_LINE}"/>')
    if sample_rate is not None and _finite(sample_rate):
        parts.append(f'<text x="{L}" y="{height - 6}" font-size="11" fill="{_AXIS}">'
                     f'sample rate: {int(sample_rate)} Hz</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _image_data_uri(path) -> str | None:
    """C3 (SPEC.md 28.3): a base64 PNG data URI for one image file, or
    `None` on any failure (PIL absent or a decode error) — the callers fall
    back to a text list. PIL is lazy-imported (SPEC.md 3/24.1) so
    `import autorefine` stays PIL-free."""
    if not path:
        return None
    try:
        import io
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
    except Exception:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _spec_get(spec, key, default=None):
    """Read a spec field from a dict (summary JSON) or a ModelSpec object."""
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def svg_architecture(spec, state_dim, n_out, width: int = 640) -> str:
    """C4 (SPEC.md 28.4): a small horizontal annotated block diagram of one
    ModelSpec — an input block (state_dim) -> the family body -> an output
    block (n_out), plus an annotations row (optimizer always; LR schedule,
    early stopping, gradient clipping, label smoothing when non-default).

    Per family: mlp one block per hidden layer (depth-0 -> "linear");
    convnet the verified conv/pool/FC pipeline with its c1/c2; tree "N trees
    (depth D), bagged"; boost "N rounds (depth D), boosted" (in both
    `N = train_steps // 100` clamped 2..50 as in the trainer,
    `D = architecture[0]`); knn "k = knn_k". ASCII-only, pure, valid XML,
    deterministic (G2)."""
    if spec is None:
        height = 160
        parts = _svg_header(width, height, "architecture (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no spec to render</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    family = str(_spec_get(spec, "model_family", "mlp"))
    arch = _spec_get(spec, "architecture") or []
    arch = [int(v) for v in arch]
    n_out = int(n_out) if _finite(n_out) else 1
    sdim = int(state_dim) if _finite(state_dim) else 0

    blocks: list[tuple[str, str]] = []  # (title, detail)
    if family == "mlp":
        if arch:
            blocks = [(f"hidden {h}", "activation " + str(_spec_get(spec, "activation", "?")))
                      for h in arch]
        else:
            blocks = [("linear", "no hidden layers")]
    elif family == "convnet":
        c1, c2 = (arch[0], arch[1]) if len(arch) >= 2 else ("?", "?")
        blocks = [
            ("conv 3x3", f"1 -> {c1} filters"),
            ("pool 2x2", ""),
            ("conv 3x3", f"{c1} -> {c2} filters"),
            ("pool 2x2", ""),
            ("FC 32", "fixed hidden"),
            ("linear", f"-> {n_out} outputs"),
        ]
    elif family in ("tree", "boost"):
        d = arch[0] if arch else "?"
        n_trees = int(max(2, min(50, int(_spec_get(spec, "train_steps", 200)) // 100)))
        kind = "trees" if family == "tree" else "rounds"
        mode = "bagged" if family == "tree" else "boosted"
        blocks = [(f"{n_trees} {kind} (depth {d})", mode)]
    elif family == "knn":
        blocks = [("k nearest neighbors", f"k = {_spec_get(spec, 'knn_k', 5)}")]
    else:
        blocks = [(family, "unknown family")]

    # layout: input -> body -> output, horizontal, one row of blocks
    bw, bh, gap = 118, 54, 26
    nblocks = len(blocks) + 2
    total = nblocks * bw + (nblocks - 1) * gap
    L = max(16.0, (width - total) / 2.0)
    T = 64.0
    height = int(T + bh + 64)  # room for the annotations row
    parts = _svg_header(width, height, f"best spec architecture ({family})")

    def block(x: float, title: str, detail: str, fill: str) -> None:
        parts.append(f'<rect x="{x:.1f}" y="{T:.1f}" width="{bw}" height="{bh}" '
                     f'fill="{fill}" stroke="{_AXIS}" rx="4"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{T + 22:.1f}" text-anchor="middle" '
                     f'font-size="12" fill="{_AXIS}">{html.escape(title)}</text>')
        if detail:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{T + 40:.1f}" '
                         f'text-anchor="middle" font-size="10" fill="{_AXIS}">'
                         f'{html.escape(detail)}</text>')

    def arrow(x0: float, x1: float) -> None:
        y = T + bh / 2
        parts.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1 - 5:.1f}" '
                     f'y2="{y:.1f}" stroke="{_AXIS}" stroke-width="1.5"/>')
        parts.append(f'<polygon points="{x1:.1f},{y:.1f} {x1 - 7:.1f},{y - 4:.1f} '
                     f'{x1 - 7:.1f},{y + 4:.1f}" fill="{_AXIS}"/>')

    x = L
    block(x, f"input ({sdim})", "state / features", "#ffffff")
    x += bw
    for title, detail in blocks:
        x += gap
        arrow(x - gap, x)
        block(x, title, detail, "#eef2f7")
        x += bw
    x += gap
    arrow(x - gap, x)
    block(x, f"output ({n_out})", "logits / value", "#ffffff")

    # annotations row (SPEC.md 28.4): optimizer always + non-default knobs
    notes = ["optimizer: " + str(_spec_get(spec, "optimizer", "?"))]
    sched = str(_spec_get(spec, "lr_schedule", "constant"))
    if sched and sched != "constant":
        notes.append("lr schedule: " + sched)
    pat = _spec_get(spec, "early_stopping_patience", 0)
    if isinstance(pat, (int, float)) and not isinstance(pat, bool) and pat > 0:
        notes.append(f"early stopping: patience {int(pat)}")
    clip = _spec_get(spec, "gradient_clipping", 0.0)
    if _finite(clip) and float(clip) > 0.0:
        notes.append(f"gradient clip: {float(clip):g}")
    ls = _spec_get(spec, "label_smoothing", 0.0)
    if _finite(ls) and float(ls) > 0.0:
        notes.append(f"label smoothing: {float(ls):g}")
    text = ";  ".join(notes)
    parts.append(f'<text x="{width / 2:.1f}" y="{T + bh + 28:.1f}" '
                 f'text-anchor="middle" font-size="11" fill="{_AXIS}">'
                 f'{html.escape(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --- v0.15 multi-run / policy views (SPEC.md 29) -----------------------------

def _quantile(sorted_vals, p: float) -> float:
    """Linear-interpolation quantile (the NumPy `linear` method, SPEC.md 29.1)
    over an already-sorted list: position `p*(n-1)`, interpolate between the
    two closest ranks (n=1 -> that value). Pure/deterministic (G2)."""
    a = list(sorted_vals)
    n = len(a)
    if n == 0:
        raise ValueError("_quantile needs at least one value")
    if n == 1:
        return float(a[0])
    pos = min(max(float(p), 0.0), 1.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(a[lo])
    frac = pos - lo
    return float(a[lo]) * (1.0 - frac) + float(a[hi]) * frac


def svg_seed_variance(seeds, target=None, width: int = 640,
                      height: int = 360) -> str:
    """D1 (SPEC.md 29.1): the final-score distribution across seeds — a
    box-and-whisker over the `final` scores (min/q1/median/q3/max, linear
    quantile) on a fixed 0-100 axis, one paired baseline->final marker per
    seed, a dashed target line when given, and a passing P/N summary.
    `seeds` is a list of dicts each with `baseline`/`final` (and optional
    `seed`). Pure, valid XML, deterministic (G2); empty -> header + message.
    ASCII-only text."""
    rows = []
    for s in seeds or []:
        if not isinstance(s, dict):
            continue
        b, f = s.get("baseline"), s.get("final")
        if _finite(b) and _finite(f):
            rows.append({"seed": s.get("seed"),
                         "baseline": float(b), "final": float(f)})
    if not rows:
        parts = _svg_header(width, height, "seed variance (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no seeds in the sweep</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    L, R, T, B = 72.0, 24.0, 28.0, 64.0
    pw, ph = width - L - R, height - T - B

    def y(s: float) -> float:
        return T + ph * (1.0 - min(max(float(s), 0.0), 100.0) / 100.0)

    finals = sorted(r["final"] for r in rows)
    q1, med, q3 = (_quantile(finals, 0.25), _quantile(finals, 0.5),
                   _quantile(finals, 0.75))
    lo, hi = finals[0], finals[-1]
    n = len(rows)
    has_target = _finite(target) and not isinstance(target, bool)
    passing = (sum(1 for r in rows if r["final"] >= float(target))
               if has_target else None)

    parts = _svg_header(width, height, f"final score across {n} seed(s)")
    # fixed 0-100 score axis (left) + faint gridlines
    parts.append(f'<line x1="{L:.1f}" y1="{T:.1f}" x2="{L:.1f}" y2="{T + ph:.1f}" stroke="{_AXIS}"/>')
    for s in (0, 25, 50, 75, 100):
        yy = y(s)
        parts.append(f'<line x1="{L - 4:.1f}" y1="{yy:.1f}" x2="{L + pw:.1f}" y2="{yy:.1f}" '
                     f'stroke="#eef1f4"/>')
        parts.append(f'<text x="{L - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_AXIS}">{s}</text>')
    # dashed target line (SPEC.md 29.1)
    if has_target:
        ty = y(float(target))
        parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" y2="{ty:.1f}" '
                     f'stroke="{_BASELINE}" stroke-dasharray="6 4" stroke-width="2"/>')
        parts.append(f'<text x="{L + pw:.1f}" y="{ty - 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_BASELINE}">target {float(target):.2f}</text>')
    # box-and-whisker, centered in the left portion (SPEC.md 29.1)
    x_box = L + 0.20 * pw
    bw = 96.0
    bx0, bx1 = x_box - bw / 2, x_box + bw / 2
    parts.append(f'<line x1="{x_box:.1f}" y1="{y(hi):.1f}" x2="{x_box:.1f}" '
                 f'y2="{y(lo):.1f}" stroke="{_LINE}" stroke-width="2"/>')
    for wv in (lo, hi):
        parts.append(f'<line x1="{bx0:.1f}" y1="{y(wv):.1f}" x2="{bx1:.1f}" '
                     f'y2="{y(wv):.1f}" stroke="{_LINE}" stroke-width="2"/>')
    parts.append(f'<rect x="{bx0:.1f}" y="{y(q3):.1f}" width="{bw:.1f}" '
                 f'height="{max(1.0, y(q1) - y(q3)):.1f}" fill="#dbeafe" '
                 f'stroke="{_LINE}" stroke-width="2"><title>box q1..q3</title></rect>')
    parts.append(f'<line x1="{bx0:.1f}" y1="{y(med):.1f}" x2="{bx1:.1f}" '
                 f'y2="{y(med):.1f}" stroke="{_LINE}" stroke-width="3"/>')
    for name, val in (("max", hi), ("q3", q3), ("median", med), ("q1", q1), ("min", lo)):
        parts.append(f'<text x="{bx1 + 6:.1f}" y="{y(val) + 4:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{name} {val:.2f}</text>')
    # one paired baseline->final marker per seed (right portion)
    region_x0 = L + 0.44 * pw
    region_pw = L + pw - region_x0
    k = len(rows)
    for i, r in enumerate(rows):
        xk = region_x0 + region_pw * (0.5 if k == 1 else (i + 0.5) / k)
        yb, yf = y(r["baseline"]), y(r["final"])
        parts.append(f'<line x1="{xk:.1f}" y1="{yb:.1f}" x2="{xk:.1f}" y2="{yf:.1f}" '
                     f'stroke="{_REJECTED}" stroke-width="1.5"/>')
        d = 6.0
        parts.append(f'<polygon points="{xk:.1f},{yb - d:.1f} {xk + d:.1f},{yb:.1f} '
                     f'{xk:.1f},{yb + d:.1f} {xk - d:.1f},{yb:.1f}" fill="{_BASELINE}">'
                     f'<title>baseline {r["baseline"]:.2f}</title></polygon>')
        parts.append(f'<circle cx="{xk:.1f}" cy="{yf:.1f}" r="5" fill="{_ACCEPTED}">'
                     f'<title>final {r["final"]:.2f}</title></circle>')
        lab = f"seed {r['seed']}" if r["seed"] is not None else f"seed {i}"
        parts.append(f'<text x="{xk:.1f}" y="{T + ph + 16:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="{_AXIS}">{html.escape(str(lab))}</text>')
        parts.append(f'<text x="{xk:.1f}" y="{T + ph + 30:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="{_ACCEPTED}">{r["final"]:.1f}</text>')
    # summary line (SPEC.md 29.1)
    summary = (f"n seeds = {n} - passing = - (no target)" if passing is None
               else f"n seeds = {n} - passing = {passing}/{n}")
    parts.append(f'<text x="{width // 2}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="12" fill="{_AXIS}">{html.escape(summary)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _action_label(i) -> str:
    """Catalog action index -> ASCII label (SPEC.md 29.2): `field=value`;
    tuple values joined (e.g. `architecture=16,8`). The catalog is imported
    lazily so the core import stays clean (SPEC.md 3); an out-of-range index
    reads as a plain `action <i>`."""
    from .improver.catalog import ACTIONS
    try:
        i = int(i)
    except (TypeError, ValueError):
        return f"action {i}"
    if not (0 <= i < len(ACTIONS)):
        return f"action {i}"
    field, value = ACTIONS[i]
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    return f"{field}={value}"


def _idx_or_none(v) -> int | None:
    try:
        if isinstance(v, bool):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def svg_action_probabilities(trace, top_k: int = 8, n_steps: int = 12,
                             width: int = 640) -> str:
    """D2 (SPEC.md 29.2): the last `n_steps` steps of a `train_policy` trace
    as horizontal bar rows — each row shows that step's top-`top_k` catalog
    actions by probability (ties broken by index) as a segmented bar plus a
    grey `rest` segment, on a fixed 0-1 axis; the sampled action's segment is
    highlighted (green) and named. Pure, valid XML, deterministic (G2);
    empty -> header + message. ASCII-only text."""
    steps = [t for t in (trace or [])
             if isinstance(t, dict) and isinstance(t.get("probs"), (list, tuple))
             and len(t["probs"]) > 0]
    if not steps:
        height = 120
        parts = _svg_header(width, height, "action probabilities (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no policy trace '
                     f'recorded (train with trace=[...])</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    n_show = max(1, int(n_steps))
    start = max(0, len(steps) - n_show)
    steps = steps[start:]
    n = len(steps)
    L, R = 24.0, 24.0
    pw = width - L - R
    row_h = 48.0
    T = 34.0
    bottom = T + n * row_h
    height = int(bottom + 46)
    parts = _svg_header(width, height,
                        f"action probabilities (last {n} of {start + n} step(s))")

    def x(p: float) -> float:
        return L + pw * min(max(float(p), 0.0), 1.0)

    for p in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(p):.1f}" y1="{T - 6:.1f}" x2="{x(p):.1f}" '
                     f'y2="{bottom:.1f}" stroke="#eef1f4"/>')
    parts.append(f'<text x="{L:.1f}" y="{bottom + 14:.1f}" text-anchor="start" '
                 f'font-size="10" fill="{_AXIS}">0</text>')
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{bottom + 14:.1f}" text-anchor="middle" '
                 f'font-size="10" fill="{_AXIS}">0.5</text>')
    parts.append(f'<text x="{L + pw:.1f}" y="{bottom + 14:.1f}" text-anchor="end" '
                 f'font-size="10" fill="{_AXIS}">1.0</text>')

    for ri, t in enumerate(steps):
        probs = [float(v) if _finite(v) else 0.0 for v in t["probs"]]
        idx = _idx_or_none(t.get("action"))
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        top = order[:max(1, int(top_k))]
        top_set = set(top)
        rest = sum(p for i, p in enumerate(probs) if i not in top_set)
        yb = T + ri * row_h
        head = f"step {start + ri}"
        if idx is not None:
            head += f"   sampled: {_action_label(idx)}"
        if _finite(t.get("reward")):
            head += f"   r={float(t['reward']):+.3f}"
        parts.append(f'<text x="{L:.1f}" y="{yb + 10:.1f}" font-size="11" '
                     f'font-weight="bold" fill="{_AXIS}">{html.escape(head)}</text>')
        # segmented bar: top-K in probability order, then the rest (SPEC.md 29.2)
        segs = [(ai, probs[ai]) for ai in top if probs[ai] > 0.0]
        segs.append(("", rest))
        cx = L
        bar_y = yb + 16.0
        bar_h = 16.0
        for ai, p in segs:
            if p <= 0.0:
                continue
            w = float(p) * pw
            if ai == "":
                fill, label = _REJECTED, "rest"
            elif idx is not None and ai == idx:
                fill, label = _ACCEPTED, _action_label(ai)
            else:
                fill, label = _LINE, _action_label(ai)
            parts.append(f'<rect x="{cx:.1f}" y="{bar_y:.1f}" '
                         f'width="{max(0.5, w):.1f}" height="{bar_h:.1f}" '
                         f'fill="{fill}"><title>{html.escape(label)} = {float(p):.4f}</title></rect>')
            cx += w
        bits = [f"{_action_label(ai)} {probs[ai]:.2f}"
                for ai in top if probs[ai] > 0.0]
        if rest > 0.0:
            bits.append(f"rest {rest:.2f}")
        parts.append(f'<text x="{L:.1f}" y="{yb + 44:.1f}" font-size="10" '
                     f'fill="{_AXIS}">{html.escape("  ".join(bits))}</text>')
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{height - 6}" text-anchor="middle" '
                 f'font-size="11" fill="{_AXIS}">probability mass (0-1); '
                 f'green segment = the sampled action; grey = rest</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_task_returns(returns, width: int = 640, height: int = 360) -> str:
    """D2 (SPEC.md 29.2): one episode-return curve per task from `returns`
    (`{task: [episode_return, ...]}`, the exact shape of
    `train_multi_policy["episode_returns"]`, SPEC.md 20.2). Data-driven
    axes, a per-task legend. A single-task trace is `{task: [..]}` (one line).
    Pure, valid XML, deterministic (G2); empty -> header + message."""
    series = []
    for task, vals in (returns or {}).items():
        vv = [float(v) for v in (vals or []) if _finite(v)]
        if vv:
            series.append((str(task), vv))
    if not series:
        parts = _svg_header(width, height, "task returns (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no episode returns recorded</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, T, R, B = 72.0, 28.0, 24.0, 58.0
    pw, ph = width - L - R, height - T - B
    nmax = max(len(v) for _, v in series)
    allvals = [v for _, vals in series for v in vals]
    lo, hi = min(allvals), max(allvals)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    x_lo, x_hi = 1, nmax
    if nmax == 1:
        x_hi = 2  # degenerate guard (a single episode)

    def x(ep: int) -> float:
        return L + pw * (ep - x_lo) / (x_hi - x_lo)

    def y(v: float) -> float:
        return T + ph * (1.0 - (v - lo) / (hi - lo))

    palette = [_LINE, _BASELINE, _LADDER, _ACCEPTED, _SCORED_REJ, _REJECTED]
    parts = _svg_header(width, height, "episode returns per task")
    parts += _svg_axes(L, T, pw, ph, lo, hi, float(x_lo), float(x_hi), "episode")
    for ti, (task, vals) in enumerate(series):
        color = palette[ti % len(palette)]
        xs = [x(i + 1) for i in range(len(vals))]
        ys = [y(v) for v in vals]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{" ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in zip(xs, ys))}"/>')
        for xx, vv in zip(xs, vals):
            parts.append(f'<circle cx="{xx:.1f}" cy="{y(vv):.1f}" r="3.5" fill="{color}">'
                         f'<title>{html.escape(task)}: {vv:.4f}</title></circle>')
    lx = L
    for ti, (task, _v) in enumerate(series):
        color = palette[ti % len(palette)]
        parts.append(f'<rect x="{lx:.1f}" y="{T + ph + 28:.1f}" width="12" '
                     f'height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{T + ph + 38:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{html.escape(task)}</text>')
        lx += 16 + 7 * len(task) + 26
    parts.append("</svg>")
    return "\n".join(parts)
