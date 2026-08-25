"""Hand-written ASCII/SVG charts for `report --plot` (SPEC.md 21.2).

The core stays NumPy-only (SPEC.md 3), so charts are generated directly as
text/XML — no matplotlib. Every function is a pure, deterministic function of
its input rows/points (G2), ASCII-safe, and escapes SVG text.

Inputs are plain dicts exactly as logged by `RunMemory` (experiments.jsonl
rows and the `summary["pareto_frontier"]` points, SPEC.md 9/15).
"""
from __future__ import annotations

import html
import json
import math

_AXIS = "#333333"
_LINE = "#1a66c2"
_BASELINE = "#c2410c"
_ACCEPTED = "#15803d"
_REJECTED = "#6b7280"


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


def html_report(summary: dict, entries) -> str:
    """One self-contained HTML5 run report (SPEC.md 22.2).

    Hand-written, no new dependency, no JS, no external assets: the summary
    table, the best spec as JSON in a <pre>, the two §21.2 SVGs embedded
    inline, and the experiment table. Pure/deterministic in (summary,
    entries) — no timestamps, byte-identical on rerun (G2). All dynamic
    text is HTML-escaped.
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

    a("<h2>Score vs experiment</h2>")
    a(svg_score_curve(entries))
    a("<h2>Pareto frontier</h2>")
    a(svg_pareto(pareto_pts))

    a("<h2>Experiments</h2>")
    a("<table><tr><th>kind</th><th>accepted</th><th>score</th><th>gen_gap</th>"
      "<th>train_seconds</th><th>mutation</th></tr>")
    for e in entries:
        cells = (
            e.get("kind", "?"),
            e.get("accepted"),
            _fmt(e.get("holdout_score")),
            _fmt(e.get("gen_gap")),
            _fmt(e.get("train_seconds")),
            ",".join(e.get("mutation") or []) or "-",
        )
        a("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in cells) + "</tr>")
    a("</table>")

    a("<p class=\"foot\">Generated by autorefine — self-contained, no external "
      "assets (SPEC.md 22.2).</p>")
    a("</body></html>")
    return "\n".join(L) + "\n"


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
