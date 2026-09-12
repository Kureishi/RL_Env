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
from contextlib import contextmanager

from .memory import KIND_BASELINE, KIND_CURRICULUM, KIND_EXPERIMENT  # 35.1 (C4)

_AXIS = "#333333"
_LINE = "#1a66c2"
_BASELINE = "#c2410c"
_ACCEPTED = "#15803d"
_REJECTED = "#6b7280"
_SCORED_REJ = "#b91c1c"  # timeline: rejected with a score (SPEC.md 26.3)
_FRONTIER_PALETTE = (  # 49.3.3: the fixed A/B overlay palette (cycling)
    "#1a66c2", "#c2410c", "#15803d", "#7c3aed", "#b91c1c", "#0e7490",
)
_LADDER = "#7c3aed"  # curriculum step-up marker (SPEC.md 27.3)
_BAND = "#93c5fd"  # CI band on the score curve (SPEC.md 30.4)
# V3 (SPEC.md 30.3): the 8-color class palette (cycling for k > 8)
_CLASS_PALETTE = (
    "#1a66c2", "#c2410c", "#15803d", "#7c3aed",
    "#b91c1c", "#0e7490", "#a16207", "#db2777",
)
# --- SPEC.md 51.4 (v0.37): accessibility palettes -----------------------------
# 51.4.1: the Okabe-Ito colorblind-safe hues (the palette="okabe" data set)
OKABE_ITO = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7",
)
# 51.4.2: the dark-mode surface colors (defaults = the pre-v0.37 light set)
_BG = "white"  # the header background rect (byte-identical default)
_LANE_BG = "#f6f8fa"  # score-strip lane panels (was inline hex)
_NOSCORE_BG = "#e8edf3"  # the score-strip no-score zone (was inline hex)
_HIGHLIGHT = "#b45309"  # 53.3 (v0.39): the champion-card highlight color (fixed —

# 51.4.2: the dark set — background, axis/text, surfaces, and every data
# color, dark-tuned (the light set above stays the byte-identical default)
_DARK = {
    "_BG": "#0e1117",
    "_AXIS": "#c9d1d9",
    "_LINE": "#4493f8",
    "_BASELINE": "#f97316",
    "_ACCEPTED": "#3fb950",
    "_REJECTED": "#8b949e",
    "_SCORED_REJ": "#f85149",
    "_BAND": "#388bfd",
    "_LADDER": "#a371f7",
    "_LANE_BG": "#21262d",
    "_NOSCORE_BG": "#30363d",
    "_FRONTIER_PALETTE": ("#4493f8", "#f97316", "#3fb950", "#a371f7",
                          "#f85149", "#39c5cf"),
    "_CLASS_PALETTE": ("#4493f8", "#f97316", "#3fb950", "#a371f7",
                       "#f85149", "#39c5cf", "#d29922", "#ff7b72"),
}
# 51.4.1: the Okabe-Ito data-color set (axis/surfaces keep the base theme —
# the 51.4.1 set is about the *series* colors; the neutral grey is
# colorblind-safe by construction)
_OKABE = {
    "_LINE": "#56B4E9",
    "_BASELINE": "#E69F00",
    "_ACCEPTED": "#009E73",
    "_REJECTED": "#7f7f7f",
    "_SCORED_REJ": "#D55E00",
    "_BAND": "#56B4E9",
    "_LADDER": "#CC79A7",
    "_FRONTIER_PALETTE": ("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                          "#0072B2", "#D55E00"),
    "_CLASS_PALETTE": ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                       "#D55E00", "#CC79A7", "#E69F00"),
}
# every color global the two sets can swap (saved/restored by _styled)
_STYLE_KEYS = (
    "_BG", "_AXIS", "_LINE", "_BASELINE", "_ACCEPTED", "_REJECTED",
    "_SCORED_REJ", "_BAND", "_LADDER", "_LANE_BG", "_NOSCORE_BG",
    "_FRONTIER_PALETTE", "_CLASS_PALETTE",
)

# --- SPEC.md 56.3 (v0.42): the formal design-token block ----------------------
# The color system, formalized as one documented registry. `TOKENS` maps
# each canonical token name to its default (light) value (the module
# constants above); `_TOKEN_TO_KEY` binds each token to the module-global
# name it aliases, so the `okabe` / `dark` override sets (keyed by those
# global names — `_OKABE` / `_DARK`) merge back onto the canonical names
# with no case/name guessing. `resolve_tokens(palette, dark)` is the pure
# read: a NEW dict of resolved values (never mutating the module globals,
# unlike the `_styled` render context), validating exactly like `_styled`
# (51.4.1: unknown palette -> `ValueError`; 51.4.2: `dark` must be a bool),
# deterministic (G2) — the A46 test pins the default set to the module
# constants and the okabe/dark sets to `_OKABE` / `_DARK`, so the registry
# and the live tokens can never drift apart.
_TOKEN_TO_KEY = {
    "bg": "_BG",
    "axis": "_AXIS",
    "line": "_LINE",
    "baseline": "_BASELINE",
    "accepted": "_ACCEPTED",
    "rejected": "_REJECTED",
    "scored_rejected": "_SCORED_REJ",
    "band": "_BAND",
    "ladder": "_LADDER",
    "lane_bg": "_LANE_BG",
    "noscore_bg": "_NOSCORE_BG",
    "frontier_palette": "_FRONTIER_PALETTE",
    "class_palette": "_CLASS_PALETTE",
}
_TOKENS_FROM_KEY = {key: token for token, key in _TOKEN_TO_KEY.items()}
TOKENS: dict = {token: globals()[key]
                for token, key in _TOKEN_TO_KEY.items()}


def resolve_tokens(palette: str = "default", dark: bool = False) -> dict:
    """SPEC.md 56.3 (v0.42): the resolved design-token dict — `TOKENS`
    with the `okabe` / `dark` overrides applied per the active theme
    (the same two-set model as `_styled`, 51.4.1/51.4.2), returned as a
    NEW dict (the module globals are never touched, so it is safe to
    call from anywhere — the app's Design-tokens expander renders this).
    Validation mirrors `_styled` exactly: an unknown palette is a
    `ValueError` (51.4.1); `dark` must be a bool (51.4.2). The default
    call is byte-identical to `dict(TOKENS)` (G2); deterministic."""
    if palette not in ("default", "okabe"):
        raise ValueError(f"palette must be 'default' or 'okabe', got "
                         f"{palette!r} (SPEC.md 51.4.1/56.3)")
    if not isinstance(dark, bool):
        raise ValueError(f"dark must be a bool, got {type(dark).__name__} "
                         f"(SPEC.md 51.4.2/56.3)")
    out = dict(TOKENS)
    # dark first, then okabe on top (the same two-set order as `_styled`,
    # 51.4); a theme only overrides the tokens it defines (the okabe set is
    # a subset of the full set — the `.get` keeps the rest at the default)
    for active, overrides in ((dark, _DARK), (palette == "okabe", _OKABE)):
        if not active:
            continue
        for key, value in overrides.items():
            token = _TOKENS_FROM_KEY.get(key)
            if token is not None:
                out[token] = value
    return out


@contextmanager
def _styled(palette: str = "default", dark: bool = False):
    """SPEC.md 51.4.1/51.4.2 (v0.37): scoped palette + dark-mode overrides.

    Temporarily swaps the module's color globals — saved and restored in
    `finally` (nesting-safe) — so every inline color reference in the
    rendering functions picks up the active theme. The default
    (``palette="default", dark=False``) leaves every global untouched, so
    the output is byte-identical to pre-v0.37 (G2). Unknown palettes are a
    `ValueError` (51.4.1).

    Implementation contract (SPEC.md 51.4): the swap mutates module
    globals, so rendering must be single-threaded — the app renders on the
    main thread (51.2.3) and the core renderers never run concurrently.
    """
    if palette not in ("default", "okabe"):
        raise ValueError(f"palette must be 'default' or 'okabe', got "
                         f"{palette!r} (SPEC.md 51.4.1)")
    if not isinstance(dark, bool):
        raise ValueError(f"dark must be a bool, got {type(dark).__name__} "
                         f"(SPEC.md 51.4.2)")
    saved = {name: globals()[name] for name in _STYLE_KEYS}
    try:
        if dark:
            for name, value in _DARK.items():
                globals()[name] = value
        if palette == "okabe":
            for name, value in _OKABE.items():
                globals()[name] = value
        yield
    finally:
        for name, value in saved.items():
            globals()[name] = value


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _score_rows(rows) -> list[tuple[str, float, bool]]:
    """Scored entries of an experiment log, in log order (SPEC.md 21.2)."""
    out = []
    for r in rows or []:
        kind = r.get("kind")
        if kind not in (KIND_BASELINE, KIND_EXPERIMENT) or not _finite(r.get("holdout_score")):
            continue  # SPEC.md 35.1 (C4): the kind registry
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
        ch = "B" if kind == KIND_BASELINE else ("+" if accepted else "o")  # 35.1
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
    # SPEC.md 51.4.3 (v0.37): ARIA on every SVG root — `role="img"` +
    # `aria-label` (the escaped title), always, so a screen reader gets the
    # chart's title; purely additive attributes (no test pins the header).
    # `fill="{_BG}"` keeps the default byte-identical (`_BG == "white"`)
    # while the dark set (51.4.2) re-colors the background.
    label = html.escape(title)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}" '
        f'font-family="monospace">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_BG}"/>',
        f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="14" '
        f'fill="{_AXIS}">{label}</text>',
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


def _band_stds(rows) -> list[float]:
    """V4 (SPEC.md 30.4): the `std` of each scored row, aligned with the
    order of `_score_rows(rows)`. A row's std counts only when it is a
    positive finite number — legacy mode (`std == 0.0`) and old logs (no
    `std` key) read 0.0 and draw no band."""
    out = []
    for r in rows or []:
        kind = r.get("kind")
        if kind not in (KIND_BASELINE, KIND_EXPERIMENT) or not _finite(r.get("holdout_score")):
            continue  # SPEC.md 35.1 (C4): the kind registry
        s = r.get("std")
        ok = (isinstance(s, (int, float)) and not isinstance(s, bool)
              and math.isfinite(s) and float(s) > 0.0)
        out.append(float(s) if ok else 0.0)
    return out


def svg_score_curve(rows, width: int = 640, height: int = 360,
                    palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_score_curve` (the pre-v0.37 body). The default call is
    byte-identical to pre-v0.37 (G2); ARIA is always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_score_curve(rows, width=width, height=height)


def _svg_score_curve(rows, width: int = 640, height: int = 360) -> str:
    """Score vs experiment index as a valid-XML SVG (SPEC.md 21.2).

    V4 (SPEC.md 30.4): each scored row with a positive logged `std` (the
    block-bootstrap holdout sigma, SPEC.md 18.3) draws a vertical
    translucent band spanning `score +/- std` behind the curve, and the
    y-range expands to cover the bands. Rows without a positive `std`
    (legacy mode, old logs) render exactly the pre-v0.16 SVG — no band
    elements, no caption; `ascii_score_curve` is untouched."""
    pts = _score_rows(rows)
    if not pts:
        parts = _svg_header(width, height, "score vs experiment (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no scored experiments in log</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    bands = [(i, s) for i, s in enumerate(_band_stds(rows)) if s > 0.0]
    scores = [p[1] for p in pts]
    lo, hi = min(scores), max(scores)
    if bands:  # expand the y-range to cover the bands (SPEC.md 30.4)
        lo = min(lo, min(scores[i] - s for i, s in bands))
        hi = max(hi, max(scores[i] + s for i, s in bands))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(pts)
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B
    xs = [L + pw * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    ys = [T + ph * (1.0 - (s - lo) / (hi - lo)) for s in scores]
    colors = [_BASELINE if k == KIND_BASELINE else (_ACCEPTED if a else _REJECTED)  # 35.1
              for k, s, a in pts]
    titles = [f"{k} #{i}: {s:.2f}" for i, (k, s, a) in enumerate(pts)]
    parts = _svg_header(width, height, "score vs experiment")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(n - 1), "experiment index")
    if bands:
        for i, s in bands:  # behind the curve (SPEC.md 30.4)
            y_lo = T + ph * (1.0 - (scores[i] - s - lo) / (hi - lo))
            y_hi = T + ph * (1.0 - (scores[i] + s - lo) / (hi - lo))
            parts.append(f'<line x1="{xs[i]:.1f}" y1="{y_hi:.1f}" x2="{xs[i]:.1f}" '
                         f'y2="{y_lo:.1f}" stroke="{_BAND}" stroke-width="7" '
                         f'stroke-linecap="round">'
                         f'<title>±std {s:.3f}</title></line>')
    _svg_plot(parts, xs, ys, colors, titles)
    if bands:
        parts.append(f'<text x="{L + pw / 2:.1f}" y="{height - 6}" text-anchor="middle" '
                     f'font-size="11" fill="{_AXIS}">bands = score ± std (SPEC.md 18.3 '
                     f'block-bootstrap; the §18.5 CI gate)</text>')
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
                gallery: list | None = None, arch_svg: str | None = None,
                cover: str | None = None) -> str:
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
      table gains a per-row curve SVG column (old runs: no column);
    * `cover` (SPEC.md 56.2, v0.42) — the brand masthead + provenance
      cover block (caller-injected, like `arch_svg`), rendered at the top
      of the body; a call without it stays byte-identical (28.5).
    """
    summary = summary or {}
    entries = list(entries or [])
    esc = html.escape

    pareto_pts = summary.get("pareto_frontier") or [
        {"score": e["holdout_score"], "train_seconds": e.get("train_seconds", 0.0)}
        for e in entries
        if e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)  # 35.1 (C4)
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
    # 56.2 (v0.42): the brand masthead + provenance cover — only when the
    # caller supplies one (the `report --certificate` path, 56.1.1); calls
    # without a cover stay byte-identical (28.5).
    if cover:
        a(cover)
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


def html_cover(summary: dict, payload: dict | None = None) -> str:
    """SPEC.md 56.2 (v0.42): the report cover block — the brand masthead
    (the `AutoRefine` wordmark + the report title) and the provenance
    table (one row per `provenance.provenance_payload` entry, in payload
    order; `extras` as indented rows; None reads as `—`), rendered as a
    self-contained `<header>` + `<table>` with inline styles (no external
    assets, 22.2; no timestamps — a re-render is byte-identical, G2).
    `payload=None` renders the masthead only (the brand chrome without the
    certificate). All dynamic text is HTML-escaped. Pure, deterministic."""
    summary = summary if isinstance(summary, dict) else {}
    esc = html.escape
    task = summary.get("task", "?")
    seed = summary.get("seed")
    import autorefine  # the version is read at call time (33.1)
    L: list[str] = []
    a = L.append
    a("<style>")
    a(".cover{border:1px solid #cbd2d9;border-left:6px solid #1a66c2;"
      "border-radius:8px;padding:14px 18px;margin:0 0 18px;background:#f7f9fc;}")
    a(".cover .brand{font-size:19px;font-weight:700;color:#0f172a;}")
    a(".cover .tag{color:#6b7280;font-size:12px;margin-top:2px;}")
    a(".cover table{border-collapse:collapse;margin:10px 0 0;}")
    a(".cover td{padding:2px 14px 2px 0;font-size:13px;color:#1f2933;}")
    a(".cover td.k{color:#6b7280;white-space:nowrap;}")
    a("</style>")
    a("<header class=\"cover\">")
    a("<div class=\"brand\">AutoRefine</div>")
    a(f"<div class=\"tag\">autonomous model improvement — report · "
      f"task {esc(str(task))} · seed {esc(str(seed)) if seed is not None else '—'} "
      f"· v{esc(autorefine.__version__)}</div>")
    if payload is not None:
        p = payload if isinstance(payload, dict) else {}
        a("<table>")
        for key, value in p.items():
            if key == "extras" and isinstance(value, dict):
                for dist, version in value.items():
                    a(f"<tr><td class=\"k\">  {esc(str(dist))}</td>"
                      f"<td>{esc(str(version)) if version is not None else '—'}</td></tr>")
            else:
                text = value if value is not None else "—"
                a(f"<tr><td class=\"k\">{esc(str(key))}</td><td>{esc(str(text))}</td></tr>")
        a("</table>")
    a("</header>")
    return "\n".join(L)


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


def svg_score_strip(rows, width: int = 640,
                    palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_score_strip` (the pre-v0.37 body); default byte-identical
    (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_score_strip(rows, width=width)


def _svg_score_strip(rows, width: int = 640) -> str:
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
                     f'height="{lane_h:.1f}" fill="{_LANE_BG}"/>')  # 51.4.2
        parts.append(f'<text x="{L - 8:.1f}" y="{lane_cy[label] - 3:.1f}" '
                     f'text-anchor="end" font-size="11" fill="{fill}">{label}</text>')
    # the labelled no-score zone at the left edge (SPEC.md 27.1)
    ny = lane_top["unscored"]
    parts.append(f'<rect x="{L:.1f}" y="{ny:.1f}" width="72" height="{lane_h:.1f}" '
                 f'fill="{_NOSCORE_BG}"/>')  # 51.4.2
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
            if reason:  # 54.1 (v0.40): the reason, when present (byte-stable
                title += f" ({html.escape(str(reason))})"  # when falsy)
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


def _gap_rows(rows) -> list[tuple[int, float, float, bool, str | None]]:
    """(step, score, gen_gap, accepted, reason) for scored updates
    (SPEC.md 27.2; the `reason` hover, SPEC.md 54.1).

    Unscored updates (duplicate / invalid_spec rejections) carry no
    `candidate_score` and contribute no dots (SPEC.md 27.2).
    """
    out = []
    for i, r in enumerate(rows or [], start=1):
        s = r.get("candidate_score")
        g = r.get("gen_gap")
        if _finite(s) and _finite(g):
            out.append((i, float(s), float(g), bool(r.get("accepted")),
                        r.get("reason")))
    return out


def svg_score_gap_scatter(rows, width: int = 640, height: int = 360,
                          palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_score_gap_scatter` (the pre-v0.37 body); default
    byte-identical (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_score_gap_scatter(rows, width=width, height=height)


def _svg_score_gap_scatter(rows, width: int = 640, height: int = 360) -> str:
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
    gaps = [g for _i, _s, g, _a, _r in pts]
    y_lo = min(0.0, min(gaps))
    y_hi = max(0.05 * max(s for _i, s, _g, _a, _r in pts), max(gaps)) * 1.15
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
    for step, s, g, accepted, reason in pts:
        fill = _ACCEPTED if accepted else _SCORED_REJ
        label = "accepted" if accepted else "scored-rejected"
        why = f" ({html.escape(str(reason))})" if reason else ""  # 54.1
        parts.append(f'<circle cx="{x(s):.1f}" cy="{y(g):.1f}" r="5" fill="{fill}">'
                     f'<title>step {step}: {label}, score {s:.2f}, '
                     f'gen_gap {g:.3f}{why}</title></circle>')
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
            if e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT,  # 35.1 (C4)
                                KIND_CURRICULUM)]
    cur_idx = [i for i, e in enumerate(rows) if e.get("kind") == KIND_CURRICULUM]
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
        if kind == KIND_BASELINE:  # SPEC.md 35.1 (C4): the kind registry
            if _finite(e.get("holdout_score")):
                best = float(e["holdout_score"])
        elif kind == KIND_EXPERIMENT:
            if e.get("accepted") and _finite(e.get("holdout_score")):
                best = float(e["holdout_score"])
        elif kind == KIND_CURRICULUM:
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
        elif isinstance(lv, dict) and lv.get("difficulty"):
            # SPEC.md 46.2 (v0.32): the sine / cartpole ladder levels carry
            # their own parameter keys, so the marker is labelled with the
            # curriculum's difficulty string (e.g. "sine-n0.10-f1.5-a0.75")
            parts.append(f'<text x="{x:.1f}" y="{T - 8:.1f}" text-anchor="middle" '
                         f'font-size="10" fill="{_LADDER}">{lv["difficulty"]}</text>')
    colors = []
    for i, k, _v in pts:
        if k == KIND_CURRICULUM:  # SPEC.md 35.1 (C4)
            colors.append(_LADDER)
        elif k == KIND_BASELINE:
            colors.append(_BASELINE)
        else:
            colors.append(_ACCEPTED if rows[i].get("accepted") else _REJECTED)
    titles = []
    for i, k, v in pts:
        if k == KIND_BASELINE:
            titles.append(f"baseline: best {v:.2f}")
        elif k == KIND_CURRICULUM:
            titles.append(f"step-up: re-baselined, best {v:.2f}")
        else:
            ok = bool(rows[i].get("accepted"))
            titles.append(f"experiment: {'accepted' if ok else 'rejected'}, "
                          f"best {v:.2f}")
    _svg_plot(parts, xs, ys, colors, titles)
    parts.append("</svg>")
    return "\n".join(parts)


def _timeline_cells(rows, field: str | None = None) -> list[tuple[int, str, bool, bool, float | None, str | None]]:
    """(step, field, accepted, scored, score-or-None, reason) cells from
    update rows (SPEC.md 26.3; the score + reason hovers, SPEC.md 54.1).

    `rows` are the runner's update dicts (SPEC.md 23.1): `mutation` fields,
    `accepted` outcome, `candidate_score` present iff the step was scored.
    `field` (SPEC.md 52.2.2, v0.38) keeps only that mutation field's cells;
    `None` (default) keeps all — byte-identical to the pre-v0.38 cells (G2).
    """
    cells = []
    for i, r in enumerate(rows or [], start=1):
        s = r.get("candidate_score")
        scored = _finite(s)
        accepted = bool(r.get("accepted"))
        reason = r.get("reason")
        for f in (r.get("mutation") or []):
            if isinstance(f, str) and (field is None or f == field):
                cells.append((i, f, accepted, scored,
                              float(s) if scored else None, reason))
    return cells


def svg_mutation_timeline(rows, width: int = 640, row_h: int = 18,
                          palette: str = "default", dark: bool = False,
                          field: str | None = None) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_mutation_timeline` (the pre-v0.37 body); default
    byte-identical (G2); ARIA always on (51.4.3).
    `field` (SPEC.md 52.2.2, v0.38): the focus-field filter — only that
    mutation field's cells render; `None` (default) is byte-identical."""
    with _styled(palette, dark):
        return _svg_mutation_timeline(rows, width=width, row_h=row_h,
                                      field=field)


def _svg_mutation_timeline(rows, width: int = 640, row_h: int = 18,
                           field: str | None = None) -> str:
    """Mutation timeline: field x update-index strip (SPEC.md 26.3).

    One cell per (step, field) mutation — green accepted, red
    scored-rejected, grey unscored (duplicate / invalid-spec). Pure,
    deterministic in `rows` (G2), valid XML like the other §21.2 charts.
    `field` (SPEC.md 52.2.2, v0.38) filters the cells to one mutation
    field (the app's cross-view focus, 52.2.2); `None` = all fields.
    """
    cells = _timeline_cells(rows, field)
    if not cells:
        height = 120
        parts = _svg_header(width, height, "mutation timeline (empty)")
        empty_msg = (f"no {html.escape(field)} mutations in the update stream"
                     if field else "no mutations in the update stream")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">{empty_msg}</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    fields = sorted({c[1] for c in cells})
    n_steps = max(c[0] for c in cells)
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
    for i, f, accepted, scored, score, reason in cells:
        if accepted:
            fill, label = _ACCEPTED, "accepted"
        elif scored:
            fill, label = _SCORED_REJ, "scored-rejected"
        else:
            fill, label = _REJECTED, "unscored"
        x = L + (i - 1) * col_w
        y = T + row[f] * row_h
        sbit = f" @ {score:.2f}" if score is not None else ""  # 54.1
        why = f" ({html.escape(str(reason))})" if reason else ""  # 54.1
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 1:.1f}" width="{max(1.0, col_w - 1):.1f}" '
            f'height="{row_h - 2:.1f}" fill="{fill}">'
            f'<title>step {i}: {html.escape(f)} ({label}){sbit}{why}</title></rect>')
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


def svg_frontier_overlay(named, target=None, width: int = 640,
                         height: int = 360, palette: str = "default",
                         dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_frontier_overlay` (the pre-v0.37 body); default
    byte-identical (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_frontier_overlay(named, target=target, width=width,
                                     height=height)


def _svg_frontier_overlay(named, target=None, width: int = 640,
                          height: int = 360) -> str:
    """SPEC.md 49.3.3 (v0.35, policy A/B): two (or more) Pareto frontiers
    on one shared axis — the policy A/B overlay.

    ``named`` is a list of ``(name, points)`` pairs whose points carry
    the ``summary["pareto_frontier"]`` shape (``score`` +
    ``train_seconds``/``seconds``). ``x`` is the *actual* train seconds
    (not per-frontier index positions, so different-length frontiers
    align truthfully); ``y`` is score. One colored polyline + point
    circles (with ``<title>``) per frontier from ``_FRONTIER_PALETTE``;
    an optional horizontal target line (only when inside the y-range);
    a legend naming each frontier. Empty/invalid series are skipped and
    all-invalid renders the standard empty header + message (the
    ``svg_pareto`` convention, 21.2). Pure, valid XML, deterministic
    (G2).
    """
    series: list[tuple[str, list[tuple[float, float]]]] = []
    for item in named or []:
        try:
            name, pts = item[0], item[1]
        except (TypeError, IndexError, KeyError):
            continue  # not a (name, points) pair — skip (49.3.3)
        pts = _pareto_points(pts)
        if pts:
            series.append((str(name), pts))
    if not series:
        parts = _svg_header(width, height, "frontier overlay (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no frontier points to overlay</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    allpts = [p for _name, pts in series for p in pts]
    lo, hi = min(p[0] for p in allpts), max(p[0] for p in allpts)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    tlo, thi = min(p[1] for p in allpts), max(p[1] for p in allpts)
    if thi - tlo < 1e-9:
        thi = tlo + 1e-3
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B

    def x(t: float) -> float:
        return L + pw * (t - tlo) / (thi - tlo)

    def y(s: float) -> float:
        return T + ph * (1.0 - (s - lo) / (hi - lo))

    parts = _svg_header(width, height, "pareto frontier overlay (policy A/B)")
    parts += _svg_axes(L, T, pw, ph, lo, hi, tlo, thi, "training seconds")
    if target is not None and _finite(target):  # 49.3.3: inside the y-range only
        ty = y(float(target))
        if lo - 1e-9 <= float(target) <= hi + 1e-9:
            parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" '
                         f'y2="{ty:.1f}" stroke="{_REJECTED}" stroke-width="1" '
                         f'stroke-dasharray="5 4"/>'
                         f'<title>target {float(target):g}</title>')
            parts.append(f'<text x="{L + pw:.1f}" y="{ty - 5:.1f}" text-anchor="end" '
                         f'font-size="11" fill="{_REJECTED}">'
                         f'target {float(target):.2f}</text>')
    for i, (name, pts) in enumerate(series):
        color = _FRONTIER_PALETTE[i % len(_FRONTIER_PALETTE)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{" ".join(f"{x(t):.1f},{y(s):.1f}" for s, t in pts)}"/>')
        for s, t in pts:
            parts.append(f'<circle cx="{x(t):.1f}" cy="{y(s):.1f}" r="4" fill="{color}">'
                         f'<title>{html.escape(name)}: score {s:.2f} @ {t:.3f}s</title>'
                         f'</circle>')
        # the legend: a swatch line + the frontier name, stacked top-right
        ly = T + 10 + i * 16
        parts.append(f'<line x1="{L + pw - 110:.1f}" y1="{ly:.1f}" '
                     f'x2="{L + pw - 90:.1f}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{L + pw - 84:.1f}" y="{ly + 4:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_run_curves(named, target=None, width: int = 640, height: int = 360,
                   palette: str = "default", dark: bool = False):
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_run_curves` (the pre-v0.37 body); default byte-identical
    (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_run_curves(named, target=target, width=width,
                               height=height)


def _svg_run_curves(named, target=None, width: int = 640, height: int = 360):
    """SPEC.md 50.1.3 (v0.36): the N-run score-curve overlay — 2..N
    running-best score curves (50.1.2) on one shared experiment-index
    axis (0..max-1, so different-length runs align by position). The
    score axis is *adaptive* (min..max across the set — cartpole-v1's
    `mean_steps` is [0, 500] and cross-task runs may be compared;
    unlike `svg_seed_curves`' fixed 0-100, 30.1). One
    `_FRONTIER_PALETTE` (49.3.3) polyline + point circles (with
    `<title>`) + a named legend entry per run, in input order; an
    optional horizontal target line (only when inside the y-range);
    empty/all-invalid input renders the empty header + message (the
    21.2 convention). `named` is a list of `(name, curve)` pairs whose
    curves carry >= 1 finite float. Valid XML, pure, deterministic (G2).
    ASCII-only text."""
    series: list[tuple[str, list[float]]] = []
    for item in named or []:
        try:
            name, curve = item[0], item[1]
        except (TypeError, IndexError, KeyError):
            continue  # not a (name, curve) pair — skip (50.1.3)
        pts = [float(v) for v in (curve or []) if _finite(v)]
        if pts:
            series.append((str(name), pts))
    if not series:
        parts = _svg_header(width, height, "run curves (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no run curves to overlay</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    m = max(len(pts) for _name, pts in series)
    lo = min(min(pts) for _name, pts in series)
    hi = max(max(pts) for _name, pts in series)
    if hi - lo < 1e-9:
        hi = lo + 1e-3
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B

    def x(i: int) -> float:
        return L + pw * (i / (m - 1) if m > 1 else 0.5)

    def y(s: float) -> float:
        return T + ph * (1.0 - (float(s) - lo) / (hi - lo))

    parts = _svg_header(width, height, "best score per run (N-run compare)")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(m - 1) if m > 1 else 1.0,
                       "experiment index")
    if target is not None and _finite(target):  # 50.1.3: inside the y-range only
        t = float(target)
        if lo - 1e-9 <= t <= hi + 1e-9:
            ty = y(t)
            parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" '
                         f'y2="{ty:.1f}" stroke="{_REJECTED}" stroke-width="1" '
                         f'stroke-dasharray="5 4"/>'
                         f'<title>target {t:g}</title>')
            parts.append(f'<text x="{L + pw:.1f}" y="{ty - 5:.1f}" text-anchor="end" '
                         f'font-size="11" fill="{_REJECTED}">'
                         f'target {t:.2f}</text>')
    for i, (name, pts) in enumerate(series):
        color = _FRONTIER_PALETTE[i % len(_FRONTIER_PALETTE)]
        xs = [x(j) for j in range(len(pts))]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{" ".join(f"{xx:.1f},{y(v):.1f}" for xx, v in zip(xs, pts))}"/>'
                     f'<title>{html.escape(name)}</title>')
        for xx, v in zip(xs, pts):
            parts.append(f'<circle cx="{xx:.1f}" cy="{y(v):.1f}" r="4" fill="{color}">'
                         f'<title>{html.escape(name)}: {v:.2f}</title>'
                         f'</circle>')
        # the legend: a swatch line + the run name, stacked top-right (49.3.3)
        ly = T + 10 + i * 16
        parts.append(f'<line x1="{L + pw - 110:.1f}" y1="{ly:.1f}" '
                     f'x2="{L + pw - 90:.1f}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{L + pw - 84:.1f}" y="{ly + 4:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_live_sparkline(best_series, target=None, width: int = 220, height: int = 48,
                        live: bool = True, reduced_motion: bool = False) -> str:
    """55.1 (v0.41) + 56.5 (v0.42): the live best-score sparkline.
    `best_series` is the running-best series (the baseline seeds point 0,
    like `running_best_curve`); the last point is drawn as a *pulse* — a
    larger flash dot with a soft halo — when `live=True` and
    `reduced_motion=False`, a plain dot otherwise (55.1.2; the
    reduced-motion pass, 56.5 — the pulse is the app's one animated
    affordance, and this flag is its accessibility opt-out). An optional
    horizontal target line (inside the y-range only). Empty input renders
    the empty header + message (the 21.2 convention). Valid XML, pure,
    deterministic (G2); ASCII-only text."""
    pts = [float(v) for v in (best_series or []) if _finite(v)]
    if not pts:
        parts = _svg_header(width, height, "best score (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" '
                     f'text-anchor="middle" font-size="12" fill="{_AXIS}">'
                     f'no experiments yet</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    lo, hi = min(pts), max(pts)
    if target is not None and _finite(target):
        lo = min(lo, float(target)); hi = max(hi, float(target))
    if hi - lo < 1e-9:
        hi = lo + 1e-3
    L, T, R, B = 8.0, 22.0, 8.0, 8.0
    pw, ph = width - L - R, height - T - B

    def x(i: int) -> float:
        return L + pw * (i / (len(pts) - 1) if len(pts) > 1 else 0.5)

    def y(s: float) -> float:
        return T + ph * (1.0 - (float(s) - lo) / (hi - lo))

    parts = _svg_header(width, height, "best score (live sparkline)")
    if (target is not None and _finite(target)
            and lo - 1e-9 <= float(target) <= hi + 1e-9):
        ty = y(target)
        parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" '
                     f'y2="{ty:.1f}" stroke="{_REJECTED}" stroke-width="1" '
                     f'stroke-dasharray="4 3"><title>target {float(target):g}'
                     f'</title></line>')
    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(pts))
    parts.append(f'<polyline fill="none" stroke="{_LINE}" stroke-width="2" '
                 f'points="{line}"/>')
    lx, ly = x(len(pts) - 1), y(pts[-1])
    last = f"current best {pts[-1]:.2f}" + (" (live)" if live else "")
    if live and not reduced_motion:  # 56.5: the reduced-motion pass
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="7" '
                     f'fill="{_ACCEPTED}" opacity="0.28"><title>{last}'
                     f'</title></circle>')
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" '
                 f'fill="{_ACCEPTED}"><title>{last}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_live_sparkline(best_series, target=None, width: int = 220, height: int = 48,
                       live: bool = True, palette: str = "default",
                       dark: bool = False, reduced_motion: bool = False) -> str:
    """55.1 (v0.41) + 56.5 (v0.42): the live best-score sparkline with a
    pulsing last point (a flash when `live=True`, a static dot when
    `live=False` — and a static dot when `reduced_motion=True`, the
    accessibility opt-out for the pulse, 56.5). `palette`/`dark` per
    SPEC.md 51.4; the default call is byte-identical to an explicit one
    (G2); ARIA always on (51.4.3). Pure, valid XML."""
    with _styled(palette, dark):
        return _svg_live_sparkline(best_series, target=target, width=width,
                                   height=height, live=live,
                                   reduced_motion=reduced_motion)


def _svg_reference_curve(current, reference, target=None, width: int = 640,
                         height: int = 360) -> str:
    """55.3 (v0.41): the reference-run overlay — the current run's
    best-score curve (solid, `_ACCEPTED`, on top) with a past run's
    best-score curve (`reference`) drawn faintly beneath it (dashed,
    `_REJECTED`, greyed, `opacity 0.75`), on one shared experiment-index
    axis (0..max-1, so different-length runs align by position) with an
    adaptive y-range over both (plus the target when given). This is the
    live-curve analogue of `svg_run_curves` (50.1.3) with one borrowed
    curve. An empty `current` renders the empty header + message (21.2);
    an empty `reference` (with a non-empty `current`) renders only the
    current curve. ARIA, valid XML, pure, deterministic (G2)."""
    cur = [float(v) for v in (current or []) if _finite(v)]
    ref = [float(v) for v in (reference or []) if _finite(v)]
    if not cur:
        parts = _svg_header(width, height, "reference overlay (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" '
                     f'text-anchor="middle" font-size="13" fill="{_AXIS}">'
                     f'no live curve to overlay</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    m = max(len(cur), len(ref))
    lo = min([min(cur)] + ([min(ref)] if ref else []))
    hi = max([max(cur)] + ([max(ref)] if ref else []))
    if target is not None and _finite(target):
        lo = min(lo, float(target)); hi = max(hi, float(target))
    if hi - lo < 1e-9:
        hi = lo + 1e-3
    L, T, R, B = 72.0, 28.0, 24.0, 44.0
    pw, ph = width - L - R, height - T - B

    def x(i: int) -> float:
        return L + pw * (i / (m - 1) if m > 1 else 0.5)

    def y(s: float) -> float:
        return T + ph * (1.0 - (float(s) - lo) / (hi - lo))

    parts = _svg_header(width, height, "live best vs reference run")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0,
                       float(m - 1) if m > 1 else 1.0, "experiment index")
    if (target is not None and _finite(target)
            and lo - 1e-9 <= float(target) <= hi + 1e-9):
        ty = y(target)
        parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" '
                     f'y2="{ty:.1f}" stroke="{_REJECTED}" stroke-width="1" '
                     f'stroke-dasharray="5 4"><title>target '
                     f'{float(target):g}</title></line>')
    if ref:  # the reference: drawn first (beneath), dashed + faint
        xs = [x(i) for i in range(len(ref))]
        parts.append(f'<polyline fill="none" stroke="{_REJECTED}" '
                     f'stroke-width="1.5" stroke-dasharray="5 4" opacity="0.75" '
                     f'points="{" ".join(f"{a:.1f},{y(v):.1f}"
                                          for a, v in zip(xs, ref))}">'
                     f'<title>reference (past run)</title></polyline>')
    xs = [x(i) for i in range(len(cur))]  # the current: on top, solid
    parts.append(f'<polyline fill="none" stroke="{_ACCEPTED}" '
                 f'stroke-width="2.5" points="{" ".join(f"{a:.1f},{y(v):.1f}"
                                                         for a, v in zip(xs, cur))}">'
                 f'<title>current run</title></polyline>')
    for a, v in zip(xs, cur):
        parts.append(f'<circle cx="{a:.1f}" cy="{y(v):.1f}" r="3.5" '
                     f'fill="{_ACCEPTED}"><title>current: {v:.2f}</title>'
                     f'</circle>')
    ly = T + 10
    parts.append(f'<line x1="{L + pw - 110:.1f}" y1="{ly:.1f}" '
                 f'x2="{L + pw - 90:.1f}" y2="{ly:.1f}" stroke="{_ACCEPTED}" '
                 f'stroke-width="3"/><text x="{L + pw - 84:.1f}" '
                 f'y="{ly + 4:.1f}" font-size="11" fill="{_AXIS}">'
                 f'current</text>')
    if ref:
        ry = T + 26
        parts.append(f'<line x1="{L + pw - 110:.1f}" y1="{ry:.1f}" '
                     f'x2="{L + pw - 90:.1f}" y2="{ry:.1f}" '
                     f'stroke="{_REJECTED}" stroke-width="3" '
                     f'stroke-dasharray="5 4"/><text x="{L + pw - 84:.1f}" '
                     f'y="{ry + 4:.1f}" font-size="11" fill="{_AXIS}">'
                     f'reference</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_reference_curve(current, reference, target=None, width: int = 640,
                        height: int = 360, palette: str = "default",
                        dark: bool = False) -> str:
    """55.3 (v0.41): the reference-run overlay — the current best-score
    curve with a past run's curve drawn faintly beneath it. `palette`/
    `dark` per SPEC.md 51.4; the default call is byte-identical to an
    explicit one (G2); ARIA always on (51.4.3). Pure, valid XML."""
    with _styled(palette, dark):
        return _svg_reference_curve(current, reference, target=target,
                                    width=width, height=height)


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
            count = int(conf[i][j])  # 54.1 (v0.40): the cell's hover title
            row_total = sum(int(conf[i][c]) for c in range(k))
            pct = (count / row_total * 100.0) if row_total else 0.0
            cell_title = (f"true {html.escape(str(labels[i]))} \u2192 predicted "
                          f"{html.escape(str(labels[j]))}: {count} "
                          f"({pct:.1f}% of row)")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{fill}" stroke="{_AXIS}">'
                         f'<title>{cell_title}</title></rect>')
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


def svg_architecture(spec, state_dim, n_out, width: int = 640,
                     highlight: str | None = None) -> str:
    """C4 (SPEC.md 28.4): a small horizontal annotated block diagram of one
    ModelSpec — an input block (state_dim) -> the family body -> an output
    block (n_out), plus an annotations row (optimizer always; LR schedule,
    early stopping, gradient clipping, label smoothing when non-default).

    Per family: mlp one block per hidden layer (depth-0 -> "linear");
    convnet the verified conv/pool/FC pipeline with its c1/c2; tree "N trees
    (depth D), bagged"; boost "N rounds (depth D), boosted" (in both
    `N = train_steps // 100` clamped 2..50 as in the trainer,
    `D = architecture[0]`); knn "k = knn_k". ASCII-only, pure, valid XML,
    deterministic (G2).

    SPEC.md 53.3 (v0.39): `highlight` (default ``None`` -> byte-identical,
    G2) names a ModelSpec field; the body block(s) the field drives get an
    amber ring (per family), while a field that drives no body block
    (the annotation-row fields — optimizer / lr schedule / early stopping
    / gradient clipping / label smoothing — or an unknown field) gets a
    `highlight: <field>` line under the annotations."""
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

    # 53.3 (v0.39): the highlight mapping (None -> byte-identical, G2)
    hl = str(highlight).strip().lower() if highlight is not None else None
    _NOTE_FIELDS = ("optimizer", "lr_schedule", "early_stopping_patience",
                    "gradient_clipping", "label_smoothing")
    _BLOCK_BY_FAMILY = {
        "mlp": ("architecture", "activation", "hidden_dim"),
        "convnet": ("architecture",),
        "tree": ("architecture", "train_steps"),
        "boost": ("architecture", "train_steps"),
        "knn": ("knn_k",),
    }
    ring = hl is not None and (
        hl == "model_family" or hl in _BLOCK_BY_FAMILY.get(family, ()))
    hl_note = hl is not None and not ring

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

    def block(x: float, title: str, detail: str, fill: str,
              body: bool = False) -> None:
        parts.append(f'<rect x="{x:.1f}" y="{T:.1f}" width="{bw}" height="{bh}" '
                     f'fill="{fill}" stroke="{_AXIS}" rx="4"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{T + 22:.1f}" text-anchor="middle" '
                     f'font-size="12" fill="{_AXIS}">{html.escape(title)}</text>')
        if detail:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{T + 40:.1f}" '
                         f'text-anchor="middle" font-size="10" fill="{_AXIS}">'
                         f'{html.escape(detail)}</text>')
        if body and ring:  # 53.3: the highlight ring (gated -> default unchanged)
            parts.append(f'<rect x="{x - 2:.1f}" y="{T - 2:.1f}" '
                         f'width="{bw + 4:.0f}" height="{bh + 4:.0f}" '
                         f'fill="none" stroke="{_HIGHLIGHT}" stroke-width="2.5"/>')

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
        block(x, title, detail, "#eef2f7", body=True)
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
    if hl_note:  # 53.3: the annotation-row highlight (gated -> default unchanged)
        parts.append(f'<text x="{width / 2:.1f}" y="{T + bh + 46:.1f}" '
                     f'text-anchor="middle" font-size="11" fill="{_HIGHLIGHT}">'
                     f'highlight: {html.escape(str(highlight))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --- v0.39 per-candidate decision views (SPEC.md 53) --------------------------

def _gate_note(candidate, best, z_se, accepted, reason) -> tuple[str, str]:
    """53.1 (v0.39): the number-line annotation — the 52.1.1 gate-math
    numbers plus the 51.3.2 reason word ("missed by 4.2", "accepted
    +3.1 over best", "inside CI band", "over gen-gap tolerance").
    ASCII-only; the reason is caller-supplied (51.3.1) so the core stays
    free of the accounting import."""
    r = str(reason or "").strip() or "score"
    c = float(candidate) if _finite(candidate) else None
    b = float(best) if _finite(best) else None
    delta = (c - b) if (c is not None and b is not None) else None
    if c is None:
        return (f"unscored ({r}) — no candidate dot (R3)", _REJECTED)
    if accepted:
        if delta is not None and b is not None:
            return (f"accepted: {delta:+.2f} over best {b:.2f}", _ACCEPTED)
        return ("accepted", _ACCEPTED)
    if r == "stopped":
        note = "stopped — final step"
        if delta is not None and b is not None:
            note = f"stopped — final step ({delta:+.2f} vs best {b:.2f})"
        return (note, _REJECTED)
    if delta is not None and delta <= 0.0 and b is not None:
        return (f"missed by {-delta:.2f} (score {c:.2f} vs best {b:.2f})",
                _SCORED_REJ)
    if r == "overfit":
        if delta is not None:
            return (f"above best by {delta:.2f}, over gen-gap tolerance (18.5)",
                    _SCORED_REJ)
        return ("over gen-gap tolerance (18.5)", _SCORED_REJ)
    if r == "ci":
        if delta is not None and z_se > 0.0:
            return (f"above best by {delta:.2f}, inside CI band (z*SE {z_se:.2f})",
                    _SCORED_REJ)
        return ("inside CI band (18.6)", _SCORED_REJ)
    if delta is not None and b is not None:
        return (f"rejected ({r}): {delta:+.2f} vs best {b:.2f}", _SCORED_REJ)
    return (f"rejected ({r})", _SCORED_REJ)


def _svg_gate_line(candidate, best_before, target, z_se, accepted, reason,
                   width: int) -> str:
    """53.1 (v0.39): the gate number-line body — a fixed 0-100 axis (the
    27.1 strip's tick pattern), vertical markers for the running best and
    the target, the CI band [best, best + z*SE] shaded when z*SE > 0
    (the 18.6 gate), the candidate's dot, and the margin annotation
    (52.1.1 numbers + the 51.3.2 reason word). An unscored (dup)
    candidate renders best/target markers + the honest "unscored"
    caption and no dot (52.1.1: the gate math is undefined when its
    input is not finite)."""
    c = float(candidate) if _finite(candidate) else None
    b = float(best_before) if _finite(best_before) else None
    t = float(target) if _finite(target) else None
    zse = float(z_se) if _finite(z_se) and float(z_se) > 0.0 else 0.0
    L = 24.0
    R = 24.0
    pw = width - L - R
    y0 = 96.0  # the axis line
    T = 56.0   # the marker tops
    height = 170
    parts = _svg_header(width, height, "gate number-line")

    def x(v: float) -> float:
        return L + pw * min(max(float(v), 0.0), 100.0) / 100.0

    if b is not None and zse > 0.0:  # the CI band (SPEC.md 18.6)
        x1, x2 = x(b), x(b + zse)
        parts.append(f'<rect x="{min(x1, x2):.1f}" y="{T:.1f}" '
                     f'width="{abs(x2 - x1):.1f}" height="{y0 - T:.1f}" '
                     f'fill="{_BAND}" fill-opacity="0.35">'
                     f'<title>CI band: best + z*SE (SPEC.md 18.6)</title></rect>')
    if b is not None:  # the running best
        parts.append(f'<line x1="{x(b):.1f}" y1="{T:.1f}" x2="{x(b):.1f}" '
                     f'y2="{y0:.1f}" stroke="{_BASELINE}" stroke-width="2"/>')
        parts.append(f'<text x="{x(b):.1f}" y="{T - 8:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="{_BASELINE}">best {b:.2f}</text>')
    if t is not None:  # the target (dashed)
        parts.append(f'<line x1="{x(t):.1f}" y1="{T:.1f}" x2="{x(t):.1f}" '
                     f'y2="{y0:.1f}" stroke="{_LADDER}" stroke-width="2" '
                     f'stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{T - 8:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="{_LADDER}">target {t:.1f}</text>')
    if c is not None:  # the candidate dot
        fill = _ACCEPTED if accepted else _SCORED_REJ
        parts.append(f'<circle cx="{x(c):.1f}" cy="{(T + y0) / 2:.1f}" r="6" '
                     f'fill="{fill}">'
                     f'<title>candidate {c:.2f} ({reason or "?"})</title></circle>')
    parts.append(f'<line x1="{L:.1f}" y1="{y0:.1f}" x2="{L + pw:.1f}" '
                 f'y2="{y0:.1f}" stroke="{_AXIS}"/>')
    for s in (0, 25, 50, 75, 100):
        xx = L + pw * s / 100.0
        parts.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" '
                     f'y2="{y0 + 4:.1f}" stroke="{_AXIS}"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="{_AXIS}">{s}</text>')
    note, ncolor = _gate_note(candidate, best_before, zse, accepted, reason)
    parts.append(f'<text x="{width / 2:.1f}" y="{y0 + 38:.1f}" '
                 f'text-anchor="middle" font-size="11" fill="{ncolor}">'
                 f'{html.escape(note)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_gate_line(candidate, best_before, target, z_se: float = 0.0,
                  accepted: bool = False, reason: str = "",
                  palette: str = "default", dark: bool = False,
                  width: int = 520) -> str:
    """SPEC.md 53.1 (v0.39): the per-candidate gate number-line — turns
    the reason column's "no" into a picture (53.1.1): a 0-100 axis with
    the best, the target, the CI band, and the candidate's dot, annotated
    with the margin. ``palette``/``dark`` per SPEC.md 51.4; the default
    is byte-identical (G2); ARIA always on (51.4.3). Pure, deterministic
    (G2); ASCII-only text."""
    with _styled(palette, dark):
        return _svg_gate_line(candidate, best_before, target, z_se,
                              accepted, reason, width)


def _frontier_rows(updates):
    """53.2 (v0.39): (step, score, train_seconds) for the scored updates
    that carry a finite train time (the G2 scatter's data contract);
    unscored (dup) steps contribute no points."""
    rows = []
    for i, u in enumerate(updates or [], start=1):
        if not isinstance(u, dict):
            continue
        s, t = u.get("candidate_score"), u.get("train_seconds")
        if _finite(s) and _finite(t):
            rows.append((i, float(s), float(t)))
    return rows


def _dominated(rows) -> set[int]:
    """53.2 (v0.39): the dominated indices — A dominates B iff
    A.score >= B.score and A.train <= B.train and A is strict in at
    least one coordinate. Pure, deterministic (G2)."""
    out: set[int] = set()
    for i, (_, si, ti) in enumerate(rows):
        for j, (_, sj, tj) in enumerate(rows):
            if i == j:
                continue
            if sj >= si and tj <= ti and (sj > si or tj < ti):
                out.add(i)
                break
    return out


def _svg_live_frontier(updates, width: int, height: int) -> str:
    """53.2 (v0.39): the live Pareto frontier body — the score-vs-train-
    time scatter that grows each step: frontier candidates in the accent
    color, dominated ones greyed, the latest candidate drawn as a flash
    (larger dot + halo ring). Empty -> header + message (the other
    empty cases' pattern)."""
    rows = _frontier_rows(updates)
    if not rows:
        parts = _svg_header(width, height, "live Pareto frontier (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no scored candidates with a train time yet</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    dom = _dominated(rows)
    L, T = 56.0, 40.0
    pw = width - L - 24.0
    ph = height - T - 64.0
    x_hi = max(t for _, _, t in rows)
    if x_hi <= 0.0:
        x_hi = 1.0
    parts = _svg_header(width, height, "live Pareto frontier")
    parts += _svg_axes(L, T, pw, ph, 0.0, 100.0, 0.0, x_hi, "train time (s)")

    def px(t: float) -> float:
        return L + pw * float(t) / x_hi

    def py(s: float) -> float:
        return T + ph * (1.0 - min(max(float(s), 0.0), 100.0) / 100.0)

    latest = len(rows) - 1
    for i, (step, s, t) in enumerate(rows):
        if i == latest:  # drawn below as the flash
            continue
        if i in dom:
            fill, r = _REJECTED, 4.0
        else:
            fill, r = _LINE, 5.0
        parts.append(f'<circle cx="{px(t):.1f}" cy="{py(s):.1f}" r="{r}" '
                     f'fill="{fill}">'
                     f'<title>candidate {step}: score {s:.2f}, {t:.2f}s '
                     f'({"dominated" if i in dom else "frontier"})</title></circle>')
    step, s, t = rows[latest]  # the latest candidate: flash (53.2.2)
    parts.append(f'<circle cx="{px(t):.1f}" cy="{py(s):.1f}" r="10" fill="none" '
                 f'stroke="{_BASELINE}" stroke-width="2"/>')
    parts.append(f'<circle cx="{px(t):.1f}" cy="{py(s):.1f}" r="7" '
                 f'fill="{_ACCEPTED}">'
                 f'<title>latest candidate {step}: score {s:.2f}, {t:.2f}s'
                 f'</title></circle>')
    ly = T + ph + 50.0  # legend (below the 34px x-unit line)
    lx = L
    for text, fill in (("frontier", _LINE), ("dominated", _REJECTED),
                       ("latest", _ACCEPTED)):
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="10" height="10" '
                     f'fill="{fill}"/>')
        parts.append(f'<text x="{lx + 14:.1f}" y="{ly:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{text}</text>')
        lx += 14 + 7 * len(text) + 20
    parts.append("</svg>")
    return "\n".join(parts)


def svg_live_frontier(updates, palette: str = "default", dark: bool = False,
                      width: int = 560, height: int = 340) -> str:
    """SPEC.md 53.2 (v0.39): the live Pareto frontier — the
    score-vs-train-time scatter that grows each step, the new candidate
    drawn as a flash, dominated ones greyed (53.2.1/53.2.2). The
    frontier data (`candidate_score` + `train_seconds`) is already in
    every update (23.1). ``palette``/``dark`` per SPEC.md 51.4; the
    default is byte-identical (G2); ARIA always on (51.4.3). Pure,
    deterministic (G2); ASCII-only text."""
    with _styled(palette, dark):
        return _svg_live_frontier(updates, width, height)


def _time_segments(updates, baseline_seconds=None):
    """54.2 (v0.40): the wall-time segments — the baseline first, then one
    per update that has a finite, non-negative `train_seconds`.

    `updates` are the runner's update dicts (SPEC.md 23.1); a step whose
    `train_seconds` is missing / None / non-finite (the free duplicate
    rejections, R3) contributes no segment. `baseline_seconds` (the reset
    baseline's train time, SPEC.md 54.2) is the first segment when finite
    and non-negative; `None` omits it. Pure, deterministic (G2)."""
    segs = []
    if _finite(baseline_seconds) and float(baseline_seconds) >= 0.0:
        segs.append(("baseline", float(baseline_seconds)))
    for i, u in enumerate(updates or [], start=1):
        if not isinstance(u, dict):
            continue
        t = u.get("train_seconds")
        if _finite(t) and float(t) >= 0.0:
            segs.append((f"exp {i}", float(t)))
    return segs


def _svg_time_strip(updates, baseline_seconds=None, width: int = 640) -> str:
    """54.2 (v0.40): the wall-time cost strip — a single horizontal stacked
    bar showing where the train time went: the baseline segment first
    (orange, `_BASELINE`), then one blue (`_LINE`) segment per update that
    has a finite train time. Every segment carries a hover `<title>`
    (SPEC.md 54.1) with its label, seconds, and % of the total.

    Edges: the empty case (no finite segments) renders a header + message;
    a zero total renders equal-width segments whose titles read `0.00 s`;
    tiny segments keep a 1.0 px minimum width so they stay hoverable.
    Pure, valid XML, deterministic (G2), ASCII-safe text."""
    segs = _time_segments(updates, baseline_seconds)
    if not segs:
        height = 104
        parts = _svg_header(width, height, "wall-time cost strip (empty)")
        parts.append(
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'font-size="13" fill="{_AXIS}">no train-time data in the update stream</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, R = 24.0, 24.0
    pw = width - L - R
    bar_y, bar_h = 44.0, 30.0
    total = sum(t for _, t in segs)
    parts = _svg_header(width, 104, "wall-time cost strip")
    x = L
    for label, t in segs:
        frac = (t / total) if total > 0.0 else 1.0 / len(segs)
        seg_w = max(1.0, pw * frac)
        fill = _BASELINE if label == KIND_BASELINE else _LINE  # 35.1 (C4)
        pct = (t / total * 100.0) if total > 0.0 else 100.0 / len(segs)
        title = f"{label}: {t:.2f} s ({pct:.1f}%)"
        parts.append(f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{seg_w:.1f}" '
                     f'height="{bar_h:.1f}" fill="{fill}" stroke="{_AXIS}">'
                     f'<title>{title}</title></rect>')
        x += seg_w
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{bar_y + bar_h + 20:.1f}" '
                 f'text-anchor="middle" font-size="12" fill="{_AXIS}">'
                 f'total {total:.2f} s</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_time_strip(updates, baseline_seconds=None, width: int = 640,
                   palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 54.2 (v0.40): the wall-time cost strip — a single horizontal
    stacked bar (the baseline first, then one segment per update with a
    finite train time), each segment carrying a hover `<title>` (SPEC.md
    54.1) with its label, seconds, and share of the total. ``palette`` /
    ``dark`` per SPEC.md 51.4; the default is byte-identical (G2); ARIA
    always on (51.4.3). Pure, deterministic (G2); ASCII-safe text."""
    with _styled(palette, dark):
        return _svg_time_strip(updates, baseline_seconds, width=width)


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
                      height: int = 360, palette: str = "default",
                      dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_seed_variance` (the pre-v0.37 body); default byte-identical
    (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_seed_variance(seeds, target=target, width=width,
                                  height=height)


def _svg_seed_variance(seeds, target=None, width: int = 640,
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


# --- v0.16 comprehension visuals II (SPEC.md 30) -------------------------------

def svg_seed_curves(seeds, target=None, width: int = 640, height: int = 360,
                    palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 51.4 (v0.37): the palette (51.4.1) + dark (51.4.2) params
    over `_svg_seed_curves` (the pre-v0.37 body); default byte-identical
    (G2); ARIA always on (51.4.3)."""
    with _styled(palette, dark):
        return _svg_seed_curves(seeds, target=target, width=width,
                                height=height)


def _svg_seed_curves(seeds, target=None, width: int = 640,
                     height: int = 360) -> str:
    """V1 (SPEC.md 30.1): the running best score of each seed over the
    sweep — one polyline per seed (input order) on a fixed 0-100 score
    axis, so the variance view (SPEC.md 29.1) answers *when* the seeds
    diverge (early noise vs late divergence). A dashed target line when
    given, a per-seed legend, and the `n seeds / steps` summary.
    `seeds` is a list of dicts each carrying `curve` (a list of >= 1
    finite floats; point 0 = the baseline) and optionally `seed` (int).
    Rows without a finite `curve` are skipped. Pure, valid XML,
    deterministic (G2); empty -> header + message. ASCII-only text."""
    rows = []
    for s in seeds or []:
        if not isinstance(s, dict):
            continue
        curve = [float(v) for v in (s.get("curve") or []) if _finite(v)]
        if curve:
            rows.append({"seed": s.get("seed"), "curve": curve})
    if not rows:
        parts = _svg_header(width, height, "seed curves (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no seed curves in the sweep</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    m = max(len(r["curve"]) for r in rows)
    L, R, T, B = 72.0, 24.0, 28.0, 78.0
    pw, ph = width - L - R, height - T - B

    def x(i: int) -> float:
        return L + pw * (i / (m - 1) if m > 1 else 0.5)

    def y(s: float) -> float:
        return T + ph * (1.0 - min(max(float(s), 0.0), 100.0) / 100.0)

    def label(i: int) -> str:
        seed = rows[i]["seed"]
        return f"seed {seed}" if seed is not None else f"seed {i}"

    has_target = _finite(target) and not isinstance(target, bool)
    palette = [_LINE, _BASELINE, _LADDER, _ACCEPTED, _SCORED_REJ, _REJECTED]
    parts = _svg_header(width, height, "best score per seed over the sweep")
    # fixed 0-100 score axis (left) + faint gridlines, like svg_seed_variance
    parts.append(f'<line x1="{L:.1f}" y1="{T:.1f}" x2="{L:.1f}" y2="{T + ph:.1f}" stroke="{_AXIS}"/>')
    for s in (0, 25, 50, 75, 100):
        yy = y(s)
        parts.append(f'<line x1="{L - 4:.1f}" y1="{yy:.1f}" x2="{L + pw:.1f}" y2="{yy:.1f}" '
                     f'stroke="#eef1f4"/>')
        parts.append(f'<text x="{L - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_AXIS}">{s}</text>')
    if has_target:
        ty = y(float(target))
        parts.append(f'<line x1="{L:.1f}" y1="{ty:.1f}" x2="{L + pw:.1f}" y2="{ty:.1f}" '
                     f'stroke="{_BASELINE}" stroke-dasharray="6 4" stroke-width="2"/>')
        parts.append(f'<text x="{L + pw:.1f}" y="{ty - 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{_BASELINE}">target {float(target):.2f}</text>')
    # one polyline per seed, input order (SPEC.md 30.1)
    for i, r in enumerate(rows):
        color = palette[i % len(palette)]
        xs = [x(j) for j in range(len(r["curve"]))]
        ys = [y(v) for v in r["curve"]]
        lab = label(i)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{" ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in zip(xs, ys))}">'
                     f'<title>{html.escape(lab)}</title></polyline>')
        for xx, yy, v in zip(xs, ys, r["curve"]):
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3" fill="{color}">'
                         f'<title>{html.escape(lab)}: {v:.2f}</title></circle>')
    # legend (swatch + `seed <n>` per line, input order)
    lx, ly = L, T + ph + 26.0
    for i in range(len(rows)):
        color = palette[i % len(palette)]
        lab = label(i)
        parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="12" height="12" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{ly + 10:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{html.escape(lab)}</text>')
        lx += 16 + 7 * len(lab) + 26
    summary = f"n seeds = {len(rows)} - steps = {m}"
    parts.append(f'<text x="{width // 2}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="12" fill="{_AXIS}">{html.escape(summary)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _fv_sort_key(k) -> tuple:
    """V2 (SPEC.md 30.2): matrix cells sort numeric-first — a value that
    float-parses sorts by its number, else alphabetically (the same order
    `field_value_stats` in dashboard.py returns)."""
    try:
        return (0.0, float(k), "")
    except (TypeError, ValueError):
        return (1.0, 0.0, str(k))


def svg_field_value_matrix(stats, width: int = 640, row_h: int = 28) -> str:
    """V2 (SPEC.md 30.2): the field x value win matrix — one row per field
    (sorted), one cell per value (numeric-first, `_fv_sort_key`): a green cell
    (`_ACCEPTED`) with
    `fill-opacity = 0.10 + 0.90*win_rate`, the `wins/trials` text centered
    (white when `win_rate >= 0.5`, axis color otherwise), and a `<title>`
    `field value: wins/trials (rate)` per cell. Field names sit left of the
    cells; cell widths shrink to fit the widest row. `stats` is the
    `field_value_stats` shape ({field: {value: {trials, wins, win_rate}}}).
    Pure, valid XML, deterministic (G2); empty -> header + message."""
    fields = sorted(f for f in (stats or {})
                    if isinstance(stats[f], dict) and stats[f])
    if not fields:
        height = 120
        parts = _svg_header(width, height, "field x value matrix (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no field values credited</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, R = 120.0, 24.0
    pw = width - L - R
    n_rows = len(fields)
    max_vals = max(len(stats[f]) for f in fields)
    cell_w = min(110.0, pw / max(1, max_vals))
    T = 34.0
    bottom = T + n_rows * row_h
    height = int(bottom + 44)
    parts = _svg_header(width, height, "field x value win matrix")
    for fi, field in enumerate(fields):
        vals = stats[field]
        y0 = T + fi * row_h
        parts.append(f'<text x="{L - 10:.1f}" y="{y0 + row_h * 0.68:.1f}" '
                     f'text-anchor="end" font-size="12" fill="{_AXIS}">'
                     f'{html.escape(str(field))}</text>')
        for vi, value in enumerate(sorted(vals, key=_fv_sort_key)):
            cell = vals[value]
            trials = int(cell.get("trials", 0) or 0)
            wins = float(cell.get("wins", 0.0) or 0.0)
            rate = min(max(float(cell.get("win_rate", 0.0) or 0.0), 0.0), 1.0)
            x0 = L + vi * cell_w
            opacity = 0.10 + 0.90 * rate
            tcolor = "white" if rate >= 0.5 else _AXIS
            title = f"{field} {value}: {wins:.0f}/{trials} ({rate:.2f})"
            parts.append(f'<rect x="{x0 + 1:.1f}" y="{y0 + 2:.1f}" '
                         f'width="{max(1.0, cell_w - 2):.1f}" height="{row_h - 4:.1f}" '
                         f'fill="{_ACCEPTED}" fill-opacity="{opacity:.3f}">'
                         f'<title>{html.escape(title)}</title></rect>')
            parts.append(f'<text x="{x0 + cell_w / 2:.1f}" y="{y0 + row_h * 0.62:.1f}" '
                         f'text-anchor="middle" font-size="11" fill="{tcolor}">'
                         f'{wins:.0f}/{trials}</text>')
    parts.append(f'<text x="{L:.1f}" y="{height - 8}" font-size="11" fill="{_AXIS}">'
                 f'cell = wins/trials · fill = win rate</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_decision_boundary(data, width: int = 640, height: int = 560) -> str:
    """V3 (SPEC.md 30.3): the decision-boundary scatter for a 2-feature
    classification task — the `n_grid x n_grid` prediction grid painted as
    class-colored cells (`fill-opacity 0.30`, the 8-color class palette
    cycling for `k > 8`), each with a `<title>` `grid <i>,<j> -> class <k>`;
    the holdout points on top — correct as green (`_ACCEPTED`) circles,
    misclassified as red (`_SCORED_REJ`), each with a `<title>`
    `true <label> -> pred <label>`; value ticks on both axes; a legend row
    (class swatches + correct/misclassified swatches + the `x:` / `y:`
    feature names); and the summary `n holdout = N - correct = C (P%)`.
    `data` is the `decision_boundary` dict (SPEC.md 30.3); None renders
    header + message. Pure, valid XML, deterministic (G2)."""
    if not isinstance(data, dict) or not data.get("grid_preds"):
        parts = _svg_header(width, height, "decision boundary (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no 2-feature classification data</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    n = max(1, int(data.get("n_grid", 0) or 0))
    preds = [int(p) for p in data["grid_preds"]]
    classes = [str(c) for c in (data.get("classes") or ["0"])]
    k = len(classes)
    x0lo, x0hi = (float(v) for v in (data.get("x0_range") or [0.0, 1.0]))
    x1lo, x1hi = (float(v) for v in (data.get("x1_range") or [0.0, 1.0]))
    x0name = str(data.get("x0") or "x0")
    x1name = str(data.get("x1") or "x1")
    L, R, T, B = 72.0, 24.0, 28.0, 92.0
    pw, ph = width - L - R, height - T - B
    span0 = x0hi - x0lo
    span1 = x1hi - x1lo

    def sx(v: float) -> float:  # feature-0 value -> svg x
        return L + pw * (min(max((float(v) - x0lo) / span0, 0.0), 1.0)
                         if span0 > 0 else 0.5)

    def sy(v: float) -> float:  # feature-1 value -> svg y (flipped)
        return T + ph * (1.0 - (min(max((float(v) - x1lo) / span1, 0.0), 1.0)
                               if span1 > 0 else 0.5))

    cell_w, cell_h = pw / n, ph / n
    parts = _svg_header(width, height, "decision boundary (2 features)")
    # painted prediction grid: cell (i, j) = (feature-0 index i, feature-1
    # index j); grid_preds is row-major, index i*n + j (SPEC.md 30.3)
    for i in range(n):
        for j in range(n):
            p = min(max(preds[i * n + j] if i * n + j < len(preds) else 0, 0), k - 1)
            x = L + i * cell_w
            y = T + (n - 1 - j) * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.5:.1f}" '
                         f'height="{cell_h + 0.5:.1f}" '
                         f'fill="{_CLASS_PALETTE[p % len(_CLASS_PALETTE)]}" '
                         f'fill-opacity="0.30"><title>grid {i},{j} -> class {p}</title></rect>')
    # holdout points on top: correct green, misclassified red
    for pt in data.get("points") or []:
        if not isinstance(pt, dict):
            continue
        t = min(max(int(pt.get("true", 0) or 0), 0), k - 1)
        p = min(max(int(pt.get("pred", 0) or 0), 0), k - 1)
        color = _ACCEPTED if t == p else _SCORED_REJ
        title = f"true {classes[t]} -> pred {classes[p]}"
        parts.append(f'<circle cx="{sx(float(pt.get("x", 0.0))):.1f}" '
                     f'cy="{sy(float(pt.get("y", 0.0))):.1f}" r="5" fill="{color}">'
                     f'<title>{html.escape(title)}</title></circle>')
    # axes frame + value ticks (SPEC.md 30.3)
    parts.append(f'<rect x="{L:.1f}" y="{T:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
                 f'fill="none" stroke="{_AXIS}"/>')
    for t in range(5):
        v0 = x0lo + span0 * t / 4
        v1 = x1lo + span1 * t / 4
        parts.append(f'<text x="{L + pw * t / 4:.1f}" y="{T + ph + 16:.1f}" '
                     f'text-anchor="middle" font-size="10" fill="{_AXIS}">{v0:.2f}</text>')
        parts.append(f'<text x="{L - 8:.1f}" y="{T + ph * (1 - t / 4) + 3:.1f}" '
                     f'text-anchor="end" font-size="10" fill="{_AXIS}">{v1:.2f}</text>')
    # legend row: class swatches + correct/misclassified + axis names
    lx, ly = L, T + ph + 30.0
    for ci, cname in enumerate(classes):
        parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="12" height="12" '
                     f'fill="{_CLASS_PALETTE[ci % len(_CLASS_PALETTE)]}"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{ly + 10:.1f}" font-size="11" '
                     f'fill="{_AXIS}">{html.escape(cname)}</text>')
        lx += 16 + 7 * len(cname) + 22
    parts.append(f'<circle cx="{lx + 5:.1f}" cy="{ly + 6:.1f}" r="5" fill="{_ACCEPTED}"/>')
    parts.append(f'<text x="{lx + 15:.1f}" y="{ly + 10:.1f}" font-size="11" '
                 f'fill="{_AXIS}">correct</text>')
    lx += 70
    parts.append(f'<circle cx="{lx + 5:.1f}" cy="{ly + 6:.1f}" r="5" fill="{_SCORED_REJ}"/>')
    parts.append(f'<text x="{lx + 15:.1f}" y="{ly + 10:.1f}" font-size="11" '
                 f'fill="{_AXIS}">misclassified</text>')
    lx += 104
    parts.append(f'<text x="{lx:.1f}" y="{ly + 10:.1f}" font-size="11" fill="{_AXIS}">'
                 f'x: {html.escape(x0name)} · y: {html.escape(x1name)}</text>')
    # summary
    n_ho = int(data.get("n", 0) or 0)
    correct = int(data.get("correct", 0) or 0)
    pct = 100.0 * correct / n_ho if n_ho else 0.0
    parts.append(f'<text x="{width // 2}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="12" fill="{_AXIS}">'
                 f'n holdout = {n_ho} - correct = {correct} ({pct:.0f}%)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_family_bars(families, width: int = 640, row_h: int = 34) -> str:
    """V5 (SPEC.md 30.5): best holdout score per model family — one
    horizontal bar per family on a fixed 0-100 axis (bar length =
    `best_score`), the top family highlighted (`_ACCEPTED`, the rest
    `_LINE`), captioned `name — best (cost, wins/trials)`, faint
    0/25/50/75/100 gridlines, and the legend `best holdout score per model
    family (all logged experiments)`. Each bar's `<title>` is
    `family <name>: best <s> in <secs>s (<wins>/<trials> accepted)`. `families`
    is the `family_stats` list (SPEC.md 30.5); rows without a finite
    `best_score` are skipped; empty -> header + message."""
    rows = [f for f in (families or [])
            if isinstance(f, dict) and _finite(f.get("best_score"))]
    if not rows:
        height = 120
        parts = _svg_header(width, height, "model family bars (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no scored experiments</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    L, R = 24.0, 24.0
    pw = width - L - R
    n = len(rows)
    T = 34.0
    bottom = T + n * row_h
    height = int(bottom + 44)

    def x(s: float) -> float:
        return L + pw * min(max(float(s), 0.0), 100.0) / 100.0

    parts = _svg_header(width, height, "best holdout score per model family")
    for g in (0, 25, 50, 75, 100):
        parts.append(f'<line x1="{x(g):.1f}" y1="{T - 6:.1f}" x2="{x(g):.1f}" '
                     f'y2="{bottom:.1f}" stroke="#eef1f4"/>')
        parts.append(f'<text x="{x(g):.1f}" y="{bottom + 14:.1f}" text-anchor="middle" '
                     f'font-size="10" fill="{_AXIS}">{g}</text>')
    for i, f in enumerate(rows):
        s = float(f["best_score"])
        yb = T + i * row_h
        name = str(f.get("family") or "?")
        secs = float(f.get("best_seconds", 0.0) or 0.0)
        wins = float(f.get("wins", 0.0) or 0.0)
        trials = int(f.get("trials", 0) or 0)
        fill = _ACCEPTED if i == 0 else _LINE  # top family highlighted
        w = max(0.5, x(s) - L)
        title = (f"family {name}: best {s:.2f} in {secs:.1f}s "
                 f"({wins:.0f}/{trials} accepted)")
        parts.append(f'<rect x="{L:.1f}" y="{yb + 8:.1f}" width="{w:.1f}" '
                     f'height="{row_h - 16:.1f}" fill="{fill}">'
                     f'<title>{html.escape(title)}</title></rect>')
        parts.append(f'<text x="{L + w + 6:.1f}" y="{yb + row_h * 0.62:.1f}" '
                     f'font-size="11" fill="{_AXIS}">'
                     f'{html.escape(f"{name} — {s:.2f} ({secs:.1f}s, {wins:.0f}/{trials})")}'
                     f'</text>')
    parts.append(f'<text x="{L:.1f}" y="{height - 8}" font-size="11" fill="{_AXIS}">'
                 f'best holdout score per model family (all logged experiments)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_policy_trace(trace, width: int = 640, height: int = 360) -> str:
    """V6 (SPEC.md 30.6): the RL policy's return-to-go / baseline trace,
    beside the action-probability bars (SPEC.md 29.2) — *why the policy
    explored*, a pure rendering of the existing per-step trace records
    `{task, action, probs, reward}` (no `rl_policy.py` change):
    one reward bar per step (green `r >= 0`, red `r < 0`), a
    return-to-go line + points (`r2g[t] = sum(rewards[t:])`), a dashed
    baseline line (the running mean of rewards up to `t`, inclusive), a
    data-driven y-range covering 0, all rewards, and all return-to-go
    values (all-zero rewards -> fixed `[-1, 1]`), a legend, and the
    summary `n steps = N - final return-to-go = R - mean reward = M`.
    Pure, valid XML, deterministic (G2); empty -> header + message."""
    recs = [t for t in (trace or [])
            if isinstance(t, dict) and _finite(t.get("reward"))]
    if not recs:
        parts = _svg_header(width, height, "policy trace (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
                     f'font-size="13" fill="{_AXIS}">no trace rewards to plot '
                     f'(train with trace=[...])</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    rewards = [float(t["reward"]) for t in recs]
    n = len(rewards)
    r2g = [sum(rewards[t:]) for t in range(n)]  # return-to-go (SPEC.md 30.6)
    baseline = [sum(rewards[:t + 1]) / (t + 1) for t in range(n)]  # running mean
    lo = min(0.0, min(rewards), min(r2g), min(baseline))
    hi = max(0.0, max(rewards), max(r2g), max(baseline))
    if hi - lo < 1e-9:  # all-zero rewards -> the fixed [-1, 1] range
        lo, hi = -1.0, 1.0
    L, T, R, B = 72.0, 28.0, 24.0, 58.0
    pw, ph = width - L - R, height - T - B

    def x(i: int) -> float:
        return L + pw * (i / (n - 1) if n > 1 else 0.5)

    def y(v: float) -> float:
        return T + ph * (1.0 - (float(v) - lo) / (hi - lo))

    parts = _svg_header(width, height, "policy trace: rewards, return-to-go, baseline")
    parts += _svg_axes(L, T, pw, ph, lo, hi, 0.0, float(n - 1), "step")
    # reward bars (green r >= 0, red r < 0)
    bw = max(3.0, pw / max(1, n) * 0.5)
    for i, t in enumerate(recs):
        r = rewards[i]
        color = _ACCEPTED if r >= 0 else _SCORED_REJ
        task = str(t.get("task") or "?")
        y0 = min(y(0.0), y(r))
        parts.append(f'<rect x="{x(i) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" '
                     f'height="{max(1.0, abs(y(r) - y(0.0))):.1f}" fill="{color}" '
                     f'fill-opacity="0.55"><title>step {i} · {task} · r={r:.3f}</title></rect>')
    # return-to-go line + points
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(r2g))
    parts.append(f'<polyline fill="none" stroke="{_LADDER}" stroke-width="2" points="{pts}"/>')
    for i, v in enumerate(r2g):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="{_LADDER}">'
                     f'<title>return-to-go {v:.3f}</title></circle>')
    # dashed baseline (running mean of rewards up to t, inclusive)
    bpts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(baseline))
    parts.append(f'<polyline fill="none" stroke="{_BASELINE}" stroke-width="2" '
                 f'stroke-dasharray="6 4" points="{bpts}"/>')
    # legend
    lx, ly = L, T + ph + 26.0
    parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="12" height="12" fill="{_ACCEPTED}" fill-opacity="0.55"/>')
    parts.append(f'<text x="{lx + 16:.1f}" y="{ly + 10:.1f}" font-size="11" fill="{_AXIS}">r >= 0</text>')
    lx += 62
    parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="12" height="12" fill="{_SCORED_REJ}" fill-opacity="0.55"/>')
    parts.append(f'<text x="{lx + 16:.1f}" y="{ly + 10:.1f}" font-size="11" fill="{_AXIS}">{html.escape("r < 0")}</text>')
    lx += 52
    parts.append(f'<line x1="{lx:.1f}" y1="{ly + 6:.1f}" x2="{lx + 14:.1f}" y2="{ly + 6:.1f}" stroke="{_LADDER}" stroke-width="2"/>')
    parts.append(f'<text x="{lx + 18:.1f}" y="{ly + 10:.1f}" font-size="11" fill="{_AXIS}">return-to-go</text>')
    lx += 100
    parts.append(f'<line x1="{lx:.1f}" y1="{ly + 6:.1f}" x2="{lx + 14:.1f}" y2="{ly + 6:.1f}" stroke="{_BASELINE}" stroke-width="2" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{lx + 18:.1f}" y="{ly + 10:.1f}" font-size="11" fill="{_AXIS}">baseline (running mean)</text>')
    mean_r = sum(rewards) / n
    summary = (f"n steps = {n} - final return-to-go = {r2g[-1]:.3f} - "
               f"mean reward = {mean_r:.3f}")
    parts.append(f'<text x="{width // 2}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="12" fill="{_AXIS}">{html.escape(summary)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
