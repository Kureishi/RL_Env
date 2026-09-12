"""The provenance certificate core (SPEC.md 56.1, v0.42) — the
"what produced this result" block as pure functions.

`cli._cmd_report --certificate` (56.1.1), the app's Provenance expander
(56.4), and `plotting.html_cover` (56.2) all render from these three, so
the certificate contract has one home. The payload is a pure merge of the
environment facts (`env_provenance`) with the run's identity (seed / task /
target / run-config hash); the card and the SVG are deterministic functions
of the payload (G2). No wall-clock timestamps — a re-render of the same
inputs is byte-identical (22.2).

House rules: stdlib only (3) + numpy (a hard dependency); import-cycle-free
(this module imports nothing from `cli` / `dashboard_app`); streamlit-free
(23.1).
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

import autorefine
from . import plotting as _plotting  # 56.1: the card's tokens + header (the
# colors are read at render time so the 51.4 `_styled` theme swap applies)
from .plotting import _svg_header  # 51.4.3: ARIA on the card root

# 56.1: the fixed extras probed — the [gui] / [image] / [audio] optionals
# (pyproject.toml), in a stable order.
_EXTRA_PACKAGES = (
    ("streamlit", "streamlit"),  # [gui]
    ("PIL", "pillow"),           # [image]
    ("soundfile", "soundfile"),  # [audio]
)

# the long-value rows (the config hash / git sha) get the truncated +
# `<title>` treatment (56.1.2)
_HASH_LABELS = ("config_hash", "git_sha")


def canonical_hash(payload) -> str:
    """56.1 (SPEC.md 56.1): the deterministic config hash — sha256 of the
    canonical JSON encoding (`sort_keys=True`, compact separators,
    `default=str` for non-JSON types). The same payload hashes to the same
    digest everywhere; key order and whitespace never change it."""
    canonical = json.dumps(payload, sort_keys=True,
                          separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def env_provenance() -> dict:
    """56.1 (SPEC.md 56.1): the environment facts — `python`, `numpy`, the
    `extras` dict (version or None per the three optional packages, in
    fixed order), and `git_sha` (the short `HEAD` of the repository the
    source tree lives in; None when there is no repository or git is
    unavailable). Deterministic for a given environment (G2); never
    raises — every probe degrades to None."""
    extras = {}
    for module_name, dist in _EXTRA_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            extras[dist] = None
            continue
        try:
            from importlib.metadata import version
            extras[dist] = version(dist)
        except Exception:
            extras[dist] = None
    git_sha = None
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3)
        sha = (out.stdout or "").strip()
        if out.returncode == 0 and sha:
            git_sha = sha
    except Exception:
        git_sha = None
    return {
        "python": platform.python_version(),
        "numpy": getattr(np, "__version__", None),
        "extras": extras,
        "git_sha": git_sha,
    }


def provenance_payload(env, seed=None, task=None, target=None,
                       run_config=None) -> dict:
    """56.1 (SPEC.md 56.1): the certificate payload — a pure merge of the
    environment facts with the run's identity, in a fixed key order:
    `tool`, `version`, `python`, `numpy`, `extras`, `task`, `seed`,
    `target`, `config_hash` (the run-config canonical hash, or None when
    the run carries no `run_config.json`), and `git_sha`. A non-dict
    `env` degrades to the empty facts (never a crash)."""
    env = env if isinstance(env, dict) else {}
    extras = env.get("extras")
    if not isinstance(extras, dict):
        extras = {dist: None for _name, dist in _EXTRA_PACKAGES}
    return {
        "tool": "autorefine",
        "version": autorefine.__version__,
        "python": env.get("python"),
        "numpy": env.get("numpy"),
        "extras": dict(extras),
        "task": task,
        "seed": seed,
        "target": target,
        "config_hash": canonical_hash(run_config) if run_config is not None
                       else None,
        "git_sha": env.get("git_sha"),
    }


def provenance_card(payload) -> str:
    """56.1 (SPEC.md 56.1): the plain-text certificate — one
    `key: value` line per payload entry, in payload order; None reads as
    `—`; each `extras` entry is its own indented line. A non-dict payload
    renders as the header alone (never a crash)."""
    p = payload if isinstance(payload, dict) else {}
    lines = ["AutoRefine provenance certificate"]
    for key, value in p.items():
        if key == "extras" and isinstance(value, dict):
            for dist, version in value.items():
                lines.append(f"  {dist}: {version if version is not None else '—'}")
        else:
            lines.append(f"{key}: {value if value is not None else '—'}")
    return "\n".join(lines)


def _svg_provenance(payload, width: int = 560) -> str:
    """56.1 (SPEC.md 56.1): the SVG certificate card — a bordered panel
    with the brand line and one `key: value` row per payload entry (fixed
    row height, data-driven total height); `extras` entries render as
    indented rows; a `None` reads `—`; the long hash rows (config_hash /
    git_sha) truncate to 16 chars with the full value in a `<title>`
    (56.1.2). A non-dict / empty payload renders the header + message
    (21.2). Valid XML, pure, deterministic (G2)."""
    p = payload if isinstance(payload, dict) else {}
    rows: list[tuple[str, object]] = []
    for key, value in p.items():
        if key == "extras" and isinstance(value, dict):
            rows.extend((("  " + dist, version)
                         for dist, version in value.items()))
        else:
            rows.append((key, value))
    axis, bg, line = _plotting._AXIS, _plotting._BG, _plotting._LINE
    if not rows:
        height = 120
        parts = _svg_header(width, height, "provenance (empty)")
        parts.append(f'<text x="{width // 2}" y="{height // 2}" '
                     f'text-anchor="middle" font-size="13" fill="{axis}">'
                     f'no provenance payload</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    row_h = 24
    top = 96
    height = top + len(rows) * row_h + 24
    parts = _svg_header(width, height, "provenance certificate")
    parts.append(f'<rect x="8" y="40" width="{width - 16}" '
                 f'height="{height - 48}" fill="{bg}" stroke="{axis}" '
                 f'stroke-width="1.5" rx="8"/>')
    parts.append(f'<text x="24" y="68" font-size="15" font-weight="bold" '
                 f'fill="{axis}">AutoRefine — provenance certificate</text>')
    y = top
    for label, value in rows:
        text = "—" if value is None else str(value)
        if label in _HASH_LABELS and len(text) > 16:
            short, full = text[:16] + "…", text
            parts.append(
                f'<text x="24" y="{y}" font-size="13" fill="{axis}">'
                f'{html.escape(label)}: '
                f'<tspan fill="{line}">{html.escape(short)}</tspan>'
                f'<title>{html.escape(full)}</title></text>')
        else:
            parts.append(f'<text x="24" y="{y}" font-size="13" '
                         f'fill="{axis}">{html.escape(label)}: '
                         f'{html.escape(text)}</text>')
        y += row_h
    parts.append("</svg>")
    return "\n".join(parts)


def svg_provenance(payload, width: int = 560, palette: str = "default",
                   dark: bool = False) -> str:
    """56.1 (SPEC.md 56.1): the provenance certificate as an SVG card —
    the 51.4 palette (`default` / `okabe`) and dark-mode theming (the card
    picks up the themed axis/surface/line colors), ARIA always on
    (51.4.3); the default call is byte-identical to an explicit one (G2);
    valid XML, pure, deterministic."""
    from .plotting import _styled  # 51.4: the scoped theme swap
    with _styled(palette, dark):
        return _svg_provenance(payload, width=width)
