"""Visuals interface — cards, spacing, side-by-side layout (v0.57,
SPEC.md 71, A61).

SPEC.md 71: every rendered SVG on the dashboard gains a standard visual
card (emphasized title, vertical spacing, a horizontal scroll window so
charts scroll instead of clipping), and the compact Learning views read
as two side-by-side pairs:
  * 71.1 — `VISUAL_CSS` + `visual_card` (pure, in `plotting`);
  * 71.2 — the app's `_svg_pair` side-by-side helper;
  * 71.3 — the Learning views arrangement (two pairs);
  * 71.4 — the card applied app-wide through the two 69.3 guard seams
    (the degrade contract preserved verbatim);
  * 71.5 — A1–A60 stay green (additive core; pinned tokens survive).

House rules: pure / deterministic (G2), stdlib + numpy core leaves, no
cross-test imports (every fixture synthesized here), the core stays
streamlit-free (the app tests touch `dashboard_app` only under AppTest,
streamlit optional-skipped). Envs stay tiny (quadrant-XOR CSV, 2
experiments) for speed.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import (
    VISUAL_CSS,
    svg_is_well_formed,
    visual_card,
)

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
SPEC = REPO / "SPEC.md"

# 35.2 app-language scans — duplicated here (no cross-test imports):
# the same two regexes `tests/test_app_language.py` applies.
BANNED = re.compile(r"SPEC\.md|SPEC_FIELDS|\u00a7|v0\.\d+")
SECTION = re.compile(r"(?<![=>\".0-9])\d{1,2}\.\d{1,2}(?:\.\d{1,2})?(?![\d.%f])")

_GOOD_SVG = ('<svg width="640" height="200" role="img" '
             'aria-label="demo"><rect width="640" height="200" '
             'fill="#ffffff"/></svg>')


# --- fixtures (synthesized here — no cross-test imports) ----------------------

def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes)."""
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --- 71.1.1 the stylesheet (A61) ----------------------------------------------

def test_visual_css_rules_and_language_clean():
    """A61 (71.1.1): `VISUAL_CSS` carries all five card rules and stays
    clean under the 35.2 app-language scans (integer px + hex colors —
    no fractional literals, no SPEC references, no bare section numbers)."""
    for rule in (".ar-visual {", ".ar-visual-title {", ".ar-visual-scroll {",
                 ".ar-visual-scroll svg {", ".ar-visual-cap {"):
        assert rule in VISUAL_CSS, rule
    assert "overflow-x: auto" in VISUAL_CSS  # the scroll window (71.1.3)
    assert "max-width: none" in VISUAL_CSS  # charts keep natural width
    assert not BANNED.search(VISUAL_CSS), BANNED.search(VISUAL_CSS).group(0)
    assert not SECTION.search(VISUAL_CSS), \
        SECTION.search(VISUAL_CSS).group(0)


# --- 71.1.2 the visual card (A61) ---------------------------------------------

def test_visual_card_healthy_wraps():
    """A61 (71.1.2): a well-formed SVG is wrapped in the card markup —
    the emphasized title div, the scroll window, and the SVG exactly
    once (never duplicated, never truncated)."""
    card = visual_card("My title", _GOOD_SVG)
    assert '<div class="ar-visual">' in card
    assert '<div class="ar-visual-title">My title</div>' in card
    assert '<div class="ar-visual-scroll">' in card
    assert card.count("<svg") == 1
    assert card.count("</svg>") == 1
    assert card.rstrip().endswith("</div>")
    # the card itself is well-formed enough to round-trip the SVG intact
    assert _GOOD_SVG in card


def test_visual_card_caption_optional():
    """A61 (71.1.2): the caption is optional — absent by default, and a
    `.ar-visual-cap` line when given."""
    assert "ar-visual-cap" not in visual_card("T", _GOOD_SVG)
    with_cap = visual_card("T", _GOOD_SVG, "a caption here")
    assert '<div class="ar-visual-cap">a caption here</div>' in with_cap
    assert visual_card("T", _GOOD_SVG, None) == visual_card("T", _GOOD_SVG)


def test_visual_card_degrades_malformed():
    """A61 (71.1.2): a truncated fragment (or a non-string) returns the
    friendly "plot unavailable" note — the 69.3.2 contract — and never
    paints a broken partial."""
    for bad in ("<svg></svg", "", "not svg at all", 42, None):
        out = visual_card("T", bad)
        assert "plot unavailable" in out, bad
        assert "ar-visual-title" not in out, bad
        assert "ar-visual-scroll" not in out, bad
    # the guard decision matches the 69.3 predicate exactly
    assert svg_is_well_formed(_GOOD_SVG)
    assert not svg_is_well_formed("<svg></svg")


def test_visual_card_deterministic():
    """A61 (71.6): identical inputs give byte-identical markup — a pure
    string function (G2)."""
    for args in (("T", _GOOD_SVG), ("T", _GOOD_SVG, "cap")):
        assert visual_card(*args) == visual_card(*args)


# --- 71.4 / 71.2 / 71.3 the app (A61) -----------------------------------------

def test_app_injects_card_css_and_cards(tmp_path):
    """A61 (71.4.1/71.4.2): after a tiny run, the page carries the
    `<style>` card block, and the Learning-views pair titles render in
    card markup (`.ar-visual-title`) — the cards are live app-wide."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "<style>" in md and ".ar-visual" in md  # 71.4.1: CSS injected

    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["experiments"] = 2
    at.session_state["max_train"] = 5.0
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    at.run()
    assert not at.exception, at.exception

    md = " ".join(m.value for m in at.markdown)
    assert ".ar-visual-title" in md  # the cards are rendering
    for title in ("Per-class accuracy", "Confusion matrix",
                  "Per-input difficulty", "Decision boundary",
                  "Training curves", "Final architecture"):
        assert f'<div class="ar-visual-title">{title}</div>' in md, title


def test_app_malformed_learning_svg_still_degrades(tmp_path, monkeypatch):
    """A61 (71.5): with a Learning-view renderer monkeypatched
    (bound-name, pre-`run()`) to emit truncated markup, the app
    completes without exception and the 69.3.2 "plot unavailable"
    caption appears — the degrade path is preserved by the card
    upgrade."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest
    import autorefine.plotting as plotting

    monkeypatch.setattr(
        plotting, "svg_loss_curves", lambda *a, **k: "<svg></svg")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.session_state["runs_dir"] = str(tmp_path / "runs")
    at.session_state["experiments"] = 2
    at.session_state["max_train"] = 5.0
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "plot unavailable" in caps


def test_app_source_wires_71():
    """A61 (71.2–71.4): the app imports the card machinery, injects the
    stylesheet in `main()`, defines the pair helper, and renders the
    two Learning-views pairs; `plotting` defines the card wrapper."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "visual_card,  # 71.1.2 (v0.57, A61)",
        "VISUAL_CSS,  # 71.1 (v0.57, A61)",
        "st.markdown(f\"<style>{VISUAL_CSS}</style>\"",
        "def _svg_pair(",
        '("Per-class accuracy", res["per_class_svg"])',
        '("Confusion matrix", res["confusion_svg"])',
        '("Per-input difficulty", res["difficulty_svg"])',
        '("Decision boundary", res["boundary_svg"])',
        "st.columns(2)",
    ):
        assert token in src, token
    plot_src = (REPO / "src" / "autorefine" / "plotting.py") \
        .read_text(encoding="utf-8")
    assert "def visual_card(" in plot_src
    assert "VISUAL_CSS" in plot_src


# --- 33.1 exports + version, SPEC cites (A61) ----------------------------------

def test_exports_and_version():
    """A61 (71.1.4 + 33.1): the new names are in `autorefine.__all__`
    and resolvable (the package re-exports the very objects); the
    version steps to `0.58.0` in both sources; SPEC §71 cites A61 and
    the index row resolves to this file."""
    for name in ("visual_card", "VISUAL_CSS"):
        assert name in autorefine.__all__, name
    assert autorefine.visual_card is visual_card
    assert autorefine.VISUAL_CSS is VISUAL_CSS
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.61.0"
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 71.7 Acceptance (A61)" in spec
    row = next(l for l in spec.splitlines()
               if l.startswith("| M60 " ) and "71" in l)
    assert "A61" in row and "tests/test_visuals_v057.py" in row
    assert "A61" in Path(__file__).read_text(encoding="utf-8")
