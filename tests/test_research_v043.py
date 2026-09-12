"""v0.43 — "B. Visually useful for researchers / niche specialized tasks"
(SPEC.md 57, A47, M46).

Covers the four A items as implemented — all *pure derivations* from data
that already exists (the ``experiments.jsonl`` entries + the update
stream's ``field_stats`` / ``ucb``), no new logged field, no loop change
(G2):

- 57.1 the **spec-lineage graph** — ``research.spec_lineage(entries)``
  (the champion chain: baseline seeds, accepted re-champions, curriculum
  re-pins the score without re-parenting, ``invalid_spec`` excluded,
  ``delta`` None when a score is missing) + ``svg_spec_lineage`` (the tidy
  tree, the bézier edge weight ``1.5 + 3.5·min(1, |Δ|/10)``, the node
  boxes, the legend + summary, the tooltips, the empty state);
- 57.2 the **per-field response surfaces** —
  ``research.field_response_stats(entries)`` (per-value n / mean / best
  over the *full* spec, the screen / curriculum / invalid exclusion, the
  ``"-"`` / ``","`` value keys, the sort order) + ``svg_field_response``
  (the per-cell bar + mean + ``n`` text, the tooltips, the empty state);
- 57.3 the **gate-decision region** —
  ``research.gate_region_candidates(entries)`` (the ``best_before`` chain
  incl. the curriculum re-pin, the ``accounting.candidate_reason`` reason
  chain, ``baseline_score`` + ``tol = 0.05``) + ``svg_gate_region`` (the
  acceptance polygon, the ``gen_gap = tol·score`` penalty line, the
  baseline line, the verdict-colored dots + tooltips, the 18.6 caption,
  the empty state);
- 57.4 the **bandit belief bars** — ``research.bandit_beliefs(field_stats,
  ucb)`` (the 95% Wilson interval, hand-pinned, the ``trials <= 0``
  exclusion, a non-finite / missing UCB → None, the field sort) +
  ``svg_bandit_beliefs`` (the CI band, the win-rate tick, the UCB
  diamond, the best-believed flag, the empty state);
- 57.5 the **app surface + exports** — the "Research views" block in
  ``_render_result`` (the four renderers + the four research functions +
  the friendly-error path + the no-field-stats caption), the four + four
  names in ``autorefine.__all__``, and the default path end-to-end (a
  finished run renders the block; no new widget);
- the A47 round regression — the version stepped to ``0.43.0`` in both
  sources (33.1) and SPEC carries the A47 block + M46 row.

House rules (A47): no cross-test imports (all fixtures synthesized here);
the pure core (``research.py`` + the four ``plotting`` renderers) is
tested by hand-computation; the app (``dashboard_app.py``) is a thin
renderer tested by source-token assertions + one end-to-end AppTest
(streamlit optional). Every renderer output is validated as XML
(``xml.dom.minidom``); the default-theme output is byte-identical (G2),
and the okabe / dark re-render + the bad-palette ``ValueError`` are
covered (51.4).
"""
from __future__ import annotations

import re
import tomllib
import xml.dom.minidom as minidom
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import (
    svg_bandit_beliefs,
    svg_field_response,
    svg_gate_region,
    svg_spec_lineage,
)
from autorefine.research import (
    bandit_beliefs,
    field_response_stats,
    gate_region_candidates,
    spec_lineage,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


def _parse(svg: str) -> None:
    minidom.parseString(svg)  # valid XML (G2)


# the canonical run fixture (log order = reset baseline → candidates → the
# curriculum step-up → more candidates → the invalid + screen rows)
ENTRIES = [
    {"kind": "baseline", "spec": {"model_family": "mlp", "hidden_dim": 64},
     "spec_hash": "B0", "mutation": None, "holdout_score": 50.0, "std": 1.0,
     "gen_score": 50.0, "gen_gap": 0.0, "train_seconds": 1.0,
     "effective_score": 50.0, "accepted": True},
    {"kind": "experiment", "spec": {"model_family": "mlp", "hidden_dim": 128},
     "spec_hash": "B1", "mutation": ["hidden_dim"], "holdout_score": 60.0,
     "std": 1.0, "gen_score": 61.0, "gen_gap": 1.0, "train_seconds": 1.2,
     "effective_score": 60.0, "accepted": True},
    {"kind": "experiment", "spec": {"model_family": "tree", "max_depth": 3},
     "spec_hash": "B2", "mutation": ["model_family"], "holdout_score": 55.0,
     "std": 1.0, "gen_score": 56.0, "gen_gap": 1.0, "train_seconds": 0.8,
     "effective_score": 55.0, "accepted": False},
    {"kind": "curriculum", "new_baseline_score": 62.0},
    {"kind": "experiment", "spec": {"model_family": "mlp", "hidden_dim": 256},
     "spec_hash": "B3", "mutation": ["hidden_dim"], "holdout_score": 65.0,
     "std": 1.0, "gen_score": 75.0, "gen_gap": 10.0, "train_seconds": 1.5,
     "effective_score": 60.0, "accepted": False},
    {"kind": "invalid_spec", "reason": "bad"},
    {"kind": "screen", "spec": {"model_family": "mlp", "hidden_dim": 64},
     "holdout_score": 40.0},
]


# --- 57.1 the spec-lineage graph core (A47) ----------------------------------

def test_spec_lineage_champion_chain():
    """A47 (SPEC.md 57.1): the parent link is reconstructed from log order —
    the baseline seeds the champion, the accepted B1 re-champions, the
    rejected B2 does not (so B3's parent is B1, not B2); the curriculum
    step-up re-pins the *score* (B3's delta is 65−62=3, not 65−60) without
    re-parenting; the ``invalid_spec`` row is excluded; node labels are
    ``family <hash>``."""
    lin = spec_lineage(ENTRIES)
    assert set(lin["nodes"]) == {"B0", "B1", "B2", "B3"}
    assert lin["nodes"]["B0"] == {
        "hash": "B0", "score": 50.0, "accepted": True, "label": "mlp B0"}
    assert lin["nodes"]["B2"] == {
        "hash": "B2", "score": 55.0, "accepted": False, "label": "tree B2"}
    edges = [(e["src"], e["dst"]) for e in lin["edges"]]
    assert edges == [("B0", "B1"), ("B1", "B2"), ("B1", "B3")]  # B1, not B2
    by = {(e["src"], e["dst"]): e for e in lin["edges"]}
    assert by[("B0", "B1")]["accepted"] is True
    assert by[("B0", "B1")]["delta"] == 10.0
    assert by[("B0", "B1")]["fields"] == ["hidden_dim"]
    assert by[("B1", "B2")]["accepted"] is False
    assert by[("B1", "B2")]["delta"] == -5.0
    # the curriculum re-pin: 65 − 62 (not 65 − 60)
    assert by[("B1", "B3")]["delta"] == 3.0
    assert by[("B1", "B3")]["accepted"] is False


def test_spec_lineage_delta_none_and_empty():
    """A47 (SPEC.md 57.1): ``delta`` is None when a score is missing; empty
    entries → empty sets (pure, deterministic — G2)."""
    entries = [
        {"kind": "baseline", "spec_hash": "X0",
         "spec": {"model_family": "mlp"}, "holdout_score": 50.0,
         "accepted": True},
        {"kind": "experiment", "spec_hash": "X1",
         "spec": {"model_family": "mlp"}, "mutation": ["hidden_dim"],
         "holdout_score": None, "accepted": False},
    ]
    lin = spec_lineage(entries)
    assert lin["nodes"]["X1"]["score"] is None
    assert lin["edges"][0]["delta"] is None
    assert spec_lineage([]) == {"nodes": {}, "edges": []}
    assert spec_lineage(None) == {"nodes": {}, "edges": []}


def test_svg_spec_lineage_render():
    """A47 (SPEC.md 57.1): the tidy-tree SVG — ARIA, the node labels +
    scores, the bézier edge paths, the verdict legend, the ``n specs /
    n mutations`` summary, the edge tooltips (mutated fields + Δscore),
    valid XML."""
    svg = svg_spec_lineage(spec_lineage(ENTRIES))
    _parse(svg)
    assert 'aria-label="spec lineage (DAG)"' in svg
    for token in ("mlp B0", "mlp B1", "tree B2", "mlp B3"):
        assert token in svg
    assert '<path d="M ' in svg  # the cubic-bézier edges
    assert "n specs = 4 · n mutations = 3" in svg
    assert "edge = one mutation · width ∝ |Δscore|" in svg
    assert "accepted" in svg and "rejected" in svg  # the legend
    assert "hidden_dim" in svg  # the edge tooltip's mutated fields
    assert "Δscore 10.00" in svg
    assert "Δscore 3.00" in svg  # the curriculum re-pin
    empty = svg_spec_lineage({"nodes": {}, "edges": []})
    _parse(empty)
    assert "no logged specs (empty run)" in empty


# --- 57.2 the per-field response surfaces (A47) -------------------------------

def test_field_response_stats_hand_computed():
    """A47 (SPEC.md 57.2): for every *scored* entry (baseline / experiment
    with a finite holdout_score + a dict spec) each (field, value) pair of
    the FULL spec is credited — the per-value n / mean / best is
    hand-computed; the screen / curriculum / invalid rows are excluded;
    the ``"-"`` / ``","`` value keys; the sorted order."""
    entries = [
        {"kind": "baseline", "spec": {"a": None, "layers": [8, 16]},
         "spec_hash": "R0", "holdout_score": 70.0, "accepted": True},
        {"kind": "experiment", "spec": {"a": None, "layers": [8, 16]},
         "spec_hash": "R1", "mutation": ["a"], "holdout_score": 80.0,
         "accepted": True},
        {"kind": "screen", "spec": {"a": 1}, "holdout_score": 99.0},
        {"kind": "curriculum", "new_baseline_score": 5.0},
        {"kind": "invalid_spec", "reason": "bad"},
    ]
    stats = field_response_stats(entries)
    assert stats == {
        "a": {"-": {"n": 2, "mean": 75.0, "best": 80.0}},
        "layers": {"8,16": {"n": 2, "mean": 75.0, "best": 80.0}},
    }
    # the canonical fixture: the full spec is credited per value
    stats = field_response_stats(ENTRIES)
    assert stats["model_family"]["mlp"] == {
        "n": 3, "mean": (50.0 + 60.0 + 65.0) / 3.0, "best": 65.0}
    assert stats["model_family"]["tree"] == {"n": 1, "mean": 55.0,
                                             "best": 55.0}
    assert stats["hidden_dim"] == {
        "64": {"n": 1, "mean": 50.0, "best": 50.0},
        "128": {"n": 1, "mean": 60.0, "best": 60.0},
        "256": {"n": 1, "mean": 65.0, "best": 65.0},
    }
    assert stats["max_depth"] == {"3": {"n": 1, "mean": 55.0, "best": 55.0}}
    assert field_response_stats([]) == {}


def test_svg_field_response_render():
    """A47 (SPEC.md 57.2): one row per field, one cell per value — the bar
    (mean, 0-100 clamped), the mean label, the ``value · n`` caption, the
    per-cell ``<title>`` (``field value: mean (n, best)``), the caption
    line, valid XML; empty → message."""
    svg = svg_field_response(field_response_stats(ENTRIES))
    _parse(svg)
    assert 'aria-label="per-field response surfaces"' in svg
    for token in ("hidden_dim", "model_family", "max_depth",
                  "bar = mean holdout score of candidates carrying that "
                  "value (0-100 clamped) · n = sample size"):
        assert token in svg
    assert "60.0" in svg  # the mean label above the bar
    assert "n=1" in svg
    assert "hidden_dim 128: mean 60.00 (n=1, best 60.00)" in svg
    assert "model_family mlp: mean 58.33 (n=3, best 65.00)" in svg
    empty = svg_field_response({})
    _parse(empty)
    assert "no scored specs to aggregate" in empty


# --- 57.3 the gate-decision region (A47) --------------------------------------

def test_gate_region_candidates_hand_computed():
    """A47 (SPEC.md 57.3): the running best is reconstructed exactly like
    ``accounting._rejections`` (the baseline seeds it, the curriculum
    re-pins it, an accepted candidate raises it); each reason is
    ``accounting.candidate_reason`` (score → overfit → ci); the baseline
    score + the §18.5 tolerance are carried."""
    gate = gate_region_candidates(ENTRIES)
    assert gate["baseline_score"] == 50.0
    assert gate["tol"] == 0.05
    cands = gate["candidates"]
    assert [(c["score"], c["reason"], c["accepted"]) for c in cands] == [
        (60.0, "accepted", True),   # 60 > 50, small gap
        (55.0, "score", False),     # 55 < 60 (the score gate first)
        (65.0, "overfit", False),   # 65 > 62 but 10 > (1.05·65 − 62)
        (40.0, "score", False),     # the screen row (40 < 62)
    ]
    assert [c["best_before"] for c in cands] == [50.0, 60.0, 62.0, 62.0]
    assert cands[2]["gen_gap"] == 10.0
    assert cands[3]["gen_gap"] is None  # old rows keep None
    empty = gate_region_candidates([])
    assert empty == {"baseline_score": None, "tol": 0.05, "candidates": []}


def test_svg_gate_region_render():
    """A47 (SPEC.md 57.3): the acceptance polygon, the ``gen_gap =
    tol·score`` penalty line, the baseline reference line, the
    verdict-colored candidate dots with their tooltips, the 18.6 CI-margin
    caption, valid XML; no scored candidates → message."""
    svg = svg_gate_region(gate_region_candidates(ENTRIES))
    _parse(svg)
    assert 'aria-label="gate-decision region (score × gen gap)"' in svg
    assert "<polygon points=" in svg  # the acceptance region
    assert "gen_gap = 5%·score (18.5)" in svg  # the penalty boundary
    # the candidate dots' tooltips (score, gap, reason, best_before)
    assert "score 60.00 · gen_gap 1.00 · accepted · best_before 50.00" in svg
    assert "score 55.00 · gen_gap 1.00 · score · best_before 60.00" in svg
    assert "score 65.00 · gen_gap 10.00 · overfit · best_before 62.00" in svg
    # the legend + the 18.6 caption
    for token in ("accepted", "overfit", "score", "ci"):
        assert token in svg
    assert "the per-step CI margin (18.6) shifts this by z·SE" in svg
    empty = svg_gate_region({"baseline_score": None, "tol": 0.05,
                             "candidates": []})
    _parse(empty)
    assert "no scored candidates yet" in empty


# --- 57.4 the bandit belief bars (A47) -----------------------------------------

# the 95% Wilson interval for wins=5, n=10, z=1.959964 (hand-computed):
# center = (p + z²/2n)/(1 + z²/n) = 0.5 (exactly); half ≈ 0.2634069109885
_WILSON_5_10_LO = 0.23659308901147935


def test_bandit_beliefs_hand_computed():
    """A47 (SPEC.md 57.4): the Wilson 95% CI is hand-pinned (z = 1.959964,
    clamped to [0, 1]); fields with ``trials <= 0`` are skipped; a missing
    / non-finite UCB reads None (the search policy); fields are sorted."""
    stats = {"lr": {"trials": 10, "wins": 5, "win_rate": 0.5},
             "dead": {"trials": 0, "wins": 0, "win_rate": 0.0},
             "neg": {"trials": 4, "wins": 1}}
    bel = bandit_beliefs(stats, {"lr": 0.7, "dead": None})
    assert list(bel) == ["lr", "neg"]  # sorted; ``dead`` (trials 0) skipped
    lr = bel["lr"]
    assert lr["trials"] == 10
    assert lr["wins"] == 5.0
    assert lr["win_rate"] == 0.5
    assert abs(lr["lo"] - _WILSON_5_10_LO) < 1e-12
    assert abs(lr["hi"] - (1.0 - _WILSON_5_10_LO)) < 1e-12
    assert abs(lr["ucb"] - 0.7) < 1e-15
    assert bel["neg"]["ucb"] is None  # no UCB reading → None
    assert bel["neg"]["win_rate"] == 0.25
    # the CI is clamped inside [0, 1] for a degenerate 100% field
    one = bandit_beliefs({"f": {"trials": 1, "wins": 1}}, None)
    assert 0.0 <= one["f"]["lo"] and one["f"]["hi"] <= 1.0
    assert one["f"]["ucb"] is None
    assert bandit_beliefs(None, None) == {}
    assert bandit_beliefs({}, {"x": 0.1}) == {}


def test_svg_bandit_beliefs_render():
    """A47 (SPEC.md 57.4): the CI band (``[lo, hi]``), the win-rate tick,
    the UCB diamond, the best-believed flag (highest UCB), the per-row
    ``<title>`` with the full numbers, the axis, the legend caption, valid
    XML; empty → the bandit-policy message."""
    stats = {"a": {"trials": 10, "wins": 5, "win_rate": 0.5},
             "b": {"trials": 10, "wins": 4, "win_rate": 0.4}}
    bel = bandit_beliefs(stats, {"a": 0.7, "b": 0.5})
    svg = svg_bandit_beliefs(bel)
    _parse(svg)
    assert 'aria-label="bandit belief bars"' in svg
    assert "a: win 5/10 (0.500) · 95% CI [0.237, 0.763] · UCB 0.700 · " \
        "best-believed (max UCB)" in svg
    assert "b: win 4/10 (0.400)" in svg
    assert "best-believed (max UCB)" in svg
    # the flag appears once per best-row element (lane/band/tick/diamond
    # tooltips) and nowhere in the non-best row's title
    assert svg.count("best-believed (max UCB)") == 4
    i = svg.find("b: win 4/10 (0.400)")
    assert i >= 0
    b_title = svg[i:svg.find("</title>", i)]
    assert "best-believed" not in b_title
    assert "band = 95% Wilson CI · tick = win rate (26.1) · diamond = UCB " \
        "(26.4) · ★ = best-believed" in svg
    empty = svg_bandit_beliefs({})
    _parse(empty)
    assert "no field stats — needs a bandit-policy run (SPEC.md 57.4)" \
        in empty


# --- the renderers: palette / dark / validation (A47) --------------------------

def test_renderers_palette_dark_and_validation():
    """A47 (SPEC.md 57.5): the four renderers honor the 51.4 palette /
    dark contract — the okabe + dark re-render is valid XML, and a bad
    palette is a ``ValueError`` (51.4.1)."""
    lin = spec_lineage(ENTRIES)
    resp = field_response_stats(ENTRIES)
    gate = gate_region_candidates(ENTRIES)
    bel = bandit_beliefs({"lr": {"trials": 10, "wins": 5}}, {"lr": 0.7})
    for fn, data in ((svg_spec_lineage, lin), (svg_field_response, resp),
                     (svg_gate_region, gate), (svg_bandit_beliefs, bel)):
        _parse(fn(data, palette="okabe", dark=True))
        with pytest.raises(ValueError):
            fn(data, palette="nope")


# --- 57.5 the app wiring (A47) -------------------------------------------------

def test_app_source_wires_research_views():
    """A47 (SPEC.md 57.5): the app is a thin renderer — the "Research
    views" subheader in ``_render_result``, the four svg renderers, the
    four research functions (module-level import), the friendly-error
    path (49.4.3), and the no-field-stats caption; the palette/dark
    passthrough feeds every render."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        # the block (57.5)
        'st.subheader("Research views")',
        "svg_spec_lineage(spec_lineage(_ren), **pal)",
        "svg_field_response(field_response_stats(_ren), **pal)",
        "svg_gate_region(gate_region_candidates(_ren), **pal)",
        "svg_bandit_beliefs(bandit_beliefs(_fs, _ucb), **pal)",
        # the core imports (57.5)
        "from autorefine.research import (",
        "    bandit_beliefs,",
        "    field_response_stats,",
        "    gate_region_candidates,",
        "    spec_lineage,",
        # the friendly-error path (49.4.3)
        "research views unavailable",
        # the no-field-stats caption
        "bandit belief bars need a bandit-policy run (field_stats)",
    ):
        assert token in src, token


def test_app_default_path_end_to_end(tmp_path):
    """A47 (SPEC.md 57.5): the *default* path runs end-to-end — a finished
    run persists its result and renders the "Research views" block (the
    spec-lineage SVG among the markdown) with no new widget (AppTest-safe).
    """
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    # a small classification CSV (quadrant XOR, 2 classes)
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    # the default path finished and persisted its result
    assert at.session_state["result"] is not None
    # the Research views block rendered (subheader + the lineage SVG)
    sub_headers = [str(s.value) for s in at.subheader]
    assert "Research views" in sub_headers, sub_headers
    assert any("spec lineage" in str(m.value) for m in at.markdown), \
        "the spec-lineage SVG renders in the Results view"


# --- A47 round regression -------------------------------------------------------

def test_version_round_v043():
    """A47 (SPEC.md 57.6, 33.1): the version stepped to ``0.43.0`` in both
    sources (v0.43 ⇒ ``0.43.0``, M46, SPEC.md 57)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"


def test_all_exports_research_round():
    """A47 (SPEC.md 57.5, 33.1): the four research functions + the four
    renderers are exported from the package top level and ``__all__``."""
    for name in ("spec_lineage", "field_response_stats",
                 "gate_region_candidates", "bandit_beliefs",
                 "svg_spec_lineage", "svg_field_response",
                 "svg_gate_region", "svg_bandit_beliefs"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_spec_cites_a47_and_round():
    """A47 (SPEC.md 57.6/57.7): SPEC.md defines the A47 acceptance block
    and the M46 index row + milestone — the A25 index machinery reads both
    (defined == set(range(1, 48)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 57.6 Acceptance (A47)" in spec
    assert re.search(r"^\s*\| M46 \| v0\.43\s*\|\s*57\s*\|\s*A47\s*\|",
                     spec, re.MULTILINE)
    assert "**M46**" in spec
