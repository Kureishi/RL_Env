"""v0.39 — show each decision, not just the running best (SPEC.md 53,
A43, M42).

Covers:

- 53.1.1 `plotting.svg_gate_line` — the per-candidate gate number-line
  hand-computed: the best/target marker positions on the fixed 0–100
  axis, the CI band present only when `z·SE > 0`, the candidate's dot,
  and the margin annotation per 51.3.2 reason ("accepted: …",
  "missed by …", "over gen-gap tolerance", "inside CI band (z*SE …)",
  "stopped"); the unscored (dup) case renders no dot + the honest
  caption; default == explicit byte-identical (G2); ARIA on the root
  (51.4.3);
- 53.2 `plotting.svg_live_frontier` — the dominated set hand-computed
  under the 53.2.1 rule (strict in at least one coordinate; equal
  pairs dominate nothing); the latest candidate is the flash (larger
  dot + halo) and is excluded from the normal pass; the empty case
  renders header + message; ARIA on the root;
- 53.3 `plotting.svg_architecture(..., highlight=)` — a body-field
  highlight adds the amber ring (per family: mlp `activation`, knn
  `knn_k`); an annotation-row field (`lr_schedule`) adds the
  `highlight: …` line; a family-irrelevant field falls back to the
  label line; `highlight=None` is byte-identical to the pre-v0.39
  output (G2);
- 53.4 the app — a finished run renders the per-candidate gate
  number-lines in the drill-downs, the live-Pareto-frontier SVG, and
  the champion spec card (the stored result carries
  `state_dim`/`n_out`); the app source wires the three new renderers;
- the A43 round regression — the version stepped to `0.39.0` in both
  sources (33.1) and SPEC carries the A43 block + M42 row.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy core; streamlit is optional (the app tests skip without
it).
"""
from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import (
    _dominated,
    _frontier_rows,
    svg_architecture,
    svg_gate_line,
    svg_live_frontier,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
HL = "#b45309"  # the 53.3 highlight color (plotting._HIGHLIGHT)

# the gate-line geometry (53.1.1): L = R = 24, width 520 -> pw = 472
L, PW = 24.0, 472.0


def _x(score: float) -> str:
    return f"{L + PW * min(max(score, 0.0), 100.0) / 100.0:.1f}"


def _assert_xml(svg: str) -> None:
    ET.fromstring(svg)  # valid XML (G2)


# --- 53.1.1 the gate number-line (A43) ----------------------------------------

def test_gate_line_markers_and_annotation():
    """A43 (SPEC.md 53.1.1): the best/target marker positions are
    hand-computed on the fixed 0–100 axis, the candidate dot sits at
    its score, and the CI-rejection annotation carries the 52.1.1
    numbers + the 51.3.2 reason word."""
    svg = svg_gate_line(68.5, 62.34, 95.0, z_se=1.7,
                        accepted=False, reason="ci")
    assert f"x1=\"{_x(62.34)}\"" in svg and "best 62.34" in svg
    assert f"x1=\"{_x(95.0)}\"" in svg and "target 95.0" in svg
    assert f"cx=\"{_x(68.5)}\"" in svg
    # the CI band spans [best, best + z*SE]
    band_w = PW * 1.7 / 100.0
    assert f"width=\"{band_w:.1f}\"" in svg and "CI band" in svg
    assert "above best by 6.16, inside CI band (z*SE 1.70)" in svg
    _assert_xml(svg)


def test_gate_line_reason_annotations():
    """A43 (SPEC.md 53.1.1): each 51.3.2 reason word yields its margin
    annotation — accepted, score (missed), overfit, stopped."""
    ok = svg_gate_line(70.0, 62.0, 95.0, accepted=True, reason="accepted")
    assert "accepted: +8.00 over best 62.00" in ok
    assert "missed by 2.34" in svg_gate_line(
        60.0, 62.34, 95.0, reason="score")
    assert "over gen-gap tolerance (18.5)" in svg_gate_line(
        70.0, 62.0, 95.0, reason="overfit")
    assert "stopped — final step (+8.00 vs best 62.00)" in svg_gate_line(
        70.0, 62.0, 95.0, reason="stopped")
    # the unscored (dup) case: honest caption, no candidate dot
    dup = svg_gate_line(None, 62.34, 95.0, reason="dup")
    assert "unscored (dup) — no candidate dot (R3)" in dup
    assert "<circle" not in dup
    _assert_xml(dup)


def test_gate_line_band_only_when_zse():
    """A43 (SPEC.md 53.1.1): the CI band is shaded only when z*SE > 0
    (the 18.6 gate is inert at legacy z = 0)."""
    with_band = svg_gate_line(68.5, 62.34, 95.0, z_se=1.7)
    no_band = svg_gate_line(68.5, 62.34, 95.0, z_se=0.0)
    assert "CI band" in with_band
    assert "CI band" not in no_band
    _assert_xml(no_band)


def test_gate_line_byte_identity_aria_and_palette():
    """A43 (SPEC.md 53.1.1/51.4): the default call is byte-identical to
    an explicit default (G2); ARIA is always on (51.4.3); the
    Okabe-Ito/dark sets re-color but stay valid XML."""
    a = svg_gate_line(68.5, 62.34, 95.0, 1.7, False, "ci")
    b = svg_gate_line(68.5, 62.34, 95.0, z_se=1.7, accepted=False,
                      reason="ci", palette="default", dark=False)
    assert a == b
    assert 'role="img"' in a and 'aria-label="gate number-line"' in a
    okabe = svg_gate_line(68.5, 62.34, 95.0, 1.7, False, "ci",
                          palette="okabe")
    assert okabe != a  # the data colors swapped (51.4.1)
    _assert_xml(okabe)
    dark = svg_gate_line(68.5, 62.34, 95.0, 1.7, False, "ci", dark=True)
    assert dark != a  # the dark surface set (51.4.2)
    _assert_xml(dark)
    with pytest.raises(ValueError):
        svg_gate_line(1.0, 0.5, 95.0, palette="nope")


# --- 53.2 the live Pareto frontier (A43) ----------------------------------------

FRONTIER = [
    {"index": 1, "candidate_score": 50.0, "train_seconds": 2.0},
    {"index": 2, "candidate_score": 60.0, "train_seconds": 1.0},
    {"index": 3, "candidate_score": 55.0, "train_seconds": 3.0},
]


def test_frontier_rows_and_dominance():
    """A43 (SPEC.md 53.2.1): the scored rows are the updates with a
    finite score + train time (dup steps excluded); the dominated set
    is hand-computed — strictly-better candidates dominate; equal
    pairs dominate nothing (the strict-in-one rule)."""
    rows = _frontier_rows(FRONTIER + [
        {"index": 4, "candidate_score": None, "train_seconds": None},
        {"index": 5, "candidate_score": 90.0, "train_seconds": None},
    ])
    assert rows == [(1, 50.0, 2.0), (2, 60.0, 1.0), (3, 55.0, 3.0)]
    # idx 0 (50, 2s) is dominated by idx 1 (60, 1s); idx 2 (55, 3s) is
    # dominated by idx 1 as well; idx 1 dominates nothing
    assert _dominated(rows) == {0, 2}
    # equal pairs: neither dominates the other
    assert _dominated([(1, 60.0, 1.0), (2, 60.0, 1.0)]) == set()
    # a strict improvement dominates everything
    assert _dominated([(1, 50.0, 2.0), (2, 50.0, 2.0), (3, 99.0, 2.0)]) \
        == {0, 1}


def test_frontier_flash_and_render():
    """A43 (SPEC.md 53.2.2): the latest candidate is drawn as the
    flash (larger dot + halo ring) and excluded from the normal pass;
    dominated points are greyed, frontier points in the accent color;
    the empty case renders header + message; ARIA on the root."""
    svg = svg_live_frontier(FRONTIER)
    assert 'aria-label="live Pareto frontier"' in svg
    assert "latest candidate 3" in svg
    assert "candidate 1" in svg and "candidate 2" in svg
    # the flash: halo ring (r=10, fill none) + larger dot (r=7)
    assert 'r="10" fill="none"' in svg and 'r="7"' in svg
    # idx 1 (60, 1s) is on the frontier; idx 0/2 are dominated
    assert '(dominated)</title>' in svg  # at least one greyed
    assert "frontier" in svg  # the accent-color points + the legend
    _assert_xml(svg)
    # empty cases: no updates / no finite train time
    for empty in ([], [{"candidate_score": None},
                       {"candidate_score": 90.0, "train_seconds": None}]):
        e = svg_live_frontier(empty)
        assert "live Pareto frontier (empty)" in e
        assert "no scored candidates" in e
        _assert_xml(e)
    # palette/dark re-color; default byte-identical (G2)
    assert svg == svg_live_frontier(FRONTIER, palette="default", dark=False)
    _assert_xml(svg_live_frontier(FRONTIER, palette="okabe"))
    _assert_xml(svg_live_frontier(FRONTIER, dark=True))


# --- 53.3 the champion spec card (A43) -------------------------------------------

MLP = {"model_family": "mlp", "architecture": [64], "activation": "relu",
       "optimizer": "adam"}
KNN = {"model_family": "knn", "knn_k": 7, "optimizer": "adam"}
TREE = {"model_family": "tree", "architecture": [4], "train_steps": 300,
        "optimizer": "adam"}


def test_architecture_highlight_ring_per_family():
    """A43 (SPEC.md 53.3.1): the body-field highlight rings the block(s)
    the field drives — per family (mlp `activation`, knn `knn_k`,
    tree `train_steps`) — and changes the output."""
    base = svg_architecture(MLP, 2, 1)
    for spec, field in ((MLP, "activation"), (MLP, "architecture"),
                        (MLP, "model_family"), (KNN, "knn_k"),
                        (TREE, "train_steps")):
        out = svg_architecture(spec, 2, 1, highlight=field)
        assert HL in out, (spec["model_family"], field)
        assert out != svg_architecture(spec, 2, 1)
        _assert_xml(out)


def test_architecture_highlight_note_and_fallback():
    """A43 (SPEC.md 53.3.1): an annotation-row field (`lr_schedule`)
    adds the `highlight: …` line; a family-irrelevant field (`activation`
    on a knn) and an unknown field fall back to the label line; the
    pre-v0.39 output (no ring, no label) is untouched by the default."""
    out = svg_architecture(MLP, 2, 1, highlight="lr_schedule")
    assert "highlight: lr_schedule" in out
    assert HL in out
    out_knn = svg_architecture(KNN, 2, 1, highlight="activation")
    assert "highlight: activation" in out_knn
    out_unk = svg_architecture(MLP, 2, 1, highlight="a_new_field")
    assert "highlight: a_new_field" in out_unk
    for s in (out, out_knn, out_unk):
        _assert_xml(s)


def test_architecture_highlight_default_byte_identical():
    """A43 (SPEC.md 53.3/G2): `highlight=None` — the default — is
    byte-identical to the pre-v0.39 call form (no ring, no label
    line), so the C4 (28.4) / 51.4 pinned outputs hold."""
    for spec in (MLP, KNN, TREE):
        default = svg_architecture(spec, 2, 1)
        explicit = svg_architecture(spec, 2, 1, highlight=None)
        assert default == explicit
        assert HL not in default
        assert "highlight:" not in default
        _assert_xml(default)
    # the empty-spec case is unchanged too
    e = svg_architecture(None, 2, 1)
    assert "no spec to render" in e
    _assert_xml(e)


# --- 53.4 the app (A43) -------------------------------------------------------------

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


def test_app_run_decision_views(tmp_path):
    """A43 (SPEC.md 53.4): a finished run renders the per-candidate
    gate number-lines inside the drill-downs, the live-Pareto-frontier
    SVG, and the champion spec card; the stored result carries the
    champion re-render's `state_dim`/`n_out` (53.4.3)."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    res = at.session_state["result"]
    assert res is not None, "the run finished and persisted its result"
    # 53.4.3: the champion re-render inputs are stored
    assert isinstance(res["state_dim"], int)
    assert isinstance(res["n_out"], int)
    assert res["res"]["best_spec"], "the run produced a best spec"

    subs = " | ".join(str(s.value) for s in at.subheader)
    assert "Decision views (SPEC.md 53.2)" in subs, subs
    assert "Champion spec (SPEC.md 53.3)" in subs, subs

    mds = [str(m.value) for m in at.markdown]
    # 53.4.1: one gate number-line per candidate drill-down (at least
    # the scored candidates; the 2-experiment run scored at least one)
    gate_lines = [m for m in mds if 'aria-label="gate number-line"' in m]
    scored = [u for u in res["stream"]
              if isinstance(u.get("candidate_score"), (int, float))]
    assert len(gate_lines) >= len(scored) >= 1, \
        (len(gate_lines), len(scored))
    # 53.4.2/53.4.3: the live-Pareto-frontier SVG (Run + Experiments)
    assert any("live Pareto frontier" in m for m in mds)
    # 53.4.3: the champion spec card (the C4 architecture diagram)
    champ = [m for m in mds if "best spec architecture" in m]
    assert champ, "the Experiments tab renders the champion spec card"
    for m in mds:  # every rendered SVG is valid XML (G2)
        if m.lstrip().startswith("<svg"):
            _assert_xml(m)


def test_app_source_wires_53():
    """A43 (SPEC.md 53.4): the app source wires the three new
    renderers — the number-line in the shared drill body (with the
    target), the two live placeholders in the drain loop (the champion
    card gated on acceptance with the mutated-field highlight), and the
    finished Experiments-tab views over the stored stream."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "svg_gate_line(",
        "svg_live_frontier(",
        "svg_architecture(",
        "highlight=hl if isinstance(hl, str) else None",
        "vfrontier = st.empty()",
        "vchamp = st.empty()",
        'if u["accepted"]:',
        "runner.env.best_spec.to_dict()",
        '"state_dim": runner.env.task.state_dim',
        '"n_out": runner.env.task.n_outputs',
        "info[\"target\"]",
        "info.get(\"target\")",
        '(payload.get("res") or {}).get("best_spec")',
        "Decision views (SPEC.md 53.2)",
        "Champion spec (SPEC.md 53.3)",
    ):
        assert token in src, token


# --- A43 round regression -----------------------------------------------------------

def test_version_round_v039():
    """A43 (SPEC.md 53.5, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"


def test_spec_cites_a43_and_round():
    """A43 (SPEC.md 53.5/53.6): SPEC.md defines the A43 acceptance block
    and the M42 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 44)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 53.5 Acceptance (A43)" in spec
    assert re.search(r"^\s*\| M42 \| v0\.39\s*\|\s*53\s*\|\s*A43\s*\|",
                     spec, re.MULTILINE)
    assert "**M42**" in spec
