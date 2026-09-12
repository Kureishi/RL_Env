"""v0.40 — hover tooltips on the four zero-hover views + the wall-time
cost strip (SPEC.md 54, A44, M43).

Covers:

- 54.1 the four zero-hover views gain ``<title>`` tooltips:
  * ``svg_score_strip`` — the scored dots gain `` (reason)`` when a reason
    is present (byte-identical when it is falsy); the unscored dots keep
    their existing reason text;
  * ``svg_score_gap_scatter`` — the dots gain `` (reason)`` when present;
  * ``svg_mutation_timeline`` — the cells gain `` @ score`` when scored and
    `` (reason)`` when present;
  * ``svg_confusion_matrix`` — every cell gains a ``true X → predicted Y:
    N (P% of row)`` title (a zero row reads 0.0%);
- 54.2 the wall-time cost strip — ``_time_segments`` (the baseline first,
  one per finite-train-second update, dup / non-finite dropped, a ``None``
  baseline omitted) and ``svg_time_strip`` (hand-computed stacked-bar
  geometry, the per-segment hover titles, the ``total`` caption, the empty
  and zero-total edges, the palette/dark byte-identity + Okabe swap,
  ARIA, valid XML);
- 54.3 the runner — ``start()`` info + ``finish()`` result both carry
  ``baseline_train_seconds`` (a finite ``>= 0`` number) and the result
  carries a valid ``time_strip_svg``;
- 54.4 the app — the source wires the live ``vtime`` placeholder + render,
  the Results-tab ``time_strip_svg`` guard, and the Experiments-tab
  ``Wall-time cost strip (SPEC.md 54.2)`` subheader; a finished run
  renders the strip and a valid ``time_strip_svg``;
- the A44 round regression — the version stepped to ``0.40.0`` in both
  sources (33.1) and SPEC carries the A44 block + M43 row.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy core; streamlit is optional (the app tests skip without
it).
"""
from __future__ import annotations

import math
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import autorefine
from autorefine.dashboard import DashboardRunner
from autorefine.plotting import (
    _time_segments,
    svg_confusion_matrix,
    svg_mutation_timeline,
    svg_score_gap_scatter,
    svg_score_strip,
    svg_time_strip,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


def _assert_xml(svg: str) -> None:
    ET.fromstring(svg)  # valid XML (G2)


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


def _runner(tmp_path, **kw) -> DashboardRunner:
    kw.setdefault("csv_path", _write_csv(tmp_path))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 4)
    kw.setdefault("policy", "bandit")
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    return DashboardRunner(**kw)


# --- 54.1 hover tooltips on the four zero-hover views (A44) -------------------

def test_strip_scored_reason_title():
    """A44 (SPEC.md 54.1.1): the scored dots gain `` (reason)`` when a
    reason is present and stay byte-stable (no suffix) when it is falsy;
    the unscored dots keep their existing reason text (27.1)."""
    stream = [
        {"accepted": True, "candidate_score": 72.5, "reason": "score"},
        {"accepted": False, "candidate_score": 60.0, "reason": None},
        {"accepted": False, "candidate_score": None, "reason": "duplicate"},
    ]
    svg = svg_score_strip(stream)
    _assert_xml(svg)
    assert svg.count("<circle") == 3
    assert "step 1: accepted @ 72.50 (score)" in svg
    # reason falsy -> no suffix, byte-stable (the pre-v0.40 title)
    assert "step 2: scored-rejected @ 60.00</title>" in svg
    # the unscored branch is unchanged (27.1)
    assert "step 3: unscored (duplicate)" in svg


def test_scatter_reason_title():
    """A44 (SPEC.md 54.1.1): the scatter dots gain `` (reason)`` when a
    reason is present and stay byte-stable when it is falsy."""
    stream = [
        {"accepted": True, "candidate_score": 70.0, "gen_gap": 1.0,
         "reason": "ci"},
        {"accepted": False, "candidate_score": 80.0, "gen_gap": 2.0,
         "reason": None},
    ]
    svg = svg_score_gap_scatter(stream)
    _assert_xml(svg)
    assert svg.count("<circle") == 2
    assert "score 70.00, gen_gap 1.000 (ci)" in svg
    # reason falsy -> no suffix, byte-stable (the pre-v0.40 title)
    assert "score 80.00, gen_gap 2.000</title>" in svg


def test_timeline_score_reason_titles():
    """A44 (SPEC.md 54.1.1): the timeline cells gain `` @ score`` when
    scored and `` (reason)`` when a reason is present; an unscored cell
    keeps only the reason (no score)."""
    rows = [
        {"mutation": ["a", "b"], "accepted": True,
         "candidate_score": 70.0, "reason": "ok"},
        {"mutation": ["a"], "accepted": False,
         "candidate_score": 60.0, "reason": None},
        {"mutation": ["b"], "accepted": False,
         "candidate_score": None, "reason": "duplicate"},
    ]
    svg = svg_mutation_timeline(rows)
    _assert_xml(svg)
    assert svg.count("<title>") == 4  # 2 + 1 + 1 cells
    assert "step 1: a (accepted) @ 70.00 (ok)" in svg
    # scored, no reason -> `` @ 60.00`` and no suffix
    assert "step 2: a (scored-rejected) @ 60.00</title>" in svg
    # unscored -> no `` @ score``, the reason only
    assert "step 3: b (unscored) (duplicate)" in svg


def test_matrix_cell_titles():
    """A44 (SPEC.md 54.1.2): every confusion cell carries a
    ``true X → predicted Y: N (P% of row)`` title (the row-normalised
    share; a zero row reads 0.0%)."""
    diag = {"confusion": [[5, 1], [0, 4]], "class_labels": ["A", "B"]}
    svg = svg_confusion_matrix(diag)
    _assert_xml(svg)
    assert svg.count("<title>") == 4  # one per cell
    # row 0 total = 6
    assert "true A \u2192 predicted A: 5 (83.3% of row)" in svg
    assert "true A \u2192 predicted B: 1 (16.7% of row)" in svg
    # row 1 total = 4
    assert "true B \u2192 predicted A: 0 (0.0% of row)" in svg
    assert "true B \u2192 predicted B: 4 (100.0% of row)" in svg


def test_matrix_zero_row_and_empty():
    """A44 (SPEC.md 54.1.2): a zero row reads 0.0% (not a divide error);
    the empty case (no diagnostics) renders the header + message with no
    cell titles."""
    diag = {"confusion": [[5, 0], [0, 0]], "class_labels": ["0", "1"]}
    svg = svg_confusion_matrix(diag)
    _assert_xml(svg)
    assert "true 1 \u2192 predicted 0: 0 (0.0% of row)" in svg
    assert "true 1 \u2192 predicted 1: 0 (0.0% of row)" in svg
    assert "true 0 \u2192 predicted 0: 5 (100.0% of row)" in svg
    empty = svg_confusion_matrix(None)
    assert "no holdout diagnostics" in empty
    assert empty.count("<title>") == 0


# --- 54.2 the wall-time cost strip (A44) ---------------------------------------

def test_time_segments_hand_computed():
    """A44 (SPEC.md 54.2.1): the baseline is the first segment, one
    segment per update with a finite non-negative ``train_seconds``, a
    ``None`` / non-finite / negative / bool train time is dropped (R3),
    non-dict entries are skipped, and a ``None`` baseline is omitted."""
    up = [
        {"index": 1, "train_seconds": 1.0},
        {"index": 2, "train_seconds": None},  # the free dup (R3)
        {"index": 3, "train_seconds": 2.0},
        "not-a-dict",  # non-dict-safe
        {"index": 5, "train_seconds": -1.0},  # negative -> dropped
        {"index": 6, "train_seconds": True},  # bool -> dropped
    ]
    assert _time_segments(up, 2.0) == [
        ("baseline", 2.0), ("exp 1", 1.0), ("exp 3", 2.0)]
    # a None baseline is omitted
    assert _time_segments(up, None) == [("exp 1", 1.0), ("exp 3", 2.0)]
    # no finite train time at all -> empty
    assert _time_segments([{"index": 1, "train_seconds": None}], None) == []


def test_time_strip_geometry_hand_computed():
    """A44 (SPEC.md 54.2.1): width 640, L = R = 24 -> pw = 592; baseline
    2.0 + exp 1.0 + exp 1.0 -> segment widths 296 / 148 / 148, the
    ``total 4.00 s`` caption, the per-segment hover titles, ARIA, and
    valid XML."""
    up = [
        {"index": 1, "train_seconds": 1.0},
        {"index": 2, "train_seconds": 1.0},
        {"index": 3, "train_seconds": None},  # dropped (R3)
    ]
    svg = svg_time_strip(up, 2.0)
    _assert_xml(svg)
    assert svg.startswith("<svg")
    assert 'width="296.0"' in svg  # baseline: 2.0 / 4.0 * 592
    assert svg.count('width="148.0"') == 2  # the two exp segments
    assert 'x="24.0"' in svg  # the bar starts at L
    assert "total 4.00 s" in svg
    assert "baseline: 2.00 s (50.0%)" in svg
    assert "exp 1: 1.00 s (25.0%)" in svg
    assert svg.count("<title>") == 3  # one per segment
    assert 'role="img"' in svg and "aria-label=" in svg  # ARIA (51.4.3)


def test_time_strip_edges():
    """A44 (SPEC.md 54.2.1): the empty case (no finite segments) renders
    a header + message with no segment titles; a zero total renders
    equal-width segments whose titles read ``0.00 s``."""
    empty = svg_time_strip([], None)
    _assert_xml(empty)
    assert "no train-time data" in empty
    assert "total" not in empty
    assert empty.count("<title>") == 0
    zero = svg_time_strip([{"index": 1, "train_seconds": 0.0}], 0.0)
    _assert_xml(zero)
    assert "baseline: 0.00 s (50.0%)" in zero
    assert "total 0.00 s" in zero
    assert zero.count("<title>") == 2


def test_time_strip_palette_dark():
    """A44 (SPEC.md 54.2.1 / 51.4): the default call is byte-identical to
    ``palette="default", dark=False`` (G2); the Okabe set re-colors the
    two series (baseline ``_BASELINE``, experiments ``_LINE``); dark
    carries the dark surface/axis; an unknown palette is a ``ValueError``.
    """
    up = [{"index": 1, "train_seconds": 1.0}, {"index": 2, "train_seconds": 1.0}]
    default = svg_time_strip(up, 2.0)
    assert default == svg_time_strip(up, 2.0, palette="default", dark=False)
    okabe = svg_time_strip(up, 2.0, palette="okabe")
    assert "#E69F00" in okabe and "#56B4E9" in okabe
    assert "#c2410c" not in okabe and "#1a66c2" not in okabe
    dark = svg_time_strip(up, 2.0, dark=True)
    assert "#0e1117" in dark and "#c9d1d9" in dark
    assert "#0e1117" not in default and "#c9d1d9" not in default
    with pytest.raises(ValueError):
        svg_time_strip(up, 2.0, palette="nope")


# --- 54.3 the runner carries the wall-time strip (A44) -------------------------

def test_runner_carries_time_strip(tmp_path):
    """A44 (SPEC.md 54.3): ``start()`` info + ``finish()`` result both
    carry ``baseline_train_seconds`` (a finite ``>= 0`` number) and the
    result carries a valid ``time_strip_svg``."""
    r = _runner(tmp_path)
    info = r.start()
    bts = info["baseline_train_seconds"]
    assert isinstance(bts, (int, float)) and not isinstance(bts, bool)
    assert bts >= 0.0 and math.isfinite(bts)
    res = r.run_all()
    fbts = res["baseline_train_seconds"]
    assert isinstance(fbts, (int, float)) and not isinstance(fbts, bool)
    assert fbts >= 0.0 and math.isfinite(fbts)
    svg = res["time_strip_svg"]
    assert svg.startswith("<svg")
    _assert_xml(svg)
    assert "wall-time cost strip" in svg  # the ARIA label


# --- 54.4 the app (A44) ---------------------------------------------------------

def test_app_source_wires_54():
    """A44 (SPEC.md 54.4): the app source wires the live ``vtime``
    placeholder + render (the drain loop), the Results-tab
    ``time_strip_svg`` guard (the restored-view-safe ``res.get``), and the
    Experiments-tab subheader."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "svg_time_strip",
        "vtime = st.empty()",
        "svg_time_strip(stream,",
        'info.get("baseline_train_seconds")',
        'res.get("time_strip_svg")',
        'svg_time_strip(res["updates"]',
        'res.get("baseline_train_seconds")',
        "Wall-time cost strip (SPEC.md 54.2)",
    ):
        assert token in src, token


def test_app_run_renders_time_strip(tmp_path):
    """A44 (SPEC.md 54.4): a finished run renders the wall-time cost
    strip (the Experiments-tab subheader + a valid ``time_strip_svg`` in
    the stored result)."""
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
    assert res["info"]["baseline_train_seconds"] is not None
    assert res["res"]["time_strip_svg"].startswith("<svg")

    subs = " | ".join(str(s.value) for s in at.subheader)
    assert "Wall-time cost strip (SPEC.md 54.2)" in subs, subs
    mds = [str(m.value) for m in at.markdown]
    strips = [m for m in mds if 'aria-label="wall-time cost strip' in m]
    assert strips, "the wall-time cost strip renders"
    for m in strips:
        _assert_xml(m)


# --- A44 round regression -------------------------------------------------------

def test_version_round_v040():
    """A44 (SPEC.md 54.5, 33.1): the version stepped to ``0.40.0`` in
    both sources (v0.40 ⇒ ``0.40.0``, M43, SPEC.md 54)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"


def test_spec_cites_a44_and_round():
    """A44 (SPEC.md 54.5/54.6): SPEC.md defines the A44 acceptance block
    and the M43 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 45)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 54.5 Acceptance (A44)" in spec
    assert re.search(r"^\s*\| M43 \| v0\.40\s*\|\s*54\s*\|\s*A44\s*\|",
                     spec, re.MULTILINE)
    assert "**M43**" in spec
