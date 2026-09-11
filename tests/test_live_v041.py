"""v0.41 — "make it feel live": the three live-view helpers (SPEC.md 55,
A45, M44).

Covers:

- 55.1 the opt-in **stream mode** — the two pure helpers it renders:
  * ``freshness_caption(n, elapsed_s, done)`` — the live freshness line
    (``live`` / ``done``, the experiment count, ``just now`` under 5 s,
    ``<n>s ago`` after, ``?`` for a non-finite elapsed);
  * ``stream_tick_due(record)`` — the non-blocking decision (the worker
    thread still alive and not drained → tick; a dead / absent thread or a
    drained record → finish);
  * ``svg_live_sparkline`` — the pulsing best-score sparkline (a flash dot +
    halo when ``live=True``, a plain dot when ``live=False``), the optional
    target line, the empty case, palette/dark byte-identity, ARIA, XML;
- 55.2 the **mid-run status snapshot** — ``status_snapshot(info, stream,
  best_series, done)``: the self-contained text card (task/head/rows/
  baseline/target/best/budget/ETA + the last 3 decisions), the live→final
  header flip, and the graceful degradation over missing / non-finite fields;
  ``_status_inputs`` (app) resolves the tuple from the finished ``result``
  or a live ``record``;
- 55.3 the **reference-run overlay** — ``reference_curve`` (a past run's
  running-best curve from its ``experiments.jsonl``), ``list_reference_runs``
  (the runs dir's scored runs, sorted, empty-safe), and
  ``svg_reference_curve`` (the current curve solid on top, the reference
  dashed + faint beneath, the shared index axis, the empty / reference-only
  edges, palette/dark byte-identity, ARIA, XML);
- the A45 round regression — the version stepped to ``0.41.0`` in both
  sources (33.1) and SPEC carries the A45 block + M44 row.

House rules: no cross-test imports (all fixtures synthesized here); the pure
core (``live.py``) is tested by hand-computation; the app (``dashboard_app.py``)
is a thin renderer tested by source-token assertions + one end-to-end AppTest
(streamlit optional).
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import autorefine
from autorefine.live import (
    freshness_caption,
    list_reference_runs,
    reference_curve,
    snapshot_inputs,
    status_snapshot,
    stream_tick_due,
)
from autorefine.plotting import svg_live_sparkline, svg_reference_curve

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


def _assert_xml(svg: str) -> None:
    ET.fromstring(svg)  # valid XML (G2)


def _alive_thread(seconds: float = 0.4) -> threading.Thread:
    t = threading.Thread(target=lambda: time.sleep(seconds))
    t.start()
    return t


# --- 55.1 freshness caption + stream tick decision (A45) ---------------------

def test_freshness_caption_hand_computed():
    """A45 (SPEC.md 55.1): the state word, the experiment count, and the
    freshness (``just now`` < 5 s, ``<n>s ago`` after, ``?`` for a
    non-finite elapsed); ``done`` flips the state word; a non-finite count
    reads 0."""
    assert (freshness_caption(3, 2.0)
            == "live · 3 experiment(s) in · updated just now")
    assert (freshness_caption(3, 12.0)
            == "live · 3 experiment(s) in · updated 12s ago")
    assert (freshness_caption(5, 2.0, done=True)
            == "done · 5 experiment(s) in · updated just now")
    # a non-finite elapsed reads `?` (never a crash)
    assert (freshness_caption(3, float("nan"))
            == "live · 3 experiment(s) in · updated ?")
    # a non-finite / missing count degrades to 0
    assert (freshness_caption(None, 2.0)
            == "live · 0 experiment(s) in · updated just now")
    assert (freshness_caption(0, 6.0)
            == "live · 0 experiment(s) in · updated 6s ago")


def test_stream_tick_due():
    """A45 (SPEC.md 55.1): the tick fires only when the worker thread is
    still alive and the record is not drained — a dead / absent thread or a
    drained record reads as "finish" (the default synchronous path)."""
    assert stream_tick_due(None) is False  # non-dict-safe
    assert stream_tick_due({}) is False  # no thread
    t = _alive_thread()
    try:
        assert stream_tick_due({"drained": False, "thread": t}) is True
        # a drained record does not tick, even with a live thread
        assert stream_tick_due({"drained": True, "thread": t}) is False
    finally:
        t.join()
    # a dead (finished) thread does not tick
    t2 = threading.Thread(target=lambda: None)
    t2.start()
    t2.join()
    assert stream_tick_due({"drained": False, "thread": t2}) is False


def test_svg_live_sparkline_live_vs_static():
    """A45 (SPEC.md 55.1.2): ``live=True`` draws the pulsing last point — a
    soft halo (``r="7" opacity="0.28"``) around the dot — while ``live=False``
    draws only the plain dot; both carry the ``current best`` title, ARIA,
    and valid XML."""
    series = [60.0, 72.5, 72.5, 80.0]
    live = svg_live_sparkline(series)
    static = svg_live_sparkline(series, live=False)
    _assert_xml(live)
    _assert_xml(static)
    # the live halo is present only when live
    assert 'r="7"' in live and 'opacity="0.28"' in live
    assert 'r="7"' not in static  # no halo when static
    assert 'r="4"' in live and 'r="4"' in static  # the dot in both
    assert "current best 80.00 (live)" in live
    assert "current best 80.00" in static
    assert "(live)" not in static
    assert 'role="img"' in live and "aria-label=" in live


def test_svg_live_sparkline_target_and_empty():
    """A45 (SPEC.md 55.1.2): an optional target line with a ``target`` title
    (the y-range expands to include it, so a finite target is always drawn);
    the empty case renders the header + message."""
    live = svg_live_sparkline([60.0, 70.0], target=65.0)
    _assert_xml(live)
    assert "target 65" in live
    # a far-outside finite target still draws (the range expands to include it)
    assert "target 999" in svg_live_sparkline([60.0, 70.0], target=999.0)
    empty = svg_live_sparkline([])
    _assert_xml(empty)
    assert "no experiments yet" in empty


def test_svg_live_sparkline_palette_dark():
    """A45 (SPEC.md 55.1.2 / 51.4): the default call is byte-identical to an
    explicit one (G2); the Okabe set re-colors; dark carries the dark
    surface; an unknown palette is a ``ValueError``."""
    series = [60.0, 70.0, 80.0]
    default = svg_live_sparkline(series)
    assert default == svg_live_sparkline(series, palette="default", dark=False)
    okabe = svg_live_sparkline(series, palette="okabe")
    assert "#56B4E9" in okabe
    dark = svg_live_sparkline(series, dark=True)
    assert "#0e1117" in dark and "#c9d1d9" in dark
    assert "#0e1117" not in default
    with pytest.raises(ValueError):
        svg_live_sparkline(series, palette="nope")


# --- 55.2 the mid-run status snapshot (A45) ----------------------------------

_INFO = {
    "label": "churn", "head": "accuracy",
    "rows": {"train": 40, "holdout": 12, "gen": 8},
    "baseline_score": 60.0, "target": 95.0,
    "budget_experiments": 10,
}
_STREAM = [
    {"index": 1, "accepted": True, "candidate_score": 72.5,
     "mutation": ["hidden_dim"], "experiments_left": 9},
    {"index": 2, "accepted": False, "candidate_score": 68.0,
     "mutation": ["lr"], "experiments_left": 8},
    {"index": 3, "accepted": False, "candidate_score": None,
     "mutation": [], "experiments_left": 7},
    {"index": 4, "accepted": True, "candidate_score": 80.0,
     "mutation": ["knn_k"], "experiments_left": 6},
]
_BEST = [60.0, 72.5, 72.5, 72.5, 80.0]


def test_status_snapshot_hand_computed():
    """A45 (SPEC.md 55.2): the self-contained card — the state header, the
    task/head, the rows, baseline/target, the running best, the budget
    spent (with the remaining), and the last 3 decisions (scored + dup)."""
    snap = status_snapshot(_INFO, _STREAM, _BEST, done=False)
    assert "AutoRefine run — LIVE" in snap
    assert "task: churn · head: accuracy" in snap
    assert "rows: train 40 / holdout 12 / gen 8" in snap
    assert "baseline: 60.00" in snap
    assert "target: 95.0" in snap
    assert "best so far: 80.00" in snap
    assert "budget: 4 of 10 experiments spent · 6 left" in snap
    assert "last 3 decisions:" in snap
    assert "#4 accepted (80.00): knn_k" in snap
    assert "#3 rejected (dup): -" in snap
    assert "#2 rejected (68.00): lr" in snap


def test_status_snapshot_done_flip():
    """A45 (SPEC.md 55.2): ``done`` flips the LIVE → FINAL header; the body
    is otherwise identical."""
    live = status_snapshot(_INFO, _STREAM, _BEST, done=False)
    final = status_snapshot(_INFO, _STREAM, _BEST, done=True)
    assert "AutoRefine run — FINAL" in final
    assert "AutoRefine run — FINAL" not in live
    # the same decisions in both (only the header differs)
    assert final.split("\n", 1)[1] == live.split("\n", 1)[1]


def test_status_snapshot_degrades():
    """A45 (SPEC.md 55.2): a missing / non-finite field degrades to an
    omitted line (never a crash); an empty stream reads ``(none yet)``."""
    empty = status_snapshot(None, [], [], done=False)
    assert "AutoRefine run — LIVE" in empty
    assert "task: ? · head: ?" in empty
    assert "last 3 decisions:" in empty
    assert "  (none yet)" in empty
    # no budget line without a budget + experiments_left
    assert "budget:" not in empty
    # a non-finite baseline is omitted (not a crash)
    info = {"baseline_score": float("nan"), "target": 95.0}
    snap = status_snapshot(info, [], [60.0], done=False)
    assert "baseline:" not in snap
    assert "best so far: 60.00" in snap


def test_snapshot_inputs_hand_computed():
    """A45 (SPEC.md 55.2): ``snapshot_inputs`` (core, ``live.py``) resolves
    the ``(info, stream, best_series, done)`` tuple from the finished
    ``result`` (done=True) or a live ``record`` (done=False); before either
    it returns the empty defaults."""
    rec = {"messages": [
        ("info", {"baseline_score": 60.0}),
        ("update", {"best_score": 72.5}),
        ("update", {"best_score": 80.0}),
    ], "drained": False}
    # live record
    info, stream, best, done = snapshot_inputs(rec, None)
    assert done is False
    assert info["baseline_score"] == 60.0
    assert best == [60.0, 72.5, 80.0]
    assert len(stream) == 2
    # finished result wins over the live record (done=True)
    result = {"info": _INFO, "stream": _STREAM, "best_series": _BEST}
    rinfo, rstream, rbest, rdone = snapshot_inputs(rec, result)
    assert rdone is True and rbest == _BEST and len(rstream) == 4
    # nothing yet -> empty defaults
    assert snapshot_inputs(None, None) == (None, [], [], False)


# --- 55.3 the reference-run overlay (A45) ------------------------------------

def _write_run(runs_dir: Path, run_id: str, scores) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "baseline", "holdout_score": scores[0]})]
    lines += [json.dumps({"kind": "experiment", "holdout_score": s})
              for s in scores[1:]]
    (d / "experiments.jsonl").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")


def test_reference_curve_hand_computed(tmp_path):
    """A45 (SPEC.md 55.3): a past run's running-best curve from its
    ``experiments.jsonl`` (a rejected dip never lowers it); ``[]`` when the
    run or dir is missing."""
    _write_run(tmp_path, "runA", [60.0, 72.0, 68.0, 80.0])
    assert reference_curve(str(tmp_path), "runA") == [60.0, 72.0, 72.0, 80.0]
    assert reference_curve(str(tmp_path), "nope") == []  # missing run
    assert reference_curve(str(tmp_path / "missing"), "runA") == []


def test_list_reference_runs_hand_computed(tmp_path):
    """A45 (SPEC.md 55.3): the runs dir's scored runs, as
    ``[(run_id, best_series)]`` sorted by run_id; a run with no scored rows
    (or a non-dir entry) is skipped; a missing dir reads ``[]``."""
    _write_run(tmp_path, "b", [60.0, 70.0])
    _write_run(tmp_path, "a", [50.0, 80.0])
    # a run dir with no scored rows -> skipped
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "experiments.jsonl").write_text(
        json.dumps({"kind": "experiment", "holdout_score": None}) + "\n",
        encoding="utf-8")
    # a plain file in the runs dir -> skipped
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    out = list_reference_runs(str(tmp_path))
    assert [name for name, _c in out] == ["a", "b"]  # sorted
    by_name = dict(out)
    assert by_name["a"] == [50.0, 80.0]
    assert by_name["b"] == [60.0, 70.0]
    assert list_reference_runs(str(tmp_path / "missing")) == []


def test_svg_reference_curve_hand_computed():
    """A45 (SPEC.md 55.3): the current curve is solid (``_ACCEPTED``) on
    top; the reference is dashed + faint (``opacity="0.75"``) beneath; the
    legend shows ``current`` always and ``reference`` only when present;
    valid XML."""
    svg = svg_reference_curve([60.0, 70.0, 80.0], [50.0, 60.0, 75.0, 85.0])
    _assert_xml(svg)
    assert 'stroke-dasharray="5 4" opacity="0.75"' in svg  # reference
    assert "reference (past run)" in svg
    assert 'stroke-width="2.5"' in svg  # the current curve
    assert "current run" in svg
    assert "current" in svg and "reference" in svg  # both legend entries
    assert svg.count('<circle') == 3  # one dot per current point


def test_svg_reference_curve_edges():
    """A45 (SPEC.md 55.3): an empty ``current`` renders the empty header +
    message; an empty ``reference`` (with a non-empty ``current``) renders
    only the current curve (no reference polyline / legend entry)."""
    empty = svg_reference_curve([], [50.0, 60.0])
    _assert_xml(empty)
    assert "no live curve to overlay" in empty
    cur_only = svg_reference_curve([60.0, 70.0], [])
    _assert_xml(cur_only)
    assert 'stroke-dasharray="5 4" opacity="0.75"' not in cur_only
    assert "reference (past run)" not in cur_only
    assert "current run" in cur_only


def test_svg_reference_curve_palette_dark():
    """A45 (SPEC.md 55.3 / 51.4): the default call is byte-identical to an
    explicit one (G2); the Okabe set re-colors; dark carries the dark
    surface; an unknown palette is a ``ValueError``."""
    cur, ref = [60.0, 70.0], [50.0, 60.0]
    default = svg_reference_curve(cur, ref)
    assert (default == svg_reference_curve(cur, ref,
                                           palette="default", dark=False))
    okabe = svg_reference_curve(cur, ref, palette="okabe")
    # the two series colors swap: current `_ACCEPTED`, reference `_REJECTED`
    assert "#009E73" in okabe and "#7f7f7f" in okabe
    assert "#15803d" not in okabe
    dark = svg_reference_curve(cur, ref, dark=True)
    assert "#0e1117" in dark and "#c9d1d9" in dark
    assert "#0e1117" not in default
    with pytest.raises(ValueError):
        svg_reference_curve(cur, ref, palette="nope")


# --- 55.4 the app (A45) -------------------------------------------------------

def test_app_source_wires_55():
    """A45 (SPEC.md 55.4): the app wires the opt-in live path (the
    ``live_mode`` checkbox, the ``stream_mode=`` drain / tick, the
    ``st.rerun``), the mid-run ``copy_status`` button + ``status_snapshot``
    render, and the ``ref_run`` selectbox + ``svg_reference_curve`` render —
    all distinct keys, all default no-op (the synchronous path is
    byte-identical)."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        'key="live_mode"',
        "stream_mode: bool = False",  # _run default (byte-identical, 55.4)
        "_drain_live(record, narrate, stream_mode=stream_mode)",
        "_drain_live(rec, narrate, stream_mode=live_mode)",
        "stream_tick_due",
        "st.rerun()",
        "freshness_caption",
        "svg_live_sparkline",
        'key="copy_status"',
        "status_snapshot(",
        "list_reference_runs(",
        'key="ref_run"',
        "svg_reference_curve(",
        "snapshot_inputs(rec, result)",
        "from autorefine.live import (",
        "    snapshot_inputs,  # 55.2: the (info, stream, best, done) resolver",
    ):
        assert token in src, token


def test_app_default_path_end_to_end(tmp_path):
    """A45 (SPEC.md 55.4): the *default* (non-stream) path runs
    end-to-end — a finished run persists its result, and the new
    ``copy_status`` button + ``ref_run`` selectbox are present (gated on a
    run existing) without breaking the pre-v0.41 surface."""
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
    # the new live-view widgets are present (gated on a run existing)
    assert any(b.key == "copy_status" for b in at.button), \
        "the copy_status button renders once a run exists"
    assert any(s.key == "ref_run" for s in at.selectbox), \
        "the ref_run selectbox renders once a run exists"
    # pressing the copy button renders the status snapshot (55.2)
    at.button(key="copy_status").set_value(True).run()
    assert not at.exception
    codes = [str(c.value) for c in at.code]
    assert any("AutoRefine run" in c for c in codes), \
        "the status snapshot text card renders"


# --- A45 round regression -----------------------------------------------------

def test_version_round_v041():
    """A45 (SPEC.md 55.5, 33.1): the version stepped to ``0.41.0`` in both
    sources (v0.41 ⇒ ``0.41.0``, M44, SPEC.md 55)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.41.0"


def test_spec_cites_a45_and_round():
    """A45 (SPEC.md 55.5/55.6): SPEC.md defines the A45 acceptance block
    and the M44 index row + milestone — the A25 index machinery reads both
    (defined == set(range(1, 46)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 55.5 Acceptance (A45)" in spec
    assert re.search(r"^\s*\| M44 \| v0\.41\s*\|\s*55\s*\|\s*A45\s*\|",
                     spec, re.MULTILINE)
    assert "**M44**" in spec
