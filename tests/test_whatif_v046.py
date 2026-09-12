"""v0.46 — "C. Parameters: visually interpreted and modified (part 2)"
(SPEC.md 60, A50, M49).

Covers the four A50 items as implemented — all *pure derivations* over data
that already exists (the ``experiments.jsonl`` entries, the ``SPEC_FIELDS``
registry, the 57.2 per-field stats) with **zero retraining** and zero new
logged fields (G2). The pre-v0.46 default path is byte-identical, and the
A1–A49 acceptance pins stay green:

- 60.1 the **live "what-if" preview** — ``whatif_preview`` (the would-be
  candidate validated against the §59.2 registry rule, the estimated
  effect read off the 57.2 logged response surface, honest ``None``
  "no data yet" for never-logged values, the pool size);
- 60.2 the **spec fingerprint / "DNA"** — ``spec_fingerprint`` (one row
  per registry field, the value normalized within its own space) and
  ``fingerprint_diff`` (two specs, the signed ``t`` delta, unset →
  ``None``);
- 60.3 the **interaction heatmap** — ``interaction_matrix`` (which field
  mutations co-occurred in the same candidate + their joint effect vs the
  run's mean, the scored-candidate pool rule, the empty shape);
- 60.4 the **objective-weight reslice** — ``weighted_reslice`` (the
  pool-relative composite over the score / train-time / model-size axes,
  renormalized over the axes each candidate has, the 40.2 final
  tie-break, loud weight / target validation, the zero shape);
- the four SVG renderers (``svg_whatif_effect`` / ``svg_spec_fingerprint``
  / ``svg_interaction_heatmap`` / ``svg_weighted_reslice``) — valid XML,
  the empty states, the okabe / dark palettes, the loud ``palette``
  validation;
- 60.5 the **app wiring + exports** — the ``What-if & comparison`` block's
  widgets / calls are present in ``dashboard_app.py`` (source tokens) and
  the nine new names are exported from the package top level;
- the A50 round regression — the version stepped to ``0.46.0`` in both
  sources (33.1) and SPEC carries the A50 acceptance block + the M49 row.

House rules (A50): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; the app wiring is tested by
source-token assertions + one light AppTest (streamlit optional).
"""
from __future__ import annotations

import re
import tomllib
import xml.dom.minidom
from pathlib import Path

import pytest

import autorefine
from autorefine import (
    DEFAULT_SPEC,
    fingerprint_diff,
    interaction_matrix,
    spec_fingerprint,
    svg_interaction_heatmap,
    svg_spec_fingerprint,
    svg_weighted_reslice,
    svg_whatif_effect,
    whatif_preview,
    weighted_reslice,
)
from autorefine.improver.specspace import SPEC_FIELDS

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
SPEC = REPO / "SPEC.md"


# --- shared fixtures (synthesized; no cross-test imports, A50) --------------

def _base() -> dict:
    return DEFAULT_SPEC.to_dict()


def _entry(kind: str, score: float, spec: dict, mutation, **kw) -> dict:
    d = {"kind": kind, "holdout_score": score, "spec": spec, "mutation": mutation}
    d.update(kw)
    return d


def _lr_entries() -> list:
    """60.1: baseline (score 50) + one experiment that lifts learning_rate
    to 3e-3 (score 80) — the logged response surface for the preview."""
    b = _base()
    e = dict(b, learning_rate=3e-3)
    return [
        _entry("baseline", 50.0, b, None, spec_hash="h_base",
               gen_gap=0.0, train_seconds=5.0, accepted=True),
        _entry("experiment", 80.0, e, ["learning_rate"], spec_hash="h_exp",
               gen_gap=0.1, train_seconds=6.0, accepted=True),
    ]


def _im_entries() -> list:
    """60.3: baseline (50) + two candidates mutating {lr, batch} (80, 60)
    + one mutating {lr} (70) — overall mean 65, one co-mutated pair."""
    b = _base()
    return [
        _entry("baseline", 50.0, b, None),
        _entry("experiment", 80.0, b, ["learning_rate", "batch_size"]),
        _entry("experiment", 60.0, b, ["learning_rate", "batch_size"]),
        _entry("experiment", 70.0, b, ["learning_rate"]),
    ]


def _reslice_entries() -> list:
    """60.4: three candidates (state_dim=4, n_out=2) — two mlp (8,16)
    (size 218) and one tree (size None); scores 50/90/70, train 10/40/25."""
    b = _base()
    c1 = dict(b, architecture=(8, 16))
    c2 = dict(b, architecture=(8, 16), learning_rate=1e-2)
    c3 = dict(b, model_family="tree", architecture=(2,))
    return [
        _entry("baseline", 50.0, c1, None, train_seconds=10.0),
        _entry("experiment", 90.0, c2, ["learning_rate"], train_seconds=40.0),
        _entry("experiment", 70.0, c3, ["model_family", "architecture"],
               train_seconds=25.0),
    ]


def _parse(svg: str) -> None:
    """A50 (60.5): the SVG must be well-formed XML (parse or raise)."""
    xml.dom.minidom.parseString(svg)


# --- 60.1 the live "what-if" preview (A50) ----------------------------------

def test_whatif_unknown_field_raises():
    """A50 (SPEC.md 60.1): a field outside the registry is a loud
    ``ValueError`` naming the registry (G2, 36.2)."""
    with pytest.raises(ValueError, match="registry"):
        whatif_preview(_lr_entries(), _base(), "nope", 1)


def test_whatif_bad_value_raises():
    """A50 (SPEC.md 60.1): a value the field's registry validator rejects
    (``"bogus"`` on an ordered field) is a ``ValueError``."""
    with pytest.raises(ValueError):
        whatif_preview(_lr_entries(), _base(), "learning_rate", "bogus")


def test_whatif_hand_computed_preview():
    """A50 (SPEC.md 60.1): the would-be candidate (the champion with the
    field force-set) is valid, and its estimated effect is read off the
    logged surface: value 3e-3 → mean 80 (n=1), current 1e-3 → mean 50,
    delta +30, pool 2 (both scored rows)."""
    r = whatif_preview(_lr_entries(), _base(), "learning_rate", 3e-3)
    assert r["candidate"] == dict(_base(), learning_rate=3e-3)
    assert r["valid"] is True
    assert r["error"] is None
    assert r["value_mean"] == 80.0
    assert r["value_n"] == 1
    assert r["current_mean"] == 50.0
    assert r["delta"] == 30.0
    assert r["pool"] == 2


def test_whatif_never_logged_value_is_honest_none():
    """A50 (SPEC.md 60.1): a value never logged has ``value_mean`` /
    ``value_n`` / ``delta`` = ``None`` (an honest "no data yet", never an
    invented number), while ``current_mean`` is still the champion's
    value — both scored rows use batch 32 → (50+80)/2 = 65."""
    r = whatif_preview(_lr_entries(), _base(), "batch_size", 128)
    assert r["value_mean"] is None
    assert r["value_n"] is None
    assert r["delta"] is None
    assert r["current_mean"] == 65.0
    assert r["pool"] == 2


def test_whatif_invalid_combo_reports_the_spec_error():
    """A50 (SPEC.md 60.1): a registry-legal value can still be a bad
    *combination* for the champion's family (a tree depth on an mlp) —
    ``valid`` is False and the ``SpecError`` text is surfaced."""
    r = whatif_preview(_lr_entries(), _base(), "architecture", (1,))
    assert r["valid"] is False
    assert "hidden layer size 1" in r["error"]


def test_whatif_none_best_spec_uses_the_default():
    """A50 (SPEC.md 60.1): a ``None`` champion falls back to
    ``DEFAULT_SPEC`` for the would-be candidate (the pool is unchanged)."""
    r = whatif_preview(_lr_entries(), None, "learning_rate", 1e-2)
    assert r["candidate"] == dict(DEFAULT_SPEC.to_dict(), learning_rate=1e-2)
    assert r["pool"] == 2


# --- 60.2 the spec fingerprint ("DNA") (A50) --------------------------------

def test_fingerprint_rows_and_order():
    """A50 (SPEC.md 60.2): one row per registry field, in registry order
    (deterministic, G2)."""
    rows = spec_fingerprint(_base())
    assert len(rows) == 15
    assert [r["name"] for r in rows] == list(SPEC_FIELDS)


def test_fingerprint_t_values():
    """A50 (SPEC.md 60.2): each value normalized within its own space —
    ordered → min-max, categorical → ordinal position, sequence → the
    max-component scale reading (space max is 64, the (64,32) pair)."""
    fp = {r["name"]: r["t"] for r in spec_fingerprint(_base())}
    assert fp["architecture"] == pytest.approx(0.25)          # 16 / 64
    assert fp["model_family"] == 0.0                          # mlp is first
    assert fp["optimizer"] == pytest.approx(0.5)              # momentum 1/2
    assert fp["learning_rate"] == pytest.approx((1e-3 - 1e-4) / (1e-1 - 1e-4))
    assert fp["batch_size"] == pytest.approx(16 / 112)        # (32-16)/(128-16)
    assert fp["init_scale"] == pytest.approx(1 / 3)           # (1.0-0.5)/(2.0-0.5)
    assert fp["knn_k"] == pytest.approx(0.5)                  # k=5, idx 2/4
    assert fp["activation"] == 0.0                            # tanh is first


def test_fingerprint_t_variant_values():
    """A50 (SPEC.md 60.2): the value only — not the field's default —
    drives ``t`` (a wide spec reads 1.0 on the scale axis, relu is the
    last activation, a tree depth reads its depth / 64)."""
    b = _base(); b["architecture"] = (64, 32); b["activation"] = "relu"
    fp = {r["name"]: r["t"] for r in spec_fingerprint(b)}
    assert fp["architecture"] == pytest.approx(1.0)           # 64 / 64
    assert fp["activation"] == 1.0                            # relu is last

    t = _base(); t["architecture"] = (2,)                      # tree depth 2
    fp2 = {r["name"]: r["t"] for r in spec_fingerprint(t)}
    assert fp2["architecture"] == pytest.approx(2 / 64)


def test_fingerprint_unset_field_is_none():
    """A50 (SPEC.md 60.2): a field absent from the spec has ``t = None``
    (the renderer shows the value with no bar, never a fake 0)."""
    b = _base(); del b["knn_k"]
    row = next(r for r in spec_fingerprint(b) if r["name"] == "knn_k")
    assert row["t"] is None
    assert row["value"] is None


def test_fingerprint_none_spec_is_empty():
    """A50 (SPEC.md 60.2): a ``None`` spec is the empty fingerprint."""
    assert spec_fingerprint(None) == []


def test_fingerprint_diff_deltas():
    """A50 (SPEC.md 60.2): the signed ``t`` delta per field — the mutated
    fields move, unchanged fields read 0.0 (relu 0→1, arch 0.25→1.0)."""
    b2 = dict(_base(), activation="relu", architecture=(64, 32))
    d = {r["name"]: r for r in fingerprint_diff(_base(), b2)}
    assert d["activation"]["delta"] == pytest.approx(1.0)
    assert d["architecture"]["delta"] == pytest.approx(0.75)
    assert d["learning_rate"]["delta"] == 0.0


def test_fingerprint_diff_one_side_unset():
    """A50 (SPEC.md 60.2): when either spec leaves a field unset the delta
    is ``None`` (never an invented number)."""
    a = _base(); del a["knn_k"]
    d = {r["name"]: r for r in fingerprint_diff(a, _base())}
    assert d["knn_k"]["delta"] is None
    assert d["knn_k"]["t_a"] is None
    assert d["knn_k"]["t_b"] == pytest.approx(0.5)


# --- 60.3 the interaction heatmap (A50) ------------------------------------

def test_interaction_overall_and_pool():
    """A50 (SPEC.md 60.3): the run's mean over the whole scored pool (50,
    80, 60, 70 → 65) and the pool size (4, incl. the baseline)."""
    m = interaction_matrix(_im_entries())
    assert m["overall_mean"] == pytest.approx(65.0)
    assert m["n_scored"] == 4


def test_interaction_pair_cell():
    """A50 (SPEC.md 60.3): the {lr, batch} pair co-mutated twice (80, 60)
    → mean 70, delta 5 vs the run mean."""
    m = interaction_matrix(_im_entries())
    assert m["cells"]["batch_size"]["learning_rate"] == {
        "n": 2, "mean": 70.0, "delta": 5.0}


def test_interaction_marginals_and_sorted_fields():
    """A50 (SPEC.md 60.3): the per-field mutation counts and the sorted
    field list (lr in 3 candidates, batch in 2)."""
    m = interaction_matrix(_im_entries())
    assert m["marginals"] == {"learning_rate": 3, "batch_size": 2}
    assert m["fields"] == ["batch_size", "learning_rate"]


def test_interaction_nonregistry_mutations_filtered():
    """A50 (SPEC.md 60.3): a non-registry name in ``mutation`` is dropped
    (G2); a single real field yields a marginal but no pair."""
    b = _base()
    m = interaction_matrix([
        _entry("baseline", 50.0, b, None),
        _entry("experiment", 70.0, b, ["learning_rate", "not_a_field"]),
    ])
    assert m["marginals"] == {"learning_rate": 1}
    assert m["cells"] == {}


def test_interaction_excludes_non_scored_kinds():
    """A50 (SPEC.md 60.3): ``screen`` / ``curriculum`` rows never enter
    the scored pool (the 57.2 rule)."""
    b = _base()
    assert interaction_matrix(
        [_entry("screen", 99.0, b, ["learning_rate"])]
    )["n_scored"] == 0
    assert interaction_matrix(
        [_entry("curriculum", 99.0, b, ["learning_rate"])]
    )["n_scored"] == 0


def test_interaction_excludes_nan_score_and_bad_spec():
    """A50 (SPEC.md 60.3): a NaN score or a non-dict spec never enters the
    pool — the whole thing degrades to the empty shape."""
    b = _base()
    m = interaction_matrix([
        _entry("experiment", float("nan"), b, ["learning_rate"]),
        {"kind": "experiment", "holdout_score": 90.0,
         "spec": "not-a-dict", "mutation": ["learning_rate"]},
    ])
    assert m["n_scored"] == 0
    assert m["overall_mean"] is None


def test_interaction_none_row_ignored():
    """A50 (SPEC.md 60.3): a non-dict row is skipped; the baseline (no
    mutation) is scored but contributes no pair."""
    b = _base()
    m = interaction_matrix([None, _entry("baseline", 50.0, b, None)])
    assert m["n_scored"] == 1
    assert m["cells"] == {}


def test_interaction_empty_pool_shape():
    """A50 (SPEC.md 60.3): an empty pool is the documented zero shape."""
    assert interaction_matrix([]) == {
        "fields": [], "cells": {}, "marginals": {},
        "overall_mean": None, "n_scored": 0}


# --- 60.4 the objective-weight reslice (A50) --------------------------------

def test_reslice_hand_computed():
    """A50 (SPEC.md 60.4): the pool-relative composite for w=(1,1,0.5),
    target 0.5 — the two mlp size 218 (degenerate size axis reads 0.0),
    the tree unsized (axis dropped); composites 0.4/0.4/0.5, only the
    tree passes, final = candidate 3."""
    r = weighted_reslice(_reslice_entries(), 1, 1, 0.5, 0.5, 4, 2)
    assert [c["composite"] for c in r["candidates"]] == [
        pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.5)]
    assert [c["pass"] for c in r["candidates"]] == [False, False, True]
    assert r["final"]["cand"] == 3
    assert r["passing"] == 1
    assert r["pool"] == 3
    assert [c["n_score"] for c in r["candidates"]] == [0.0, 1.0, 0.5]
    assert [c["n_train"] for c in r["candidates"]] == [1.0, 0.0, 0.5]
    assert [c["n_size"] for c in r["candidates"]] == [0.0, 0.0, None]
    assert [c["size"] for c in r["candidates"]] == [218, 218, None]


def test_reslice_tie_break_by_train_time():
    """A50 (SPEC.md 60.4): with equal composites (w=(1,1,0), target 0.4)
    the 40.2 final tie-break is lower train time → candidate 1 (10s)."""
    r = weighted_reslice(_reslice_entries(), 1, 1, 0, 0.4, 4, 2)
    assert [c["composite"] for c in r["candidates"]] == [pytest.approx(0.5)] * 3
    assert all(c["pass"] for c in r["candidates"])
    assert r["final"]["cand"] == 1


def test_reslice_renormalizes_over_present_axes():
    """A50 (SPEC.md 60.4): the composite renormalizes over the axes each
    candidate has (w=(0,1,1)) — composites 0.5/0.0/0.5, candidate 2
    fails, final = candidate 1."""
    r = weighted_reslice(_reslice_entries(), 0, 1, 1, 0.25, 4, 2)
    assert [c["composite"] for c in r["candidates"]] == [
        pytest.approx(0.5), pytest.approx(0.0), pytest.approx(0.5)]
    assert [c["pass"] for c in r["candidates"]] == [True, False, True]
    assert r["final"]["cand"] == 1


def test_reslice_no_geometry_drops_the_size_axis():
    """A50 (SPEC.md 60.4): with no geometry every candidate is unsized
    (``n_size = None``); the size axis drops and the composite is
    score/train only (0.5 for all) → final = candidate 1."""
    r = weighted_reslice(_reslice_entries(), 1, 1, 0.5, 0.5, None, None)
    assert [c["n_size"] for c in r["candidates"]] == [None, None, None]
    assert [c["composite"] for c in r["candidates"]] == [pytest.approx(0.5)] * 3
    assert r["final"]["cand"] == 1


def test_reslice_degenerate_pool():
    """A50 (SPEC.md 60.4): a pool with identical score / train / size is
    degenerate on every axis → each composite is 1/3 (w=(1,1,1)), both
    pass target 1/3, final = candidate 1 (log order)."""
    b = _base(); c1 = dict(b, architecture=(8, 16))
    deg = [
        _entry("baseline", 70.0, c1, None, train_seconds=20.0),
        _entry("experiment", 70.0, c1, ["learning_rate"], train_seconds=20.0),
    ]
    r = weighted_reslice(deg, 1, 1, 1, 1 / 3, 4, 2)
    assert [c["composite"] for c in r["candidates"]] == [
        pytest.approx(1 / 3), pytest.approx(1 / 3)]
    assert all(c["pass"] for c in r["candidates"])
    assert r["final"]["cand"] == 1


def test_reslice_weight_validation():
    """A50 (SPEC.md 60.4): all-zero weights and any non-finite / negative /
    bool / string weight are a loud ``ValueError`` (match ``weight``)."""
    pool = _reslice_entries()
    with pytest.raises(ValueError, match="weight"):
        weighted_reslice(pool, 0, 0, 0, 0.5, 4, 2)
    for bad in (-1, True, float("inf"), float("nan"), "1"):
        with pytest.raises(ValueError, match="weight"):
            weighted_reslice(pool, bad, 1, 0, 0.5, 4, 2)
        with pytest.raises(ValueError, match="weight"):
            weighted_reslice(pool, 1, bad, 0, 0.5, 4, 2)
        with pytest.raises(ValueError, match="weight"):
            weighted_reslice(pool, 1, 0, bad, 0.5, 4, 2)


def test_reslice_target_validation():
    """A50 (SPEC.md 60.4): a target outside ``[0, 1]`` (or non-finite /
    non-numeric) is a loud ``ValueError`` (match ``target``)."""
    pool = _reslice_entries()
    for bad in (-0.1, 1.5, float("inf"), "0.5", None):
        with pytest.raises(ValueError, match="target"):
            weighted_reslice(pool, 1, 1, 0, bad, 4, 2)


def test_reslice_empty_pool_shape():
    """A50 (SPEC.md 60.4): an empty scored pool is the documented zero
    shape (``pass: False``, ``final: None``)."""
    assert weighted_reslice([], 1, 1, 0.5, 0.5, 4, 2) == {
        "weights": {"score": 1.0, "train": 1.0, "size": 0.5},
        "target": 0.5, "pool": 0, "passing": 0,
        "pass": False, "final": None, "candidates": []}


def test_reslice_excludes_unscored_rows():
    """A50 (SPEC.md 60.4): a ``screen`` row (or any non-baseline /
    experiment kind) never enters the pool."""
    b = _base(); c1 = dict(b, architecture=(8, 16))
    r = weighted_reslice(
        [_entry("screen", 99.0, c1, None, train_seconds=1.0)],
        1, 1, 0.5, 0.5, 4, 2)
    assert r["pool"] == 0
    assert r["pass"] is False


# --- 60.5 the SVG renderers (A50) ------------------------------------------

def test_svg_whatif_effect_tokens():
    """A50 (SPEC.md 60.5): the effect bar chart carries the signed delta,
    the "vs current value" caption, and a per-bar ``<title>``."""
    s = svg_whatif_effect(
        whatif_preview(_lr_entries(), _base(), "learning_rate", 3e-3))
    _parse(s)
    for tok in ("+30.00", "vs current value", "<title>"):
        assert tok in s, tok


def test_svg_whatif_effect_empty_state():
    """A50 (SPEC.md 60.5): a never-logged value is an honest empty state —
    "no logged candidates with this value yet", "unknown — not zero"."""
    s = svg_whatif_effect(
        whatif_preview(_lr_entries(), _base(), "batch_size", 128))
    _parse(s)
    for tok in ("no logged candidates with batch_size=128 yet",
                "unknown", "not zero"):
        assert tok in s, tok


def test_svg_spec_fingerprint_tokens():
    """A50 (SPEC.md 60.5): the single-spec fingerprint shows the field
    labels and the "no bar = field unset" caption."""
    s = svg_spec_fingerprint(_base())
    _parse(s)
    assert "learning_rate" in s
    assert "no bar = field unset" in s


def test_svg_spec_fingerprint_paired():
    """A50 (SPEC.md 60.5): the paired fingerprint is labeled "A vs B" with
    the "A (top) vs B (bottom)" caption."""
    b2 = dict(_base(), activation="relu", architecture=(64, 32))
    s = svg_spec_fingerprint(_base(), b2)
    _parse(s)
    assert "A vs B" in s
    assert "A (top) vs B (bottom)" in s


def test_svg_spec_fingerprint_empty():
    """A50 (SPEC.md 60.5): a spec with no set fields is the empty state."""
    s = svg_spec_fingerprint({})
    _parse(s)
    assert "no fingerprintable fields set" in s


def test_svg_interaction_heatmap_tokens():
    """A50 (SPEC.md 60.5): the heatmap carries the pair count, the run
    mean, the "scored candidate(s) in the pool" caption, a per-cell
    ``<title>``."""
    s = svg_interaction_heatmap(interaction_matrix(_im_entries()))
    _parse(s)
    for tok in ("n=2", "65.00", "4 scored candidate(s) in the pool",
                "<title>"):
        assert tok in s, tok


def test_svg_interaction_heatmap_empty():
    """A50 (SPEC.md 60.5): no co-mutated pair is the empty state."""
    s = svg_interaction_heatmap(interaction_matrix([]))
    _parse(s)
    assert "no co-mutated field pairs" in s


def test_svg_weighted_reslice_tokens():
    """A50 (SPEC.md 60.5): the reslice carries the target line, the
    counterfactual final star, the weight header, the pass count, and a
    per-dot ``<title>``."""
    s = svg_weighted_reslice(
        weighted_reslice(_reslice_entries(), 1, 1, 0.5, 0.5, 4, 2))
    _parse(s)
    for tok in ("target 0.50", "\u2605", "w(score)=1",
                "1/3 pass the weighted gate", "<title>"):
        assert tok in s, tok


def test_svg_weighted_reslice_empty():
    """A50 (SPEC.md 60.5): an empty pool is the empty state."""
    s = svg_weighted_reslice(weighted_reslice([], 1, 1, 0.5, 0.5, 4, 2))
    _parse(s)
    assert "no scored candidates to re-gate" in s


def test_renderers_palette_and_dark():
    """A50 (SPEC.md 60.5): the okabe (colorblind-safe) and dark palettes
    each change the rendered output (the accessibility surfaces)."""
    base_svg = svg_spec_fingerprint(_base())
    assert svg_spec_fingerprint(_base(), palette="okabe") != base_svg
    assert svg_spec_fingerprint(_base(), dark=True) != base_svg


def test_renderers_bad_palette_raises():
    """A50 (SPEC.md 60.5): an unknown palette is a loud ``ValueError``
    (match ``palette``) on every renderer."""
    preview = whatif_preview(_lr_entries(), _base(), "learning_rate", 3e-3)
    with pytest.raises(ValueError, match="palette"):
        svg_whatif_effect(preview, palette="bogus")
    with pytest.raises(ValueError, match="palette"):
        svg_spec_fingerprint(_base(), palette="bogus")
    with pytest.raises(ValueError, match="palette"):
        svg_interaction_heatmap(interaction_matrix(_im_entries()),
                                palette="bogus")
    with pytest.raises(ValueError, match="palette"):
        svg_weighted_reslice(
            weighted_reslice(_reslice_entries(), 1, 1, 0.5, 0.5, 4, 2),
            palette="bogus")


# --- 60.5 the app wiring + exports (A50) -----------------------------------

def test_app_wires_whatif_block():
    """A50 (SPEC.md 60.5): the ``What-if & comparison`` block's widgets
    and calls are present in the app (the thin renderer)."""
    src = APP.read_text(encoding="utf-8")
    for tok in (
        "What-if & comparison (SPEC.md 60)",
        'key="wf_field"',
        'key="wf_value"',
        'key="fp_compare"',
        'key="wobj_score"',
        'key="wobj_train"',
        'key="wobj_size"',
        'key="wobj_target"',
        "whatif_preview(",
        "interaction_matrix(",
        "weighted_reslice(",
        "svg_whatif_effect(",
        "svg_spec_fingerprint(",
        "svg_interaction_heatmap(",
        "svg_weighted_reslice(",
    ):
        assert tok in src, tok


def test_app_default_path_renders_whatif(tmp_path):
    """A50 (SPEC.md 60.5): the default path runs end-to-end and the
    Results tab renders the What-if & comparison block (AppTest-safe; the
    block adds only selectboxes / a checkbox / sliders)."""
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

    assert at.session_state["result"] is not None
    sub_headers = [str(s.value) for s in at.subheader]
    assert "What-if & comparison (SPEC.md 60)" in sub_headers, sub_headers


def test_all_exports_whatif_round():
    """A50 (SPEC.md 60.5, 33.1): the five whatif functions + four SVG
    renderers are exported from the package top level and in ``__all__``."""
    for name in ("whatif_preview", "spec_fingerprint", "fingerprint_diff",
                 "interaction_matrix", "weighted_reslice",
                 "svg_whatif_effect", "svg_spec_fingerprint",
                 "svg_interaction_heatmap", "svg_weighted_reslice"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


# --- A50 round regression ----------------------------------------------------

def test_version_round_v046():
    """A50 (SPEC.md 60.6, 33.1): the version stepped to ``0.46.0`` in both
    sources (v0.46 ⇒ ``0.46.0``, M49, SPEC.md 60)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"


def test_spec_cites_a50_and_round():
    """A50 (SPEC.md 60.6): SPEC.md defines the A50 acceptance block and the
    M49 index row + milestone — the A25 index machinery reads both
    (``defined == set(range(1, 51))`` includes this round)."""
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 60.6 Acceptance (A50)" in spec
    assert re.search(r"^\s*\| M49 \| v0\.46\s*\|\s*60\s*\|\s*A50\s*\|",
                     spec, re.MULTILINE)
    assert "**M49**" in spec
