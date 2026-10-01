"""Model comparison — dynamic multi-model graphics (SPEC.md 81, A71).

SPEC.md 81 (v0.67): "compare different models on all metrics" — the
run's scored candidates grouped by model family (or the top-K distinct
specs), each group's best candidate read on the full metric catalog
(the six logged metrics + the 58.2 size axis), rendered in three
dynamic encodings:

  * 81.2  — `research.model_comparison`: the family/spec grouping, the
    best-candidate representative + per-metric mean, the scored-
    candidate rule (81.2.2), the loud bad-arg `ValueError`s (81.2.3);
  * 81.3  — `svg_model_radar` / `svg_model_bars` / `svg_model_matrix`:
    beneficial-direction normalization (81.3.1), the n/a encoding for
    the size-less families (81.3.2–81.3.4), the empty degradation
    (81.3.5), pure/deterministic (G2), valid XML, okabe/dark themes
    (51.4);
  * 81.4  — the app's Model comparison panel (the app-language scan in
    `tests/test_app_language.py` keeps the panel strings clean);
  * 81.5  — the version steps to `0.71.0` in both sources (33.1) and
    the A-index advances (asserted in `tests/test_coherency_v021.py`,
    A25).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import math
import tomllib
import xml.dom.minidom
from pathlib import Path

import pytest

import autorefine
from autorefine.config import MODEL_FAMILIES, spec_n_params
from autorefine.plotting import (
    svg_is_well_formed,
    svg_model_bars,
    svg_model_matrix,
    svg_model_radar,
)
from autorefine.research import MODEL_METRICS, model_comparison

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

_MLP = {"model_family": "mlp", "architecture": [16, 8], "activation": "tanh"}
_TREE = {"model_family": "tree", "architecture": [2]}
_KNN = {"model_family": "knn", "architecture": []}


def _strict(svg: str) -> None:
    """A71 (81.3.5): strict XML parse — the renderer must emit a
    complete, well-formed document, not just a string."""
    xml.dom.minidom.parseString(svg)


def _entry(kind: str, spec: dict, holdout: float, **kw) -> dict:
    """A71 (81.2.2): one logged candidate row — the full metric set the
    catalog reads (18.3/30.4 fields), a dict spec, and a kind."""
    e = {
        "kind": kind,
        "spec": spec,
        "spec_hash": kw.pop("spec_hash", ""),
        "holdout_score": holdout,
        "std": 0.5,
        "gen_score": holdout - 1.0,
        "gen_gap": 1.0,
        "train_seconds": 1.5,
        "effective_score": holdout,
        "accepted": kind == "baseline",
    }
    e.update(kw)
    return e


def _entries() -> list:
    """A71 (81.2.3): the synthetic multi-family run — two mlp candidates
    (90.0 then 80.0, the first is the representative), one tree, one
    knn, in log order (the families interleave)."""
    return [
        _entry("baseline", _MLP, 90.0, spec_hash="hmlp1"),
        _entry("experiment", _TREE, 70.0, spec_hash="htree1"),
        _entry("experiment", _MLP, 80.0, spec_hash="hmlp2"),
        _entry("experiment", _KNN, 60.0, spec_hash="hknn1"),
    ]


# --- 81.2 the derivation (A71) --------------------------------------------------

def test_family_grouping_order_and_representative():
    """A71 (81.2.3): family mode walks `MODEL_FAMILIES` (empty families
    skipped), the representative is the best-holdout candidate, and its
    `values`/`mean` are the catalog reads (81.2.1)."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    assert d["group_by"] == "family"
    assert [m["name"] for m in d["models"]] == ["mlp", "tree", "knn"]
    fam = {f: i for i, f in enumerate(MODEL_FAMILIES)}
    assert [fam[m["name"]] for m in d["models"]] == sorted(
        fam[m["name"]] for m in d["models"]), "registry order"
    mlp = d["models"][0]
    assert mlp["label"] == "mlp (16,8)"
    assert mlp["family"] == "mlp"
    assert mlp["n_candidates"] == 2
    # the representative is the 90.0 candidate (best holdout, 81.2.3)
    assert mlp["values"]["holdout_score"] == 90.0
    assert mlp["best_hash"] == "hmlp1"
    # the mean over the group's candidates (the 85.0 of 90/80)
    assert mlp["mean"]["holdout_score"] == pytest.approx(85.0)
    assert mlp["mean"]["train_seconds"] == pytest.approx(1.5)
    tree = d["models"][1]
    assert tree["values"]["holdout_score"] == 70.0
    assert tree["label"] == "tree (2)"  # the depth architecture


def test_size_axis_and_catalog():
    """A71 (81.2.1/81.2.3): the catalog is the seven fixed rows with
    directions; `model_size` is the 58.2 axis — finite for mlp, `None`
    for the data-dependent families."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    assert [(m["name"], m["direction"]) for m in d["metrics"]] == [
        (name, direction) for name, direction, _l in MODEL_METRICS]
    expected_mlp = spec_n_params(_MLP, 4, 2)
    assert expected_mlp == 4 * 16 + 16 + 16 * 8 + 8 + 8 * 2 + 2  # 80+136+18
    assert d["models"][0]["values"]["model_size"] == expected_mlp
    for m in d["models"][1:]:  # tree, knn: data-dependent (58.2)
        assert m["values"]["model_size"] is None
        assert m["mean"]["model_size"] is None
    assert d["n_scored"] == 4


def test_scored_candidate_rule_and_garbage():
    """A71 (81.2.2): only baseline/experiment rows with a finite
    holdout and a dict spec enter — screen/curriculum/invalid rows,
    missing scores, and non-dict garbage are excluded from `n_scored`
    and from the groups."""
    rows = _entries() + [
        _entry("screen", _MLP, 99.0),
        _entry("curriculum", _MLP, 98.0),
        _entry("invalid_spec", _MLP, 97.0),
        {"kind": "experiment", "spec": _MLP},  # no holdout score
        "garbage",  # non-dict
        42,
        None,
        {"kind": "experiment", "spec": None, "holdout_score": 50.0},
    ]
    d = model_comparison(rows, state_dim=4, n_out=2)
    assert d["n_scored"] == 4
    assert [m["name"] for m in d["models"]] == ["mlp", "tree", "knn"]
    assert d["models"][0]["n_candidates"] == 2  # the 99/98/97 rows did not enter


def test_spec_mode_order_and_cap():
    """A71 (81.2.3): spec mode groups by `spec_hash`, orders best
    holdout desc then hash asc, caps at `top_k`, and labels with the
    57.1 family-hash label."""
    d = model_comparison(_entries(), state_dim=4, n_out=2,
                         group_by="spec", top_k=2)
    # each row is its own spec — the top-2 best holdouts (90.0, 80.0)
    assert [m["name"] for m in d["models"]] == ["hmlp1", "hmlp2"]
    assert d["models"][0]["label"].startswith("mlp ")  # the 57.1 label
    assert d["models"][0]["label"].endswith("hmlp1")
    assert d["models"][1]["values"]["holdout_score"] == 80.0
    # a tie on the best holdout breaks on the hash (deterministic)
    tied = [
        _entry("experiment", _MLP, 75.0, spec_hash="zb"),
        _entry("experiment", _TREE, 75.0, spec_hash="za"),
        _entry("experiment", _KNN, 40.0, spec_hash="zz"),
    ]
    d2 = model_comparison(tied, group_by="spec", top_k=2)
    assert [m["name"] for m in d2["models"]] == ["za", "zb"]


def test_bad_args_raise():
    """A71 (81.2.3): a bad `group_by` and a non-positive / non-int
    `top_k` are loud `ValueError`s, not silent defaults."""
    with pytest.raises(ValueError, match="group_by"):
        model_comparison(_entries(), group_by="bogus")
    with pytest.raises(ValueError, match="top_k"):
        model_comparison(_entries(), top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        model_comparison(_entries(), top_k=True)


def test_empty_degradation():
    """A71 (81.2.3): no scored candidates -> `models: []`,
    `n_scored: 0` (the renderers' empty path, 81.3.5)."""
    d = model_comparison([], state_dim=4, n_out=2)
    assert d["models"] == []
    assert d["n_scored"] == 0
    d2 = model_comparison([None, "junk"], group_by="spec", top_k=3)
    assert d2["models"] == [] and d2["n_scored"] == 0


# --- 81.3 the renderers (A71) ----------------------------------------------------

def test_renderers_wellformed_strict_deterministic():
    """A71 (81.3.5): all three renderers are well-formed, strict-
    XML-parseable, and byte-identical across repeated calls (G2)."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    for fn in (svg_model_radar, svg_model_bars, svg_model_matrix):
        a, b = fn(d), fn(d)
        assert a == b, f"{fn.__name__} is not deterministic"
        assert svg_is_well_formed(a)
        _strict(a)
        assert "<title>" in a  # the tooltip idiom (81.3.2–81.3.4)


def test_renderers_themes():
    """A71 (81.3.5/51.4): the okabe palette and dark mode are byte-
    different from the default (the _styled swap works)."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    for fn in (svg_model_radar, svg_model_bars, svg_model_matrix):
        default = fn(d)
        assert fn(d, palette="okabe") != default
        assert fn(d, dark=True) != default
        with pytest.raises(ValueError, match="palette"):
            fn(d, palette="bogus")
        with pytest.raises(ValueError, match="dark"):
            fn(d, dark="yes")


def test_radar_polygons_and_na_vertex():
    """A71 (81.3.2): one polygon per model (the four grid rings are the
    other polygons), the missing size pins the tree/knn vertex at the
    center (an `n/a` title), and the legend names every model."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    svg = svg_model_radar(d)
    assert svg.count('<polygon points="') == 4 + len(d["models"])
    assert "model size: n/a" in svg  # the tree/knn size vertex (81.3.2)
    assert ">mlp (16,8)</text>" in svg  # the legend label
    # a single-axis catalog still renders (degenerate but valid)
    one = dict(d)
    one["metrics"] = d["metrics"][:1]
    assert svg_is_well_formed(svg_model_radar(one))


def test_bars_blocks_and_na():
    """A71 (81.3.3): one block per metric (the 81.2.1 labels), the raw
    values at the bar ends, and the `n/a` read for the size-less
    families."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    svg = svg_model_bars(d)
    for label in ("holdout score", "effective score", "gen score",
                  "holdout std", "gen gap", "train time", "model size"):
        assert f">{label} <title>" in svg, label  # the block header
    assert ">90.0</text>" in svg      # the mlp representative's holdout
    assert ">70.0</text>" in svg      # the tree's holdout
    assert ">n/a</text>" in svg       # the tree/knn size (81.3.3)


def test_matrix_cells():
    """A71 (81.3.4): the short metric header, every raw value in a cell
    (including the `n/a` cells for the size-less families), and the
    row-swatch colors matching the radar's palette."""
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    svg = svg_model_matrix(d)
    for short in ("holdout", "effective", "gen", "std", "gap", "time",
                  "size"):
        assert f">{short} <title>" in svg, short  # the column header
    # the representatives' values (the 80.0 mlp is not a representative)
    for raw in ("90.0", "70.0", "60.0", "1.50", "0.50"):
        assert f">{raw}</text>" in svg, raw
    assert ">n/a</text>" in svg      # the tree/knn size cells (81.3.4)
    # the size cell of the mlp: 234 params (80 + 136 + 18, 58.2)
    assert ">234</text>" in svg


def test_renderers_empty():
    """A71 (81.3.5): no scored candidates -> the header + the empty
    message, still well-formed and strict-XML-parseable."""
    d = model_comparison([])
    for fn in (svg_model_radar, svg_model_bars, svg_model_matrix):
        svg = fn(d)
        assert svg_is_well_formed(svg)
        _strict(svg)
        assert "no scored candidates to compare" in svg


def test_normalize_directions():
    """A71 (81.3.1): the shared normalization runs in each metric's
    beneficial direction — higher: (v−min)/(max−min), lower:
    (max−v)/(max−min), constant → 1.0, missing → None."""
    from autorefine.plotting import _mc_normalize
    d = model_comparison(_entries(), state_dim=4, n_out=2)
    norm = _mc_normalize(d["metrics"], d["models"])
    # holdout (higher): mlp 90 → 1.0, tree 70 → (70−60)/(90−60), knn 60 → 0
    assert norm[("holdout_score", 0)] == 1.0
    assert norm[("holdout_score", 1)] == pytest.approx(1.0 / 3.0)
    assert norm[("holdout_score", 2)] == 0.0
    # train time (lower): all 1.5 → constant → 1.0 (all equally good)
    assert norm[("train_seconds", 0)] == 1.0
    # the missing sizes → None for the size-less families
    assert norm[("model_size", 0)] == pytest.approx(1.0)  # 234 ≤ 234, lower
    assert norm[("model_size", 1)] is None
    assert norm[("model_size", 2)] is None


def test_hostile_labels_stay_well_formed():
    """A71 (81.3.5): hostile spec labels (markup in a hash) stay
    well-formed after html.escape — the document cannot be broken by
    the data."""
    rows = [_entry("baseline", _MLP, 90.0, spec_hash='<b>&"x"')]
    d = model_comparison(rows, group_by="spec", top_k=1)
    for fn in (svg_model_radar, svg_model_bars, svg_model_matrix):
        svg = fn(d)
        assert svg_is_well_formed(svg)
        _strict(svg)


# --- 81.5 regression (A71) -------------------------------------------------------

def test_spec_and_version():
    """A71 (SPEC.md 81.5): SPEC.md defines the A71 acceptance block,
    the M70 index row + milestone, and the version stepped to
    `0.71.0` in both sources (33.1)."""
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 81.5 Acceptance (A71)" in spec
    assert "### 81.6 Milestone (M70)" in spec
    assert ("| M70 | v0.67   | 81     | A71 | "
            "tests/test_model_compare_v067.py |") in spec
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.71.0"
