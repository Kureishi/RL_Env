"""v0.44 — "B. Visually useful for researchers / niche specialized tasks
(round 2)" (SPEC.md 58, A48, M47).

Covers the three A items as implemented — all *pure derivations* over
artifacts that already exist (the ``experiments.jsonl`` entries, the
task's holdout split, the pre-rendered diagnostics), no new logged
field, no loop change (G2):

- 58.1 the **per-input difficulty ranking** (the error-analysis pack) —
  ``diagnostics.holdout_difficulty(task, model, n=200)`` (the signed
  softmax margin, the mse ``|pred − y|`` + 5% tolerance rule, the
  easiest→hardest ordering incl. tie-by-index, the ``None`` cases) +
  ``svg_difficulty_ranking`` (the lollipop rows, the verdict colors,
  the tooltips, the head+tail cap + omitted marker, the signed /
  non-signed axis captions, the empty state);
- 58.2 the **3-objective frontier** — ``config.spec_n_params`` (hand
  computed: mlp 218, convnet 690, the data-dependent families → None,
  the missing-input → None cases) + ``research.frontier3`` (the
  scored-candidate rule, the 3D non-dominance incl. a ``None``-size
  point, ties not dominated, screen-kind / missing-time exclusions) +
  ``svg_frontier3`` (the two panels, the sized / unsized points, the
  frontier vs dominated colors, the tooltips, the legend, the caption,
  the empty states);
- 58.3 the **run dossier** — ``dossier.build_dossier(run_dir, …)``
  (two renders byte-identical, G2, every section present, the recipe
  line + JSON, the error-analysis pack incl. the media gallery, the
  no-classification caption, the broken-run-dir ``ValueError``);
- 58.4 the **surfaces + exports** — the app wires the difficulty SVG +
  the fifth artifacts download button (with the friendly-error path),
  the CLI wires the ``dossier`` subcommand, the six new names are in
  ``__all__``, and the default path renders end-to-end (one light
  AppTest — streamlit optional);
- the A48 round regression — the version stepped to ``0.44.0`` in both
  sources (33.1) and SPEC carries the A48 block + M47 row.

House rules (A48): no cross-test imports (all fixtures synthesized
here); the pure core is tested by hand-computation; the app / dashboard
/ CLI wiring is tested by source-token assertions + one end-to-end
AppTest. Every renderer output is validated as XML
(``xml.dom.minidom``); the okabe / dark re-render + the bad-palette
``ValueError`` are covered (51.4).
"""
from __future__ import annotations

import json
import re
import tomllib
import xml.dom.minidom as minidom
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.config import ModelSpec, spec_n_params
from autorefine.diagnostics import holdout_difficulty
from autorefine.dossier import build_dossier
from autorefine.plotting import svg_difficulty_ranking, svg_frontier3
from autorefine.research import frontier3
from autorefine.runconfig import RunConfig

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
DASH = REPO / "src" / "autorefine" / "dashboard.py"
CLI = REPO / "src" / "autorefine" / "cli.py"


def _parse(svg: str) -> None:
    minidom.parseString(svg)  # valid XML (G2)


def _approx(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


# the frontier3 fixture: a sized dominated point (A1, mlp), a sized
# frontier point (A2, convnet), an unsized frontier point (A3, tree),
# plus the exclusion rows (a screen-kind row and an experiment row
# without train_seconds — both must stay out of the frontier)
F3_ENTRIES = [
    {"kind": "baseline",
     "spec": {"model_family": "mlp", "architecture": [8, 16]},
     "spec_hash": "A1", "holdout_score": 50.0, "train_seconds": 1.0},
    {"kind": "experiment",
     "spec": {"model_family": "convnet", "architecture": [4, 8]},
     "spec_hash": "A2", "holdout_score": 70.0, "train_seconds": 2.0},
    {"kind": "experiment",
     "spec": {"model_family": "tree"},
     "spec_hash": "A3", "holdout_score": 60.0, "train_seconds": 0.5},
    {"kind": "screen",
     "spec": {"model_family": "mlp", "architecture": [8, 16]},
     "holdout_score": 95.0, "train_seconds": 0.1},
    {"kind": "experiment",
     "spec": {"model_family": "mlp", "architecture": [8, 16]},
     "spec_hash": "A4", "holdout_score": 90.0},  # no train_seconds
]

GRID = (1, 8, 8)


# --- 58.2 the size axis: spec_n_params (A48) ---------------------------------

def test_spec_n_params_mlp_hand_computed():
    """A48 (SPEC.md 58.2): mlp ``sizes = [state_dim, *arch, n_out]``; each
    layer contributes ``fan_in·fan_out + fan_out``. arch (8,16),
    state_dim 4, n_out 2 → sizes [4, 8, 16, 2]: 4·8+8 + 8·16+16 + 16·2+2
    = 40 + 144 + 34 = 218. A depth-0 spec is one linear layer: 4·2+2 = 10."""
    assert spec_n_params({"model_family": "mlp", "architecture": [8, 16]},
                         4, 2) == 218
    assert spec_n_params({"model_family": "mlp", "architecture": []},
                         4, 2) == 10


def test_spec_n_params_convnet_hand_computed():
    """A48 (SPEC.md 58.2): convnet arch (4,8) on grid (1,8,8), n_out 2 —
    conv1 4·1·9+4 = 40; conv2 8·4·9+8 = 296; h1 = 6, h1p = 3,
    h2p = max(1, 1//2) = 1 → flat 8·1·1 = 8 → fc 8·32+32 = 288; head
    32·2+2 = 66. Total 40 + 296 + 288 + 66 = 690. A grid with
    h1p = max(1, (4−2)//2) = 1 < 3 is an invalid pipeline → None; a
    missing grid degrades too (the count is not an error)."""
    spec = {"model_family": "convnet", "architecture": [4, 8]}
    assert spec_n_params(spec, 1, 2, grid=(1, 8, 8)) == 690
    assert spec_n_params(spec, 1, 2, grid=(1, 4, 4)) is None
    assert spec_n_params(spec, 1, 2) is None


def test_spec_n_params_data_dependent_and_missing():
    """A48 (SPEC.md 58.2): tree / boost / knn sizes are data-dependent →
    None; missing state_dim / n_out → None; a ``ModelSpec`` instance is
    accepted (the dict convention and the dataclass agree)."""
    for fam in ("tree", "boost", "knn"):
        assert spec_n_params({"model_family": fam}, 4, 2) is None, fam
    assert spec_n_params({"model_family": "mlp", "architecture": [8]},
                         None, 2) is None
    assert spec_n_params({"model_family": "mlp", "architecture": [8]},
                         4, None) is None
    ms = ModelSpec.from_dict({
        "architecture": [8, 16], "optimizer": "momentum",
        "learning_rate": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
        "train_steps": 400, "input_noise": 0.02, "activation": "tanh"})
    assert spec_n_params(ms, 4, 2) == 218


# --- 58.2 the 3-objective frontier: frontier3 (A48) ---------------------------

def test_frontier3_dominance_hand_computed():
    """A48 (SPEC.md 58.2): the scored-candidate rule keeps A1/A2/A3 (the
    screen row and the missing-time row are excluded); sizes 218 / 690 /
    None. The unsized A3 dominates the sized A1 (score 60 ≥ 50, time
    0.5 ≤ 1.0, the ``None`` size neither violates nor establishes — one
    strict on a comparable axis); A2 and A3 stand (A3 is slower? no —
    A2 is *slower*: 70 ≥ 60 but 2.0 > 0.5, and 60 < 70 the other way).
    So frontier = {A2, A3}, unsized = 1."""
    r = frontier3(F3_ENTRIES, 4, 2, grid=GRID)
    assert [p["hash"] for p in r["points"]] == ["A1", "A2", "A3"]
    assert [p["size"] for p in r["points"]] == [218, 690, None]
    assert r["frontier"] == [1, 2]
    assert [p["frontier"] for p in r["points"]] == [False, True, True]
    assert r["unsized"] == 1


def test_frontier3_ties_are_not_dominating():
    """A48 (SPEC.md 58.2): identical points (same score, time, size) carry
    no *strict* inequality on any axis, so neither dominates — both
    stand on the frontier."""
    e = [
        {"kind": "baseline", "spec": {"model_family": "mlp",
                                      "architecture": [4]},
         "spec_hash": "T1", "holdout_score": 50.0, "train_seconds": 1.0},
        {"kind": "experiment", "spec": {"model_family": "mlp",
                                        "architecture": [4]},
         "spec_hash": "T2", "holdout_score": 50.0, "train_seconds": 1.0},
    ]
    r = frontier3(e, 3, 2)
    assert [p["size"] for p in r["points"]] == [26, 26]  # 3·4+4 + 4·2+2
    assert r["frontier"] == [0, 1]
    assert r["unsized"] == 0


# --- 58.1 the per-input difficulty ranking core (A48) --------------------------

class _FakeModel:
    def __init__(self, out: np.ndarray) -> None:
        self._out = out

    def forward(self, x) -> np.ndarray:
        return self._out


class _SoftmaxTask:
    head = "softmax"

    def __init__(self, y: list[int]) -> None:
        self._y = np.asarray(y, dtype=np.int64)

    def holdout_rows(self, n, model):
        return np.zeros((len(self._y), 2)), self._y


class _MseTask:
    head = "mse"

    def __init__(self, y: np.ndarray) -> None:
        self._y = y

    def holdout_rows(self, n, model):
        return np.zeros((len(self._y), 2)), self._y


def test_holdout_difficulty_softmax_hand_computed():
    """A48 (SPEC.md 58.1): the signed margin ``logit[y] −
    max_{j≠y} logit[j]`` — item0 2.0−0.5 = 1.5 (easy correct),
    item1 1.0−0.9 ≈ 0.1 (barely correct), item2 1.2−1.5 ≈ −0.3
    (misclassified, predicted 1). Sorted margin DESCENDING."""
    logits = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, 0.9], [1.0, 1.5, 1.2]])
    out = holdout_difficulty(_SoftmaxTask([0, 1, 2]), _FakeModel(logits))
    assert out["n"] == 3
    assert out["head"] == "softmax"
    items = out["items"]
    assert [it["index"] for it in items] == [0, 1, 2]
    assert items[0] == {"index": 0, "true": 0, "predicted": 0,
                        "correct": True, "difficulty": 1.5}
    assert items[1]["predicted"] == 1 and items[1]["correct"] is True
    assert _approx(items[1]["difficulty"], 0.1)
    assert items[2]["true"] == 2 and items[2]["predicted"] == 1
    assert items[2]["correct"] is False
    assert _approx(items[2]["difficulty"], -0.3)


def test_holdout_difficulty_ties_by_index():
    """A48 (SPEC.md 58.1): equal margins keep their split order (the sort
    key is (−margin, index)) — 1.0, 1.0, 0.5 → indices 0, 1, 2."""
    logits = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.5, 1.0, 0.5]])
    out = holdout_difficulty(_SoftmaxTask([1, 1, 1]), _FakeModel(logits))
    assert [it["index"] for it in out["items"]] == [0, 1, 2]
    assert _approx(out["items"][0]["difficulty"], 1.0)
    assert _approx(out["items"][1]["difficulty"], 1.0)
    assert _approx(out["items"][2]["difficulty"], 0.5)


def test_holdout_difficulty_mse_hand_computed():
    """A48 (SPEC.md 58.1): mse head — ``difficulty = max_j |pred_j − y_j|``
    ASCENDING (small = easy); ``correct = difficulty ≤ 0.05·max(1,
    max_j |y_j|)`` → 0.02 ≤ 0.05 correct, 0.5 > 0.05 not. true/predicted
    carry the scalar target/prediction values."""
    preds = np.array([[1.02], [0.5]])
    target = np.array([[1.0], [1.0]])
    out = holdout_difficulty(_MseTask(target), _FakeModel(preds))
    assert out["head"] == "mse"
    i0, i1 = out["items"]
    assert [it["index"] for it in out["items"]] == [0, 1]
    assert i0["correct"] is True and _approx(i0["difficulty"], 0.02)
    assert i0["true"] == 1.0 and i0["predicted"] == 1.02
    assert i1["correct"] is False and _approx(i1["difficulty"], 0.5)


def test_holdout_difficulty_none_cases():
    """A48 (SPEC.md 58.1): ``None`` when the task exposes no
    ``holdout_rows`` (episode tasks) or the holdout split is empty."""
    class NoRows:
        head = "softmax"

    class Empty:
        head = "softmax"

        def holdout_rows(self, n, model):
            return np.zeros((0, 2)), np.asarray([], dtype=np.int64)

    assert holdout_difficulty(NoRows(), _FakeModel(np.zeros((1, 2)))) is None
    assert holdout_difficulty(Empty(), _FakeModel(np.zeros((0, 2)))) is None


# --- 58.1 the difficulty ranking renderer (A48) --------------------------------

SOFTMAX_DATA = {
    "n": 3, "head": "softmax",
    "items": [
        {"index": 0, "true": 0, "predicted": 0, "correct": True,
         "difficulty": 1.5},
        {"index": 1, "true": 1, "predicted": 1, "correct": True,
         "difficulty": 0.1},
        {"index": 2, "true": 2, "predicted": 1, "correct": False,
         "difficulty": -0.3},
    ],
}


def test_svg_difficulty_ranking_softmax():
    """A48 (SPEC.md 58.1): valid XML; one lollipop row per item with the
    ``#index true → predicted`` label and the full-number tooltip
    (``correct`` vs ``MISCLASSIFIED``, ``difficulty {d:.3f}``); the
    signed axis caption names the softmax-margin metric."""
    svg = svg_difficulty_ranking(SOFTMAX_DATA)
    _parse(svg)
    assert "per-input difficulty ranking" in svg  # the aria-label/title
    assert "signed margin (softmax)" in svg
    assert "item #0 · true 0 · predicted 0 · correct · difficulty 1.500" \
        in svg
    assert "item #2 · true 2 · predicted 1 · MISCLASSIFIED · " \
           "difficulty -0.300" in svg
    assert "easiest → hardest" in svg
    assert "3 holdout items" in svg


def test_svg_difficulty_ranking_head_tail_cap():
    """A48 (SPEC.md 58.1): beyond ``_DIFF_ROWS_MAX = 30`` only the head +
    tail render, with the ``… n omitted (head + tail shown) …`` marker —
    35 items → 5 omitted."""
    items = [{"index": i, "true": 0, "predicted": 0, "correct": True,
              "difficulty": float(i)} for i in range(35)]
    svg = svg_difficulty_ranking({"n": 35, "head": "softmax", "items": items})
    _parse(svg)
    assert "… 5 items omitted (head + tail shown) …" in svg


def test_svg_difficulty_ranking_mse_axis():
    """A48 (SPEC.md 58.1): the mse mode is non-signed (no negative values)
    → the ``|pred − y| (mse)`` caption, not the signed one."""
    data = {"n": 2, "head": "mse", "items": [
        {"index": 0, "true": 1.0, "predicted": 1.02, "correct": True,
         "difficulty": 0.02},
        {"index": 1, "true": 1.0, "predicted": 0.5, "correct": False,
         "difficulty": 0.5},
    ]}
    svg = svg_difficulty_ranking(data)
    _parse(svg)
    assert "|pred − y| (mse)" in svg
    assert "signed margin (softmax)" not in svg


def test_svg_difficulty_ranking_empty():
    """A48 (SPEC.md 58.1): no items → header + the empty-state message."""
    svg = svg_difficulty_ranking({"n": 0, "head": "softmax", "items": []})
    _parse(svg)
    assert "no holdout items (non-classification task?)" in svg


def test_svg_difficulty_ranking_themes():
    """A48 (SPEC.md 51.4): okabe / dark re-renders stay valid XML; an
    unknown palette is a ValueError (51.4.1)."""
    data = {"n": 1, "head": "softmax", "items": [
        {"index": 0, "true": 0, "predicted": 0, "correct": True,
         "difficulty": 1.0}]}
    for kw in ({"palette": "okabe"}, {"dark": True},
               {"palette": "okabe", "dark": True}):
        _parse(svg_difficulty_ranking(data, **kw))
    with pytest.raises(ValueError):
        svg_difficulty_ranking(data, palette="rainbow")


# --- 58.2 the 3-objective frontier renderer (A48) ------------------------------

def test_svg_frontier3_panels():
    """A48 (SPEC.md 58.2): valid XML; both panel titles; the sized
    tooltips (``size 218 params``), the unsized note, the FRONTIER /
    dominated verdicts, the legend, and the unsized count in the
    caption."""
    r = frontier3(F3_ENTRIES, 4, 2, grid=GRID)
    svg = svg_frontier3(r)
    _parse(svg)
    assert "score × training time" in svg
    assert "score × parameter count" in svg
    assert "size 218 params" in svg          # A1 (panel 1 tooltip)
    assert "size 690 params" in svg          # A2
    assert "size n/a (data-dependent)" in svg  # A3
    assert "FRONTIER" in svg and "dominated" in svg
    assert "1 unsized point(s) shown in panel 1 only" in svg
    assert "frontier" in svg and "3D non-dominance" in svg


def test_svg_frontier3_empty():
    """A48 (SPEC.md 58.2): no scored candidates with train time → the
    header + empty-state message."""
    svg = svg_frontier3({"points": [], "frontier": [], "unsized": 0})
    _parse(svg)
    assert "no scored candidates with train time" in svg


def test_svg_frontier3_no_sizes():
    """A48 (SPEC.md 58.2): all points unsized (tree) → panel 2 degrades
    to the ``no parameter counts`` caption with the 58.2 note."""
    r = frontier3([
        {"kind": "baseline", "spec": {"model_family": "tree"},
         "spec_hash": "T", "holdout_score": 50.0, "train_seconds": 1.0},
    ])
    svg = svg_frontier3(r)
    _parse(svg)
    assert "no parameter counts" in svg
    assert "tree/boost/knn sizes are data-dependent (58.2)" in svg


def test_svg_frontier3_themes():
    """A48 (SPEC.md 51.4): okabe / dark re-renders stay valid XML; a bad
    palette is a ValueError (51.4.1)."""
    r = frontier3(F3_ENTRIES, 4, 2, grid=GRID)
    for kw in ({"palette": "okabe"}, {"dark": True}):
        _parse(svg_frontier3(r, **kw))
    with pytest.raises(ValueError):
        svg_frontier3(r, palette="rainbow")


# --- 58.3 the run dossier (A48) -------------------------------------------------

def _write_run(run_dir: Path, *, with_config: bool = True) -> None:
    """A synthetic finished run dir (the ``_run_task_and_model`` shape):
    summary + experiments + (optionally) the canonical run_config."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "task": "csv", "seed": 7, "baseline_score": 50.0,
        "final_best_score": 70.0, "improvement_factor": 1.4,
        "experiments_run": 3, "wall_seconds": 12.0,
        "finished_reason": "budget",
    }), encoding="utf-8")
    with (run_dir / "experiments.jsonl").open("w", encoding="utf-8") as f:
        for e in F3_ENTRIES:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    if with_config:
        rc = RunConfig(task="csv", seed=7, max_experiments=2,
                       autorefine_version="0.46.0").to_dict()
        (run_dir / "run_config.json").write_text(
            json.dumps(rc, indent=2, sort_keys=True), encoding="utf-8")


def test_build_dossier_sections_and_g2(tmp_path):
    """A48 (SPEC.md 58.3): every section heading is present; the recipe
    is the ``format_recipe`` line (non-default flags only) + the full
    canonical JSON; and a re-render of the same inputs is
    byte-identical (G2 — no timestamps)."""
    run = tmp_path / "run"
    _write_run(run)
    html1 = build_dossier(run, state_dim=4, n_out=2, grid=GRID)
    html2 = build_dossier(run, state_dim=4, n_out=2, grid=GRID)
    assert html1 == html2  # G2
    for heading in ("<h2>Summary</h2>", "<h2>Recipe (run_config)</h2>",
                    "<h2>Provenance</h2>", "<h2>Frontier</h2>",
                    "<h2>Spec lineage</h2>",
                    "<h2>Sensitivity (per-field response)</h2>",
                    "<h2>Error analysis</h2>"):
        assert heading in html1, heading
    assert "autorefine run --task csv" in html1
    assert "--experiments 2" in html1
    assert "autorefine.run_config/1" in html1  # the canonical schema tag
    assert "<!DOCTYPE html>" in html1 and "</html>" in html1


def test_build_dossier_error_pack_and_gallery(tmp_path):
    """A48 (SPEC.md 58.3): the error-analysis pack — the caller's
    pre-rendered per-class / confusion SVGs are inlined as-is, the
    difficulty ranking renders from its dict, and the 28.3 media gallery
    renders each item (file caption + the waveform SVG)."""
    run = tmp_path / "run"
    _write_run(run)
    html = build_dossier(
        run,
        difficulty={"n": 1, "head": "softmax", "items": [
            {"index": 0, "true": 0, "predicted": 0, "correct": True,
             "difficulty": 1.0}]},
        per_class_svg="<svg id='pc'></svg>",
        confusion_svg="<svg id='cm'></svg>",
        gallery=[{"file": "clip.wav", "label": "0", "predicted": "1",
                  "waveform": [0.1, -0.2, 0.3], "sample_rate": 8000}],
    )
    assert "<svg id='pc'></svg>" in html
    assert "<svg id='cm'></svg>" in html
    assert "per-input difficulty ranking" in html
    # the file appears in the gallery caption AND the waveform SVG's
    # aria-label (the dossier passes the file as the waveform title)
    assert html.count("clip.wav") >= 2
    assert "no holdout classification data" not in html


def test_build_dossier_degrades_without_classification(tmp_path):
    """A48 (SPEC.md 58.3): a run with no classification data degrades the
    Error analysis section to the caption (episode / mse tasks)."""
    run = tmp_path / "run"
    _write_run(run)
    html = build_dossier(run)
    assert "<h2>Error analysis</h2>" in html
    assert "no holdout classification data" in html


def test_build_dossier_missing_summary_raises(tmp_path):
    """A48 (SPEC.md 58.3/42): a run dir without ``summary.json`` is a
    ValueError (the ``_run_task_and_model`` contract)."""
    (tmp_path / "broken").mkdir()
    with pytest.raises(ValueError):
        build_dossier(tmp_path / "broken")


def test_build_dossier_without_run_config(tmp_path):
    """A48 (SPEC.md 58.3): a run that predates the config artifact (or
    lost it) still renders — the Recipe section degrades to the
    pre-v0.23 caption."""
    run = tmp_path / "run"
    _write_run(run, with_config=False)
    html = build_dossier(run)
    assert "<h2>Recipe (run_config)</h2>" in html
    assert "no run_config.json in this run (pre-v0.23)" in html


# --- 58.4 the surfaces (app / dashboard / CLI) (A48) ---------------------------

def test_app_wires_surfaces():
    """A48 (SPEC.md 58.4): the app imports the dossier builder, renders
    the difficulty SVG in the Learning views (the shared condition
    tuple), and offers the fifth artifacts download button with the
    49.4.3 friendly-error path; the dashboard computes the 58.1
    difficulty + the 58.3 geometry keys; the CLI wires the
    ``dossier`` subcommand (the command, the epilog, the handler, the
    imports, the flags)."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "from autorefine.dossier import build_dossier",
        "build_dossier(",
        '"difficulty_svg"',
        'c5.download_button("dossier.html"',
        "dossier unavailable:",
    ):
        assert token in src, token

    dash = DASH.read_text(encoding="utf-8")
    for token in (
        "holdout_difficulty, holdout_diagnostics",
        '"difficulty": diff',
        '"difficulty_svg": svg_difficulty_ranking(diff) if diff else None',
        '"state_dim": self.env.task.state_dim',
        '"n_out": self.env.task.n_outputs',
        '"grid": getattr(self.env.task, "feature_grid", None)',
    ):
        assert token in dash, token

    cli = CLI.read_text(encoding="utf-8")
    for token in (
        "from .dossier import build_dossier",
        "def _cmd_dossier",
        "_EP_DOSSIER",
        'p_dossier = sub.add_parser(\n        "dossier"',
        '"--run"',
        '"--out"',
        "holdout_difficulty(task, model)",
        "dossier.html",
    ):
        assert token in cli, token


# --- 58.4 the default path end-to-end (A48) ------------------------------------

def test_app_default_path_end_to_end(tmp_path):
    """A48 (SPEC.md 58.4): the *default* path runs end-to-end — a
    finished CSV run persists its result, the Learning views render the
    difficulty ranking SVG (the csv task exposes holdout rows → the
    softmax-margin mode), and the Artifacts row carries the
    ``dossier.html`` download button (AppTest-safe; no new widget)."""
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
    sub_headers = [str(s.value) for s in at.subheader]
    assert "Learning views" in sub_headers, sub_headers
    assert any("difficulty ranking" in str(m.value) for m in at.markdown), \
        "the difficulty ranking SVG renders in the Learning views"
    labels = [e.label for e in at.get("download_button")]
    assert "dossier.html" in labels, labels


# --- A48 round regression -------------------------------------------------------

def test_version_round_v044():
    """A48 (SPEC.md 58.6, 33.1): the version stepped to ``0.44.0`` in both
    sources (v0.44 ⇒ ``0.44.0``, M47, SPEC.md 58)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"


def test_all_exports_surfaces_round():
    """A48 (SPEC.md 58.4, 33.1): the six new names are exported from the
    package top level and in ``__all__``."""
    for name in ("spec_n_params", "holdout_difficulty", "frontier3",
                 "svg_difficulty_ranking", "svg_frontier3", "build_dossier"):
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_spec_cites_a48_and_round():
    """A48 (SPEC.md 58.6/58.7): SPEC.md defines the A48 acceptance block
    and the M47 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 49)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 58.5 Acceptance (A48)" in spec
    assert re.search(r"^\s*\| M47 \| v0\.44\s*\|\s*58\s*\|\s*A48\s*\|",
                     spec, re.MULTILINE)
    assert "**M47**" in spec
