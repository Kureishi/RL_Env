"""v0.16 comprehension visuals II — V1 + V2 + V3 + V4 + V5 + V6 (SPEC.md 30,
A20):

- V1 per-seed best-score curves — `svg_seed_curves` (SPEC.md 30.1),
  `DashboardRunner.seed_sweep`'s `curve` key, the `autorefine variance`
  CLI's `seed_curves.svg`, and the app's Seed variance panel.
- V2 field × value win matrix — `field_value_stats` (SPEC.md 30.2),
  `svg_field_value_matrix`, the `next()` / `finish()` keys, and the app's
  Decision views (live + result).
- V3 decision-boundary scatter (2-feature tabular) —
  `decision_boundary` (SPEC.md 30.3), `svg_decision_boundary`, the
  `finish()` `boundary` / `boundary_svg` keys, and the app's Learning
  views.
- V4 CI band on the score curve — the logged `std` key (SPEC.md 30.4,
  `meta_env.py`) and `svg_score_curve`'s bands / legacy byte-identity.
- V5 model-family score bars — `family_stats` (SPEC.md 30.5),
  `svg_family_bars`, the `finish()` `family_stats` / `family_bars_svg`
  keys, and the app's Decision views.
- V6 RL return-to-go / baseline trace — `svg_policy_trace`
  (SPEC.md 30.6), the `autorefine policy-report` `policy_trace_curve.svg`,
  and the app's RL policy panel (any-of-three render, A19 warning kept).

House rules: fixtures are duplicated from the other test modules (no
cross-test imports); every SVG is asserted as valid XML; budgets are kept
small (quadrant-XOR CSV, 2-3 experiments) for speed.
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget, DashboardRunner
from autorefine.cli import main as cli_main
from autorefine.dashboard import (
    decision_boundary,
    family_stats,
    field_stats,
    field_value_stats,
)
from autorefine.improver.meta_env import search_quality_v04
from autorefine.plotting import (
    ascii_score_curve,
    svg_decision_boundary,
    svg_family_bars,
    svg_field_value_matrix,
    svg_policy_trace,
    svg_score_curve,
    svg_seed_curves,
)
from autorefine.tasks.csv import CsvTask

APP = Path(__file__).resolve().parents[1] / "src" / "autorefine" / "dashboard_app.py"


def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    """Deterministic 60-row quadrant-XOR classification CSV (2 classes).
    (Duplicated fixture — tests/test_dashboard.py and
    tests/test_multi_run_policy_views.py.)"""
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
    kw.setdefault("csv_path", str(_write_csv(tmp_path)))
    kw.setdefault("seed", 7)
    kw.setdefault("experiments", 3)
    kw.setdefault("policy", "bandit")
    kw.setdefault("max_train_seconds", 10.0)
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    return DashboardRunner(**kw)


def _polylines(root) -> list:
    """All <polyline> elements — namespace-agnostic child scan (the SVG
    declares xmlns, so plain tag lookups miss)."""
    return [el for el in root.iter() if el.tag.endswith("polyline")]


def _cells_with_titles(root) -> dict:
    """{title text: element} over every <rect> carrying a <title> child."""
    cells = {}
    for el in root.iter():
        if not el.tag.endswith("rect"):
            continue
        for ch in el:
            if ch.tag.endswith("title") and ch.text:
                cells.setdefault(ch.text, el)
                break
    return cells


# --- V1 (SPEC.md 30.1): svg_seed_curves ---------------------------------------

_SEEDS = [
    {"seed": 7, "curve": [50.0, 62.0, 62.0, 71.0, 74.0]},
    {"seed": 8, "curve": [50.0, 48.0, 55.0, 60.0, 60.0]},
    {"seed": 9, "curve": [50.0, 51.0, 51.0]},  # shorter: diverges later
]


def test_svg_seed_curves_valid_xml_polylines_legend_target_and_summary():
    svg = svg_seed_curves(_SEEDS, target=85.0)
    root = ET.fromstring(svg)  # valid XML (A20)
    assert len(_polylines(root)) == 3  # one polyline per seed (SPEC.md 30.1)
    for s in _SEEDS:
        assert f"seed {s['seed']}" in svg  # the legend names each seed
    assert "target 85.00" in svg  # the dashed target line
    assert "n seeds = 3 - steps = 5" in svg  # M = the longest curve
    assert svg == svg_seed_curves(_SEEDS, target=85.0)  # deterministic (G2)


def test_svg_seed_curves_no_target_empty_and_skips():
    svg = svg_seed_curves(_SEEDS)
    ET.fromstring(svg)  # valid XML without a target
    assert "target " not in svg
    empty = svg_seed_curves([])
    ET.fromstring(empty)  # empty -> header + message, still valid XML
    assert "no seed curves in the sweep" in empty
    # rows without a finite curve are skipped (SPEC.md 30.1)
    svg2 = svg_seed_curves([
        {"seed": 1, "curve": None},
        {"seed": 2, "curve": [float("nan"), float("inf")]},
        {"seed": 3, "curve": [50.0, 60.0]},
    ])
    root = ET.fromstring(svg2)
    assert len(_polylines(root)) == 1
    assert "n seeds = 1 - steps = 2" in svg2
    assert "seed 3" in svg2


# --- V1 (SPEC.md 30.1): the variance CLI --------------------------------------

def test_variance_cli_writes_seed_curves_svg(tmp_path, capsys):
    csv = _write_csv(tmp_path)
    runs = tmp_path / "vc"
    rc = cli_main(["variance", "--data", str(csv), "--seeds", "3",
                   "--seed", "7", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    svg = (runs / "seed_curves.svg").read_text(encoding="utf-8")
    ET.fromstring(svg)  # valid XML (A20)
    assert "best score per seed over the sweep" in svg
    assert "n seeds = 3" in svg
    assert str(runs / "seed_curves.svg") in out  # named in the listing
    assert (runs / "seed_variance.svg").exists()  # the A19 artifact stays
    sweep = json.loads((runs / "seed_sweep.json").read_text(encoding="utf-8"))
    for s in sweep:  # the JSON carries the curves too (SPEC.md 30.1)
        assert "curve" in s and len(s["curve"]) >= 2
        assert s["curve"][0] == s["baseline"]
        assert s["curve"][-1] == s["final"]


# --- V2 (SPEC.md 30.2): field_value_stats --------------------------------------

def _synthetic_updates() -> list[dict]:
    """A hand-computable stream: 4 updates crediting 4 (field, value) cells,
    covering numeric values, a joined list, and a `None` value."""
    return [
        {"mutation": ["hidden_dim"], "accepted": True,
         "spec_diff": [{"field": "hidden_dim", "old": 8, "new": 16}]},
        {"mutation": ["hidden_dim", "learning_rate"], "accepted": False,
         "spec_diff": [{"field": "hidden_dim", "old": 16, "new": 64},
                       {"field": "learning_rate", "old": 1e-3, "new": 1e-4}]},
        {"mutation": ["layers"], "accepted": True,
         "spec_diff": [{"field": "layers", "old": 2, "new": [8, 16]}]},
        {"mutation": ["knn_k"], "accepted": True,
         "spec_diff": [{"field": "knn_k", "old": 5, "new": None}]},
    ]


def test_field_value_stats_hand_computed_normalization_and_sums():
    stats = field_value_stats(_synthetic_updates())
    assert stats == {
        "hidden_dim": {
            "16": {"trials": 1, "wins": 1.0, "win_rate": 1.0},
            "64": {"trials": 1, "wins": 0.0, "win_rate": 0.0},
        },
        "knn_k": {"-": {"trials": 1, "wins": 1.0, "win_rate": 1.0}},
        "layers": {"8,16": {"trials": 1, "wins": 1.0, "win_rate": 1.0}},
        "learning_rate": {"0.0001": {"trials": 1, "wins": 0.0, "win_rate": 0.0}},
    }
    # deterministic key order: fields alphabetical, values numeric-first
    assert list(stats) == ["hidden_dim", "knn_k", "layers", "learning_rate"]
    assert list(stats["hidden_dim"]) == ["16", "64"]
    # the per-field trial/wins sums equal `field_stats`' (SPEC.md 30.2)
    fs = field_stats(_synthetic_updates())
    assert set(stats) == set(fs)
    for f, cells in stats.items():
        assert sum(c["trials"] for c in cells.values()) == fs[f]["trials"]
        assert sum(c["wins"] for c in cells.values()) == fs[f]["wins"]
    assert field_value_stats(_synthetic_updates()) == stats  # deterministic
    assert field_value_stats([]) == {}  # empty-safe
    assert field_value_stats(None) == {}  # forgiving None, like field_stats


def test_svg_field_value_matrix_valid_cells_titles_and_empty():
    stats = field_value_stats(_synthetic_updates())
    svg = svg_field_value_matrix(stats)
    root = ET.fromstring(svg)  # valid XML (A20)
    cells = _cells_with_titles(root)
    # one cell per credited (field, value): hidden_dim 2 + 3 single-value fields
    assert len(cells) == 5
    for field, vals in stats.items():
        for value, cell in vals.items():
            title = (f"{field} {value}: "
                     f"{cell['wins']:.0f}/{cell['trials']} "
                     f"({cell['win_rate']:.2f})")
            assert title in cells
            assert float(cells[title].get("fill-opacity")) == pytest.approx(
                0.10 + 0.90 * cell["win_rate"], abs=0.001)  # the §30.2 fill
    assert "1/1" in svg  # the wins/trials text
    assert "cell = wins/trials" in svg  # the legend line
    assert svg == svg_field_value_matrix(stats)  # deterministic (G2)
    # the renderer sorts rows/cells itself: an unsorted input renders
    # bit-identically (SPEC.md 30.2 "one row per field (sorted)")
    unsorted = {
        "layers": {"8,16": stats["layers"]["8,16"]},
        "hidden_dim": {"64": stats["hidden_dim"]["64"],
                       "16": stats["hidden_dim"]["16"]},
        "knn_k": {"-": stats["knn_k"]["-"]},
        "learning_rate": {"0.0001": stats["learning_rate"]["0.0001"]},
    }
    assert svg_field_value_matrix(unsorted) == svg
    empty = svg_field_value_matrix({})
    ET.fromstring(empty)  # empty -> header + message, still valid XML
    assert "no field values credited" in empty


# --- V2 (SPEC.md 30.2): the runner --------------------------------------------

def test_next_updates_carry_field_value_stats(tmp_path):
    r = _runner(tmp_path)
    r.start()
    credited = 0
    while not r.done:
        u = r.next()
        assert "field_value_stats" in u  # attached like field_stats
        fs, fvs = u["field_stats"], u["field_value_stats"]
        assert set(fs) == set(fvs)  # same fields, refined to values
        for f in fs:  # the per-field sums equal `field_stats`' (SPEC.md 30.2)
            assert sum(c["trials"] for c in fvs[f].values()) == fs[f]["trials"]
            assert sum(c["wins"] for c in fvs[f].values()) == fs[f]["wins"]
        credited += bool(fvs)
    assert credited >= 1  # the bandit credited at least one value


def test_finish_returns_field_value_keys(tmp_path):
    r = _runner(tmp_path)
    r.start()
    while not r.done:
        r.next()
    res = r.finish()
    assert "field_value_stats" in res and "field_value_svg" in res
    assert res["field_value_stats"] == field_value_stats(res["updates"])
    root = ET.fromstring(res["field_value_svg"])  # valid XML (A20)
    assert "field x value win matrix" in res["field_value_svg"]
    assert len(_cells_with_titles(root)) == len(
        {f: v for f, vals in res["field_value_stats"].items()
         for v in vals})


# --- app (A20): the two new panels (streamlit optional) ------------------------

def _app_setup(tmp_path):
    """Render the app with a CSV configured (the run + opt-in panels show)."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest
    csv = _write_csv(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    return at


def test_app_seed_variance_panel_renders_boxplot_and_curves(tmp_path):
    at = _app_setup(tmp_path)
    at.number_input(key="var_seeds").set_value(2)
    at.button(key="var_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "final score across 2 seed(s)" in md  # svg_seed_variance (A19)
    assert "best score per seed over the sweep" in md  # svg_seed_curves (A20)


def test_app_decision_views_render_field_value_matrix(tmp_path):
    at = _app_setup(tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert md.count("field x value win matrix") >= 2  # live + result (A20)
    assert "Decision views" in " ".join(h.value for h in at.subheader)


# --- V3 (SPEC.md 30.3): the decision boundary ---------------------------------

def _write_csv_1feat(tmp_path: Path, name: str = "one.csv") -> Path:
    """40-row 1-feature binary CSV (softmax head, state_dim 1)."""
    lines = ["a,churn"]
    for i in range(40):
        a = i / 40.0
        lines.append(f"{a:.3f},{1.0 if a > 0.5 else 0.0:.0f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_csv_mse(tmp_path: Path, name: str = "mse.csv") -> Path:
    """40-row 2-feature CSV with a continuous label (mse head)."""
    lines = ["a,b,y"]
    for i in range(40):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        lines.append(f"{a:.2f},{b:.2f},{(a + b) * 3.7:.3f}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class _RuleModel:
    """Deterministic fake model: class 1 iff x0 + x1 > 0.9 (no training,
    no RNG — G2)."""

    def forward(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        c = (x[:, 0] + x[:, 1]) > 0.9
        return np.where(c[:, None], [[0.2, 0.8]], [[0.8, 0.2]])


def test_decision_boundary_none_cases(tmp_path):
    # A20: None for a 1-feature CSV, an mse-head CSV, and model=None
    model = _RuleModel()
    one = CsvTask(seed=7, path=_write_csv_1feat(tmp_path))
    assert one.head == "softmax" and one.state_dim == 1
    assert decision_boundary(one, model) is None
    mse = CsvTask(seed=7, path=_write_csv_mse(tmp_path))
    assert mse.head == "mse"
    assert decision_boundary(mse, model) is None
    task = CsvTask(seed=7, path=_write_csv(tmp_path))
    assert decision_boundary(task, None) is None


def _padded(lo: float, hi: float) -> list[float]:
    """The SPEC.md 30.3 range rule: padded 5%; degenerate → ±1."""
    span = hi - lo
    if span > 0:
        return [lo - 0.05 * span, hi + 0.05 * span]
    return [lo - 1.0, hi + 1.0]


def test_decision_boundary_2feature_contract(tmp_path):
    task = CsvTask(seed=7, path=_write_csv(tmp_path))
    x, _y = task.holdout_rows(200, None)
    n_grid = 8
    db = decision_boundary(task, _RuleModel(), n_grid=n_grid)
    assert db is not None
    assert db["n_grid"] == n_grid
    assert len(db["grid_preds"]) == n_grid * n_grid  # n_grid^2 (A20)
    assert all(0 <= p < 2 for p in db["grid_preds"])
    assert db["n"] == len(x) == len(db["points"])  # the holdout size (A20)
    for pt in db["points"]:
        assert set(pt) == {"x", "y", "true", "pred"}
        assert 0 <= pt["true"] < 2 and 0 <= pt["pred"] < 2  # in range (A20)
    assert db["correct"] == sum(  # correct = #{true == pred} (A20)
        pt["true"] == pt["pred"] for pt in db["points"])
    assert db["classes"] == ["0", "1"]  # A20 (class_label_str, SPEC.md 28.3)
    assert (db["x0"], db["x1"]) == ("a", "b")  # feature names
    assert db["x0_range"] == pytest.approx(
        _padded(float(x[:, 0].min()), float(x[:, 0].max())))
    assert db["x1_range"] == pytest.approx(
        _padded(float(x[:, 1].min()), float(x[:, 1].max())))
    # the grid spans the ranges: its corners are the range endpoints and
    # follow the model's rule
    lo0, hi0 = db["x0_range"]
    lo1, hi1 = db["x1_range"]
    rule = lambda v0, v1: 1 if v0 + v1 > 0.9 else 0
    assert db["grid_preds"][0] == rule(lo0, lo1)
    assert db["grid_preds"][(n_grid - 1) * n_grid] == rule(hi0, lo1)  # row-major
    assert db["grid_preds"][-1] == rule(hi0, hi1)
    assert db == decision_boundary(task, _RuleModel(), n_grid=n_grid)  # G2


def test_svg_decision_boundary_xml_contract():
    data = {
        "x0": "a", "x1": "b",
        "x0_range": [0.0, 1.0], "x1_range": [0.0, 1.0],
        "n_grid": 4,
        "grid_preds": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        "points": [
            {"x": 0.1, "y": 0.1, "true": 0, "pred": 0},
            {"x": 0.9, "y": 0.1, "true": 0, "pred": 1},
            {"x": 0.5, "y": 0.5, "true": 1, "pred": 1},
        ],
        "classes": ["0", "1"],
        "n": 3, "correct": 2,
    }
    svg = svg_decision_boundary(data)
    root = ET.fromstring(svg)  # valid XML (A20)
    titled = {}
    for el in root.iter():
        for ch in el:
            if ch.tag.endswith("title") and ch.text:
                titled.setdefault(el.tag.rsplit("}", 1)[-1], set()).add(ch.text)
                break
    # n^2 grid-cell titles (A20)
    cells = {f"grid {i},{j} -> class {data['grid_preds'][i * 4 + j]}"
             for i in range(4) for j in range(4)}
    assert cells <= titled.get("rect", set())
    # one true -> pred title per holdout point (A20)
    pts = titled.get("circle", set())
    assert {"true 0 -> pred 0", "true 0 -> pred 1", "true 1 -> pred 1"} <= pts
    assert len(pts) == 3
    # the summary line + the axis names (SPEC.md 30.3)
    assert "n holdout = 3 - correct = 2 (67%)" in svg
    assert "x: a · y: b" in svg
    assert svg == svg_decision_boundary(data)  # deterministic (G2)
    # None-safe (A20)
    empty = svg_decision_boundary(None)
    ET.fromstring(empty)
    assert "no 2-feature classification data" in empty


def test_finish_returns_boundary_keys(tmp_path):
    r = _runner(tmp_path)
    r.start()
    while not r.done:
        r.next()
    res = r.finish()
    assert res["boundary"] is not None  # 2-feature softmax CSV (A20)
    assert json.dumps(res["boundary"], sort_keys=True)  # JSON-safe
    root = ET.fromstring(res["boundary_svg"])  # valid XML (A20)
    assert "decision boundary" in res["boundary_svg"]
    assert len([el for el in root.iter() if el.tag.endswith("rect")]) > 25


def test_app_learning_views_render_boundary(tmp_path):
    at = _app_setup(tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "decision boundary" in md  # the app renders it (A20)
    assert "Learning views" in " ".join(h.value for h in at.subheader)


# --- V4 (SPEC.md 30.4): the logged std + the CI band ---------------------------

def _drive_to_done(env, pol):
    state = env.reset()
    while not env.done:
        state, _r, _d, _info = env.step(pol.propose(state))


def test_log_entries_carry_std(tmp_path):
    # legacy mode: exactly one zero-valued key added (SPEC.md 30.4)
    env = AutoRefineEnv(task="parity-v1", seed=7,
                        budget=Budget(1, 300, 30),
                        runs_dir=str(tmp_path / "leg"), dataset_episodes=60)
    _drive_to_done(env, BanditPolicy(seed=7, mode="uniform"))
    entries = env.memory.load_experiments()
    assert [e["kind"] for e in entries] == ["baseline", "experiment"]
    for e in entries:
        assert e["std"] == 0.0  # legacy: std == 0.0 (A20)

    # v0.4 mode: the §18.3 block-bootstrap sigma, finite (A20)
    env2 = AutoRefineEnv(task="parity-v1", seed=7,
                         budget=Budget(1, 300, 30),
                         runs_dir=str(tmp_path / "v04"),
                         dataset_episodes=60, **search_quality_v04())
    _drive_to_done(env2, BanditPolicy(seed=7, mode="uniform"))
    entries2 = env2.memory.load_experiments()
    assert [e["kind"] for e in entries2] == ["baseline", "experiment"]
    for e in entries2:
        assert math.isfinite(e["std"]) and e["std"] >= 0.0
    assert entries2[0]["std"] > 0.0  # 8 blocks disagree on parity (18.3)


def test_svg_score_curve_bands_and_legacy_unchanged():
    rows = [
        {"kind": "baseline", "holdout_score": 50.0, "accepted": True},
        {"kind": "experiment", "holdout_score": 60.0, "accepted": True},
        {"kind": "experiment", "holdout_score": 40.0, "accepted": False},
    ]
    # old logs (no `std` key) and legacy mode (`std == 0.0`) render exactly
    # the pre-v0.16 SVG — no band elements, no caption (SPEC.md 30.4)
    svg_old = svg_score_curve(rows)
    assert svg_score_curve([dict(r, std=0.0) for r in rows]) == svg_old
    assert "#93c5fd" not in svg_old
    assert "bands = score ± std" not in svg_old

    # positive std: one band line per such row + the §18.3 caption (A20)
    rows_ci = [
        {"kind": "baseline", "holdout_score": 50.0,
         "accepted": True, "std": 0.0},
        {"kind": "experiment", "holdout_score": 60.0,
         "accepted": True, "std": 5.0},
        {"kind": "experiment", "holdout_score": 40.0,
         "accepted": False, "std": 10.0},
    ]
    svg = svg_score_curve(rows_ci)
    ET.fromstring(svg)  # valid XML
    assert svg.count('stroke="#93c5fd"') == 2  # one band per positive-std row
    assert "±std 5.000" in svg and "±std 10.000" in svg
    assert "bands = score ± std (SPEC.md 18.3" in svg
    # the y-range expanded to cover the bands: 60+5=65 ... 40-10=30
    assert ">65.00<" in svg and ">30.00<" in svg
    # ascii_score_curve is untouched (SPEC.md 30.7)
    assert ascii_score_curve(rows_ci) == ascii_score_curve(rows)


# --- V5 (SPEC.md 30.5): model-family score bars --------------------------------

def test_family_stats_hand_computed_groups_and_sort():
    # A20 (SPEC.md 30.5): grouped by spec.model_family, "unknown" when
    # absent; best_score/best_seconds from the best member (ties keep the
    # first); wins = accepted count; sorted by (-best_score, family).
    entries = [
        {"kind": "baseline", "holdout_score": 60.0, "train_seconds": 1.5,
         "accepted": True, "spec": {"model_family": "mlp"}},
        {"kind": "experiment", "holdout_score": 72.0, "train_seconds": 2.5,
         "accepted": True, "spec": {"model_family": "mlp"}},
        {"kind": "experiment", "holdout_score": 65.0, "train_seconds": 0.9,
         "accepted": False, "spec": {"model_family": "tree"}},
        # tie on the family's best: the FIRST member keeps its cost
        {"kind": "experiment", "holdout_score": 65.0, "train_seconds": 0.4,
         "accepted": True, "spec": {"model_family": "tree"}},
        {"kind": "curriculum", "holdout_score": 99.0,
         "accepted": True, "spec": {"model_family": "mlp"}},  # skipped
        {"kind": "experiment", "holdout_score": None,  # skipped
         "accepted": False, "spec": {"model_family": "mlp"}},
        {"kind": "invalid_spec", "holdout_score": 99.0,  # skipped
         "accepted": False, "spec": {"model_family": "mlp"}},
        {"kind": "experiment", "holdout_score": 50.0, "train_seconds": 0.5,
         "accepted": False, "spec": None},  # -> unknown
    ]
    out = family_stats(entries)
    assert [f["family"] for f in out] == ["mlp", "tree", "unknown"]
    mlp, tree, unk = out
    assert mlp == {"family": "mlp", "best_score": 72.0, "best_seconds": 2.5,
                   "trials": 2, "wins": 2.0}
    assert tree == {"family": "tree", "best_score": 65.0,
                    "best_seconds": 0.9,  # the first best member (tie)
                    "trials": 2, "wins": 1.0}
    assert unk == {"family": "unknown", "best_score": 50.0,
                   "best_seconds": 0.5, "trials": 1, "wins": 0.0}
    # deterministic (G2) and empty-safe
    assert family_stats(entries) == out
    assert family_stats([]) == [] and family_stats(None) == []


def test_svg_family_bars_valid_bars_highlight_and_empty():
    fams = [
        {"family": "mlp", "best_score": 72.0, "best_seconds": 2.5,
         "trials": 2, "wins": 2.0},
        {"family": "tree", "best_score": 65.0, "best_seconds": 0.9,
         "trials": 2, "wins": 1.0},
    ]
    svg = svg_family_bars(fams)
    root = ET.fromstring(svg)  # valid XML (A20)
    bars = _cells_with_titles(root)  # the one titled bar per family
    assert len(bars) == 2
    assert ("family mlp: best 72.00 in 2.5s (2/2 accepted)") in bars
    assert ("family tree: best 65.00 in 0.9s (1/2 accepted)") in bars
    # the top family is highlighted (_ACCEPTED), the rest _LINE (SPEC.md 30.5)
    first = bars["family mlp: best 72.00 in 2.5s (2/2 accepted)"]
    assert first.get("fill") == "#15803d"
    assert bars["family tree: best 65.00 in 0.9s (1/2 accepted)"].get(
        "fill") == "#1a66c2"
    assert "best holdout score per model family" in svg
    # rows without a finite best_score are skipped; empty -> message
    svg2 = svg_family_bars(fams + [{"family": "ghost", "best_score": None,
                                    "trials": 1, "wins": 0.0}])
    ET.fromstring(svg2)
    assert "ghost" not in svg2
    empty = svg_family_bars([])
    ET.fromstring(empty)  # still valid XML
    assert "no scored experiments" in empty
    assert svg == svg_family_bars(fams)  # deterministic (G2)


def test_finish_returns_family_keys(tmp_path):
    # A20 (SPEC.md 30.5): finish() returns both keys, over the log entries
    r = _runner(tmp_path)
    r.start()
    while not r.done:
        r.next()
    res = r.finish()
    assert "family_stats" in res and "family_bars_svg" in res
    entries = [json.loads(line) for line in
               (Path(res["run_dir"]) / "experiments.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    assert res["family_stats"] == family_stats(entries)  # pure over the log
    svg = res["family_bars_svg"]
    ET.fromstring(svg)  # valid XML (A20)
    assert len(_cells_with_titles(ET.fromstring(svg))) == len(
        res["family_stats"])  # one bar per family row
    for f in res["family_stats"]:  # the baseline scored -> never empty
        assert f["family"]


def test_app_decision_views_render_family_bars(tmp_path):
    at = _app_setup(tmp_path)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "best holdout score per model family" in md  # V5 bars (A20)


# --- V6 (SPEC.md 30.6): the RL return-to-go / baseline trace ------------------

_TRACE = [
    {"task": "sine-v1", "action": 0, "probs": [1.0, 0.0], "reward": 1.0},
    {"task": "parity-v1", "action": 1, "probs": [0.0, 1.0], "reward": -0.5},
    {"task": "sine-v1", "action": 2, "probs": [0.5, 0.5], "reward": 0.25},
]


def test_svg_policy_trace_valid_bars_r2g_baseline_and_empty():
    svg = svg_policy_trace(_TRACE)
    ET.fromstring(svg)  # valid XML (A20)
    bars = _cells_with_titles(ET.fromstring(svg))
    # one reward bar per step, titled `step <i> · <task> · r=<r>`
    assert len(bars) == 3
    assert "step 0 · sine-v1 · r=1.000" in bars
    assert "step 1 · parity-v1 · r=-0.500" in bars
    assert "step 2 · sine-v1 · r=0.250" in bars
    # hand-computed return-to-go: [1.0-0.5+0.25, -0.5+0.25, 0.25]
    assert "return-to-go 0.750" in svg
    assert "return-to-go -0.250" in svg
    assert "return-to-go 0.250" in svg
    assert 'stroke-dasharray="6 4"' in svg  # the dashed baseline (A20)
    # summary line (SPEC.md 30.6)
    assert "n steps = 3 - final return-to-go = 0.250 - mean reward = 0.250" \
        in svg
    # records without a finite reward are skipped; empty -> message
    svg2 = svg_policy_trace([{"reward": None}, _TRACE[0]])
    ET.fromstring(svg2)
    assert "step 0 · sine-v1 · r=1.000" in svg2
    assert "parity-v1" not in svg2
    empty = svg_policy_trace([])
    ET.fromstring(empty)  # still valid XML
    assert "no trace rewards to plot" in empty
    assert svg == svg_policy_trace(_TRACE)  # deterministic (G2)


def test_policy_report_cli_writes_trace_curve(tmp_path, capsys):
    # A20 (SPEC.md 30.6): policy-report names + writes the third SVG
    runs = tmp_path / "pr6"
    rc = cli_main(["policy-report", "--task", "sine-v1", "--seed", "7",
                   "--episodes", "1", "--experiments", "3",
                   "--max-train-seconds", "8", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0
    pc = runs / "policy_trace_curve.svg"
    ET.fromstring(pc.read_text(encoding="utf-8"))  # valid XML (A20)
    assert "policy_trace_curve.svg" in out  # named in the listing
    trace = json.loads((runs / "policy_trace.json").read_text(encoding="utf-8"))
    assert len(trace) >= 1  # the curve renders the same trace
    assert "policy trace: rewards, return-to-go, baseline" in pc.read_text(
        encoding="utf-8")


def test_app_rl_panel_renders_any_of_three(tmp_path):
    # A20 (SPEC.md 30.6): whichever of the three SVGs exist render; a
    # v0.15 two-file dir still renders; an empty dir still warns (A19 text)
    at = _app_setup(tmp_path)

    def _load(d: Path) -> list:
        at.text_input(key="pol_dir").set_value(str(d))
        at.button(key="pol_button").set_value(True).run()
        assert not at.exception
        return [m.value for m in at.markdown]

    def _svg(title: str) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="10" '
                f'height="10"><text>{title}</text></svg>')

    three = tmp_path / "pr3"
    three.mkdir()
    (three / "action_probabilities.svg").write_text(_svg("AP-TITLE"),
                                                    encoding="utf-8")
    (three / "task_returns.svg").write_text(_svg("TR-TITLE"),
                                            encoding="utf-8")
    (three / "policy_trace_curve.svg").write_text(_svg("PT-TITLE"),
                                                  encoding="utf-8")
    md = " ".join(_load(three))
    assert "AP-TITLE" in md and "TR-TITLE" in md and "PT-TITLE" in md

    two = tmp_path / "pr2"  # a v0.15-era directory: still renders (A20)
    two.mkdir()
    (two / "action_probabilities.svg").write_text(_svg("AP-TITLE"),
                                                  encoding="utf-8")
    (two / "task_returns.svg").write_text(_svg("TR-TITLE"), encoding="utf-8")
    md = " ".join(_load(two))
    assert "AP-TITLE" in md and "TR-TITLE" in md and "PT-TITLE" not in md

    none_ = tmp_path / "pr0"
    none_.mkdir()
    at.text_input(key="pol_dir").set_value(str(none_))
    at.button(key="pol_button").set_value(True).run()
    assert not at.exception
    warns = " ".join(w.value for w in at.warning)
    # the A19 warning text is preserved verbatim (SPEC.md 30.6)
    assert f"{none_} has no action_probabilities.svg / " in warns
    assert "task_returns.svg — run `autorefine policy-report " in warns
