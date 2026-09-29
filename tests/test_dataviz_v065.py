"""Data flow — the data-utilization diagram (SPEC.md 79, A69).

SPEC.md 79 (v0.65): the complementary half of the model-structure view —
`svg_data_flow` shows the data's journey through the pipeline with its
shape at each stage:

    DATA -> FEATURES -> SPLIT -> STANDARDIZE -> [SPECTRAL] -> MODEL -> SCORE

  * 79.2.1 — the renderer signature + the block/arrow idiom of
    `svg_architecture`;
  * 79.2.2 — graceful degradation: `features`/`rows` = None (episode
    tasks) renders the policy-episode wording, the diagram stays complete;
  * 79.2.3 — the SPECTRAL stage renders only when
    `fourier_features > 0` (78.2.1);
  * 79.2.4 — pure / deterministic (G2), valid XML (strict-parseable),
    ASCII-safe via html.escape (hostile names cannot break the document);
  * 79.3.1 — `DashboardRunner.finish()` returns the additive
    `data_flow_svg` key (CSV run: feature names + row counts + the
    no-leakage note);
  * 79.3.2 — the app renders the "Data flow" visual in the result view.

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import tempfile
import tomllib
import xml.dom.minidom
from pathlib import Path

import numpy as np

import autorefine
from autorefine.dashboard import DashboardRunner
from autorefine.plotting import svg_data_flow, svg_is_well_formed

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

_MLP = {"model_family": "mlp", "architecture": (64, 32), "activation": "tanh"}
_ROWS = {"train": 48, "holdout": 6, "gen": 6}
_FEATS = ["alpha", "beta", "gamma", "delta"]


def _strict(svg: str) -> None:
    """A69 (79.2.4): strict XML parse — the renderer must emit a complete,
    well-formed document, not just a string."""
    xml.dom.minidom.parseString(svg)


# --- 79.2.1/79.2.2 the stage blocks (A69) --------------------------------------

def test_csv_shape_renders_all_stages():
    """A69 (79.2.1/79.2.2): the CSV-shaped input (features + rows) renders
    the six unconditional stages, well-formed and strict-XML-parseable."""
    svg = svg_data_flow(_MLP, 4, 3, "softmax", "accuracy",
                        features=_FEATS, rows=_ROWS)
    assert svg_is_well_formed(svg)
    _strict(svg)
    for stage in ("DATA", "FEATURES", "SPLIT", "STANDARDIZE", "MODEL", "SCORE"):
        assert f">{stage}</text>" in svg, stage
    # the data's shape at each stage (79.1)
    assert "60 rows" in svg            # DATA: 48 + 6 + 6
    assert "3 classes" in svg          # DATA: softmax head
    assert "alpha, beta, gamma, ..." in svg  # FEATURES: names, truncated
    assert "tr 48 | ho 6 | ge 6" in svg      # SPLIT: the counts
    assert "random permutation" in svg       # SPLIT: the rule
    assert "train-only mean/std" in svg      # STANDARDIZE
    assert "hidden [64, 32]" in svg          # MODEL: the family body
    assert "softmax 3" in svg                # SCORE: the head
    assert "score: accuracy" in svg          # SCORE: the metric
    assert "no leakage" in svg               # 79.2.5: the annotation row


def test_temporal_split_wording():
    """A69 (79.2.2): split_mode='temporal' (45.1) renders the walk-forward
    wording in the SPLIT stage."""
    svg = svg_data_flow(_MLP, 4, 3, features=_FEATS, rows=_ROWS,
                        split_mode="temporal")
    _strict(svg)
    assert "temporal walk-forward" in svg


def test_episode_shape_degrades_gracefully():
    """A69 (79.2.2): the episode-shaped input (no features, no rows —
    CartPole/GridNav) renders the policy-episode wording; the diagram is
    complete (all six stages, no crash on the missing attributes)."""
    svg = svg_data_flow(_MLP, 4, 1, "mse", "r2")
    assert svg_is_well_formed(svg)
    _strict(svg)
    for stage in ("DATA", "FEATURES", "SPLIT", "STANDARDIZE", "MODEL", "SCORE"):
        assert f">{stage}</text>" in svg, stage
    assert "episode task" in svg
    assert "state vector" in svg
    assert "seed-derived episodes" in svg
    assert "clip / normalize" in svg
    assert "regression" in svg
    assert "mse 1" in svg
    assert "score: r2" in svg
    assert "60 rows" not in svg  # no row counts without split arrays


def test_none_spec_renders_placeholder_model():
    """A69 (79.2.4): spec=None renders the pipeline with a placeholder
    MODEL block (no crash, well-formed)."""
    svg = svg_data_flow(None, 4, 3)
    _strict(svg)
    assert "(untrained)" in svg


# --- 79.2.3 the conditional SPECTRAL stage (A69) -------------------------------

def test_spectral_stage_only_when_fourier_positive():
    """A69 (79.2.3): the SPECTRAL stage renders only when
    fourier_features K > 0 (78.2.1), showing d -> d·(2K+1); at K = 0
    (legacy) the pipeline is the six unconditional stages."""
    legacy = dict(_MLP, fourier_features=0)
    assert "SPECTRAL" not in svg_data_flow(legacy, 4, 3)
    spec_k2 = dict(_MLP, fourier_features=2)
    svg = svg_data_flow(spec_k2, 4, 3)
    _strict(svg)
    assert "SPECTRAL" in svg
    # html.escape renders '>' as '&gt;' (79.2.4) — the d -> d·(2K+1) shape
    assert "4 -&gt; 20 dims" in svg
    assert "K = 2 (cos/sin)" in svg


# --- 79.2.4 purity (A69) --------------------------------------------------------

def test_deterministic_byte_identical():
    """A69 (G2): repeated calls with equal inputs are byte-identical —
    the renderer is pure (no RNG, no clock, no global state)."""
    a = svg_data_flow(_MLP, 4, 3, "softmax", "accuracy",
                      features=_FEATS, rows=_ROWS)
    b = svg_data_flow(dict(_MLP), 4, 3, "softmax", "accuracy",
                      features=list(_FEATS), rows=dict(_ROWS))
    assert a == b


def test_hostile_values_stay_well_formed():
    """A69 (79.2.4): hostile feature names / metrics cannot break the
    document — every data-derived string passes html.escape."""
    svg = svg_data_flow(_MLP, 1, 2, "softmax", "acc&<x>",
                        features=['a<b>&"q"'], rows=_ROWS)
    assert svg_is_well_formed(svg)
    _strict(svg)  # strict parse proves the escaping held


def test_rows_values_sanitized():
    """A69 (79.2.4): non-finite / negative row counts fall back to 0
    rather than producing NaN text in the SVG."""
    svg = svg_data_flow(_MLP, 4, 3, features=_FEATS,
                        rows={"train": float("nan"), "holdout": -5, "gen": 2})
    _strict(svg)
    assert "nan" not in svg.lower().replace("no leakage", "")
    assert "tr 0 | ho 0 | ge 2" in svg


# --- 79.3.1 finish() wiring (A69) ----------------------------------------------

def test_finish_returns_data_flow_svg(tmp_path):
    """A69 (79.3.1): `DashboardRunner.finish()` on a small CSV run returns
    the additive `data_flow_svg` key carrying the feature names, the row
    counts, and the no-leakage note (and `arch_svg` stays as before)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    p = tmp_path / "d.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("f0,f1,f2,f3,label\n")
        for i in range(100):
            f.write(",".join(repr(float(v)) for v in X[i]) + f",{y[i]}\n")
    runner = DashboardRunner(csv_path=str(p), label="label",
                             experiments=1, runs_dir=str(tmp_path))
    runner.start()
    while runner._phase != "done":
        runner.next()
    res = runner.finish()
    df = res["data_flow_svg"]
    assert svg_is_well_formed(df)
    _strict(df)
    assert "f0" in df                  # the feature names
    assert "100 rows" in df            # the total (79.1 DATA)
    assert "tr 80 | ho 10 | ge 10" in df  # the split counts (80/10/10)
    assert "no leakage" in df          # the 79.2.5 annotation
    assert res["arch_svg"]             # the structure view is untouched


def test_finish_data_flow_none_without_spec(tmp_path):
    """A69 (79.3.1): `data_flow_svg` is an additive key present in the
    result dict (None-safe contract: consumers read by key)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 2))
    y = (X[:, 0] > 0).astype(int)
    p = tmp_path / "d.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("a,b,label\n")
        for i in range(60):
            f.write(f"{X[i, 0]},{X[i, 1]},{y[i]}\n")
    runner = DashboardRunner(csv_path=str(p), label="label",
                             experiments=1, runs_dir=str(tmp_path))
    runner.start()
    while runner._phase != "done":
        runner.next()
    assert "data_flow_svg" in runner.finish()


# --- 79.3.2 the app renders it (A69) -------------------------------------------

def test_app_imports_and_renders_data_flow():
    """A69 (79.3.2): `dashboard_app` imports `svg_data_flow` and renders
    the "Data flow" visual (result view + live champion + experiments tab
    champion section)."""
    src = (REPO / "src" / "autorefine" / "dashboard_app.py").read_text(
        encoding="utf-8")
    assert "svg_data_flow" in src
    assert 'svg_data_flow,' in src  # the import
    assert '"Data flow"' in src     # the visual title (at least one render)
    assert src.count('"Data flow"') >= 2  # live + result (experiments uses the res key)
    assert '_svg_block("Data flow", res["data_flow_svg"])' in src


# --- 33.1 version + SPEC index (A69) --------------------------------------------

def test_version_and_spec_cite_a69():
    """A69 (33.1): the version steps to `0.65.0` in both sources; SPEC §79
    cites A69; the A25 index row for M68 points at this file."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.65.0"
    init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
        encoding="utf-8")
    assert '"0.65.0"' in init
    spec = SPEC.read_text(encoding="utf-8")
    assert "## 79. Data flow" in spec
    assert "Acceptance (A69)" in spec
    assert ("| M68 | v0.65   | 79     | A69 | tests/test_dataviz_v065.py |"
            in spec)
