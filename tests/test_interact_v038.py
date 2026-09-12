"""v0.38 — interact with the run while it's alive (SPEC.md 52, A42, M41).

Covers:

- 52.1.1 `dashboard.gate_math` — the per-candidate gate numbers hand-
  computed: `delta = candidate − best_before` (the 39.2.2 score gate),
  `gen_tol = GEN_GAP_TOL · candidate` (the 18.5 overfit tolerance),
  `z_se = z · se` (the 18.6 CI gate threshold); every output `None` for
  an unscored (dup) update; the runner's update dict carries the
  CI-gate inputs `se` + `effective_score` (JSON-safe) for a v04 run;
- 52.2 `dashboard.fields_seen` — the sorted unique mutation fields over
  the stream; `plotting.svg_mutation_timeline(..., field=...)` keeps
  only that field's cells (the default call byte-identical, G2) and the
  empty case names the field;
- 52.3.1 `dashboard.eta_seconds` / `plateau_streak` — hand-computed
  (mean train seconds × budget remaining; `None` when nothing left / no
  train time yet; the 31.1 stall semantics: acceptance resets,
  rejection extends, unscored steps ignored), with the display
  threshold `PLATEAU_HINT == 5`;
- 52.1.2/52.2.2/52.3.1/52.4 the app — a finished run renders the
  per-candidate drill-down expanders (gate math + curves); the
  `focus_field` selectbox exists and, when set, the Experiments table
  drops the non-matching rows; the live ETA caption renders; the Run
  tab's zero-height iframe carries the `keydown` JS naming both button
  labels (S = Stop, R = Run); the app source wires the 52.x helpers;
- the A42 round regression — the version stepped to `0.39.0` in both
  sources (33.1) and SPEC carries the A42 block + M41 row.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy core; streamlit is optional (the app tests skip without
it).
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine.dashboard import (
    DashboardRunner,
    PLATEAU_HINT,
    eta_seconds,
    fields_seen,
    gate_math,
    plateau_streak,
)
from autorefine.improver.meta_env import GEN_GAP_TOL
from autorefine.plotting import svg_mutation_timeline

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"


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


# 52.x inputs — update-dict rows (SPEC.md 23.1) with distinct mutation
# fields, one scored-accepted, one scored-rejected, one unscored dup
UPDATES = [
    {"index": 1, "accepted": True, "candidate_score": 70.0, "gen_gap": 0.4,
     "mutation": ["hidden_dim"], "reason": None, "train_seconds": 0.5},
    {"index": 2, "accepted": False, "candidate_score": 55.0, "gen_gap": 0.8,
     "mutation": ["lr"], "reason": None, "train_seconds": 1.0},
    {"index": 3, "accepted": False, "candidate_score": None, "gen_gap": None,
     "mutation": [], "reason": "duplicate", "train_seconds": None},
]


# --- 52.1.1 gate math (A42) ----------------------------------------------------

def test_gate_math_scored_update():
    """A42 (SPEC.md 52.1.1): the three gates' numbers over a scored
    update — `delta` (39.2.2 score gate), `gen_tol = GEN_GAP_TOL ·
    candidate` (18.5), `z_se = z · se` (18.6)."""
    u = {"accepted": False, "candidate_score": 70.0, "gen_gap": 0.4,
         "se": 1.0}
    gm = gate_math(u, 60.0, z_accept=1.0)
    assert gm["candidate"] == 70.0
    assert gm["best_before"] == 60.0
    assert gm["delta"] == 10.0
    assert gm["gen_gap"] == 0.4
    assert gm["gen_tol"] == round(GEN_GAP_TOL * 70.0, 6)
    assert gm["se"] == 1.0
    assert gm["z"] == 1.0
    assert gm["z_se"] == round(1.0 * 1.0, 6)
    assert gm["accepted"] is False
    json.dumps(gm)  # JSON-safe (G2)


def test_gate_math_accepted_and_missing_best():
    """A42 (SPEC.md 52.1.1): `accepted` is carried through; a missing
    `best_before` (not a finite number) leaves `delta` `None` while the
    overfit/CI columns still compute from the candidate's own numbers."""
    u = {"accepted": True, "candidate_score": 80.0, "gen_gap": 0.1,
         "se": 0.5}
    gm = gate_math(u, None, z_accept=2.0)
    assert gm["accepted"] is True
    assert gm["best_before"] is None
    assert gm["delta"] is None
    assert gm["gen_tol"] == round(GEN_GAP_TOL * 80.0, 6)
    assert gm["z"] == 2.0
    assert gm["z_se"] == round(2.0 * 0.5, 6)


def test_gate_math_unscored_update():
    """A42 (SPEC.md 52.1.1): an unscored (dup) candidate renders "—"
    across the board — every score-derived output is `None`."""
    u = {"accepted": False, "candidate_score": None, "gen_gap": None,
         "se": None, "mutation": [], "reason": "duplicate"}
    gm = gate_math(u, 60.0, z_accept=1.0)
    for key in ("candidate", "delta", "gen_gap", "gen_tol", "se", "z_se"):
        assert gm[key] is None, key
    assert gm["best_before"] == 60.0  # the running best is still shown
    assert gm["z"] == 1.0


def test_runner_updates_carry_ci_inputs(tmp_path):
    """A42 (SPEC.md 52.1.1): the runner's update dict carries the two
    CI-gate inputs (`se` + `effective_score`) for a v04 run, and every
    update stays JSON-safe (SPEC.md 23.1, G2)."""
    csv = _write_csv(tmp_path)
    runner = DashboardRunner(
        csv_path=csv, seed=7, experiments=2, max_train_seconds=5.0,
        runs_dir=str(tmp_path / "runs"), search_quality="v04")
    runner.start()
    updates: list[dict] = []
    runner.run_all(on_update=updates.append)
    assert len(updates) >= 2
    for u in updates:
        assert "se" in u, u["index"]
        assert "effective_score" in u, u["index"]
        json.dumps(u)  # JSON-safe (G2)
    # the v04 preset's z is what the drill-down's CI columns use (18.6)
    assert runner.env.z_accept == 1.0


# --- 52.2 focus field (A42) -----------------------------------------------------

def test_fields_seen_sorted_unique():
    """A42 (SPEC.md 52.2.1): the sorted unique mutation fields over the
    stream — the focus-field options."""
    ups = [
        {"mutation": ["hidden_dim", "lr"]},
        {"mutation": ["lr"]},
        {"mutation": []},
        {"mutation": ["dropout"]},
        "not a dict",
        {"mutation": None},
    ]
    assert fields_seen(ups) == ["dropout", "hidden_dim", "lr"]
    assert fields_seen(UPDATES) == ["hidden_dim", "lr"]
    assert fields_seen([]) == []
    assert fields_seen(None) == []


def test_timeline_field_filter_default_byte_identical():
    """A42 (SPEC.md 52.2.2): `field=None` (the default) is byte-identical
    to the pre-v0.38 call (G2)."""
    default = svg_mutation_timeline(UPDATES)
    explicit = svg_mutation_timeline(UPDATES, field=None)
    assert default == explicit


def test_timeline_field_filter_keeps_only_that_field():
    """A42 (SPEC.md 52.2.2): `field=` keeps only that mutation field's
    cells — the other fields' row labels do not render."""
    out = svg_mutation_timeline(UPDATES, field="hidden_dim")
    assert ">hidden_dim</text>" in out
    assert ">lr</text>" not in out
    # the other field's cells are gone: only one field row renders
    assert out.count("</text>") >= 1
    out_lr = svg_mutation_timeline(UPDATES, field="lr")
    assert ">lr</text>" in out_lr
    assert ">hidden_dim</text>" not in out_lr


def test_timeline_field_filter_empty_names_the_field():
    """A42 (SPEC.md 52.2.2): the empty case (no cells for the field)
    names the field in its message."""
    out = svg_mutation_timeline(UPDATES, field="knn_k")
    assert "knn_k" in out, out
    out_default = svg_mutation_timeline(
        [{"index": 1, "mutation": [], "accepted": False,
          "candidate_score": None}])
    assert "no mutations in the update stream" in out_default
    assert "knn_k" not in out_default


# --- 52.3.1 ETA + stall sentinel (A42) -----------------------------------------

def test_eta_seconds_hand_computed():
    """A42 (SPEC.md 52.3.1): mean `train_seconds` over finite entries ×
    budget remaining, rounded to 0.1 s; `None` when nothing is left to
    spend or no train time is known yet (the dup steps never skew the
    mean — R3)."""
    ups = [{"train_seconds": 1.0}, {"train_seconds": 3.0},
           {"train_seconds": None}]
    assert eta_seconds(ups, 2) == 4.0   # mean 2.0 × 2
    assert eta_seconds(ups, 1) == 2.0   # mean 2.0 × 1
    assert eta_seconds(ups, 3.5) == 7.0  # mean 2.0 × 3.5
    assert eta_seconds(ups, 0) is None
    assert eta_seconds(ups, -1) is None
    assert eta_seconds(ups, None) is None
    assert eta_seconds(ups, "x") is None
    assert eta_seconds([], 3) is None
    assert eta_seconds(None, 3) is None
    assert eta_seconds([{"train_seconds": None}], 3) is None


def test_plateau_streak_hand_computed():
    """A42 (SPEC.md 52.3.1): the trailing run of scored non-accepted
    candidates under the 31.1 semantics — a scored acceptance resets,
    a scored rejection extends, unscored (dup) steps are ignored."""
    ups = [
        {"candidate_score": 70.0, "accepted": True},    # resets to 0
        {"candidate_score": 65.0, "accepted": False},   # 1
        {"candidate_score": None, "accepted": False},   # dup — ignored
        {"candidate_score": 64.0, "accepted": False},   # 2
        {"candidate_score": 63.0, "accepted": False},   # 3
        {"candidate_score": 62.0, "accepted": False},   # 4
        {"candidate_score": 61.0, "accepted": False},   # 5
    ]
    assert plateau_streak(ups) == 5
    # a scored acceptance resets the streak
    ups2 = ups + [{"candidate_score": 90.0, "accepted": True}]
    assert plateau_streak(ups2) == 0
    assert plateau_streak(
        ups2 + [{"candidate_score": 85.0, "accepted": False}]) == 1
    # all-unscored / empty streams never stall the sentinel
    assert plateau_streak([{"candidate_score": None, "accepted": False}]) == 0
    assert plateau_streak([]) == 0
    assert plateau_streak(None) == 0
    # the display threshold is the 31.1 patience value
    assert PLATEAU_HINT == 5


# --- the app (A42) ---------------------------------------------------------------

def _tab_labels(at) -> list[str]:
    """The raw-tree walk for `tab` node labels (51.1.1)."""
    out: list[str] = []

    def walk(node) -> None:
        if getattr(node, "type", None) == "tab":
            lab = getattr(node, "label", None)
            if lab is not None:
                out.append(lab)
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            for el in ch.values():
                walk(el)

    walk(at.main)
    return out


def _iframes(at) -> list:
    """The raw-tree walk for zero-height `components.html` iframes
    (52.4)."""
    out: list = []

    def walk(node) -> None:
        if getattr(node, "type", None) == "iframe":
            out.append(node)
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            for el in ch.values():
                walk(el)

    walk(at.main)
    return out


def _setup_app(at, tmp_path: Path, experiments: int = 2) -> None:
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(experiments)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception


def test_app_run_drilldown_eta_focus_keyboard(tmp_path):
    """A42 (SPEC.md 52.1.2/52.2.2/52.3.1/52.4): a finished run renders
    the per-candidate drill-down expanders (gate math + curves), the
    live ETA caption, the `focus_field` selectbox, and the Run tab's
    zero-height keyboard iframe; focusing a field drops the
    non-matching Experiments-table rows."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    _setup_app(at, tmp_path, experiments=2)
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    # 52.1.2: the per-candidate drill-down expanders (gate math + curves)
    exp_labels = [str(e.label) for e in at.expander]
    assert any(l.startswith("candidate #") and "gate math" in l
               for l in exp_labels), exp_labels

    # 52.3.1: the live ETA caption ("ETA … · N experiment(s) left")
    caps = " ".join(str(c.value) for c in at.caption)
    assert re.search(r"ETA .+ experiment\(s\) left", caps), caps

    # 52.2.1/52.2.2: the focus-field selectbox exists
    assert any(e.key == "focus_field" for e in at.selectbox), \
        "the Run tab carries the focus_field selectbox"

    # 52.4: the zero-height keyboard iframe names both buttons + keydown
    frames = _iframes(at)
    assert frames, "the Run tab renders the zero-height keyboard iframe"
    srcdoc = frames[0].proto.srcdoc
    assert "Stop the run" in srcdoc
    assert "Run the improvement loop" in srcdoc
    assert "keydown" in srcdoc

    # 52.2.2: focus a mutated field — the Experiments table keeps only
    # the candidates that mutated it (the baseline row drops out)
    res = at.session_state["result"]
    stream = res["stream"]
    field = next(f for u in stream if isinstance(u.get("mutation"), list)
                 for f in u["mutation"] if isinstance(f, str))
    keep_idx = {u.get("index") for u in stream
                if isinstance(u.get("mutation"), list) and field in u["mutation"]}
    expected_rows = [r for r in res["rows"] if r.get("#") in keep_idx]
    assert expected_rows, "the focused field was mutated by a candidate"
    assert len(expected_rows) < len(res["rows"]), \
        "the focus must drop at least the baseline row"
    at.session_state["focus_field"] = field
    at.run()
    assert not at.exception, at.exception
    found = False
    for e in at.dataframe:
        df = e.value
        if "reason" in getattr(df, "columns", []) and "#" in \
                getattr(df, "columns", []):
            if len(df) == len(expected_rows):
                idxs = {int(v) for v in df["#"]}
                assert idxs == keep_idx, idxs
                found = True
    assert found, "the Experiments table honors the focus_field filter"


def test_app_source_wires_52():
    """A42 (SPEC.md 52.1.2/52.2.2/52.3.1/52.4): the app source wires
    the 52.x pure helpers, the focus selectbox key, the drill body, and
    the zero-height keyboard iframe."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "gate_math(",
        "fields_seen(",
        "plateau_streak(",
        "eta_seconds(",
        "PLATEAU_HINT",
        "components.html(_KEYBOARD_HTML",
        'key="focus_field"',
        "_drill_body",
        "findButton('Stop the run')",
        "findButton('Run the improvement loop')",
        "keydown",
        '"stream": stream',
    ):
        assert token in src, token


# --- A42 round regression ---------------------------------------------------------

def test_version_round_v038():
    """A42 (SPEC.md 52.5, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"


def test_spec_cites_a42_and_round():
    """A42 (SPEC.md 52.5/52.6): SPEC.md defines the A42 acceptance block
    and the M41 index row + milestone — the A25 index machinery reads
    both (defined == set(range(1, 43)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 52.5 Acceptance (A42)" in spec
    assert re.search(r"^\s*\| M41 \| v0\.38\s*\|\s*52\s*\|\s*A42\s*\|",
                     spec, re.MULTILINE)
    assert "**M41**" in spec
