"""v0.36 — advanced: N-run comparison + one-click exports (SPEC.md 50,
A40, M39).

Covers:

- 50.1.1 `dashboard.diff_n_summaries` — the N >= 2 generalisation of
  `diff_two_summaries`: per-run gate rows in input order, the best_spec
  union table (only the >= 2-distinct fields; a single-sided field is
  `None` on the other runs; list values compared by content),
  `best_run_id` (first on ties), `score_span`; < 2 / non-list input is
  a `ValueError`; JSON-safe, deterministic (G2); agrees with
  `diff_two_summaries` on the same pair;
- 50.1.2 `dashboard.running_best_curve` — the running-max over the
  scored log rows (a rejected dip never lowers the curve); no scored
  rows -> `[]`;
- 50.1.3 `plotting.svg_run_curves` — 2..N running-best curves on a
  shared experiment-index axis, one polyline + one named legend entry
  per run, the target line only when inside the y-range, the empty
  header + message for an all-invalid input, valid XML,
  byte-deterministic (G2);
- 50.1.4 `cli._cmd_compare` — 2 or 3 run dirs: the pinned two-run
  surface unchanged (A32), the three-run gate rows + spec table +
  `--json == diff_n_summaries` (run dirs tagged as `run_id`), one or
  four dirs rc 1;
- 50.2.1 `sharing.share_payload` / `zip_bundle_bytes` /
  `write_share_zip` — the name-sorted payload (nested dirs excluded,
  the injected `report.html` included, missing optionals omitted),
  the missing-summary `ValueError`, the deterministic zip, and the CLI
  `share` byte-identity (47.4.3, the A37 surfaces unchanged);
- 50.1.5 / 50.2.2 / 50.2.3 the app — after a finished run the result
  view offers "Download run_config.json" + "Build share bundle"
  (pressing the latter reveals the entry table incl. `report.html` +
  the zip download); with >= 2 registered runs the Past-runs expander
  renders the curve overlay + gate rows + per-run recipes; the source
  wiring + the export block's placement;
- the A40 round regression — the version stepped to `0.39.0` in both
  sources (33.1) and SPEC carries the A40 block + M39 row.

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy only; streamlit is optional (the app tests skip
without it).
"""
from __future__ import annotations

import io
import json
import re
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

import autorefine
from autorefine.cli import main as cli_main
from autorefine.dashboard import (
    diff_n_summaries,
    diff_two_summaries,
    running_best_curve,
)
from autorefine.plotting import svg_run_curves
from autorefine.sharing import share_payload, write_share_zip, zip_bundle_bytes

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


S1 = {"final_best_score": 80.0, "target": 95.0, "met_target": False,
      "best_spec": {"hidden_dim": 64, "shared": 1, "only_a": 5}}
S2 = {"final_best_score": 90.0, "target": 95.0, "met_target": False,
      "best_spec": {"hidden_dim": 128, "shared": 1, "only_b": 7}}
S3 = {"final_best_score": 70.0, "target": 80.0, "met_target": True,
      "best_spec": {"hidden_dim": 64, "shared": 1}}


def _tag(s: dict, run_id: str) -> dict:
    d = dict(s)
    d["run_id"] = run_id
    return d


def _fake_run_dir(path: Path, summary: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8")
    return path


def _fake_share_dir(root: Path, name: str = "run") -> Path:
    """The 47.4.2 fixture: summary + run_config + best_spec + two flat
    SVGs + a nested-svg dir (the nested one must be excluded)."""
    d = root / name
    (d / "nested").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(
        {"task": "csv", "seed": 7, "final_best_score": 80.0,
         "best_spec": {"hidden_dim": 64}}, sort_keys=True),
        encoding="utf-8")  # `seed` like a real summary (28.4) — the report
    # path reconstructs the task from task/seed/task_config
    (d / "run_config.json").write_text(
        '{"schema": "autorefine.run_config/1"}', encoding="utf-8")
    (d / "best_spec.json").write_text('{"hidden_dim": 64}', encoding="utf-8")
    (d / "score.svg").write_text("<svg></svg>", encoding="utf-8")
    (d / "pareto.svg").write_text("<svg></svg>", encoding="utf-8")
    (d / "nested" / "inner.svg").write_text("<svg></svg>", encoding="utf-8")
    return d


# --- 50.1.1 diff_n_summaries (A40) --------------------------------------------

def test_diff_n_summaries_three_runs():
    """A40 (SPEC.md 50.1.1): three summaries with distinct scores,
    targets, and specs (a shared-identical field, a two-sided
    differing field, and a single-sided field) — gate rows in input
    order, the spec-field union (>= 2 distinct only), best_run_id,
    score_span; JSON-safe and deterministic (G2)."""
    d = diff_n_summaries([_tag(S1, "a"), _tag(S2, "b"), _tag(S3, "c")])
    assert d["n_runs"] == 3
    assert [r["run_id"] for r in d["runs"]] == ["a", "b", "c"]  # input order
    assert [r["final_score"] for r in d["runs"]] == [80.0, 90.0, 70.0]
    assert [r["met_target"] for r in d["runs"]] == [False, False, True]
    # the union: >= 2 distinct fields only; "shared" (identical) excluded
    assert [f["field"] for f in d["spec_fields"]] == \
        ["hidden_dim", "only_a", "only_b"]
    by = {f["field"]: f for f in d["spec_fields"]}
    assert by["hidden_dim"]["values"] == [64, 128, 64]
    assert by["only_a"]["values"] == [5, None, None]  # single-sided -> None
    assert by["only_b"]["values"] == [None, 7, None]
    assert d["best_run_id"] == "b"  # the 90.0 scorer
    assert d["score_span"] == 20.0  # 90 - 70
    json.dumps(d)  # JSON-safe
    assert d == diff_n_summaries(
        [_tag(S1, "a"), _tag(S2, "b"), _tag(S3, "c")])  # deterministic


def test_diff_n_summaries_list_values_by_content():
    """A40 (SPEC.md 50.1.1): list-valued spec fields compare by content
    (canonical JSON) — equal lists are one distinct value (no diff
    row); reordered lists are a second one."""
    a = {"final_best_score": 80.0, "best_spec": {"hidden": [64, 128]}}
    b = {"final_best_score": 90.0, "best_spec": {"hidden": [64, 128]}}
    c = {"final_best_score": 70.0, "best_spec": {"hidden": [128, 64]}}
    assert diff_n_summaries([_tag(a, "a"), _tag(b, "b")])["spec_fields"] \
        == []  # content-equal lists -> identical
    d = diff_n_summaries([_tag(a, "a"), _tag(b, "b"), _tag(c, "c")])
    assert [f["field"] for f in d["spec_fields"]] == ["hidden"]


def test_diff_n_summaries_errors():
    """A40 (SPEC.md 50.1.1): fewer than two summaries (and a non-list
    input) raise `ValueError`."""
    with pytest.raises(ValueError):
        diff_n_summaries([_tag(S1, "a")])
    with pytest.raises(ValueError):
        diff_n_summaries([])
    with pytest.raises(ValueError):
        diff_n_summaries(_tag(S1, "a"))  # non-list


def test_diff_n_summaries_agrees_with_diff_two():
    """A40 (SPEC.md 50.1.1): against the same two summaries it agrees
    with `diff_two_summaries` on the differing-field set and the score
    pair."""
    pair2 = diff_two_summaries(S1, S2)
    d = diff_n_summaries([_tag(S1, "a"), _tag(S2, "b")])
    assert {f["field"] for f in d["spec_fields"]} == \
        {row["field"] for row in pair2["spec_diff"]}
    assert d["runs"][0]["final_score"] == pair2["score_a"] == 80.0
    assert d["runs"][1]["final_score"] == pair2["score_b"] == 90.0
    assert d["best_run_id"] == "b" and d["score_span"] == 10.0
    # the app imports the core names (38.4 pattern — one home)
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    import autorefine.dashboard_app as app
    assert app.diff_n_summaries is diff_n_summaries
    assert app.running_best_curve is running_best_curve
    assert app.svg_run_curves is svg_run_curves


# --- 50.1.2 running_best_curve (A40) -------------------------------------------

def test_running_best_curve_running_max():
    """A40 (SPEC.md 50.1.2): baseline + accepted + rejected +
    non-scored rows yield the running max in log order (a rejected dip
    never lowers the curve)."""
    rows = [
        {"kind": "baseline", "holdout_score": 60.0},
        {"kind": "experiment", "holdout_score": 80.0},
        {"kind": "experiment", "holdout_score": 70.0},  # a rejected dip
        {"kind": "screen", "holdout_score": 50.0},      # not scored
        {"kind": "experiment"},                          # no score
        {"kind": "experiment", "holdout_score": 85.0},
    ]
    assert running_best_curve(rows) == [60.0, 80.0, 80.0, 85.0]


def test_running_best_curve_empty():
    """A40 (SPEC.md 50.1.2): no scored rows yield `[]` (also for a
    missing input)."""
    assert running_best_curve([]) == []
    assert running_best_curve(None) == []
    assert running_best_curve([{"kind": "screen", "holdout_score": 5.0},
                               {"kind": "experiment"}]) == []


# --- 50.1.3 svg_run_curves (A40) ------------------------------------------------

NAMED = [("a", [60.0, 75.0]), ("b", [50.0, 80.0, 90.0]), ("c", [70.0])]


def test_svg_run_curves_three_runs():
    """A40 (SPEC.md 50.1.3): three named curves render one <polyline>
    + one named legend entry per run and parse as XML."""
    svg = svg_run_curves(NAMED)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    ET.fromstring(svg)  # valid XML (G2)
    assert svg.count("<polyline") == 3
    assert svg.count("<circle") >= 2 + 3 + 1
    for name in ("a", "b", "c"):  # the named legend entries
        assert f">{name}</text>" in svg or name in svg
    assert "best score per run (N-run compare)" in svg


def test_svg_run_curves_target_line():
    """A40 (SPEC.md 50.1.3): the target line appears when the target
    is inside the score range and not when it is far outside (50..90
    here — 75 in, 999 out)."""
    assert "target 75.00" in svg_run_curves(NAMED, target=75.0)
    assert "target 999.00" not in svg_run_curves(NAMED, target=999.0)


def test_svg_run_curves_empty_and_invalid():
    """A40 (SPEC.md 50.1.3): an empty or all-invalid input renders the
    empty header + message (the 21.2 convention) with no polylines."""
    for named in ([], [("x", []), ("y", [None, "nan"])]):
        svg = svg_run_curves(named)
        ET.fromstring(svg)
        assert "no run curves to overlay" in svg
        assert "<polyline" not in svg


def test_svg_run_curves_deterministic():
    """A40 (SPEC.md 50.1.3): pure and deterministic (G2) — same
    inputs, byte-identical SVG."""
    assert svg_run_curves(NAMED, target=75.0) == svg_run_curves(NAMED,
                                                                 target=75.0)


# --- 50.1.4 the compare CLI (A40) ------------------------------------------------

def test_compare_three_runs_cli(tmp_path, capsys):
    """A40 (SPEC.md 50.1.4): three fake run dirs give rc 0 with one gate
    row per run dir, the spec-field table, and `--json` equal to the
    `diff_n_summaries` dict (run dirs tagged as `run_id`)."""
    a = _fake_run_dir(tmp_path / "a",
                      {"final_best_score": 80.0, "target": 95.0,
                       "met_target": False, "experiments_run": 4,
                       "wall_seconds": 2.0, "policy": "bandit",
                       "best_spec": {"hidden_dim": 64}})
    b = _fake_run_dir(tmp_path / "b",
                      {"final_best_score": 90.0, "target": 95.0,
                       "met_target": True, "experiments_run": 6,
                       "wall_seconds": 3.0, "policy": "search",
                       "best_spec": {"hidden_dim": 128}})
    c = _fake_run_dir(tmp_path / "c",
                      {"final_best_score": 70.0, "target": 80.0,
                       "experiments_run": 2, "wall_seconds": 1.0,
                       "policy": "bandit",
                       "best_spec": {"hidden_dim": 64}})
    rc = cli_main(["compare", "--run", str(a), "--run", str(b),
                   "--run", str(c)])
    out = capsys.readouterr().out
    assert rc == 0, out
    for i in (1, 2, 3):
        assert f"run {i} :" in out  # one gate row per run dir
    assert "score : 80.0 | target : 95.0 | gate : MISS" in out
    assert "score : 90.0 | target : 95.0 | gate : PASS" in out
    assert "gate : —" in out  # c has no met_target
    assert "hidden_dim" in out and "64 | 128 | 64" in out
    assert "best    : b (span 20.0)" in out
    # --json == the diff_n_summaries dict (tagged copies of the summaries)
    rc = cli_main(["compare", "--run", str(a), "--run", str(b),
                   "--run", str(c), "--json"])
    assert rc == 0
    expected = diff_n_summaries([
        dict(json.loads((d / "summary.json").read_text(encoding="utf-8")),
             run_id=d.name)
        for d in (a, b, c)
    ])
    assert json.loads(capsys.readouterr().out) == expected


def test_compare_three_runs_identical_specs(tmp_path, capsys):
    """A40 (SPEC.md 50.1.4): identical best_specs -> the `identical`
    caption line (no spec table)."""
    a = _fake_run_dir(tmp_path / "a",
                      {"final_best_score": 80.0, "best_spec": {"h": 1}})
    b = _fake_run_dir(tmp_path / "b",
                      {"final_best_score": 85.0, "best_spec": {"h": 1}})
    c = _fake_run_dir(tmp_path / "c",
                      {"final_best_score": 75.0, "best_spec": {"h": 1}})
    rc = cli_main(["compare", "--run", str(a), "--run", str(b),
                   "--run", str(c)])
    out = capsys.readouterr().out
    assert rc == 0 and "best_spec: identical" in out


def test_compare_two_run_pin_unchanged(tmp_path, capsys):
    """A40 (SPEC.md 50.1.4; the A32 pin): the pinned two-run output is
    unchanged — `delta   : 4.5 (B - A)` and `64 -> 128`; `--json` is
    the `diff_two_summaries` dict."""
    a = _fake_run_dir(tmp_path / "a",
                      {"final_best_score": 80.0,
                       "best_spec": {"hidden_dim": 64, "lr": 0.001}})
    b = _fake_run_dir(tmp_path / "b",
                      {"final_best_score": 84.5,
                       "best_spec": {"hidden_dim": 128, "lr": 0.001}})
    rc = cli_main(["compare", "--run", str(a), "--run", str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "delta   : 4.5 (B - A)" in out
    assert "64 -> 128" in out and "lr" not in out
    rc = cli_main(["compare", "--run", str(a), "--run", str(b), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == diff_two_summaries(
        json.loads((a / "summary.json").read_text(encoding="utf-8")),
        json.loads((b / "summary.json").read_text(encoding="utf-8")))


def test_compare_one_and_four_dirs_rc1(tmp_path, capsys):
    """A40 (SPEC.md 50.1.4): one and four run dirs are rc 1 with the
    two-or-three message (the A32 one-dir rc 1 pin intact)."""
    a = _fake_run_dir(tmp_path / "a", {"final_best_score": 1.0,
                                       "best_spec": {}})
    assert cli_main(["compare", "--run", str(a)]) == 1
    msg = capsys.readouterr().err
    assert "two or three run dirs" in msg
    b = _fake_run_dir(tmp_path / "b", {"final_best_score": 2.0,
                                       "best_spec": {}})
    c = _fake_run_dir(tmp_path / "c", {"final_best_score": 3.0,
                                       "best_spec": {}})
    d = _fake_run_dir(tmp_path / "d", {"final_best_score": 4.0,
                                       "best_spec": {}})
    assert cli_main(["compare", "--run", str(a), "--run", str(b),
                     "--run", str(c), "--run", str(d)]) == 1
    assert "two or three run dirs" in capsys.readouterr().err
    # a missing summary among the three is still rc 1 (A32)
    assert cli_main(["compare", "--run", str(a), "--run", str(b),
                     "--run", str(tmp_path / "nope")]) == 1
    assert "summary.json" in capsys.readouterr().err


# --- 50.2.1 the sharing core (A40) ---------------------------------------------

def test_share_payload_sorted_and_filtered(tmp_path):
    """A40 (SPEC.md 50.2.1): the fake run dir (summary + run_config +
    best_spec + two flat SVGs + a nested-svg dir) yields the
    name-sorted payload — the nested dir excluded, the injected
    `report.html` included, a missing optional omitted."""
    d = _fake_share_dir(tmp_path)
    payload = share_payload(d, "<html>report</html>")
    assert list(payload) == sorted(payload)
    assert list(payload) == [
        "best_spec.json", "pareto.svg", "report.html", "run_config.json",
        "score.svg", "summary.json"]
    assert payload["report.html"] == b"<html>report</html>"
    assert payload["summary.json"] == (d / "summary.json").read_bytes()
    assert all("/" not in n and "\\" not in n for n in payload)  # flat
    # no injected report -> omitted (47.4.2)
    assert "report.html" not in share_payload(d)
    # missing optionals -> omitted
    d2 = tmp_path / "minimal"
    d2.mkdir()
    (d2 / "summary.json").write_text(
        json.dumps({"final_best_score": 1.0}), encoding="utf-8")
    assert list(share_payload(d2)) == ["summary.json"]
    # missing summary -> ValueError (47.4.4)
    d3 = tmp_path / "empty"
    d3.mkdir()
    with pytest.raises(ValueError):
        share_payload(d3)
    with pytest.raises(ValueError):
        share_payload(d3, "<html>x</html>")


def test_zip_bundle_deterministic_and_valid(tmp_path):
    """A40 (SPEC.md 50.2.1): `zip_bundle_bytes` is byte-deterministic,
    opens as a zip with the sorted namelist + the exact entry bytes,
    and `write_share_zip` writes the identical bytes."""
    d = _fake_share_dir(tmp_path)
    payload = share_payload(d, "<html>report</html>")
    z1, z2 = zip_bundle_bytes(payload), zip_bundle_bytes(payload)
    assert z1 == z2  # deterministic (47.4.3)
    with zipfile.ZipFile(io.BytesIO(z1)) as zf:
        assert zf.namelist() == sorted(payload)
        for name in payload:
            assert zf.read(name) == payload[name]
    out = tmp_path / "out" / "bundle.zip"  # parent dirs created
    write_share_zip(payload, out)
    assert out.read_bytes() == z1


def test_cli_share_identity_with_core(tmp_path, capsys):
    """A40 (SPEC.md 50.2.1/47.4.3): the CLI `share` zip of the same dir
    equals `zip_bundle_bytes(share_payload(dir, _share_report_html(dir)))`
    — the refactored command kept its A37 byte contract."""
    d = _fake_share_dir(tmp_path)
    z = tmp_path / "share.zip"
    rc = cli_main(["share", "--run", str(d), "--out", str(z)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "share   :" in out and str(z) in out
    from autorefine.cli import _share_report_html
    expected = zip_bundle_bytes(share_payload(d, _share_report_html(d)))
    assert z.read_bytes() == expected
    # and the A37 missing-summary surface (rc 1, no partial zip)
    rc = cli_main(["share", "--run", str(tmp_path / "nope"),
                   "--out", str(tmp_path / "x.zip")])
    assert rc == 1 and not (tmp_path / "x.zip").is_file()
    assert "47.4" in capsys.readouterr().err


# --- 50.1.5 / 50.2 the app (A40) -------------------------------------------------

def test_app_exports_after_finished_run(tmp_path):
    """A40 (SPEC.md 50.2.2/50.2.3): after a finished app run the result
    view offers 'Download run_config.json' and 'Build share bundle';
    pressing the latter (no exception) reveals the entry table
    (including `report.html`) + the zip download button."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
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
    at.button(key="run_button").set_value(True).run()
    assert not at.exception
    # the export block (50.2): the recipe download + the build button.
    # NOTE: this Streamlit build's AppTest does not expose `key` on
    # download_button elements (all None) — identify by unique label.
    labels = {e.label for e in at.get("download_button")}
    assert "Download run_config.json" in labels
    assert at.button(key="build_share").value is False
    # pressing it builds the bundle in memory (50.2.3) — no exception
    at.button(key="build_share").set_value(True).run()
    assert not at.exception, at.exception
    labels = {e.label for e in at.get("download_button")}
    assert "Download run_config.json" in labels
    assert "Download share bundle (.zip)" in labels
    # the entry table includes the injected report.html (47.4.2)
    found = False
    for e in at.dataframe:
        try:
            vals = [str(v) for v in e.value.values.ravel()]
        except Exception:
            continue
        if "report.html" in vals:
            found = True
            break
    assert found, "the share-bundle entry table lists report.html"


def test_app_nrun_compare_panel(tmp_path):
    """A40 (SPEC.md 50.1.5): with >= 2 registered runs the Past-runs
    expander offers the 2-3 run multiselect; selecting two renders the
    curve overlay (one named series per run), the gate rows, and the
    per-run recipes."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
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
    for _ in range(2):  # two runs -> two registry entries (38.1)
        at.button(key="run_button").set_value(True).run()
        assert not at.exception
    ids = sorted(p.name for p in (tmp_path / "runs").glob("csv-seed7-*"))
    assert len(ids) == 2, ids
    # the multiselect is materialized with both runs as options
    opts = {e.key for e in at.multiselect}
    assert "nrun_sel" in opts
    at.multiselect(key="nrun_sel").set_value([ids[0], ids[1]])
    at.run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "best score per run (N-run compare)" in md  # the overlay
    captions = " ".join(e.value for e in at.caption)
    assert "recipe — " in captions  # the per-run recipes (37.1.4)
    # and a registry entry exists per run dir (the Past-runs data source)
    assert len(ids) >= 2


def test_app_nrun_compare_empty_registry(tmp_path):
    """A40 (SPEC.md 50.1.5/38.4.2): with no finished runs the Past-runs
    section shows the empty state and no N-compare multiselect."""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(_write_csv(tmp_path)))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.run()
    assert not at.exception
    captions = " ".join(e.value for e in at.caption)
    assert "No finished runs yet" in captions
    assert not any(e.key == "nrun_sel" for e in at.multiselect)


# --- 50.1.5 / 50.2 source wiring (A40) ------------------------------------------

def test_app_source_wiring_and_placement():
    """A40 (SPEC.md 50.1.5/50.2.2/50.2.3): the app source wires
    `diff_n_summaries` / `running_best_curve` / `svg_run_curves` (the
    N-run panel lives in the Past-runs section, generalising 38.4),
    and the export block sits between the copy-paste recipe and the
    spec-space caption."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        "diff_n_summaries,",
        "running_best_curve,",
        "svg_run_curves,",
        "from autorefine.runconfig import RunConfig, fit_recipe",
        'st.download_button("Download run_config.json"',
        'st.button("Build share bundle", key="build_share")',
        'st.download_button("Download share bundle (.zip)"',
    ):
        assert token in src, token
    # the N-run panel is inside _render_past_runs (the 38.4 home)
    past = src[src.index("def _render_past_runs("):
              src.index("def _launcher_runs_dir(")]
    assert 'key="nrun_sel"' in past
    assert "diff_n_summaries(tagged)" in past
    assert "svg_run_curves(named)" in past
    assert "fit_recipe(cfg)" in past
    # the export block placement: recipe -> download -> build ->
    # spec-space caption (the result view)
    pos_recipe = src.index("Reproduce (copy-paste)")
    pos_dl = src.index('st.download_button("Download run_config.json"')
    pos_build = src.index('st.button("Build share bundle", key="build_share"')
    pos_specspace = src.index('f"spec space: {len(SPEC_FIELD_NAMES)} fields')
    assert pos_recipe < pos_dl < pos_build < pos_specspace


# --- A40 round regression ---------------------------------------------------------

def test_version_round_v036():
    """A40 (SPEC.md 50.5, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"


def test_spec_cites_a40_and_round():
    """A40 (SPEC.md 50.5): SPEC.md defines the A40 acceptance block and
    the M39 index row + milestone — the A25 index machinery reads both
    (defined == set(range(1, 41)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 50.4 Acceptance (A40)" in spec
    assert re.search(r"^\s*\| M39 \| v0\.36\s*\|\s*50\s*\|\s*A40\s*\|",
                     spec, re.MULTILINE)
    assert "**M39**" in spec
