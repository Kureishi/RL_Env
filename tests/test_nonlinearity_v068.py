"""Nonlinearity intelligence for nonlinear tasks (SPEC.md 82, A72).

SPEC.md 82 (v0.68): "Add more functionality/interconnections for
nonlinear tasks." Nonlinearity becomes a measured, tiered, reportable
quantity from the already-logged candidates:

  * 82.2  — `research.nonlinearity_profile`: the depth-0 linear best vs
    the overall best (the gap), the verdict tiers (1.0/5.0 points),
    the per-family deltas in `MODEL_FAMILIES` order, the measured
    spectral-expansion (``fourier_features``) split, and the
    token-clean steer hint;
  * 82.3  — `svg_nonlinearity`: the verdict chip, the per-family bars
    (the shared palette cycle), the dashed linear-baseline reference,
    the fourier-effect line, the threshold caption; pure/deterministic
    (G2), valid XML, okabe/dark themes (51.4), empty degradation;
  * 82.4  — the app's Nonlinearity panel (the app-language scan in
    `tests/test_app_language.py` keeps the panel strings clean);
  * 82.5  — `report --nonlinearity` (rc 0 on a real tiny run;
    mutually exclusive with ``--history``);
  * 82.6  — the version steps to `0.71.0` in both sources (33.1) and
    the A-index advances (asserted in `tests/test_coherency_v021.py`,
    A25).

House rules: pure / deterministic (G2), stdlib + numpy only, no
cross-test imports (every fixture is synthesized here).
"""
from __future__ import annotations

import json
import tomllib
import xml.dom.minidom
from pathlib import Path

import pytest

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.cli import main as cli_main
from autorefine.config import MODEL_FAMILIES
from autorefine.plotting import svg_is_well_formed, svg_nonlinearity
from autorefine.research import (
    NL_GAP_MODERATE,
    NL_GAP_STRONG,
    is_linear_spec,
    nonlinearity_profile,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

_LIN = {"model_family": "mlp", "architecture": []}
_LIN_NOFAM = {"architecture": []}  # pre-v0.3 spec JSON (family absent)
_MLP = {"model_family": "mlp", "architecture": [16, 8]}
_TREE = {"model_family": "tree", "architecture": [2]}
_KNN = {"model_family": "knn", "architecture": []}


def _entry(kind: str, spec: dict, holdout: float, **kw) -> dict:
    """A72 (81.2.2): one logged candidate row — the fields the profile
    reads, a dict spec, and a kind."""
    e = {
        "kind": kind,
        "spec": spec,
        "spec_hash": kw.pop("spec_hash", ""),
        "holdout_score": holdout,
        "accepted": kind == "baseline",
    }
    e.update(kw)
    return e


def _strict(svg: str) -> None:
    """A72 (82.3): strict XML parse — the renderer must emit a complete,
    well-formed document, not just a string."""
    xml.dom.minidom.parseString(svg)


# --- 82.2 the derivation (A72) --------------------------------------------------

def test_linear_predicate():
    """A72 (82.2.1): the depth-0 `mlp` (empty architecture, family
    `mlp` or absent) is linear; every other family or depth is
    nonlinear; a non-dict spec is never the linear baseline."""
    assert is_linear_spec(_LIN) is True
    assert is_linear_spec(_LIN_NOFAM) is True
    assert is_linear_spec({"model_family": "mlp", "architecture": ()})
    assert is_linear_spec(_MLP) is False
    assert is_linear_spec(_TREE) is False
    assert is_linear_spec(_KNN) is False  # empty arch, but the knn family
    assert is_linear_spec({"model_family": "mlp", "architecture": "x"}) is False
    assert is_linear_spec("garbage") is False
    assert is_linear_spec(None) is False


def test_verdict_tiers():
    """A72 (82.2.1): the gap tiers at the 1.0/5.0 point cuts —
    near-linear < 1.0, moderately 1.0–5.0, strongly ≥ 5.0."""
    lin = _entry("baseline", _LIN, 50.0, spec_hash="hlin")
    for extra, expected in ((50.5, "near-linear"),
                            (53.0, "moderately nonlinear"),
                            (60.0, "strongly nonlinear")):
        p = nonlinearity_profile([lin, _entry("experiment", _MLP, extra)])
        assert p["gap"] == pytest.approx(extra - 50.0)
        assert p["verdict"] == expected
    assert NL_GAP_MODERATE == 1.0 and NL_GAP_STRONG == 5.0
    # the boundary values: 1.0 is moderate, 5.0 is strong (strict <)
    p = nonlinearity_profile([lin, _entry("experiment", _MLP, 51.0)])
    assert p["verdict"] == "moderately nonlinear"
    p = nonlinearity_profile([lin, _entry("experiment", _MLP, 55.0)])
    assert p["verdict"] == "strongly nonlinear"


def test_unknown_when_no_linear_candidate():
    """A72 (82.2.1): a run with no linear candidate reports
    `unknown` with `gap: None`, an empty linear block, and all family
    deltas `None`."""
    rows = [_entry("baseline", _TREE, 45.0, spec_hash="ht1"),
            _entry("experiment", _KNN, 48.0, spec_hash="hk1")]
    p = nonlinearity_profile(rows)
    assert p["verdict"] == "unknown"
    assert p["gap"] is None
    assert p["linear"] == {"best": None, "n": 0, "hash": ""}
    assert [f["delta"] for f in p["families"]] == [None, None]
    assert "no linear-baseline" in p["steer"]
    # the overall block still reports the run's best row
    assert p["overall"]["best"] == 48.0
    assert p["overall"]["family"] == "knn"
    assert p["overall"]["hash"] == "hk1"


def test_families_order_and_deltas():
    """A72 (82.2.1/81.2.3): the `families` rows walk the
    `MODEL_FAMILIES` registry order (empty families skipped), with
    hand-computed best/n/delta over the linear best."""
    rows = [
        _entry("baseline", _LIN, 50.0, spec_hash="hlin"),
        _entry("experiment", _MLP, 40.0),
        _entry("experiment", _TREE, 45.0),
        _entry("experiment", _KNN, 48.0),
        _entry("experiment", _MLP, 52.0, spec_hash="hmlp1"),
    ]
    p = nonlinearity_profile(rows)
    assert [f["name"] for f in p["families"]] == ["mlp", "tree", "knn"]
    order = {name: i for i, name in enumerate(MODEL_FAMILIES)}
    assert [order[f["name"]] for f in p["families"]] == sorted(
        order[f["name"]] for f in p["families"])  # registry order
    got = {f["name"]: f for f in p["families"]}
    assert got["mlp"]["best"] == 52.0 and got["mlp"]["n"] == 3  # lin+2 deep
    assert got["mlp"]["delta"] == pytest.approx(2.0)
    assert got["tree"]["best"] == 45.0 and got["tree"]["n"] == 1
    assert got["tree"]["delta"] == pytest.approx(-5.0)
    assert got["knn"]["best"] == 48.0 and got["knn"]["delta"] == pytest.approx(-2.0)
    assert p["n_families"] == 3
    assert p["overall"] == {"best": 52.0, "n": 5, "hash": "hmlp1",
                            "family": "mlp"}


def test_fourier_split():
    """A72 (82.2.1): the `fourier_features` on/off split — the on side
    is the finite `> 0` values, the off side is `0`/absent; the delta
    is on − off; all `None` when either side is empty (a bool value
    counts as off, never on)."""
    # the fourier_features field lives on the *spec* (a ModelSpec field,
    # 78.2.1) — the profile reads it from `entry["spec"]`
    rows = [
        _entry("baseline", _LIN, 50.0),
        _entry("experiment", {**_MLP, "fourier_features": 16}, 80.0),
        _entry("experiment", {**_MLP, "fourier_features": 32}, 90.0),
        _entry("experiment", {**_MLP, "fourier_features": 0}, 60.0),
        _entry("experiment", _MLP, 55.0),  # absent -> off
        _entry("experiment", {**_MLP, "fourier_features": True}, 54.0),
    ]
    p = nonlinearity_profile(rows)
    f = p["fourier"]
    assert f["on_mean"] == pytest.approx(85.0)    # (80 + 90) / 2
    assert f["off_mean"] == pytest.approx(54.75)  # (50+60+55+54) / 4
    assert f["delta"] == pytest.approx(30.25)
    off_only = [r for r in rows
                if r["spec"] is _LIN
                or r["spec"].get("fourier_features") in (0, True)]
    p2 = nonlinearity_profile(off_only)
    assert p2["fourier"]["on_mean"] is None
    assert p2["fourier"]["delta"] is None
    assert p2["fourier"]["off_mean"] is not None


def test_scored_candidate_rule():
    """A72 (81.2.2): screen/curriculum rows, non-dict rows, and rows
    without a finite holdout / dict spec never enter the profile."""
    rows = [
        _entry("baseline", _LIN, 50.0),
        _entry("screen", _MLP, 99.0),            # excluded (kind)
        _entry("curriculum", _MLP, 98.0),        # excluded (kind)
        _entry("experiment", _MLP, float("nan")),  # excluded (non-finite)
        {"kind": "experiment", "spec": "not-a-dict", "holdout_score": 97.0},
        "garbage-row",                            # excluded (non-dict)
        _entry("experiment", _MLP, 52.0),
    ]
    p = nonlinearity_profile(rows)
    assert p["n_scored"] == 2  # the baseline + the one scored mlp row
    assert p["overall"]["best"] == 52.0


def test_determinism_and_empty_degradation():
    """A72 (82.2.1/G2): repeated calls are byte-identical (via
    `json.dumps`), and empty/garbage input degrades to the empty
    profile — never an exception."""
    rows = [_entry("baseline", _LIN, 50.0), _entry("experiment", _MLP, 53.0)]
    a = json.dumps(nonlinearity_profile(rows), sort_keys=True)
    b = json.dumps(nonlinearity_profile(rows), sort_keys=True)
    assert a == b
    for bad in ([], None, ["garbage", {"kind": "screen"}], 42):
        p = nonlinearity_profile(bad)
        assert p["n_scored"] == 0
        assert p["verdict"] == "unknown"
        assert p["gap"] is None
        assert p["families"] == []


def test_steer_hint_is_token_clean():
    """A72 (82.1): the steer hint is deterministic and free of
    SPEC/§/round tokens — the app may display it verbatim (the
    tests/test_app_language.py contract)."""
    lin = _entry("baseline", _LIN, 50.0)
    for score in (50.5, 53.0, 60.0):
        p = nonlinearity_profile([lin, _entry("experiment", _MLP, score)])
        assert p["steer"] == nonlinearity_profile(
            [lin, _entry("experiment", _MLP, score)])["steer"]
        for token in ("SPEC", "§", "v0.", "SPEC.md"):
            assert token not in p["steer"]
    p = nonlinearity_profile([_entry("baseline", _TREE, 45.0)])
    for token in ("SPEC", "§", "v0.", "SPEC.md"):
        assert token not in p["steer"]


def test_unknown_family_is_counted_but_not_barred():
    """A72 (82.2.1): a spec with a family outside the registry is
    counted in `n_scored` / `overall`, but its bar is skipped (the
    81.2.3 registry-order convention) — the profile never invents a
    bar for an unknown family."""
    rows = [_entry("baseline", _LIN, 50.0),
            _entry("experiment", {"model_family": "bogus",
                                  "architecture": []}, 99.0)]
    p = nonlinearity_profile(rows)
    assert p["n_scored"] == 2
    assert p["overall"]["best"] == 99.0
    assert p["overall"]["family"] == "bogus"
    assert [f["name"] for f in p["families"]] == ["mlp"]
    assert p["n_families"] == 1


# --- 82.3 the renderer (A72) ----------------------------------------------------

def _profile_with_linear() -> dict:
    return nonlinearity_profile([
        _entry("baseline", _LIN, 50.0, spec_hash="hlin"),
        _entry("experiment", {**_MLP, "fourier_features": 16}, 85.0),
        _entry("experiment", _MLP, 60.0),
        _entry("experiment", _TREE, 45.0),
        _entry("experiment", _KNN, 48.0),
    ])


def test_svg_well_formed_strict_and_deterministic():
    """A72 (82.3): `svg_nonlinearity` is well-formed and
    strict-XML-parseable, and byte-identical across repeated calls
    (G2)."""
    p = _profile_with_linear()
    s1 = svg_nonlinearity(p)
    assert svg_is_well_formed(s1)
    _strict(s1)
    assert s1 == svg_nonlinearity(p)
    # non-dict input degrades to the empty document, still parseable
    s0 = svg_nonlinearity("garbage")
    assert svg_is_well_formed(s0)
    _strict(s0)


def test_svg_linear_baseline_and_fourier_states():
    """A72 (82.3): the dashed linear-baseline line + gap annotation
    render when a linear best exists, and both are absent when the
    run has no linear candidate; the fourier-effect line renders in
    both states."""
    with_lin = svg_nonlinearity(_profile_with_linear())
    assert "linear baseline" in with_lin
    assert "stroke-dasharray" in with_lin
    assert "gap = best - linear = 35.0 points" in with_lin
    assert "on 85.0 vs off 50.8" in with_lin  # the renderer's one-decimal read
    assert "(delta +34.2)" in with_lin
    no_lin = svg_nonlinearity(nonlinearity_profile(
        [_entry("baseline", _TREE, 45.0),
         _entry("experiment", _KNN, 48.0)]))
    assert "<line " not in no_lin  # no reference line (the caption may name it)
    assert "stroke-dasharray" not in no_lin
    assert "gap = n/a (no linear candidate in this run)" in no_lin
    assert "was not tried in this run" in no_lin
    # the verdict chip reads the tier
    assert ">strongly nonlinear<" in with_lin


def test_svg_empty_message():
    """A72 (82.3): no scored candidates -> the header + the empty
    message (never a broken document)."""
    s = svg_nonlinearity(nonlinearity_profile([]))
    assert svg_is_well_formed(s)
    _strict(s)
    assert "no scored candidates to profile" in s


def test_svg_themes_and_bad_args():
    """A72 (51.4/82.3): okabe and dark are byte-different from the
    default; a bad palette / non-bool dark is a loud ValueError."""
    base = svg_nonlinearity(_profile_with_linear())
    assert svg_nonlinearity(_profile_with_linear(), palette="okabe") != base
    assert svg_nonlinearity(_profile_with_linear(), dark=True) != base
    with pytest.raises(ValueError):
        svg_nonlinearity(_profile_with_linear(), palette="nope")
    with pytest.raises(ValueError):
        svg_nonlinearity(_profile_with_linear(), dark="yes")


# --- 82.5 CLI (A72) --------------------------------------------------------------

@pytest.fixture
def real_run(tmp_path):
    """A72 (82.5): a tiny, self-consistent live run (parity-v1, 2
    experiments, G2)."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state = env.step(policy.propose(state))[0]
    assert env.memory.load_summary()
    return Path(env.run_dir)


def test_cli_report_nonlinearity(real_run, capsys):
    """A72 (82.5): ``report --run DIR --nonlinearity`` is rc 0 with
    the verdict block, the family table, and the steer hint."""
    rc = cli_main(["report", "--run", str(real_run), "--nonlinearity"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("\nnonlinearity profile (")
    assert "verdict      :" in out
    assert "overall best :" in out
    assert "families     :" in out
    assert "steer        :" in out


def test_cli_report_nonlinearity_history_exclusive(capsys):
    """A72 (82.5): ``--nonlinearity`` and ``--history`` are mutually
    exclusive (rc 1)."""
    rc = cli_main(["report", "--nonlinearity", "--history"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--nonlinearity and --history are mutually exclusive" in err


# --- 82.6 regression (A72) -------------------------------------------------------

def test_spec_and_version():
    """A72 (SPEC.md 82.6): SPEC.md defines the A72 acceptance block,
    the M71 index row + milestone, and the version stepped to
    `0.71.0` in both sources (33.1)."""
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 82.6 Acceptance (A72)" in spec
    assert "### 82.7 Milestone (M71)" in spec
    assert ("| M71 | v0.68   | 82     | A72 | "
            "tests/test_nonlinearity_v068.py |") in spec
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.71.0"
