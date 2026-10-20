"""A85 (SPEC.md 95, M84): the narrative — the environment's identity in
one screen (`examples/narrative.py`).

House rules (A25): stdlib only, no cross-test imports, deterministic,
tiny budgets (4 experiments), every docstring cites its A#.
"""
import ast
import importlib.util
import io
import re
import tempfile
import tomllib
from contextlib import redirect_stdout
from pathlib import Path

import autorefine

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
README = ROOT / "README.md"
NARRATIVE = ROOT / "examples" / "narrative.py"


def _load_narrative():
    """Load the example by path (A85.1: the test imports it as a spec)."""
    spec = importlib.util.spec_from_file_location(
        "narrative_v081_under_test", NARRATIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(mod=None):
    """Run `main([])` with stdout captured; return (rc, output)."""
    mod = mod or _load_narrative()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main([])
    return rc, buf.getvalue()


def test_narrative_runs_all_five_beats():
    """A85.1 (SPEC.md 95.3.1): all five beats run end-to-end and the
    verdict is verify-PASS; the stable story facts (baseline below
    majority-class, final 100.0, the mid-run steer event) are printed."""
    rc, out = _run_main()
    assert rc == 0, out
    for marker in (
        "[1/5] plan",
        "[2/5] run",
        "[3/5] effects",
        "[4/5] what-if",
        "[5/5] verify",
        "dry-run: plan only",
        "STEER: pin activation -> relu (applied after experiment 1)",
        "steering effects (1 event in this run)",
        "verdict   :",
        "PASS (",  # verify: "PASS (7/7 checks passed)"
        "checks passed",
        "Verdict: all five beats ran; verify says PASS",
    ):
        assert marker in out, f"missing {marker!r}\n{out}"
    m = re.search(r"final best: ([0-9.]+) \(baseline was ([0-9.]+)", out)
    assert m, out
    final, baseline = float(m.group(1)), float(m.group(2))
    assert final > baseline, (final, baseline, out)
    assert final > 50.0 > baseline  # the "classically hard" story (95.1.1)


def test_narrative_fails_loud_on_error_rc():
    """A85.1 (SPEC.md 95.3.1): a beat with an error CLI rc makes
    `main()` return 1 (not an exception)."""
    mod = _load_narrative()

    def fake_cli(args):  # noqa: ANN001
        print("boom")
        return 1

    mod.cli_main = fake_cli
    rc, out = _run_main(mod)
    assert rc == 1, out
    assert "Verdict: FAILED" in out, out


def test_story_facts_deterministic():
    """A85.2 (SPEC.md 95.3.2): two fresh runs produce the same baseline
    score, the same final best score, and the same steer event."""
    rc1, out1 = _run_main()
    rc2, out2 = _run_main()
    assert rc1 == rc2 == 0
    m1 = re.search(r"final best: ([0-9.]+) \(baseline was ([0-9.]+)", out1)
    m2 = re.search(r"final best: ([0-9.]+) \(baseline was ([0-9.]+)", out2)
    assert m1 and m2, (out1, out2)
    assert m1.groups() == m2.groups(), (m1.groups(), m2.groups())
    steer = "STEER: pin activation -> relu (applied after experiment 1)"
    assert steer in out1 and steer in out2
    assert "PASS (" in out1 and "PASS (" in out2


def test_hygiene_leaves_no_artifacts():
    """A85.3 (SPEC.md 95.3.3): the default run removes its temp dir
    (no `autorefine_narrative_*` left); the script parses and cites A85."""
    tmp_root = Path(tempfile.gettempdir())
    before = set(tmp_root.glob("autorefine_narrative_*"))
    rc, _ = _run_main()
    assert rc == 0
    leftover = set(tmp_root.glob("autorefine_narrative_*")) - before
    assert not leftover, sorted(str(p) for p in leftover)
    src = NARRATIVE.read_text(encoding="utf-8")
    ast.parse(src)
    assert "A85" in src


def test_version_ceremony():
    """A85.4 (SPEC.md 95.3.4): 0.82.0 in both version sources, the
    SPEC 95.3 heading + the M84 index row, and the README pointer."""
    assert autorefine.__version__ == "0.82.0"
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == "0.82.0"
    spec = SPEC.read_text(encoding="utf-8")
    assert "### 95.3 Acceptance (A85)" in spec
    assert re.search(r"^\s*\|\s*M84\s*\|\s*v0\.81\s*\|\s*95\s*\|\s*A85\s*\|",
                     spec, flags=re.MULTILINE)
    assert "**M84**" in spec
    assert "examples/narrative.py" in README.read_text(encoding="utf-8")
