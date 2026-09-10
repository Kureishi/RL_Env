"""v0.21 coherency III — SPEC.md 35, A25:

- 35.1 (C4) log kind registry — `memory.py` (the log module) defines
  the five `KIND_*` constants and `LOG_KINDS`; the values are
  byte-identical to the historical literals (existing logs, pins, and
  the A22 screen-row tests are untouched); the emitter
  (`improver/meta_env.py`) and the consumers (`plotting.py`, `cli.py`,
  `dashboard.py`) route through the registry — source scans fail on a
  raw `"kind"` literal, an unregistered `KIND_` name, or a bare
  kind-literal comparison; and a live run's logged rows all carry
  kinds in `LOG_KINDS`.
- 35.2 (C5) acceptance index — the index table at the top of SPEC.md
  is the single source of truth for the M# → § → A# → test-file map;
  every A# defined in SPEC is cited by ≥ 1 `tests/test_*.py`; every
  `tests/test_*.py` cites ≥ 1 A#; each index row's listed test file
  exists and cites that row's A# (the M0–M3 scaffold rows carry no
  acceptance number and are skipped).
- 35.3 (regression) — full suite green (A1–A24); the version stepped
  to `0.21.0` in both sources (33.1) with the A23 round assertion
  advanced (v0.21 ⇒ `0.21.0`).

House rules: no cross-test imports (fixtures duplicated); the source
scans are read-only (G2-safe); the live run uses a 2-experiment budget
(tiny).
"""
import re
import tomllib
from pathlib import Path

import autorefine
from autorefine import AutoRefineEnv, Budget, SearchPolicy
from autorefine.memory import (
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
    KIND_INVALID_SPEC,
    KIND_SCREEN,
    LOG_KINDS,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "autorefine"
TESTS = REPO / "tests"

# SPEC.md 35.1: the five historical literals — the values the registry
# must keep byte-identical (A25: existing logs and pins stay green).
HISTORICAL_KINDS = ("baseline", "experiment", "screen",
                    "curriculum", "invalid_spec")

# SPEC.md 35.1(3): the consumer files that match row kinds (A25).
CONSUMER_FILES = ("plotting.py", "cli.py", "dashboard.py")

# A kind value written directly in a membership/equality comparison —
# the exact gotcha the v0.18 screen row hit (35.1). Dict keys like
# `"baseline":` and summary-key access like `get("curriculum")` are a
# different namespace and are deliberately allowed.
_BARE_KIND_CMP = re.compile(
    r'(?:in|==|!=)\s*(?:\(\s*)?"(baseline|experiment|screen|curriculum|invalid_spec)"')
# A raw kind value next to the `"kind"` key — the emitter's failure mode.
_RAW_KIND_LITERAL = re.compile(
    r'"kind"\s*:\s*"(baseline|experiment|screen|curriculum|invalid_spec)"')
_KIND_NAME = re.compile(r"\bKIND_[A-Z_]+\b")
_A_NUM = re.compile(r"\bA(\d+)\b")


# --- 35.1 log kind registry (C4, A25) -----------------------------------------

def test_kind_values_byte_identical_and_registry_closed():
    """A25 (SPEC.md 35.1): the five `KIND_*` values are byte-identical
    to the historical literals; `LOG_KINDS` is exactly that set; and
    the registry is closed — the `KIND_*` constants of the memory
    module are exactly the five, so a sixth kind must be added to
    `LOG_KINDS` in the same edit (caught here, not in the report)."""
    assert (KIND_BASELINE, KIND_EXPERIMENT, KIND_SCREEN,
            KIND_CURRICULUM, KIND_INVALID_SPEC) == HISTORICAL_KINDS
    assert LOG_KINDS == frozenset(HISTORICAL_KINDS)
    import autorefine.memory as mem
    const_names = [n for n in dir(mem) if _KIND_NAME.fullmatch(n)]
    assert set(const_names) == {
        "KIND_BASELINE", "KIND_EXPERIMENT", "KIND_SCREEN",
        "KIND_CURRICULUM", "KIND_INVALID_SPEC"}, const_names
    for name in const_names:
        assert getattr(mem, name) in LOG_KINDS, name


def test_emitter_routes_through_registry():
    """A25 (SPEC.md 35.1): the emitter never writes a raw `"kind"`
    literal, and every `KIND_*` name it uses resolves to a registry
    constant — a typo'd name is a suite failure, not a silent unknown
    kind in the log."""
    src = (SRC / "improver" / "meta_env.py").read_text(encoding="utf-8")
    m = _RAW_KIND_LITERAL.search(src)
    assert m is None, f"raw kind literal left in the emitter: {m.group(0)!r}"
    import autorefine.memory as mem
    for name in sorted(set(_KIND_NAME.findall(src))):
        assert hasattr(mem, name), \
            f"emitter uses {name}, which is not in the kind registry"
        assert getattr(mem, name) in LOG_KINDS, name


def test_consumers_route_through_registry():
    """A25 (SPEC.md 35.1): the consumer kind comparisons route through
    the registry — no bare kind-literal membership/equality comparison
    in plotting.py / cli.py / dashboard.py."""
    for rel in CONSUMER_FILES:
        src = (SRC / rel).read_text(encoding="utf-8")
        m = _BARE_KIND_CMP.search(src)
        assert m is None, f"{rel}: bare kind-literal comparison {m.group(0)!r}"


def test_live_run_kinds_in_registry(tmp_path):
    """A25 (SPEC.md 35.1): every kind a live run logs is in
    `LOG_KINDS` — 'every kind emitted by the env is in the registry'
    as a behavioral fact, not only a source scan (2-experiment budget,
    tiny; G2)."""
    env = AutoRefineEnv(seed=7, budget=Budget(2, 300, 30), runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    entries = env.memory.load_experiments()
    assert entries  # at least the baseline row
    for e in entries:
        assert e.get("kind") in LOG_KINDS, e.get("kind")


# --- 35.2 acceptance index (C5, A25) ------------------------------------------

def _test_files() -> list[Path]:
    return sorted(TESTS.glob("test_*.py"))


def test_spec_acceptance_numbers_cited_by_tests():
    """A25 (SPEC.md 35.2): every A# defined in SPEC.md — a
    `- **A#.**` bullet or an `Acceptance (A#)` / `Tests (A#)` heading —
    is cited by ≥ 1 `tests/test_*.py`. An acceptance number without a
    test fails the suite."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    defined = set()
    defined.update(int(n) for n in re.findall(r"\*\*A(\d+)\.\*\*", spec))
    defined.update(int(n) for n in
                   re.findall(r"\(A(\d+)\)\s*$", spec, re.MULTILINE))
    assert defined == set(range(1, 41)), sorted(defined)  # A1..A40 (this round)
    cited: set[int] = set()
    for p in _test_files():
        cited.update(int(n) for n in _A_NUM.findall(p.read_text(encoding="utf-8")))
    missing = sorted(defined - cited)
    assert not missing, f"SPEC-defined A# not cited by any test: {[f'A{n}' for n in missing]}"


def test_every_test_file_cites_an_acceptance_number():
    """A25 (SPEC.md 35.2): every `tests/test_*.py` cites ≥ 1 A# — a
    test file without an acceptance number fails the suite."""
    bare = [p.name for p in _test_files()
            if not _A_NUM.search(p.read_text(encoding="utf-8"))]
    assert not bare, f"test files citing no A#: {bare}"


def test_index_table_rows_resolve():
    """A25 (SPEC.md 35.2): each row of the acceptance index table
    lists test file(s) that exist under `tests/` and cite that row's
    A#; the M0–M3 scaffold rows carry no acceptance number and are
    skipped. (36 acceptance rows: M4–M39.)"""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    lines = spec.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("## 0."))
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    except StopIteration:
        raise AssertionError("the acceptance index section is missing from SPEC.md")
    rows = 0
    for line in lines[start + 1:end]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].startswith("M"):
            continue
        a_nums = {int(n) for n in _A_NUM.findall(cells[3])}
        if not a_nums:
            continue  # scaffold row (M0–M3, no acceptance number)
        rows += 1
        for fname in cells[4].split(","):
            f = REPO / fname.strip()
            assert f.is_file(), f"index row {cells[0]}: {fname} does not exist"
            text = f.read_text(encoding="utf-8")
            missing = [f"A{n}" for n in sorted(a_nums)
                       if not re.search(rf"\bA{n}\b", text)]
            assert not missing, \
                f"index row {cells[0]}: {fname} does not cite {missing}"
    assert rows == 36, f"expected 36 acceptance rows (M4–M39), got {rows}"


# --- 35.3 regression (A25) ------------------------------------------------------

def test_version_round_v021():
    """A25 (SPEC.md 35.3): the version stepped with the round (33.1) —
    the round assertion advanced in place with each round (v0.21 ⇒
    `0.21.0`; now v0.36 ⇒ `0.36.0`, M39, SPEC.md 50)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.36.0"
