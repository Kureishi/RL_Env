"""App language contract (coherency, A25): the Streamlit app and its
renderers must not leak developer-internal SPEC references into
user-facing strings.

A25 (SPEC.md 35.2) makes the app's user-facing text the product surface —
captions, subheaders, expander titles, error messages, and rendered SVG
text. Developer-internal references (`SPEC.md N.N`, `SPEC_FIELDS`, `§`,
`v0.N` version tags, and bare section numbers like `(33.2)`) belong in
docstrings and comments (the developer layer), never in what a user
reads. This is a read-only AST scan of the 11 app/render modules. It
deliberately excludes `cli.py`, whose SPEC tags are a pinned CLI surface
(`test_ergonomics_v033.py`).

Two checks, both over non-docstring string constants (including f-string
literal parts, which `ast.walk` yields as their own Constants):

  1. hard ban — no user-facing string may contain `SPEC.md`,
     `SPEC_FIELDS`, `§`, or a `v0.N` tag;
  2. bare section refs — no user-facing string may carry a bare `N.N`
     section number, except a small allowlist (a CLI example that
     legitimately contains `95.0`, and the overfit-tolerance axis label
     that carries the literal `0.05`).

Docstrings are excluded from both checks: the developer layer keeps its
SPEC references by design.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "autorefine"

# SPEC.md 35.2 / A25: the app + render modules. `cli.py` is intentionally
# excluded — its SPEC tags are a pinned CLI surface (test_ergonomics_v033).
APP_MODULES = (
    "dashboard_app", "dashboard", "onboarding", "narrate", "quickstart",
    "steering", "advanced", "sharing", "live", "research", "plotting",
    "rl_dashboard",
    # 69.4 (v0.55, A59): the procedural workflow core — user-facing strings
    "workflow",
)

# A25 (SPEC.md 35.2): the developer-internal references banned from
# user-facing strings. `SPEC_FIELD_NAMES` (a valid import) does NOT match
# `SPEC_FIELDS`; `__version__` f-string placeholders do NOT match `v0.N`.
BANNED = re.compile(r"SPEC\.md|SPEC_FIELDS|\u00a7|v0\.\d+")

# A25 (SPEC.md 35.2): a bare `N.N` (or `N.N.N`) section reference. The
# lookarounds skip SVG attribute values (`="1.5"`), SVG text (`>0.5</`),
# and format specs (`8.2f`).
SECTION = re.compile(r"(?<![=>\".0-9])\d{1,2}\.\d{1,2}(?:\.\d{1,2})?(?![\d.%f])")

# A25 (SPEC.md 35.2): the only non-docstring string constants allowed to
# carry a bare `N.N` — a CLI example (`--target 95.0`) and the
# overfit-tolerance axis label (the literal `0.05`).
SECTION_ALLOWLIST = (
    "--target 95.0",
    "0.05 * score (overfit tolerance)",
)

_DEFS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_ids(tree) -> set[int]:
    """`id()` of every docstring Constant, so the scans can skip them."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DEFS) or not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            skip.add(id(first.value))
    return skip


def _string_constants(tree, skip):
    """Non-docstring string Constants of `tree` (f-string parts included)."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip):
            yield node


def _parse(name):
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))


def test_app_modules_present():
    """A25 (SPEC.md 35.2): every scanned module exists — a rename fails
    loudly instead of passing silently with zero strings scanned."""
    for name in APP_MODULES:
        assert (SRC / f"{name}.py").is_file(), f"app module missing: {name}.py"


def test_no_banned_spec_references_in_app_strings():
    """A25 (SPEC.md 35.2): no user-facing (non-docstring) string in the
    app may carry a developer-internal SPEC reference."""
    for name in APP_MODULES:
        tree = _parse(name)
        skip = _docstring_ids(tree)
        for node in _string_constants(tree, skip):
            m = BANNED.search(node.value)
            assert not m, (f"{name}.py: user-facing string leaks a SPEC "
                           f"reference ({m.group(0)!r}): {node.value!r}")


def test_no_bare_section_refs_in_app_strings():
    """A25 (SPEC.md 35.2): no user-facing (non-docstring) string in the
    app may carry a bare section number, except the documented allowlist."""
    for name in APP_MODULES:
        tree = _parse(name)
        skip = _docstring_ids(tree)
        for node in _string_constants(tree, skip):
            if any(tok in node.value for tok in SECTION_ALLOWLIST):
                continue
            m = SECTION.search(node.value)
            assert not m, (f"{name}.py: user-facing string carries a bare "
                           f"section ref ({m.group(0)!r}): {node.value!r}")
