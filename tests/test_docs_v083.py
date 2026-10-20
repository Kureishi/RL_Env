"""v0.83 (SPEC.md 97, A87): documentation agreement and succinctness.

The audit (97.1) found the README, SPEC, CLI, and code had drifted:
stale version line, a five-of-nine family list, a task list missing the
temporal / domain / v0.78 tasks, a stale catalog count, a layout block
missing ~40 modules, a feature table skipping rounds, and ten
subcommands without a row. The fixes (97.2) are now invariants — this
file pins the agreement surfaces so the per-round ceremony cannot
silently drift them again:

- A87.1 — README version line == pyproject.toml == autorefine.__version__
- A87.2 — every SPEC reference in the README resolves to a SPEC section
- A87.3 — every file named in the README project layout exists
- A87.4 — no stale acceptance-range / catalog-count literals
- A87.5 — the four one-home quickstart commands appear in the README
- A87.6 — every MODEL_FAMILIES family and every TASKS task is named
- A87.7 — the SPEC header is not "draft / awaiting approval"
- A87.8 — every top-level CLI subcommand is documented in the README

House rules: stdlib + the package only (no new dependency); no
cross-test imports; pure text test (tiny, no budgets); cites A87.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import autorefine
from autorefine.config import MODEL_FAMILIES
from autorefine.quickstart import quickstart_commands
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
SPEC = (REPO / "SPEC.md").read_text(encoding="utf-8")


def _spec_sections() -> set[str]:
    """Section numbers with a real `##`/`###` header in SPEC.md."""
    return {m.group(1)
            for m in re.finditer(r"^#{2,3}\s+([\d]+(?:\.\d+)*)\.?", SPEC, re.M)}


def _resolves(sec: str) -> bool:
    sec = sec.rstrip(".")
    if sec in _spec_sections():
        return True
    if "." in sec:
        return sec.split(".")[0] in _spec_sections()
    return False


def test_version_agrees_everywhere():
    """A87.1 (SPEC.md 97.3): the README's "Current version" line,
    pyproject.toml, and autorefine.__version__ all state the same
    version — the systemic staleness the audit caught (README said
    0.81.0 while the package was 0.83.0)."""
    m = re.search(r"\*\*Current version: ([\d.]+)\.\*\*", README)
    assert m, "README: the 'Current version' line is missing"
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert m.group(1) == py["project"]["version"] == autorefine.__version__, (
        f"README {m.group(1)} vs pyproject "
        f"{py['project']['version']} vs package {autorefine.__version__}"
    )


def test_readme_spec_refs_resolve():
    """A87.2 (SPEC.md 97.3): every `SPEC §` / `SPEC.md NN` reference in
    the README resolves to an existing SPEC section header (sub-section
    refs resolve to their parent)."""
    refs: set[str] = set()
    for m in re.finditer(r"SPEC(?:\.md)?\s*[§]?\s*([\d]+(?:\.\d+)*)", README):
        refs.add(m.group(1))
    for line in README.splitlines():  # the feature table's SPEC § column
        if line.startswith("|"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) == 4 and cols[3] not in ("SPEC §",):
                for tok in cols[3].split(","):
                    m = re.match(r"^(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?$",
                                 tok.strip())
                    if m:
                        refs.add(m.group(1))
                        if m.group(2):
                            refs.add(m.group(2))
    assert refs, "README: no SPEC references found at all (test broken?)"
    bad = sorted(r for r in refs if not _resolves(r))
    assert not bad, f"README SPEC refs with no section: {bad}"


def test_layout_files_exist():
    """A87.3 (SPEC.md 97.3): every `*.py` file named in the README's
    project-layout block exists in the repository (layout is real, not
    aspirational)."""
    start = README.index("## Project layout")
    block = README[start:README.index("## Extending", start)]
    files = set(re.findall(r"[\w./-]+\.py", block))
    assert files, "layout block: no .py files found (test broken?)"
    missing = []
    for f in sorted(files):
        cands = (REPO / f, REPO / "src" / "autorefine" / f,
                 REPO / "examples" / f)
        if not any(c.is_file() for c in cands):
            missing.append(f)
    assert not missing, f"layout names files that do not exist: {missing}"


def test_no_stale_literals():
    """A87.4 (SPEC.md 97.3): no stale acceptance-range claim (the old
    "A1-A55" / "A1–A55" literals) and no stale catalog count (the old
    "77 catalog actions" / "(77 actions") in the README."""
    m = re.search(r"A1[-–]A?\d+", README)
    assert m is None, f"stale acceptance-range literal: {m.group(0)!r}"
    assert "77 catalog" not in README, "stale catalog count: '77 catalog'"
    assert "(77 actions" not in README, "stale catalog count: '(77 actions'"


def test_quickstart_one_home_in_readme():
    """A87.5 (SPEC.md 97.3): all four one-home quickstart commands
    appear in the README (65.1.2: the README Quickstart invocations are
    one home) — both the `autorefine …` and the `python -m autorefine …`
    spellings count."""
    normalized = README.replace("python -m autorefine ", "autorefine ")
    missing = [c for c in quickstart_commands() if c not in normalized]
    assert not missing, f"quickstart one-home commands absent from README: {missing}"


def test_model_space_lists_complete():
    """A87.6 (SPEC.md 97.3): every family in config.MODEL_FAMILIES is
    named in the README's model-families bullet, and every task
    registered in tasks.TASKS is named in the README (the audit found
    five-of-nine families and a task list missing six tasks)."""
    line = next(l for l in README.splitlines()
                if l.startswith("- **Model families**"))
    i = README.index(line)
    bullet = README[i:README.index("\n- **", i + 1)]
    missing_fam = [f for f in MODEL_FAMILIES if f not in bullet]
    assert not missing_fam, f"model-families bullet misses: {missing_fam}"
    missing_tasks = [t for t in sorted(TASKS) if t not in README]
    assert not missing_tasks, f"README misses registered tasks: {missing_tasks}"


def test_spec_header_not_draft():
    """A87.7 (SPEC.md 97.3): the SPEC header no longer claims the
    document is a draft awaiting approval — it is a living, approved
    spec (the audit's last stale claim)."""
    header = "\n".join(SPEC.splitlines()[:10])
    assert "draft for review" not in header
    assert "Awaiting approval" not in header


def test_cli_commands_documented():
    """A87.8 (SPEC.md 97.3): every top-level CLI subcommand (parsed
    from cli.py's `sub.add_parser` calls — the nested `plugins list`
    is excluded) is named in the README; the audit found ten without a
    row."""
    cli = (REPO / "src" / "autorefine" / "cli.py").read_text(encoding="utf-8")
    cmds = {m.group(2) for m in
            re.finditer(r"(\w+)\s*=\s*sub\.add_parser\(\s*[\"']([\w-]+)[\"']",
                        cli)}
    assert len(cmds) >= 20, f"expected the full CLI, parsed {len(cmds)}"
    missing = sorted(c for c in cmds if not re.search(rf"\b{re.escape(c)}\b",
                                                      README))
    assert not missing, f"CLI subcommands undocumented in README: {missing}"
