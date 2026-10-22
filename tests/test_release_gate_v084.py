"""SPEC.md 98 (v0.84) — the release gate (A88).

The round before "release" is a checklist, not a hope:

* 98.1/98.2 — the package ships (metadata, license, changelog) and the
  CLI renders where users first touch it (every `--help`, cp1252-safe,
  %-safe);
* 98.3 — the live loop smoke: run → report → verify → export;
* 98.5 — the CLI surface covers the registered task space and the
  model-family space;
* 98.7 — repo hygiene (`.gitattributes` / `.gitignore`).

House rules: stdlib + numpy only; no new dependency; no pinned surface
touched (G2)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import autorefine
from autorefine.cli import build_parser
from autorefine.config import MODEL_FAMILIES
from autorefine.improver.specspace import SPEC_FIELDS
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parent.parent

# the top-level command table, as SPEC 98.2 enumerates it (29 commands +
# the nested `plugins list`; `watch` is excluded from the help-render
# sweep because it is a long-running tailer by design)
TOP_COMMANDS = [
    "run", "report", "watch", "fit", "ask", "go", "feed", "refresh",
    "datasets", "schedule", "ingest", "card", "manual", "dashboard",
    "quickstart", "plugins", "eval", "predict", "adapt", "compare",
    "explain", "doctor", "variance", "backend", "policy-report", "share",
    "export", "dossier", "verify",
]


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return a.choices
    return {}


def _cli(cwd: Path, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "autorefine", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


# --- A88.1 the package ships ---------------------------------------------------

def test_pyproject_metadata_is_complete():
    """A88.1 (SPEC.md 98.2): the wheel metadata a release needs —
    name / version / description / requires-python / dependencies /
    authors / license / readme / classifiers — is all declared."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    proj = py["project"]
    for key in ("name", "version", "description", "requires-python",
                "dependencies", "authors", "license", "readme",
                "classifiers"):
        assert key in proj, f"pyproject [project] is missing {key!r}"
    assert proj["name"] == "autorefine"
    assert any("MIT" in str(c) for c in proj["classifiers"])


def test_license_file_exists():
    """A88.1 (SPEC.md 98.2): a non-empty MIT LICENSE at the repo root."""
    lic = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in lic and "Permission is hereby granted" in lic
    assert len(lic) > 500


def test_version_triple_agrees():
    """A88.1 (SPEC.md 98.2): pyproject == package == README (the 97.1
    invariant, re-checked on the release gate)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Current version: ([\d.]+)\.\*\*", readme)
    assert m, "README lost its Current version line"
    assert py["project"]["version"] == autorefine.__version__ == m.group(1)


# --- A88.2 every help renders --------------------------------------------------

def test_every_command_accepts_help():
    """A88.2 (SPEC.md 98.2): every top-level subcommand accepts
    `--help` with exit 0 — this is the regression pin for the five
    crashing `--help`s (run/fit/ask/manual/variance) and the `ask`
    %-format crash found by the v0.84 audit."""
    parser = build_parser()
    choices = _subparsers(parser)
    for cmd in TOP_COMMANDS:
        assert cmd in choices, f"top-level command missing: {cmd}"
        try:
            parser.parse_args([cmd, "--help"])
        except SystemExit as exc:
            assert exc.code == 0, f"`{cmd} --help` exited {exc.code}"


def test_every_help_string_is_console_safe():
    """A88.2 (SPEC.md 98.2): every action's help string and each
    command's epilog is cp1252-encodable (the 48.5 console rule), and
    every action help string is safe under argparse's `help % params`
    formatting (no bare `%`). Epilogs are exempt from the %-check: argparse
    does not %-format epilog text, so a bare `%` there (e.g. "96%") is safe."""
    parser = build_parser()
    parsers = [parser, *_subparsers(parser).values()]
    parsers += [_subparsers(p)["list"] for p in parsers
                if "list" in _subparsers(p)]  # `plugins list`
    for p in parsers:
        helps = [a.help for a in p._actions if a.help is not None]
        texts = helps + ([p.epilog] if p.epilog else [])
        for t in texts:
            try:
                t.encode("cp1252")
            except UnicodeEncodeError as exc:
                raise AssertionError(
                    f"non-cp1252 U+{ord(exc.object[exc.start]):04X} in help "
                    f"{t[:60]!r}") from None
        for t in helps:  # argparse applies `help % params` to action help only
            try:
                t % {"default": "X", "metavar": "Y"}
            except (KeyError, TypeError, ValueError) as exc:
                raise AssertionError(f"unsafe %-format in help {t[:60]!r}") from exc


# --- A88.3 the loop ships -------------------------------------------------------

def test_run_report_verify_export_smoke(tmp_path):
    """A88.3 (SPEC.md 98.2): in a throwaway dir the whole user path
    works end to end — a live run, the JSON report, the independent
    re-derivation, and the portable model export."""
    rd = tmp_path / "runs"
    r = _cli(tmp_path, "run", "--task", "parity-v1", "--experiments", "2",
             "--max-seconds", "60", "--runs-dir", "runs")
    assert r.returncode == 0, r.stderr
    run_dirs = [d for d in rd.iterdir() if d.is_dir()
                and d.name.startswith("parity-v1")]
    assert run_dirs, "the run left no run dir"
    run = run_dirs[0]

    rep = _cli(tmp_path, "report", "--run", str(run), "--json")
    assert rep.returncode == 0, rep.stderr
    body = json.loads(rep.stdout[rep.stdout.index("{"):])
    assert "final_best_score" in body

    ver = _cli(tmp_path, "verify", "--run", str(run))
    assert ver.returncode == 0, ver.stderr

    out = tmp_path / "model.zip"
    ex = _cli(tmp_path, "export", "--run", str(run), "--out", str(out))
    assert ex.returncode == 0, ex.stderr
    assert out.is_file() and out.stat().st_size > 0


# --- A88.4 the plan ships -------------------------------------------------------

def test_fit_dry_run_prints_the_plan(tmp_path):
    """A88.4 (SPEC.md 98.2): `fit --dry-run` on a generated CSV exits 0
    and prints the task / dataset / budget / catalog lines (the
    zero-training plan)."""
    csv = tmp_path / "t.csv"
    csv.write_text("a,b,label\n0,0,0\n1,0,1\n0,1,1\n1,1,0\n0,0,0\n"
                   "1,1,0\n0,1,1\n1,0,1\n", encoding="utf-8")
    r = _cli(tmp_path, "fit", "--data", "t.csv", "--dry-run")
    assert r.returncode == 0, r.stderr
    for token in ("task", "dataset", "budget", "catalog"):
        assert token in r.stdout, f"plan missing the {token!r} line"


# --- A88.5 the surface is complete ----------------------------------------------

def test_cli_covers_tasks_families_and_rl():
    """A88.5 (SPEC.md 98.2): every registered task is an accepted
    `--task` choice of `run` and `manual`; every family in
    `config.MODEL_FAMILIES` is in the `model_family` field's space;
    `--policy rl` parses on `run`."""
    parser = build_parser()
    for cmd in ("run", "manual"):
        sub = _subparsers(parser)[cmd]
        act = next(a for a in sub._actions if a.dest == "task")
        assert set(TASKS) <= set(act.choices), \
            f"{cmd} --task misses: {set(TASKS) - set(act.choices)}"
    fam_space = set(SPEC_FIELDS["model_family"].space)
    assert set(MODEL_FAMILIES) <= fam_space
    parser.parse_args(["run", "--policy", "rl"])


# --- A88.6 the history ships ----------------------------------------------------

def test_changelog_covers_every_milestone():
    """A88.6 (SPEC.md 98.2): CHANGELOG.md lists the current version on
    its top row and carries one row per milestone (M4–M86)."""
    cl = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*M\d+\s*\|\s*(v[\d.]+)\s*\|", spec, re.M)
    assert len(rows) >= 80, f"expected ≥80 milestone rows, got {len(rows)}"
    top = next((l for l in cl.splitlines()
                if l.startswith("|") and autorefine.__version__ in l), None)
    assert top, ("the changelog's top row does not list the current "
                 f"version {autorefine.__version__}")
    for v in rows:
        assert v in cl, f"changelog missing milestone version {v}"


# --- A88.7 the repo is clean ----------------------------------------------------

def test_repo_hygiene_files():
    """A88.7 (SPEC.md 98.2): `.gitattributes` pins LF line endings and
    binary types; `.gitignore` covers the generated artifacts."""
    ga = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "text=auto" in ga and "eol=lf" in ga
    for pat in ("*.npz", "*.zip", "*.wav"):
        assert pat in ga, f".gitattributes missing {pat}"
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pat in ("runs/", "__pycache__/", ".pytest_cache/", "dist/",
                "*.egg-info/"):
        assert pat in gi, f".gitignore missing {pat}"
