"""v0.33 — ergonomics: --config on any command, help polish, --quiet,
autorefine share.

SPEC.md 47, A37, M36.

47.1 `--config FILE` — every subcommand accepts a JSON config: a plain
kebab-case flag object, or a canonical run_config.json (schema
`autorefine.run_config/1`) translated via RunConfig.from_dict +
runconfig_to_flags (47.1.4). Conversion reuses each parser action's own
type/choices/store_true machinery (47.1.3); precedence is explicit CLI
flags > config file > defaults, `--flag=value` included (47.1.5);
unknown keys, bad JSON, and bad values are rc 1 before the command runs.

47.2 help polish — top-level `--version` (47.2.1), per-subcommand
`examples:` epilog blocks rendered verbatim (47.2.2), and the
exit-code table in the top-level --help (47.2.3).

47.3 `--quiet` on run/fit — the per-experiment lines, the run-dir /
baseline echoes, and fit's diagnostics block are suppressed; the
`=== summary ===` block, fit's gate verdict, and the rc semantics are
kept (47.3.2).

47.4 `autorefine share` — one deterministic zip: report.html (always
regenerated in memory, 47.4.2) + summary.json + run_config.json +
best_spec.json + flat SVGs; byte-equal on re-share (47.4.3); a missing
run dir is rc 1 (47.4.4).

Regression — A1–A36 stay green (the rest of the suite); the A25 index
advances (defined == set(range(1, 38))); the version stepped to
`0.33.0` in both sources (33.1).

House rules: no cross-test imports (fixtures duplicated per file);
stdlib + numpy only; every test cites A37 + its SPEC §.
"""
import csv as _csv
import json
import re
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from autorefine.cli import build_parser, main as cli_main

import autorefine

REPO = Path(__file__).resolve().parent.parent


# --- fixtures (duplicated per file, no cross-test imports) --------------------

def _cls_csv(base: Path, name: str = "cls.csv") -> Path:
    """60 rows (30 per class, exact 50/50), 2 features, linearly
    separable: label 1 iff x1 > 0 (the A36 battery fixture) — an MLP at
    any budget fits it, so the `fit` gate PASSes at target 70."""
    rnd = np.random.default_rng(0)
    p = base / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["x1", "x2", "label"])
        for i in range(60):
            y = 1 if i < 30 else 0
            x1 = float(rnd.uniform(0.01, 1.0)) if y \
                else float(rnd.uniform(-1.0, -0.01))
            x2 = float(rnd.uniform(-1.0, 1.0))
            w.writerow([f"{x1:.6f}", f"{x2:.6f}", y])
    return p


def _parity_args(runs: str) -> list:
    """The tiny parity budget (experiments 2 / 60s / 10s) as an explicit
    argv — the config-equivalence reference (SPEC.md 47.1)."""
    return ["run", "--task", "parity-v1", "--seed", "7", "--experiments",
            "2", "--max-seconds", "60", "--max-train-seconds", "10",
            "--runs-dir", runs]


def _best_spec(runs_dir: Path) -> dict:
    for d in Path(runs_dir).iterdir():
        if d.is_dir():
            return json.loads((d / "best_spec.json").read_text(encoding="utf-8"))
    raise AssertionError(f"no run dir with best_spec.json in {runs_dir}")


def _first_run_dir(runs_dir: Path) -> Path:
    for d in Path(runs_dir).iterdir():
        if d.is_dir():
            return d
    raise AssertionError(f"no run dir in {runs_dir}")


# --- 47.1 `--config` -----------------------------------------------------------

def test_config_plain_run_reproduces_argv(tmp_path, capsys):
    """A37 (SPEC.md 47.1.2/47.1.3): a plain kebab-case JSON config on
    `run` (parity, the tiny 2/60/10 budget) reproduces the best_spec.json
    of the equivalent explicit argv — the config is plumbing, not a new
    driver."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "task": "parity-v1", "seed": 7, "experiments": 2,
        "max-seconds": 60, "max-train-seconds": 10}), encoding="utf-8")
    runs_cfg = tmp_path / "runs_cfg"
    rc = cli_main(["run", "--config", str(cfg),
                   "--runs-dir", str(runs_cfg)])
    assert rc == 0, capsys.readouterr().out
    spec_cfg = _best_spec(runs_cfg)
    runs_plain = tmp_path / "runs_plain"
    rc = cli_main(_parity_args(str(runs_plain)))
    assert rc == 0
    assert spec_cfg == _best_spec(runs_plain)


def test_config_plain_fit(tmp_path):
    """A37 (SPEC.md 47.1.2): `fit` with a plain config
    {"data": csv, "target": 70} runs the full gated loop (rc 0 PASS on
    the separable fixture) — the flag file is the user's script surface."""
    p = _cls_csv(tmp_path)
    cfg = tmp_path / "fit.json"
    cfg.write_text(json.dumps({"data": str(p), "target": 70}),
                   encoding="utf-8")
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--config", str(cfg), "--experiments", "4",
                   "--policy", "bandit", "--seed", "7",
                   "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    assert (runs / "registry.json").is_file()


def test_config_canonical_fit_rerun(tmp_path):
    """A37 (SPEC.md 47.1.4): a canonical run_config.json read from a
    finished fit run dir is accepted by `fit --config` (the run dir is
    directly re-runnable) and re-runs to the same gate verdict."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--target", "70",
                   "--experiments", "4", "--policy", "bandit", "--seed",
                   "7", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    rdcfg = _first_run_dir(runs) / "run_config.json"
    assert rdcfg.is_file()
    assert json.loads(rdcfg.read_text(encoding="utf-8"))["schema"] \
        == "autorefine.run_config/1"
    runs2 = tmp_path / "runs2"
    rc = cli_main(["fit", "--config", str(rdcfg), "--runs-dir", str(runs2),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    assert _first_run_dir(runs2).name.startswith("csv-seed7-")


def test_config_cli_seed_wins(tmp_path):
    """A37 (SPEC.md 47.1.5): an explicit --seed — both the `--seed N`
    and `--seed=N` forms — overrides the config's seed (the run dir
    name carries the CLI seed, not the file's)."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"task": "parity-v1", "seed": 7,
                               "experiments": 2, "max-seconds": 60,
                               "max-train-seconds": 10}), encoding="utf-8")
    runs1 = tmp_path / "r1"
    rc = cli_main(["run", "--config", str(cfg), "--runs-dir", str(runs1),
                   "--seed", "99", "--quiet"])
    assert rc == 0
    assert "seed99" in _first_run_dir(runs1).name
    runs2 = tmp_path / "r2"
    rc = cli_main(["run", "--config", str(cfg), "--runs-dir", str(runs2),
                   "--seed=55", "--quiet"])
    assert rc == 0
    assert "seed55" in _first_run_dir(runs2).name


def test_config_store_true_from_json(tmp_path, capsys):
    """A37 (SPEC.md 47.1.3): a store_true flag takes a JSON boolean —
    {"quiet": true} behaves exactly like the --quiet flag (no per-
    experiment lines, the summary block kept)."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"task": "parity-v1", "seed": 7,
                               "experiments": 2, "quiet": True,
                               "max-seconds": 60, "max-train-seconds": 10}),
                   encoding="utf-8")
    rc = cli_main(["run", "--config", str(cfg),
                   "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "exp " not in out
    assert "=== summary ===" in out


def test_config_bad_bool_store_true(tmp_path, capsys):
    """A37 (SPEC.md 47.1.3): a non-boolean for a store_true flag is
    rc 1 citing the flag (no silent coercion, no command run)."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"task": "parity-v1", "quiet": "yes"}),
                   encoding="utf-8")
    rc = cli_main(["run", "--config", str(cfg),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "--quiet" in err


def test_config_unknown_key(tmp_path, capsys):
    """A37 (SPEC.md 47.1.4): a flag the subcommand does not own is
    rc 1, citing the key and listing the command's valid flags (a
    canonical recipe pairs with the command that owns its flags)."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"task": "parity-v1", "bogus-flag": 3}),
                   encoding="utf-8")
    rc = cli_main(["run", "--config", str(cfg),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "bogus-flag" in err and "47.1.4" in err


def test_config_bad_json(tmp_path, capsys):
    """A37 (SPEC.md 47.1.5): an invalid JSON file is rc 1 with a stderr
    message, before any training."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{not json", encoding="utf-8")
    rc = cli_main(["run", "--config", str(cfg),
                   "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "JSON" in err


# --- 47.2 help polish ----------------------------------------------------------

def test_version_flag(capsys):
    """A37 (SPEC.md 47.2.1): `autorefine --version` is rc 0 printing
    `autorefine 0.33.0` from the 33.1 single version source."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = cli_main(["--version"])
        except SystemExit as exc:
            rc = exc.code
    assert rc == 0
    assert buf.getvalue().strip() == f"autorefine {autorefine.__version__}"
    assert buf.getvalue().strip() == "autorefine 0.33.0"


def test_help_exit_table_and_examples():
    """A37 (SPEC.md 47.2.2/47.2.3): the top-level --help ends with the
    exit-code table, and every subcommand's --help renders its examples:
    block verbatim (RawDescriptionHelpFormatter per subparser)."""
    p = build_parser()
    top = p.format_help()
    assert "exit codes (SPEC.md 47.2.3)" in top
    for code in ("130", "gate MISS"):
        assert code in top
    sub = next(a for a in p._actions
               if getattr(a, "dest", None) == "cmd")
    cases = {
        "run": "autorefine run --task cartpole-v1",
        "fit": "autorefine fit --data sales.csv",
        "variance": "autorefine variance --data sales.csv",
        "predict": "autorefine predict --run runs/",
        "share": "autorefine share --run runs/",
        "doctor": "autorefine doctor",
    }
    for name, frag in cases.items():
        h = sub.choices[name].format_help()
        # the epilog (examples block) is the last section of the help: every
        # epilog here ends with a bare `autorefine ...` example command, so
        # the final non-blank help line must be one (SPEC.md 47.2.2).
        last = [ln.strip() for ln in h.splitlines() if ln.strip()][-1]
        assert last.startswith("autorefine"), (name, last)
        assert frag in h, (name, frag)
        assert "examples" in h, name


# --- 47.3 `--quiet` --------------------------------------------------------------

def test_quiet_run(tmp_path, capsys):
    """A37 (SPEC.md 47.3.2): `run --quiet` keeps the === summary ===
    block and rc 0, drops the per-experiment lines and the run-dir /
    baseline echoes."""
    runs = tmp_path / "runs"
    rc = cli_main(["run", "--task", "parity-v1", "--seed", "7",
                   "--experiments", "2", "--max-seconds", "60",
                   "--max-train-seconds", "10",
                   "--runs-dir", str(runs), "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "exp " not in out
    assert "run dir:" not in out
    assert "=== summary ===" in out


def test_quiet_fit_keeps_gate(tmp_path, capsys):
    """A37 (SPEC.md 47.3.2): `fit --quiet` suppresses the per-experiment
    lines and the 28.2 diagnostics block, keeps the gate verdict +
    PASS line, and the rc semantics (0 PASS here)."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--target", "70",
                   "--experiments", "4", "--policy", "bandit", "--seed",
                   "7", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10",
                   "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "exp " not in out
    assert "holdout diagnostics" not in out
    assert "gate    :" in out
    assert "PASS" in out


# --- 47.4 `share` ---------------------------------------------------------------

def test_share_contents(tmp_path, capsys):
    """A37 (SPEC.md 47.4.2): the share zip of a finished fit run
    contains report.html (regenerated — the run dir itself has none),
    summary.json, run_config.json, best_spec.json; the listing +
    `share   : <path>` line are printed."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--target", "70",
                   "--experiments", "4", "--policy", "bandit", "--seed",
                   "7", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    rundir = _first_run_dir(runs)
    assert not (rundir / "report.html").is_file()
    out_zip = tmp_path / "share.zip"
    rc = cli_main(["share", "--run", str(rundir), "--out", str(out_zip)])
    out = capsys.readouterr().out
    assert rc == 0
    names = set(zipfile.ZipFile(out_zip).namelist())
    assert {"report.html", "summary.json", "run_config.json",
            "best_spec.json"} <= names, names
    html = zipfile.ZipFile(out_zip).read("report.html").decode("utf-8")
    assert "<html" in html and len(html) > 1000
    assert "share   :" in out and str(out_zip) in out


def test_share_deterministic(tmp_path, capsys):
    """A37 (SPEC.md 47.4.3): two shares of the same run dir are
    byte-for-byte equal (fixed ZipInfo timestamps, sorted entries,
    ZIP_DEFLATED)."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--target", "70",
                   "--experiments", "4", "--policy", "bandit", "--seed",
                   "7", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    rundir = _first_run_dir(runs)
    z1, z2 = tmp_path / "a.zip", tmp_path / "b.zip"
    assert cli_main(["share", "--run", str(rundir), "--out", str(z1)]) == 0
    assert cli_main(["share", "--run", str(rundir), "--out", str(z2)]) == 0
    assert z1.read_bytes() == z2.read_bytes()


def test_share_default_out_name(tmp_path, capsys):
    """A37 (SPEC.md 47.4.2): without --out the archive lands next to the
    run dir as <run_dir>-share.zip."""
    p = _cls_csv(tmp_path)
    runs = tmp_path / "runs"
    rc = cli_main(["fit", "--data", str(p), "--target", "70",
                   "--experiments", "4", "--policy", "bandit", "--seed",
                   "7", "--runs-dir", str(runs),
                   "--max-seconds", "120", "--max-train-seconds", "10"])
    assert rc == 0
    rundir = _first_run_dir(runs)
    rc = cli_main(["share", "--run", str(rundir)])
    assert rc == 0
    assert (rundir.parent / f"{rundir.name}-share.zip").is_file()


def test_share_missing_dir(tmp_path, capsys):
    """A37 (SPEC.md 47.4.4): a missing run dir is rc 1 with a stderr
    message (no partial zip)."""
    rc = cli_main(["share", "--run", str(tmp_path / "nope"),
                   "--out", str(tmp_path / "x.zip")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "47.4" in err
    assert not (tmp_path / "x.zip").is_file()


# --- 47.5 regression --------------------------------------------------------------

def test_version_round_v033():
    """A37 (SPEC.md 47.5, 33.1): the version stepped to `0.33.0` in
    both sources (v0.33 ⇒ `0.33.0`, M36)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.33.0"


def test_spec_cites_a37_and_round():
    """A37 (SPEC.md 47.5/47.6): SPEC.md defines the A37 acceptance
    block and the M36 index row + milestone — the A25 index machinery
    reads both (defined == set(range(1, 38)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 47.5 Acceptance (A37)" in spec
    assert re.search(r"^\s*\| M36 \| v0\.33\s*\|\s*47\s*\|\s*A37\s*\|",
                     spec, re.MULTILINE)
    assert "**M36**" in spec
