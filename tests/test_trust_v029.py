"""Trust-before-you-train tests (SPEC.md 43, A33, v0.29).

A (43.1) the data-health preflight — the pure leaf `autorefine.preflight`:
`data_health(task, target=None)` reports the class balance (counts +
shares, majority first) with the headroom note against the run's target
(both ways: headroom left, or the majority class already meets it), a
constant column (`min == max`), a near-constant column (>= 99% one
value, with the share; a 95% column is NOT flagged), per-column missing
counts (an *ignored* column with empties is named as ignored), and
min/max/mean per feature; an mse CSV carries no balance / note; a media
(audio) task degrades to the media shape (item count + class balance
from the subfolders) without failing. `fit --dry-run` on the same CSV
prints the health lines inside the plan with rc 0 and creates no run
dir; a bad label column is still rc 1 (40.1.3 unchanged).

B (43.2) `autorefine doctor` — rc 0 on a healthy install; the five
checks (six lines) present: version, numpy, both extras (the lines are
asserted, not the install state), the smoke-train line reporting a
finite wall under 1 s, and the `result:` line counting ok/warn/FAIL;
`--runs-dir` under a *file* is `FAIL` with rc 1; doctor never imports
the optional extras (the A23 invariant — `find_spec` + dist metadata).

C (43.3) regression — A1–A32 stay green in their own files (no run
behavior, summary key set, artifact, or app-view change; the 40.1
dry-run plan keeps its rc contract); the A25 index table advances
(29 -> 30 acceptance rows; `defined == set(range(1, 35))`); the version
stepped to `0.39.0` in both sources (33.1, advanced again with v0.32,
SPEC.md 45).

House rules: no cross-test imports (the WAV / tone-dir fixture is
duplicated from test_media_tasks.py / test_usecase_v028.py); stdlib +
numpy only.
"""
import math
import re
import tomllib
import wave
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine.cli import main as cli_main
from autorefine.preflight import data_health, format_health
from autorefine.tasks import AudioTask
from autorefine.tasks.csv import CsvTask

REPO = Path(__file__).resolve().parent.parent


# --- fixtures ------------------------------------------------------------------

def _imbalanced_csv(tmp_path: Path) -> Path:
    """A33 (43.1): a 100-row CSV with every preflight stat at once —
    a 90/10 label split, a constant column (`c`, all 7), a
    near-constant one (`nc`, 99% = 1.0), a 95% one (`nz`, NOT flagged),
    and an ignored non-numeric column (`junk`) with 2 empties."""
    rows = ["x1,x2,c,nc,nz,junk,label"]
    for i in range(100):
        label = "1" if i < 90 else "0"
        x1 = i / 10.0
        nc = "1.0" if i < 99 else "9.0"       # 99% one value (near-constant)
        nz = "1.0" if i < 95 else "2.0"       # 95% one value (not flagged)
        junk = "" if i in (5, 50) else ("a" if i < 98 else "b")
        rows.append(f"{x1:g},{-x1:g},7,{nc},{nz},{junk},{label}")
    p = tmp_path / "imbalanced.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def _mse_csv(tmp_path: Path) -> Path:
    """A33 (43.1): a float-labelled CSV — the mse head (no balance)."""
    rows = ["x,y"]
    for i in range(40):
        rows.append(f"{i / 10.0:g},{1.0 + 0.5 * i / 10.0:g}")
    p = tmp_path / "mse.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 0.4,
               amp: float = 0.6, phase: float = 0.0) -> None:
    """16-bit PCM mono WAV (SPEC.md 24.4) — duplicated from
    test_media_tasks.py / test_usecase_v028.py (house rule: no
    cross-test imports)."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = amp * np.sin(2 * math.pi * freq * t + phase)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _tone_dir(tmp_path: Path, n: int = 20, seed: int = 0) -> Path:
    """Two tone classes (220 vs 440 Hz) — duplicated from
    test_media_tasks.py / test_usecase_v028.py."""
    d = tmp_path / "tones"
    rng = np.random.default_rng(seed)
    for name, freq in (("low", 220.0), ("high", 440.0)):
        sub = d / name
        sub.mkdir(parents=True)
        for i in range(n):
            _write_wav(sub / f"c{i:02d}.wav", freq,
                       amp=0.5 + 0.3 * rng.random(),
                       phase=rng.random() * 2 * math.pi)
    return d


# --- SPEC.md 43.1 (A33): the pure `preflight` leaf -----------------------------

def test_data_health_reports_every_stat(tmp_path):
    """A33 (43.1.1): on the 90/10 fixture the leaf reports the balance
    (counts + shares, majority first), the headroom note, the constant
    column, the near-constant one (with the share) — and not the 95%
    one — the ignored column's missing count, and the feature ranges."""
    t = CsvTask(seed=7, path=_imbalanced_csv(tmp_path))
    h = data_health(t, target=95.0)
    assert h["kind"] == "csv" and h["n_rows"] == 100 and h["label"] == "label"
    assert h["balance"] == [
        {"label": "1", "count": 90, "share": 0.9},
        {"label": "0", "count": 10, "share": 0.1}]
    assert h["majority_share"] == 0.9
    assert "5.0 points of headroom below target 95" in h["note"]
    assert h["constant"] == [{"name": "c", "value": 7.0, "share": 1.0}]
    assert h["near_constant"] == [{"name": "nc", "value": 1.0, "share": 0.99}]
    assert h["missing"] == [{"column": "junk", "count": 2, "role": "ignored"}]
    feats = {f["name"]: f for f in h["features"]}
    assert [f["name"] for f in h["features"]] == ["x1", "x2", "c", "nc", "nz"]
    assert feats["x1"]["min"] == 0.0 and feats["x1"]["max"] == 9.9
    assert feats["x2"]["min"] == -9.9 and feats["x2"]["max"] == 0.0


def test_headroom_note_both_ways_and_absent_without_target(tmp_path):
    """A33 (43.1.1): a target at/below the majority score states the
    majority class already meets it; no target -> no note (the line is
    not printed)."""
    p = _imbalanced_csv(tmp_path)
    at_bar = data_health(CsvTask(seed=7, path=p), target=90.0)
    assert "a constant predictor already meets target 90" in at_bar["note"]
    none = data_health(CsvTask(seed=7, path=p))
    assert none["note"] is None


def test_mse_csv_has_no_balance(tmp_path):
    """A33 (43.1.1): an mse CSV carries no balance and no headroom
    note — `format_health` omits both lines."""
    t = CsvTask(seed=7, path=_mse_csv(tmp_path))
    h = data_health(t, target=95.0)
    assert h["head"] == "mse"
    assert h["balance"] is None and h["majority_share"] is None
    assert h["note"] is None
    out = "\n".join(format_health(h))
    assert "balance" not in out and "headroom" not in out


def test_media_task_degrades_to_media_shape(tmp_path):
    """A33 (43.1.1): a media task (no raw rows) degrades to the media
    shape — item count + head + the class balance from the subfolders —
    without failing."""
    d = _tone_dir(tmp_path)
    t = AudioTask(seed=7, path=d)
    h = data_health(t, target=95.0)
    assert h["kind"] == "media"
    assert h["n_rows"] == 40  # 20 items per class
    assert h["balance"] == [
        {"label": "high", "count": 20, "share": 0.5},
        {"label": "low", "count": 20, "share": 0.5}]
    out = "\n".join(format_health(h))
    assert "40 items" in out and "(head softmax)" in out
    assert "high: 20 (50.0%)" in out


def test_format_health_lines(tmp_path):
    """A33 (43.1.2): the plan block — one line per stat, `none` for an
    empty list, in the plan's label style (asserted verbatim)."""
    t = CsvTask(seed=7, path=_imbalanced_csv(tmp_path))
    out = format_health(data_health(t, target=95.0))
    assert out[0] == ("  health     : 100 rows, label 'label', "
                      "5 feature column(s)")
    assert "  balance    : 1: 90 (90.0%) | 0: 10 (10.0%)" in out
    assert "  constant   : c (all values 7)" in out
    assert "  near-const : nc (99.0% of rows = 1)" in out
    assert "nz" not in out  # the 95% column is not flagged
    assert "  missing    : junk (2 empty, ignored)" in out


# --- SPEC.md 43.1 (A33): the `fit --dry-run` wiring ----------------------------

def test_dry_run_prints_health_block_rc0_no_run_dir(tmp_path, capsys):
    """A33 (43.1.2/43.1.3): the health lines print inside the plan, rc
    0, no run dir — bad health is a warning in the plan, not a failure."""
    p = _imbalanced_csv(tmp_path)
    runs = str(tmp_path / "runs")
    rc = cli_main(["fit", "--dry-run", "--data", str(p), "--runs-dir", runs])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run: plan only" in out
    assert "  health     : 100 rows, label 'label'" in out
    assert "1: 90 (90.0%)" in out
    assert "  constant   : c (all values 7)" in out
    assert "  near-const : nc (99.0% of rows = 1)" in out
    assert "  missing    : junk (2 empty, ignored)" in out
    assert "5.0 points of headroom below target 95" in out
    assert not Path(runs).exists()  # 40.1: no run dir / artifacts


def test_dry_run_bad_label_still_rc1(tmp_path, capsys):
    """A33 (43.1.3): the 40.1.3 contract is unchanged — a wrong label
    column is still rc 1 (the health block never masks it)."""
    p = _imbalanced_csv(tmp_path)
    rc = cli_main(["fit", "--dry-run", "--data", str(p),
                   "--label", "NOPE", "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert rc == 1 and "NOPE" in err


# --- SPEC.md 43.2 (A33): `autorefine doctor` ----------------------------------

def test_doctor_rc0_healthy(tmp_path, capsys):
    """A33 (43.2): rc 0 on a healthy install; the five checks (six
    lines) present — the extras lines are asserted, not the install
    state; the smoke line reports a finite wall under 1 s; the
    `result:` line counts ok/warn/FAIL."""
    rc = cli_main(["doctor", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert autorefine.__version__ in out
    assert "numpy " in out
    for extra in ("[image]", "[audio]"):
        assert f"optional extra {extra}" in out
    m = re.search(r"smoke train: parity-v1, 1 experiment, (\d+\.\d+) s", out)
    assert m is not None, out
    assert 0.0 < float(m.group(1)) < 1.0
    assert re.search(r"result: \d+ ok, \d+ warn, 0 FAIL", out)


def test_doctor_unwritable_runs_dir_fail_rc1(tmp_path, capsys):
    """A33 (43.2.2/43.2.3): a `--runs-dir` that cannot be created (under
    a *file*) is `FAIL`, and the command is rc 1."""
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    rc = cli_main(["doctor", "--runs-dir", str(blocker / "runs")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "not writable" in out
    assert re.search(r"result: \d+ ok, \d+ warn, 1 FAIL", out)


def test_doctor_never_imports_the_optional_extras():
    """A33 (43.2.2): the A23 invariant holds for doctor too — the cli
    source never imports PIL/soundfile (presence via `find_spec`, the
    version via dist metadata)."""
    src = (REPO / "src" / "autorefine" / "cli.py").read_text(encoding="utf-8")
    m = re.search(r"^\s*(?:import|from)\s+(?:PIL|soundfile)\b",
                  src, re.MULTILINE)
    assert m is None, f"cli.py imports an optional extra: {m.group(0)!r}"


# --- SPEC.md 33.1 / 43.3 (A33): version -----------------------------------------

def test_version_029_both_sources():
    """A33 (43.3, 33.1): the version stepped to `0.39.0` in both sources
    (advanced again with v0.39 ⇒ `0.39.0`, M42, SPEC.md 53, both together)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.41.0"
