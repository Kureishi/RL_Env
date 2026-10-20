"""AutoRefine, in one screen: the environment's identity, walked through.

A85 (SPEC.md 95). One executable narrative that shows, in order, what
this environment is and does — the same commands a user would type, on
one dataset, in one screen of terminal output:

  1. plan     ``fit --dry-run`` — data, task, split, budget, data health.
              Zero training, zero artifacts.
  2. run      a tiny live search loop over a "classically hard" dataset
              (quadrant XOR: the baseline lands *below* the majority
              class), steered mid-run (``activation`` pinned to ``relu``)
  3. effects  ``report --steer`` — what the steering changed, measured
              against the run's own best-score curve
  4. what-if  ``report --what-if`` — re-gate the logged history under a
              new bar, with zero retraining
  5. verify   ``verify --run`` — independently re-derive every reported
              number from the raw artifacts

No new dependencies, no new commands: the story is the existing
surface, told in sequence.

Usage:
    python examples/narrative.py            # runs in a temp dir, cleaned up
    python examples/narrative.py --keep     # keep the data + run dir
Exit code 0 when all five beats ran and ``verify`` says PASS.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Repo-root import fallback (SPEC 95.2): the example must run from the
# repo root *or* be importable by path without an install.
try:
    import autorefine  # noqa: F401
except ImportError:  # pragma: no cover - depends on how it is launched
    _REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO / "src"))
    import autorefine  # noqa: F401

from autorefine.cli import main as cli_main  # noqa: E402
from autorefine.dashboard import DashboardRunner  # noqa: E402

# 95.1.1: the "classically hard" dataset — quadrant XOR over two unit
# features. Deterministic (no RNG), 60 rows, perfectly balanced, and
# hard for linear-ish models: the baseline lands below the majority
# class, which the narrative points out by name.
CSV_HEADER = "a,b,churn"


def write_dataset(directory: Path) -> Path:
    lines = [CSV_HEADER]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1 if (a > 0.4) ^ (b > 0.4) else 0
        lines.append(f"{a:.2f},{b:.2f},{churn}")
    csv_path = directory / "quadrant_xor.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _run_cli(args: list[str]) -> tuple[int, str]:
    """95.1.2: run a CLI command and capture its stdout for framing."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(args)
    return rc, buf.getvalue()


def _from_marker(text: str, marker: str) -> tuple[bool, str]:
    """95.1.2: keep from the marker line onward; (False, whole text)
    when the marker is missing (the output shape changed — fail the
    beat, don't raise)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            return True, "\n".join(lines[i:]).rstrip()
    print("  [narrative] the what-if block was not found in the CLI "
          "output — the command's shape changed; update the narrative.")
    return False, text.strip()


def _beat(number: int, title: str, command: str) -> None:
    print(f"\n[{number}/5] {title}   ($ {command})")


def beat_plan(csv: Path) -> bool:
    """95.1.3 beat 1: ``fit --dry-run`` — the plan, zero training."""
    _beat(1, "plan — what would this run do?",
          'fit --data quadrant_xor.csv --dry-run --target 95')
    rc, out = _run_cli(["fit", "--data", str(csv), "--dry-run",
                        "--target", "95"])
    print(out.strip())
    return rc == 0


def beat_run(csv: Path, runs_dir: Path) -> tuple[bool, Path]:
    """95.1.3 beat 2: a tiny steered search loop on the hard dataset."""
    _beat(2, "run — a small autonomous search, steered mid-run",
          "4-experiment loop (DashboardRunner), steering pin activation=relu")
    runner = DashboardRunner(str(csv), runs_dir=str(runs_dir),
                             experiments=4, target=50.0, seed=7)
    runner.start()

    step = 1
    first = runner.next()
    first_note = ("ACCEPTED" if first.get("accepted")
                  else f"rejected ({first.get('reason')})")
    print(f"  experiment 1: mutation {first.get('mutation')} -> "
          f"holdout {first.get('candidate_score'):.1f}  [{first_note}]")

    events = runner.steer(pin=("activation", "relu"))
    ev = events[0]
    print(f"  STEER: pin {ev['field']} -> {ev['after']} "
          f"(applied after experiment {ev['at_experiment']})")

    while not runner.done:
        step += 1
        u = runner.next()
        note = ("ACCEPTED" if u.get("accepted")
                else f"rejected ({u.get('reason')})")
        print(f"  experiment {step}: mutation {u.get('mutation')} -> "
              f"holdout {u.get('candidate_score'):.1f}  [{note}]")

    runner.finish()
    run_dir = Path(runner.env.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"  final best: {summary['final_best_score']:.1f} "
          f"(baseline was {summary['baseline_score']:.1f} — "
          f"below the 50.0 a majority-class guess gets; "
          f"finished: {summary['finished_reason']})")
    return summary["final_best_score"] > summary["baseline_score"], run_dir


def beat_steer(run_dir: Path) -> bool:
    """95.1.3 beat 3: ``report --steer`` — the measured effect."""
    _beat(3, "effects — what did the steering change?",
          "report --run RUN --steer")
    rc, out = _run_cli(["report", "--run", str(run_dir), "--steer"])
    print(out.strip())
    return rc == 0


def beat_whatif(run_dir: Path) -> bool:
    """95.1.3 beat 4: ``report --what-if`` — counterfactual re-gating.

    95.1.4: the command prints the run summary first; the narrative
    keeps only the what-if block (the story is the re-gate, not the
    recap). rc 0 = PASS, rc 2 = MISS — both are "the gate answered";
    only an error rc fails the beat.
    """
    _beat(4, "what-if — re-gate the logged history at a higher bar",
          "report --run RUN --what-if 'score>=97'")
    rc, out = _run_cli(["report", "--run", str(run_dir),
                        "--what-if", "score>=97"])
    found, block = _from_marker(out, "what-if")
    print(block)
    if not found or rc not in (0, 2):
        return False
    if rc == 0:
        print("  (verdict: PASS — the run has candidates at the bar)")
    else:
        print("  (verdict: MISS — honest: the bar is above this run)")
    return True


def beat_verify(run_dir: Path) -> bool:
    """95.1.3 beat 5: ``verify --run`` — independent re-derivation."""
    _beat(5, "verify — re-derive every reported number",
          "verify --run RUN")
    rc, out = _run_cli(["verify", "--run", str(run_dir)])
    print(out.strip())
    return rc == 0


def main(argv: list[str] | None = None) -> int:
    # cp1252 consoles: never crash on an unencodable glyph (95.2).
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(
        prog="narrative.py",
        description="AutoRefine's identity, walked through in one screen.")
    parser.add_argument("--keep", action="store_true",
                        help="keep the generated data and run directory")
    args = parser.parse_args(argv)

    print("AutoRefine, in one screen — plan, run, steer, what-if, verify")
    print("data: a 'classically hard' churn CSV (quadrant XOR, 60 rows)")

    workdir = Path(tempfile.mkdtemp(prefix="autorefine_narrative_"))
    ok = True
    try:
        csv = write_dataset(workdir)
        ok &= beat_plan(csv)
        ran, run_dir = beat_run(csv, workdir / "runs")
        ok &= ran and beat_steer(run_dir)
        ok &= beat_whatif(run_dir)
        ok &= beat_verify(run_dir)
    finally:
        if args.keep:
            print(f"\n(--keep) artifacts retained at: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    if ok:
        print("\nVerdict: all five beats ran; verify says PASS — the loop's")
        print("story is plan -> run+steer -> measured effects -> what-if -> verify.")
        return 0
    print("\nVerdict: FAILED — a beat did not run clean; see output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
