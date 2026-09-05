"""Run artifacts: JSONL experiment log + best-spec/model/summary (SPEC.md 9).

Also the experiment-log kind registry (SPEC.md 35.1, v0.21): the `kind`
of an `experiments.jsonl` row is one of LOG_KINDS, and the emitter
(improver/meta_env.py) plus every consumer (plotting.py, cli.py,
dashboard.py) route through these constants — the values are
byte-identical to the historical literals (A25).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# SPEC.md 35.1 (C4, A25): the log kind registry — one set, five kinds.
# Values are the byte-identical historical literals (existing logs,
# pins, and the A22 screen-row tests are untouched; 35.1).
KIND_BASELINE = "baseline"
KIND_EXPERIMENT = "experiment"
KIND_SCREEN = "screen"
KIND_CURRICULUM = "curriculum"
KIND_INVALID_SPEC = "invalid_spec"
LOG_KINDS = frozenset({
    KIND_BASELINE,
    KIND_EXPERIMENT,
    KIND_SCREEN,
    KIND_CURRICULUM,
    KIND_INVALID_SPEC,
})


class RunMemory:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._exp_path = self.run_dir / "experiments.jsonl"

    def log(self, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("ts", time.time())
        with self._exp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def save_best(self, spec_dict: dict, model) -> None:
        (self.run_dir / "best_spec.json").write_text(
            json.dumps(spec_dict, indent=2, sort_keys=True), encoding="utf-8"
        )
        model.save(str(self.run_dir / "best_model.npz"))

    def save_summary(self, summary: dict) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    # --- readers (CLI report/eval) -----------------------------------------
    def load_summary(self) -> dict:
        return json.loads((self.run_dir / "summary.json").read_text(encoding="utf-8"))

    def load_experiments(self) -> list[dict]:
        lines = self._exp_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
