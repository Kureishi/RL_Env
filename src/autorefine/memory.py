"""Run artifacts: JSONL experiment log + best-spec/model/summary (SPEC.md 9)."""
from __future__ import annotations

import json
import time
from pathlib import Path


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
