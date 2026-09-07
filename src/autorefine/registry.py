"""SPEC.md 38 (v0.24, T1/T2): the run registry — one appended entry per
finished run, so runs over time become comparable data.

House rules: stdlib only (no numpy/streamlit); the module is
import-cycle-free (it reads the env structurally, never imports it);
finish must never crash because of registry state (38.1.4).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

# SPEC.md 38.1.1: the registry lives in the runs dir, next to the run dirs
REGISTRY_FILENAME = "registry.json"

# SPEC.md 38.1.2: the exact entry contract — 13 keys
ENTRY_KEYS = (
    "run_id", "task", "seed", "policy", "final_score", "target",
    "met_target", "finished_reason", "experiments_run", "wall_seconds",
    "config_fp", "parent_run", "timestamp",
)


def config_fingerprint(run_config_dict: dict) -> str:
    """SPEC.md 38.1.3: first 12 hex chars of the SHA-256 of the
    canonical (sort_keys) JSON of a `run_config` dict (37.1). Same
    config → same fp; any changed field → different fp."""
    canon = json.dumps(run_config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _registry_path(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / REGISTRY_FILENAME


def load_registry(runs_dir: str | Path) -> list[dict]:
    """SPEC.md 38.1.1/38.1.4: read the registry; a missing file is `[]`;
    a corrupt one (invalid JSON or not a list) is moved aside to
    `registry.json.corrupt-<stamp>` and a fresh array is started — the
    old bytes are preserved for inspection."""
    path = _registry_path(runs_dir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("registry is not a list")
        return data
    except (ValueError, json.JSONDecodeError):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{REGISTRY_FILENAME}.corrupt-{stamp}")
        k = 1
        while backup.exists():  # never clobber an existing backup
            k += 1
            backup = path.with_name(f"{REGISTRY_FILENAME}.corrupt-{stamp}-{k}")
        path.rename(backup)
        return []


def append_entry(runs_dir: str | Path, entry: dict) -> Path:
    """SPEC.md 38.1.1: append one entry (recovering a corrupt registry
    first, 38.1.4) and write it back; returns the registry path."""
    path = _registry_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_registry(runs_dir)
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def entry_from_env(env: object, finished_reason: str | None,
                   wall_seconds: float | None = None) -> dict:
    """SPEC.md 38.1.2: build the 13-key registry entry from a finished
    `AutoRefineEnv` (read structurally — no import, no cycle). The caller
    passes the *same* `wall_seconds` it wrote to `summary.json` (the wall
    clock keeps advancing between the two, so computing it twice could
    round differently)."""
    target = env.target
    met = None if target is None else bool(env.best_score >= target)
    if wall_seconds is None:
        wall_seconds = round(
            env.bm.budget.max_wall_seconds - env.bm.wall_seconds_left, 3)
    cfg = env.run_config
    fp = config_fingerprint(cfg.to_dict()) if cfg is not None else None
    return {
        "run_id": Path(env.run_dir).name,
        "task": env.task_name,
        "seed": int(env.seed),
        "policy": env.policy,
        "final_score": float(env.best_score),
        "target": None if target is None else float(target),
        "met_target": met,
        "finished_reason": finished_reason,
        "experiments_run": int(env.bm.used_experiments),
        "wall_seconds": wall_seconds,
        "config_fp": fp,
        "parent_run": env.parent_run,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def gate_label(met_target: bool | None) -> str:
    """SPEC.md 38.3.1: the table's gate column — PASS/MISS/—."""
    if met_target is None:
        return "—"
    return "PASS" if met_target else "MISS"


def format_table(entries: list[dict]) -> str:
    """SPEC.md 38.3.1: the `report --history` table (newest last,
    append order)."""
    header = (f"{'run_id':38s} {'task':10s} {'seed':>5s} {'policy':8s} "
              f"{'score':>8s} {'target':>7s} {'gate':5s} {'wall_s':>8s} "
              f"{'exps':>5s}  parent")
    lines = [header, "-" * len(header)]
    for e in entries:
        lines.append(
            f"{str(e.get('run_id', '?')):38s} {str(e.get('task', '?')):10s} "
            f"{str(e.get('seed', '?')):>5s} {str(e.get('policy') or '-'):8s} "
            f"{e.get('final_score', 0):8.2f} "
            f"{('%g' % e['target']) if e.get('target') is not None else '-':>7s} "
            f"{gate_label(e.get('met_target')):5s} "
            f"{e.get('wall_seconds', 0.0):8.2f} {e.get('experiments_run', 0):>5d}  "
            f"{e.get('parent_run') or '-'}"
        )
    return "\n".join(lines)


__all__ = [
    "REGISTRY_FILENAME",
    "ENTRY_KEYS",
    "config_fingerprint",
    "load_registry",
    "append_entry",
    "entry_from_env",
    "gate_label",
    "format_table",
]
