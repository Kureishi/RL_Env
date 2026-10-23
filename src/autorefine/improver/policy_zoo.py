"""Policy zoo: a named, hash-pinned index of saved search policies (A89.15).

A finished search policy (a ``MetaRLPolicy`` file written by
``save_policy``) is a portable artifact — but once several exist, "which
one, and is it the one I think it is?" matters. The zoo answers both:

* ``export_policy`` — add a policy to the zoo under a name: it is written
  with ``save_policy`` and registered in ``zoo.json`` with its ``sha256``
  (the provenance pin), its action dimension, and its task / source
  metadata.
* ``import_policy`` — load a named policy after re-verifying the file's
  ``sha256`` against the index (a tampered / truncated file is a loud
  ``ValueError``), with an optional copy to a target path.
* ``list_zoo`` / ``zoo_info`` — inspect the index.

Deterministic and dependency-free (stdlib json + hashlib): the index holds
NO timestamps (G2 — two identical exports produce a byte-identical
``zoo.json``), and every name must be a safe file stem (fail loud).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .rl_policy import MetaRLPolicy, load_policy, save_policy

INDEX_NAME = "zoo.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"zoo name {name!r} is not a safe file stem "
            f"(start with a letter/digit; letters, digits, '.', '_', '-' only)")


def _index_path(zoo_dir: str | Path) -> Path:
    return Path(zoo_dir) / INDEX_NAME


def _load_index(zoo_dir: str | Path) -> dict[str, dict[str, Any]]:
    p = _index_path(zoo_dir)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"zoo index {p} must be a JSON object")
    return data


def _write_index(zoo_dir: str | Path, index: dict[str, dict[str, Any]]) -> None:
    p = _index_path(zoo_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_policy(policy: MetaRLPolicy, name: str,
                  zoo_dir: str | Path = "policies/zoo",
                  task: str | None = None,
                  source: str | None = None,
                  note: str = "") -> dict[str, Any]:
    """Add (or replace) a named policy in the zoo; return its index entry.

    The file is written with ``save_policy`` (the SPEC.md 67.1 format) and
    the index row carries its ``sha256`` — the provenance pin that
    ``import_policy`` re-verifies. Re-exporting under the same name
    replaces both the file and the row (deterministic; no timestamps).
    """
    _check_name(name)
    if not isinstance(policy, MetaRLPolicy):
        raise TypeError(
            f"export_policy expects a MetaRLPolicy, got {type(policy).__name__}")
    dest_dir = Path(zoo_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}.npz"
    save_policy(policy, path)
    entry = {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "task": task,
        "source": source,
        "note": note,
        "n_actions": int(policy.n_actions),
        "n_updates": int(policy.n_updates),
        "task_names": list(policy.task_names) if policy.task_names else None,
    }
    index = _load_index(zoo_dir)
    index[name] = entry
    _write_index(zoo_dir, index)
    return entry


def list_zoo(zoo_dir: str | Path = "policies/zoo") -> list[dict[str, Any]]:
    """The zoo's entries, sorted by name (deterministic)."""
    index = _load_index(zoo_dir)
    return [index[name] for name in sorted(index)]


def _entry_or_raise(zoo_dir: str | Path, name: str) -> dict[str, Any]:
    _check_name(name)
    index = _load_index(zoo_dir)
    if name not in index:
        known = ", ".join(sorted(index)) or "(empty zoo)"
        raise ValueError(
            f"zoo: {name!r} is not registered (known: {known})")
    return index[name]


def zoo_info(name: str, zoo_dir: str | Path = "policies/zoo") -> dict[str, Any]:
    """The index row for `name`, with ``sha256_ok`` verifying the file."""
    entry = dict(_entry_or_raise(zoo_dir, name))
    path = Path(entry["path"])
    if path.exists():
        entry["sha256_ok"] = (_sha256(path) == entry["sha256"])
    else:
        entry["sha256_ok"] = False
    return entry


def import_policy(name: str, zoo_dir: str | Path = "policies/zoo",
                  target: str | Path | None = None,
                  seed: int = 0) -> tuple[MetaRLPolicy, Path]:
    """Load a named policy after re-verifying its ``sha256`` (loud on
    mismatch / missing file); optionally copy it to `target`.

    Returns ``(policy, file_path)`` — the policy is bit-exact (the
    SPEC.md 67.1.3 ``load_policy`` contract) and `file_path` is the copy
    at `target` when given, else the zoo's own file.
    """
    entry = _entry_or_raise(zoo_dir, name)
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"zoo: {name!r} is registered but its file is missing: {path}")
    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(
            f"zoo: {name!r} failed provenance verification — sha256 "
            f"{actual} != indexed {entry['sha256']}")
    policy = load_policy(path, seed=seed)
    if target is not None:
        t = Path(target)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(path.read_bytes())
        return policy, t
    return policy, path