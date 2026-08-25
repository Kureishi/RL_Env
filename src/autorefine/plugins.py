"""Optional entry-point plugin loader (SPEC.md 21.3).

External packages contribute tasks/policies without importing this repo:

    [project.entry-points."autorefine.tasks"]
    my-task = "my_pkg.tasks:MyTask"

    [project.entry-points."autorefine.policies"]
    my-policy = "my_pkg.policies:MyPolicy"

Only stdlib `importlib.metadata` is used (no plugin framework). `discover()`
accepts an `eps` injection — any iterable of entry-point-like objects with a
`name` attribute and a `load()` method — so the loader is testable without
installing a package. Discovery is deterministic (sorted by entry name, G2).
"""
from __future__ import annotations

import importlib.metadata as _md

TASKS_GROUP = "autorefine.tasks"
POLICIES_GROUP = "autorefine.policies"


class PluginError(ValueError):
    """A plugin entry point is malformed, fails to load, or lacks the protocol."""


def _iter_eps(group: str, eps):
    if eps is not None:
        return sorted(eps, key=lambda e: str(getattr(e, "name", "")))
    return sorted(_md.entry_points(group=group), key=lambda e: e.name)


def discover(group: str, eps=None) -> dict:
    """Load every entry point in `group`; return `{name: object}` (sorted)."""
    out = {}
    for ep in _iter_eps(group, eps):
        name = getattr(ep, "name", None)
        try:
            obj = ep.load()
        except Exception as exc:
            raise PluginError(
                f"entry point {name!r} in group {group!r} failed to load: {exc}"
            ) from exc
        out[name] = obj
    return out


def _require_protocol(name, obj, methods: tuple[str, ...], group: str) -> None:
    if not isinstance(obj, type):
        raise PluginError(
            f"{group} entry {name!r} must be a class, got {type(obj).__name__}"
        )
    for m in methods:
        if not callable(getattr(obj, m, None)):
            raise PluginError(f"{group} entry {name!r} is missing required method {m}()")


def register_tasks(eps=None) -> dict:
    """Load task plugins into the `TASKS` registry (SPEC.md 21.3).

    Each entry must be a class with callable `make_dataset`/`score` (the §15
    task protocol). Returns `{name: class}`; raises `PluginError` otherwise."""
    from .tasks import TASKS
    found = {}
    for name, obj in discover(TASKS_GROUP, eps).items():
        _require_protocol(name, obj, ("make_dataset", "score"), TASKS_GROUP)
        TASKS[name] = obj
        found[name] = obj
    return found


def register_policies(eps=None) -> dict:
    """Load policy plugins; return `{name: class}` (SPEC.md 21.3).

    Each entry must be a class with callable `propose` (the §17 policy
    contract). Policies are not merged into a registry — the contract is
    `propose(env_state) -> spec_dict` — so callers use the returned dict."""
    found = {}
    for name, obj in discover(POLICIES_GROUP, eps).items():
        _require_protocol(name, obj, ("propose",), POLICIES_GROUP)
        found[name] = obj
    return found
