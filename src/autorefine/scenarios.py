"""Regime / stress scenarios (SPEC.md 61.3, v0.47, A3).

A **pure config** that composes with any task + budget + gate — the core
loop is unchanged. A ``Scenario`` names a regime (few-shot, drifting,
trap), carries a concrete ``Task`` instance, and records the
``budget_overrides`` / ``gate_overrides`` / ``extras`` that pin the
regime. Applying a preset to a task is pure (no training, no env state);
``resolve_budget`` merges the overrides onto a base ``Budget`` and
``resolved_gate`` returns the gate dict — so a preset composes with any
existing ``Budget`` / gate (the §18.5 CI/gen-gap gate and the §20.1
curriculum machinery already exist; a preset just pins one of them).

Deterministic given the seed (G2); stdlib + numpy (via the tasks) only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tasks.base import Task
from .tasks.parity import ParityTask


@dataclass(frozen=True)
class Scenario:
    """SPEC.md 61.3 (A3): one stress regime as pure config.

    ``task`` is a concrete ``Task`` instance (a ``Task`` instance, not a
    factory — a preset is a *resolved* regime, not a template).
    ``budget_overrides`` is a dict of ``Budget`` fields (see
    ``resolve_budget``); ``gate_overrides`` a dict of gate fields (e.g. a
    tighter ``gen_gap``); ``extras`` a free dict (e.g. A1's constraint
    ``objectives``, or a tiny ``n_points`` train pool)."""

    name: str
    description: str
    task: Task
    budget_overrides: dict = field(default_factory=dict)
    gate_overrides: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def resolve_budget(self, base) -> "object":
        """Merge ``budget_overrides`` onto a base ``Budget`` (a ``Budget``
        or a dict of its fields) → a new ``Budget`` (61.3.4). Pure."""
        from .config import Budget
        base_fields = (base.__dict__ if hasattr(base, "__dict__")
                       else dict(base))
        merged = dict(base_fields)
        merged.update(self.budget_overrides)
        return Budget(**merged)

    def resolved_gate(self) -> dict:
        """The gate config the preset pins (61.3.4) — a plain dict (e.g.
        ``{"gen_gap": 0.0}`` for the trap regime). Pure."""
        return dict(self.gate_overrides)

    def resolved(self, base_budget) -> dict:
        """A copy-pasteable recipe for the regime (61.3.4):
        ``{"task", "budget", "gate", "extras"}``. Pure (no training, no
        env state)."""
        return {
            "task": self.task.name if hasattr(self.task, "name") else str(self.task),
            "budget": self.resolve_budget(base_budget).__dict__,
            "gate": self.resolved_gate(),
            "extras": dict(self.extras),
        }


def fewshot(seed: int = 1) -> Scenario:
    """SPEC.md 61.3.1 (A3): a **tiny train pool** (64 points) against a
    full holdout — does the CI / gen-gap gate (18.5) catch an overfit
    candidate? A small experiment budget keeps it a fast stress probe."""
    return Scenario(
        name="fewshot",
        description=("tiny train pool (64 pts) vs a full holdout — does the "
                     "18.5 CI / gen-gap gate catch overfit?"),
        task=ParityTask(seed=seed),
        budget_overrides={"max_experiments": 5, "max_train_seconds": 20.0},
        gate_overrides={},
        extras={"n_points": 64},
    )


def drifting(seed: int = 1, p_flip: float = 0.30) -> Scenario:
    """SPEC.md 61.3.2 (A3): a ``ParityTask`` whose observation noise
    (``p_flip``) is **raised** at the gate level (0.08 → ``p_flip``) — the
    improver must re-adapt; composes with the §20.1 curriculum (this pins
    one harder level)."""
    if not 0.0 <= float(p_flip) < 0.5:
        raise ValueError(f"p_flip must be in [0.0, 0.5), got {p_flip!r}")
    return Scenario(
        name="drifting",
        description=("observation noise raised at the gate level — the "
                     "improver must re-adapt (20.1 curriculum level)."),
        task=ParityTask(seed=seed, p_flip=float(p_flip)),
        budget_overrides={"max_experiments": 8, "max_train_seconds": 25.0},
        gate_overrides={"p_flip": float(p_flip)},
        extras={"n_points": 4096},
    )


def trap(seed: int = 1) -> Scenario:
    """SPEC.md 61.3.3 (A3): a spec field that looks good on val but hurts
    holdout — a gate the §18.5 **gen-gap penalty** exists to catch. The
    preset pins a tight ``gen_gap`` threshold (0.0) so a val-holdout
    divergent candidate is rejected."""
    return Scenario(
        name="trap",
        description=("val-good / holdout-bad field — the 18.5 gen-gap "
                     "penalty rejects it (tight gen_gap threshold)."),
        task=ParityTask(seed=seed),
        budget_overrides={"max_experiments": 6, "max_train_seconds": 20.0},
        gate_overrides={"gen_gap": 0.0},
        extras={"n_points": 4096},
    )
