"""SPEC.md 59 (v0.45): parameters visually interpreted and modified.

The "C. Parameters" round — the spec space was read (chips, architecture
diagram, win-rate matrix) but never steered. Three additive surfaces,
all over the machinery that already exists:

- ``SteeringState`` (59.2) — the three human-in-the-loop verbs per
  spec field: ``pin`` (freeze a field to a value on every candidate and
  on the reset baseline), ``bias`` (redirect a mutation of the field to
  a value), ``constrain`` (reject candidates outside an allowed value
  set as ``invalid_spec``). Validation is registry-level (the C2 knob
  registry pattern, SPEC.md 33.2): the field must be in
  ``improver.specspace.SPEC_FIELDS`` (G2, SPEC.md 36.2) and every value
  must pass the field's own validator.
- ``manual_spec`` / ``train_manual`` (59.3) — manual/expert mode: set
  the exact spec over the registry, train one pass, report — the loop
  validates + reports, it does not discover.
- ``parameter_inspection`` (59.1) — the inspector's data: one row per
  field over the two real layers the parameters already have — the
  model-architecture layer (the ``SPEC_FIELDS`` registry) and the loop
  layer (the ``KNOBS`` registry) — with current value, baseline,
  best-seen + Δ, and the win-rate / Wilson CI recomposed from
  ``research.field_response_stats`` / ``research.bandit_beliefs``
  (57.2 / 57.4). Pure derivation, no new logged field (G2).

Opt-in everywhere: ``AutoRefineEnv(steering=None)`` (default) is the
byte-identical pre-v0.45 path (G2); the policies' ``exclude_fields=()``
default is the same proposal stream (A1–A4 pins green).

House rules: stdlib + the registry only (the heavy task/trainer/evaluator
imports are function-local, the ``advanced.py`` local-loader pattern);
no import cycle — this module's top-level imports never include
``improver.meta_env`` (KNOBS is imported inside ``parameter_inspection``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_SPEC, ModelSpec, SpecError
from .improver.specspace import SPEC_FIELDS

__all__ = [
    "SteeringState",
    "apply_steering",
    "manual_spec",
    "parameter_inspection",
    "train_manual",
]


def _spec_field(name: str) -> str:
    """59.2.1: the field must be in the spec registry (G2, 36.2)."""
    if name not in SPEC_FIELDS:
        raise ValueError(
            f"steering field {name!r} is not in the spec registry "
            f"(available: {', '.join(SPEC_FIELDS)}; SPEC.md 59.2)")
    return name


def _validate_value(name: str, value: Any) -> Any:
    """59.2.1: the value must pass the field's registry validator
    (loud, per-field 'why is this rejected')."""
    return SPEC_FIELDS[name].validator(value)  # raises ValueError (36.2)


def _fv_key(value: Any) -> str:
    """57.2 value-key convention (the V2 matrix, SPEC.md 30.2): `None` →
    `"-"`, list/tuple joined with `,` (e.g. `[8, 16]` → `"8,16"`)."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _value_from_key(key: str) -> Any:
    """59.1: the display value behind a 57.2 value key — a numeric
    `,`-list parses back to a list, a bare number to a number, anything
    else stays the raw string."""
    if "," in key:
        try:
            return [int(x) for x in key.split(",")]
        except ValueError:
            return key
    try:
        return float(key)
    except ValueError:
        return key


@dataclass(frozen=True)
class SteeringState:
    """SPEC.md 59.2: the human-in-the-loop steering rules over the spec
    space. A frozen, JSON-safe record:

    - ``pins``: ``(field, value)`` pairs — the field is force-set to
      ``value`` on **every** candidate (and on the reset baseline);
    - ``biases``: ``(field, value)`` pairs — a mutation of ``field`` is
      redirected to ``value`` (a candidate already equal is untouched);
    - ``constraints``: ``(field, (value, ...))`` pairs — a candidate
      whose value is outside the allowed set is rejected as
      ``invalid_spec`` (logged, SPEC.md 25.4).

    The builder methods validate at the registry (59.2.1) and return a
    **new** state (immutability, the RunConfig 37.1.2 pattern); a second
    rule for the same field replaces the first (last one wins).
    """
    pins: tuple = ()
    biases: tuple = ()
    constraints: tuple = ()

    @property
    def active(self) -> bool:
        """True when at least one rule exists (the env's fast path)."""
        return bool(self.pins or self.biases or self.constraints)

    def pin(self, name: str, value: Any) -> "SteeringState":
        """Add/replace the pin for `name` (validated, 59.2.1)."""
        _spec_field(name)
        v = _validate_value(name, value)
        pins = tuple((name, v) if f == name else (f, x)
                     for f, x in self.pins if f != name) + ((name, v),)
        return SteeringState(pins, self.biases, self.constraints)

    def bias(self, name: str, value: Any) -> "SteeringState":
        """Add/replace the bias for `name` (validated, 59.2.1)."""
        _spec_field(name)
        v = _validate_value(name, value)
        biases = tuple((name, v) if f == name else (f, x)
                       for f, x in self.biases if f != name) + ((name, v),)
        return SteeringState(self.pins, biases, self.constraints)

    def constrain(self, name: str, values) -> "SteeringState":
        """Add/replace the allowed set for `name` (non-empty; every value
        validated, 59.2.1)."""
        _spec_field(name)
        try:
            vs = tuple(values)
        except TypeError:
            raise ValueError(
                f"constrain({name!r}): values must be an iterable of "
                f"allowed values, got {type(values).__name__} (SPEC.md 59.2)") \
                from None
        if not vs:
            raise ValueError(
                f"constrain({name!r}): the allowed set must be non-empty "
                f"(SPEC.md 59.2)")
        vs = tuple(_validate_value(name, v) for v in vs)
        constraints = tuple((name, vs) if f == name else (f, x)
                            for f, x in self.constraints if f != name) \
            + ((name, vs),)
        return SteeringState(self.pins, self.biases, constraints)

    # --- JSON-safe round trip (RunConfig 59.4 pairs) -------------------------
    def to_dict(self) -> dict:
        """JSON-safe dict: lists of ``[field, value]`` (pins/biases) and
        ``[field, [value, ...]]`` (constraints) pairs."""
        return {
            "pins": [[f, v] for f, v in self.pins],
            "biases": [[f, v] for f, v in self.biases],
            "constraints": [[f, list(vs)] for f, vs in self.constraints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringState":
        """Reconstruct + re-validate (a mutated/unknown dict is rejected)."""
        if not isinstance(d, dict):
            raise ValueError(
                f"steering state must be a JSON object, got "
                f"{type(d).__name__} (SPEC.md 59.2)")
        state = cls()
        for pair in d.get("pins") or []:
            f, v = pair
            state = state.pin(f, v)
        for pair in d.get("biases") or []:
            f, v = pair
            state = state.bias(f, v)
        for pair in d.get("constraints") or []:
            f, vs = pair
            state = state.constrain(f, vs)
        return state


def apply_steering(spec_dict: dict, state: "SteeringState | None",
                   champion: dict | None = None
                   ) -> "tuple[dict, list[str]]":
    """SPEC.md 59.2.2: enforce the steering rules on one candidate spec
    dict (pure — returns a new dict, never mutates the input).

    * **pins** are force-set to the pinned value (unconditionally);
    * **biases** redirect a *mutated* field — one whose value differs
      from ``champion`` (the champion in force) — to the biased value;
      a candidate already equal (or not mutated) is untouched;
    * **constraints** produce a violation string per field whose value
      falls outside the allowed set (the caller rejects it as
      ``invalid_spec``, SPEC.md 25.4 — honest and visible, not a
      silent rewrite).

    ``state=None`` (or an empty state) returns the input unchanged with
    no violations (the pre-v0.45 path, G2)."""
    if state is None or not state.active:
        return dict(spec_dict), []
    out = dict(spec_dict)
    champ = champion or {}
    for f, v in state.pins:
        out[f] = v
    for f, v in state.biases:
        if out.get(f) != v and out.get(f) != champ.get(f):
            out[f] = v
    violations: list[str] = []
    for f, allowed in state.constraints:
        if f in out and out[f] not in allowed:
            violations.append(
                f"constrain({f}): {out[f]!r} outside the allowed set "
                f"{list(allowed)}")
    return out, violations


def manual_spec(values: dict) -> ModelSpec:
    """SPEC.md 59.3.1: the exact spec from ``field -> value`` overrides on
    top of ``DEFAULT_SPEC`` (manual/expert mode).

    Unknown fields are a `ValueError` naming the registry; every value
    passes its registry validator (loud per-field 'why rejected',
    59.2.1); the *combination* then passes ``ModelSpec`` validation
    (e.g. an mlp depth-4 architecture or a tree with a bad depth is a
    `SpecError` — the same typed error the loop rejects, 25.4). An empty
    ``values`` dict is ``DEFAULT_SPEC`` itself."""
    if not isinstance(values, dict):
        raise ValueError(
            f"manual spec values must be a dict of field -> value, got "
            f"{type(values).__name__} (SPEC.md 59.3)")
    for k in values:
        _spec_field(k)  # unknown field -> ValueError (loud, 59.2.1)
    d = DEFAULT_SPEC.to_dict()
    for k, v in values.items():
        d[k] = _validate_value(k, v)
    try:
        return ModelSpec.from_dict(d)
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError(f"invalid spec: {exc} (SPEC.md 59.3)") from exc


def train_manual(task_name: str, values: dict, seed: int = 7,
                 task_config: dict | None = None,
                 max_train_seconds: float | None = None) -> dict:
    """SPEC.md 59.3.2: train + evaluate **one exact spec** (manual/expert
    mode) — ``manual_spec`` → ``train_from_task`` → ``evaluate_full``,
    the exact pattern of ``advanced.retrain_spec`` (49.2.1).

    No loop, no budget, **no run dir** (read-only, the 49.2.1
    semantics): the researcher knows the architecture and wants the
    loop to validate + report, not discover. ``task_config`` (e.g.
    ``{"path", "label"}`` for the csv task) is the optional
    data-driven constructor kwargs (22.1). Returns the JSON-safe dict
    ``{"spec", "score", "gen_score", "gen_gap", "train_seconds",
    "final_loss", "time_capped"}``. An unknown task name is a
    `ValueError` (the registry, SPEC.md 15).
    """
    # local imports: keep the module surface small and cycle-free
    # (the advanced.py local-loader pattern)
    from .evaluator import evaluate_full
    from .tasks import TASKS
    from .trainer import train_from_task
    if task_name not in TASKS:
        raise ValueError(
            f"unknown task {task_name!r} (available: {sorted(TASKS)}; "
            f"SPEC.md 59.3)")
    spec = manual_spec(values)
    cls = TASKS[task_name]
    if task_config:
        task = cls(seed=int(seed), **dict(task_config))
    else:
        task = cls(seed=int(seed))
    result = train_from_task(
        task, spec, int(seed),
        time_limit_seconds=(None if max_train_seconds is None
                            else float(max_train_seconds)))
    ev = evaluate_full(task, result.model)
    return {
        "spec": spec.to_dict(),
        "score": float(ev["score"]),
        "gen_score": float(ev["gen_score"]),
        "gen_gap": float(ev["gen_gap"]),
        "train_seconds": float(result.train_seconds),
        "final_loss": float(result.final_loss),
        "time_capped": bool(result.time_capped),
    }


def parameter_inspection(entries, best_spec: dict | None = None,
                         run_config: dict | None = None,
                         field_stats: dict | None = None,
                         ucb: dict | None = None) -> list[dict]:
    """SPEC.md 59.1: the parameter inspector's data — one row per field
    over the **two real layers** the parameters already have:

    * **architecture** (the ``SPEC_FIELDS`` registry, 36.2) — name,
      kind, space, families, spec_ref, the *current* value (from
      ``best_spec``), the baseline value (``DEFAULT_SPEC``), the
      best-seen value + Δ recomposed from ``research.
      field_response_stats`` (57.2: per-value means over the scored
      candidates — best-seen = the value with the highest mean, Δ =
      that mean minus the baseline value's mean), and the win rate +
      95% Wilson CI + final UCB from ``research.bandit_beliefs``
      (57.4);
    * **loop** (the ``KNOBS`` registry, 33.2) — name, default (the
      baseline), spec_ref, the CLI flags, the app widget, and the
      current value (from ``run_config``). Loop params are not bandit
      arms, so their rows carry no win-rate.

    Pure derivation over data that already exists (G2): registry order
    is the row order (deterministic); missing inputs degrade to
    ``None`` (an old run without field_stats still renders every row).
    """
    # lazy: KNOBS lives in improver.meta_env (this module must stay
    # importable without it — no cycle), and research reuses the 57.x
    # derivations (57.2 / 57.4)
    from .improver.meta_env import KNOBS
    from .research import bandit_beliefs, field_response_stats

    resp = field_response_stats(entries)
    beliefs = bandit_beliefs(field_stats, ucb)
    base = DEFAULT_SPEC.to_dict()
    best = best_spec if isinstance(best_spec, dict) else {}
    cfg = run_config if isinstance(run_config, dict) else {}

    rows: list[dict] = []
    for name, f in SPEC_FIELDS.items():
        per_value = resp.get(name) or {}
        best_seen = best_mean = delta = n_best = None
        if per_value:
            top_key, top = max(per_value.items(),
                               key=lambda kv: kv[1]["mean"])
            best_seen = _value_from_key(top_key)
            best_mean = float(top["mean"])
            n_best = int(top["n"])
            base_key = _fv_key(base.get(name))
            if base_key in per_value:
                delta = float(top["mean"] - per_value[base_key]["mean"])
        bel = beliefs.get(name) or {}
        rows.append({
            "name": name,
            "layer": "architecture",
            "kind": f.kind,
            "space": [list(v) if isinstance(v, tuple) else v
                      for v in f.space],
            "families": list(f.families),
            "spec_ref": f.spec_ref,
            "current": best.get(name),
            "baseline": base.get(name),
            "best_seen": best_seen,
            "best_seen_mean": best_mean,
            "best_seen_n": n_best,
            "delta": delta,
            "win_rate": bel.get("win_rate"),
            "ci_lo": bel.get("lo"),
            "ci_hi": bel.get("hi"),
            "ucb": bel.get("ucb"),
        })
    for name, k in KNOBS.items():
        rows.append({
            "name": name,
            "layer": "loop",
            "kind": None,
            "space": None,
            "families": [],
            "spec_ref": k.spec_ref,
            "cli": list(k.cli),
            "app_widget": k.app_widget,
            "current": cfg.get(name),
            "baseline": k.default,
            "best_seen": None,
            "best_seen_mean": None,
            "best_seen_n": None,
            "delta": None,
            "win_rate": None,
            "ci_lo": None,
            "ci_hi": None,
            "ucb": None,
        })
    return rows
