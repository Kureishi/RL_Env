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
import math
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_SPEC, ModelSpec, SpecError
from .improver.specspace import SPEC_FIELDS

__all__ = [
    "SteeringState",
    "apply_steering",
    "best_score_curve",
    "effect_windows",
    "manual_spec",
    "parameter_inspection",
    "train_manual",
]

# SPEC.md 94.1.1 (v0.80): the live-steering event verbs — the three 59.2
# verbs plus ``clear`` (the rule set is removed wholesale). Closed set:
# a new verb must be added here AND in `record`'s validation.
STEER_VERBS = ("pin", "bias", "constrain", "clear")


def _spec_field(name: str) -> str:
    """59.2.1: the field must be in the spec registry (G2, 36.2)."""
    if name not in SPEC_FIELDS:
        raise ValueError(
            f"steering field {name!r} is not in the spec registry "
            f"(available: {', '.join(SPEC_FIELDS)})")
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

    ``history`` (SPEC.md 94.1.1, v0.80) is the append-only log of
    **live** steering changes — one event dict per ``record()`` call, in
    change order. It is the 4th field with a default, so positional
    construction and ``SteeringState() == SteeringState.from_dict({})``
    (the 59.2 pins) stay green; only ``record()`` (the live ``steer()``
    path, 94.1.3) ever appends to it — pre-run rules (CLI / the app's
    Setup panel) leave it empty, which is what keeps the
    ``steering_events`` summary key absent (94.1.6, the A49/A84 pin).
    """
    pins: tuple = ()
    biases: tuple = ()
    constraints: tuple = ()
    history: tuple = ()

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
        # 94.1.1 (v0.80): the builders preserve the history (the
        # log of changes is never rewritten -- only `record`
        # appends to it)
        return SteeringState(pins, self.biases, self.constraints,
                             self.history)

    def bias(self, name: str, value: Any) -> "SteeringState":
        """Add/replace the bias for `name` (validated, 59.2.1)."""
        _spec_field(name)
        v = _validate_value(name, value)
        biases = tuple((name, v) if f == name else (f, x)
                       for f, x in self.biases if f != name) + ((name, v),)
        return SteeringState(self.pins, biases, self.constraints,
                             self.history)  # 94.1.1: preserved

    def record(self, field: Any, verb: str, before: Any, after: Any,
               at_experiment: int) -> "SteeringState":
        """SPEC.md 94.1.1 (v0.80): append one live-steering event to the
        history — the **only** writer of ``history`` (the pre-run path
        never records, 94.1.6).

        The event is the JSON-safe dict ``{field, verb, before, after,
        at_experiment}``: ``verb`` is in ``STEER_VERBS`` (``clear`` has
        ``field=None``; the three rule verbs require a registry field,
        59.2.1); ``at_experiment`` is the 1-based experiment count the
        change followed (``bm.used_experiments`` at the call, 94.1.3);
        ``before`` / ``after`` are the rule values around the change
        (``None`` where there was none). Tuples are normalized to lists
        at insertion (JSON-safe round trip, the 94.1.1 rule).
        """
        if verb not in STEER_VERBS:
            raise ValueError(
                f"steering event verb must be one of {list(STEER_VERBS)}, "
                f"got {verb!r}")
        if verb == "clear":
            if field is not None:
                raise ValueError(
                    f"a 'clear' steering event has no field "
                    f"(field must be None, got {field!r})")
        else:
            _spec_field(field)  # registry-level, fail loud (59.2.1)
        if (isinstance(at_experiment, bool)
                or not isinstance(at_experiment, int) or at_experiment < 0):
            raise ValueError(
                f"steering event at_experiment must be an int >= 0, got "
                f"{at_experiment!r}")
        b = list(before) if isinstance(before, tuple) else before
        a = list(after) if isinstance(after, tuple) else after
        event = {"field": field, "verb": verb, "before": b, "after": a,
                 "at_experiment": int(at_experiment)}
        return SteeringState(self.pins, self.biases, self.constraints,
                             self.history + (event,))

    def constrain(self, name: str, values) -> "SteeringState":
        """Add/replace the allowed set for `name` (non-empty; every value
        validated, 59.2.1)."""
        _spec_field(name)
        try:
            vs = tuple(values)
        except TypeError:
            raise ValueError(
                f"constrain({name!r}): values must be an iterable of "
                f"allowed values, got {type(values).__name__} ") \
                from None
        if not vs:
            raise ValueError(
                f"constrain({name!r}): the allowed set must be non-empty "
                f"")
        vs = tuple(_validate_value(name, v) for v in vs)
        constraints = tuple((name, vs) if f == name else (f, x)
                            for f, x in self.constraints if f != name) \
            + ((name, vs),)
        return SteeringState(self.pins, self.biases, constraints,
                             self.history)  # 94.1.1: preserved

    # --- JSON-safe round trip (RunConfig 59.4 pairs) -------------------------
    def to_dict(self) -> dict:
        """JSON-safe dict: lists of ``[field, value]`` (pins/biases) and
        ``[field, [value, ...]]`` (constraints) pairs.

        ``history`` (94.1.1) is a **conditional** key — present only when
        non-empty, so the empty state serializes to the exact pre-v0.80
        three-key dict (the 59.2 round-trip pin stays byte-identical).
        """
        d = {
            "pins": [[f, v] for f, v in self.pins],
            "biases": [[f, v] for f, v in self.biases],
            "constraints": [[f, list(vs)] for f, vs in self.constraints],
        }
        if self.history:
            d["history"] = [dict(e) for e in self.history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringState":
        """Reconstruct + re-validate (a mutated/unknown dict is rejected).

        ``history`` is optional (a pre-v0.80 dict without the key loads
        as the empty history); each entry is re-validated through
        ``record()`` (the verb, the field, the experiment index)."""
        if not isinstance(d, dict):
            raise ValueError(
                f"steering state must be a JSON object, got "
                f"{type(d).__name__} ")
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
        for e in d.get("history") or []:
            if not isinstance(e, dict):
                raise ValueError(
                    f"steering history entries must be JSON objects, got "
                    f"{type(e).__name__}")
            state = state.record(
                e.get("field"), e.get("verb"), e.get("before"),
                e.get("after"), e.get("at_experiment"))
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
            f"{type(values).__name__} ")
    for k in values:
        _spec_field(k)  # unknown field -> ValueError (loud, 59.2.1)
    d = DEFAULT_SPEC.to_dict()
    for k, v in values.items():
        d[k] = _validate_value(k, v)
    try:
        return ModelSpec.from_dict(d)
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError(f"invalid spec: {exc} ") from exc


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
            f")")
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


def _finite(x: Any) -> bool:
    """A finite real number (bools are not numbers here)."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def best_score_curve(entries) -> list[float]:
    """SPEC.md 94.1.5 (v0.80): the run's running-best score curve —
    point 0 is the scored baseline, then the running best after each
    scored experiment (a rejected dip never lowers the curve): the
    50.1.2 ``running_best_curve`` semantics exactly (``kind`` in
    baseline/experiment with a finite ``holdout_score``, in log order).
    ``[]`` when no row is scored. Pure, deterministic (G2); this is the
    single curve both the CLI and the app feed to ``effect_windows``.
    """
    # local import: the kind registry is the single source (35.1); a
    # leaf module, but keep this one's import surface small
    from .memory import KIND_BASELINE, KIND_EXPERIMENT
    scores: list[float] = []
    for r in entries or []:
        if not isinstance(r, dict):
            continue
        if (r.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)
                and _finite(r.get("holdout_score"))):
            scores.append(float(r["holdout_score"]))
    out: list[float] = []
    best: float | None = None
    for s in scores:  # running best (the 50.1.2 rule)
        best = s if best is None else max(best, s)
        out.append(best)
    return out


def effect_windows(curve, events, window: int = 3,
                   eps: float = 0.05) -> list[dict]:
    """SPEC.md 94.1.5 (v0.80): the measured effect of each live-steering
    event — the single G2 source of the effect numbers (the CLI's
    ``report --steer`` and the app's effects panel both call this).

    ``curve`` is a ``best_score_curve`` (point 0 = the baseline; point
    ``i`` = the running best after the i-th experiment). For an event at
    experiment ``i`` (1-based, "after the i-th experiment"):

    * ``pre_mean``  = the mean of ``curve[max(0, i-window) : i]`` (a
      partial window at the start; the empty case — an event at
      experiment 0 — falls back to ``curve[0]``);
    * ``post_mean`` = the mean of ``curve[i : i+window]`` (clamped by
      the curve's end; an event *past* the logged curve fails loud);
    * ``delta``     = ``post_mean − pre_mean``;
    * ``direction`` = ``flat`` when ``|delta| <= eps`` (the default
      0.05 = 0.5 score points — noise, not a signal), else ``up`` / ``down``.

    Each row carries the event's identity through unchanged: ``{
    field, verb, before, after, at_experiment, pre_mean, post_mean,
    delta, direction}``. Fail loud on a bad window / eps / curve /
    event (94.1.5); an empty event list is the empty list.
    """
    if (isinstance(window, bool) or not isinstance(window, int)
            or window < 1):
        raise ValueError(
            f"effect_windows: window must be an int >= 1, got "
            f"{window!r}")
    if (isinstance(eps, bool) or not isinstance(eps, (int, float))
            or not math.isfinite(float(eps)) or float(eps) < 0):
        raise ValueError(
            f"effect_windows: eps must be a finite number >= 0, got "
            f"{eps!r}")
    try:
        cv = [float(v) for v in curve]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"effect_windows: curve must be a sequence of numbers, got "
            f"{type(curve).__name__}") from exc
    if not cv or not all(math.isfinite(v) for v in cv):
        raise ValueError(
            "effect_windows: curve must be a non-empty sequence of "
            "finite numbers  ")
    try:
        evs = list(events)
    except TypeError as exc:
        raise ValueError(
            f"effect_windows: events must be a sequence, got "
            f"{type(events).__name__}") from exc
    eps_f = float(eps)
    out: list[dict] = []
    for e in evs:
        if not isinstance(e, dict):
            raise ValueError(
                f"effect_windows: each event must be a dict, got "
                f"{type(e).__name__}")
        i = e.get("at_experiment")
        if (isinstance(i, bool) or not isinstance(i, int) or i < 0):
            raise ValueError(
                f"effect_windows: event at_experiment must be an int >= 0, "
                f"got {i!r}")
        pre = cv[max(0, i - window):i] or [cv[0]]  # 94.1.5: the i=0 fallback
        post = cv[i:i + window]
        if not post:
            raise ValueError(
                f"effect_windows: the event at experiment {i} has no "
                f"post window — the logged curve has {len(cv)} points "
                f" ")
        pre_mean = sum(pre) / len(pre)
        post_mean = sum(post) / len(post)
        delta = post_mean - pre_mean
        direction = ("flat" if abs(delta) <= eps_f
                     else "up" if delta > 0 else "down")
        out.append({
            "field": e.get("field"),
            "verb": e.get("verb"),
            "before": e.get("before"),
            "after": e.get("after"),
            "at_experiment": i,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "delta": delta,
            "direction": direction,
        })
    return out


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
