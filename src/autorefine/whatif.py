"""SPEC.md 60 (v0.46): what-if & comparison — steer the read surface.

The spec space was *interpreted* (the §57 response surfaces, the §50
architecture diagram, the §59 inspector) but never *what-iffed*. This
round adds four pure derivations over data that already exists — the
`experiments.jsonl` entries (16/25.4), the registry (36.2), and the §57
per-field stats — with **zero retraining** and zero new logged fields
(G2). All four are the counterfactual machinery of `simulate.what_if`
(40.2) made interactive:

- ``whatif_preview`` (60.1) — drag a value, see the would-be candidate
  (validated, the §59.2 registry rule) and its *estimated* effect read
  off the logged response surface (57.2);
- ``spec_fingerprint`` / ``fingerprint_diff`` (60.2) — the spec "DNA":
  one normalized bar per registry field (within its space), so two
  specs are eyeball-comparable at a glance;
- ``interaction_matrix`` (60.3) — which field mutations co-occurred in
  the same candidate and their joint effect vs the run's mean (the
  single-field views hide these);
- ``weighted_reslice`` (60.4) — re-weight score vs train-time vs
  model-size and re-slice which *logged* candidates would pass — the
  §40.2 counterfactual re-gater with a composite objective.

House rules: stdlib + numpy (a core dependency) only; import-cycle-free
(`memory` for the kind registry — 35.1 — `config` for `ModelSpec` /
`spec_n_params` — 36.2 / 58.2 — `research` for the 57.2 re-derivation,
`improver.specspace` for the registry — none import this module);
pure and deterministic (G2); the SVG renderers live in `plotting.py`
(60.5) and the app renders the block in `_render_result` (60.5).
"""
from __future__ import annotations

import math

from .config import DEFAULT_SPEC, ModelSpec, SpecError, spec_n_params
from .improver.specspace import SPEC_FIELDS
from .memory import KIND_BASELINE, KIND_EXPERIMENT
from .research import field_response_stats

__all__ = [
    "whatif_preview",
    "spec_fingerprint",
    "fingerprint_diff",
    "interaction_matrix",
    "weighted_reslice",
]


def _fv_key(value) -> str:
    """57.2 value-key convention (the V2 matrix, SPEC.md 30.2): `None` →
    `"-"`, list/tuple joined with `,` (e.g. `[8, 16]` → `"8,16"`),
    else `str(v)`. The same key `research._fv_key` builds, so a preview
    value and a logged value always match."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _num(value) -> float | None:
    """A finite number, else `None` (old rows may lack a field)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


def _scored(entries):
    """The 57.2 scored-candidate rule: kind baseline/experiment, a finite
    `holdout_score`, a dict `spec` (screen / curriculum / invalid rows
    never enter)."""
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):
            continue
        score = _num(e.get("holdout_score"))
        spec = e.get("spec")
        if score is None or not isinstance(spec, dict):
            continue
        out.append((e, score))
    return out


# --- 60.1 the live "what-if" preview ---------------------------------------

def whatif_preview(entries, best_spec, field, value) -> dict:
    """SPEC.md 60.1: the live "what-if" preview — zero retraining.

    Given the run's logged `entries`, the current champion `best_spec`
    (a dict, or `None` → `DEFAULT_SPEC`), a registry `field`, and a
    candidate `value`, return the would-be picture:

    - **candidate** — `dict(best_spec)` with `field` force-set to
      `value` (the §59.2 pin semantics on one field);
    - **valid / error** — the *combination* validated against
      `ModelSpec` (the §59.3 `manual_spec` rule: a registry-legal value
      can still be a `SpecError` for the champion's family, e.g. a
      tree depth on an mlp);
    - **estimated effect** — read off the logged response surface
      (57.2 `field_response_stats`): `value_mean` / `value_n` for the
      candidate `(field, value)` pair, `current_mean` for the champion's
      current value of the field, and `delta` = `value_mean −
      current_mean` (each `None` when never logged — an honest
      "no data yet", never an invented number);
    - **pool** — the number of scored candidates the surface was built
      from (0 → no logged data at all).

    `field` must be in the `SPEC_FIELDS` registry (G2, 36.2 — a `ValueError`
    naming the registry) and `value` must pass the field's registry
    validator (a loud `ValueError`). Pure and deterministic (G2)."""
    if field not in SPEC_FIELDS:
        raise ValueError(
            f"what-if field {field!r} is not in the spec registry "
            f"(available: {', '.join(SPEC_FIELDS)}; SPEC.md 60.1)")
    SPEC_FIELDS[field].validator(value)  # raises ValueError (36.2)
    base = dict(best_spec) if isinstance(best_spec, dict) \
        else DEFAULT_SPEC.to_dict()
    candidate = dict(base)
    candidate[field] = value
    valid, error = True, None
    try:
        ModelSpec.from_dict(candidate)
    except (SpecError, ValueError) as exc:
        valid, error = False, str(exc)
    scored = _scored(entries)
    per_value = field_response_stats(entries).get(field) or {}
    v = per_value.get(_fv_key(value))
    c = per_value.get(_fv_key(base.get(field)))
    return {
        "field": field,
        "value": value,
        "current_value": base.get(field),
        "candidate": candidate,
        "valid": bool(valid),
        "error": error,
        "value_mean": float(v["mean"]) if v else None,
        "value_n": int(v["n"]) if v else None,
        "current_mean": float(c["mean"]) if c else None,
        "delta": float(v["mean"] - c["mean"]) if (v and c) else None,
        "pool": len(scored),
    }


# --- 60.2 the spec fingerprint ("DNA") --------------------------------------

def _norm_components(value) -> list[float]:
    """The numeric components of a spec value (bools excluded)."""
    out = []
    seq = value if isinstance(value, (list, tuple)) else [value]
    for v in seq:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


def _fingerprint_t(field_name: str, value) -> float | None:
    """60.2: one field value → `t ∈ [0, 1]` normalized within its
    registry space (the field's own `space` — never an ad-hoc bound):

    - **ordered** — min-max over the space, clamped to `[0, 1]` (a
      single-point space → `0.0`);
    - **categorical** — the ordinal position in the space tuple:
      `index / (len − 1)` (`0.0` for a single-value space); a value
      outside the space → `None` (the renderer skips the bar);
    - **sequence** — the *scale* reading: `max(numeric components) /
      max(all numeric components in the space)` (the "wide/narrow"
      axis; depth is metadata the renderer shows in the tooltip).

    `None` value → `None` (the field was never set — e.g. `knn_k` on an
    mlp). Pure (G2)."""
    if value is None:
        return None
    f = SPEC_FIELDS[field_name]
    try:
        if f.kind == "ordered":
            vals = _norm_components(f.space)
            if not vals:
                return None
            lo, hi = min(vals), max(vals)
            x = float(value)
            if hi == lo:
                return 0.0
            return min(1.0, max(0.0, (x - lo) / (hi - lo)))
        if f.kind == "categorical":
            try:
                idx = f.space.index(value)
            except ValueError:
                return None
            if len(f.space) <= 1:
                return 0.0
            return idx / (len(f.space) - 1)
        if f.kind == "sequence":
            comps = _norm_components(value)
            if not comps:
                return None
            space_maxs = []
            for s in f.space:
                cs = _norm_components(s)
                if cs:
                    space_maxs.append(max(cs))
            if not space_maxs:
                return None
            m = max(space_maxs)
            if m == 0:
                return 0.0
            return min(1.0, max(0.0, max(comps) / m))
    except (TypeError, ValueError):
        return None
    return None


def spec_fingerprint(spec) -> list[dict]:
    """SPEC.md 60.2: the spec fingerprint ("DNA") — one row per registry
    field, in registry order (deterministic, G2):
    `{"name", "kind", "value", "t"}` with `t ∈ [0, 1]` the value
    normalized within the field's own space (the `_fingerprint_t` rule)
    and `t` `None` when the field was never set in the spec. `spec` is
    a dict or a `ModelSpec` (`None` → `[]`). Two specs on the same task
    become two comparable bar rows — "wide + shallow" vs "narrow +
    deep" at a glance (60.2.1). Pure (G2)."""
    d = spec.to_dict() if isinstance(spec, ModelSpec) \
        else (dict(spec) if isinstance(spec, dict) else None)
    if d is None:
        return []
    return [
        {"name": name,
         "kind": f.kind,
         "value": d.get(name),
         "t": _fingerprint_t(name, d.get(name))}
        for name, f in SPEC_FIELDS.items()
    ]


def fingerprint_diff(a, b) -> list[dict]:
    """SPEC.md 60.2: the fingerprint comparison of two specs — one row
    per registry field (registry order, deterministic, G2):
    `{"name", "value_a", "value_b", "t_a", "t_b", "delta"}` with
    `delta = t_b − t_a` (`None` when either spec leaves the field unset).
    Pure (G2)."""
    fa = {r["name"]: r for r in spec_fingerprint(a)}
    fb = {r["name"]: r for r in spec_fingerprint(b)}
    out = []
    for name in SPEC_FIELDS:
        ra, rb = fa.get(name), fb.get(name)
        ta = ra["t"] if ra else None
        tb = rb["t"] if rb else None
        out.append({
            "name": name,
            "value_a": ra["value"] if ra else None,
            "value_b": rb["value"] if rb else None,
            "t_a": ta,
            "t_b": tb,
            "delta": float(tb - ta) if (ta is not None and tb is not None)
            else None,
        })
    return out


# --- 60.3 the interaction heatmap -------------------------------------------

def interaction_matrix(entries) -> dict:
    """SPEC.md 60.3: which field mutations **co-occurred** in the same
    candidate and their joint effect — the interaction the single-field
    views (57.2 / 57.4) hide ("hidden_dim only helps when lr is low").

    The candidate pool is the 57.2 scored rule (baseline + experiment
    rows with a finite `holdout_score` and a dict `spec`). A candidate's
    mutation set is its logged `mutation` list (26.3 — the fields that
    differ from the champion in force), restricted to registry fields
    (G2, 36.2). For every *unordered pair* `{f1, f2}` both mutated in
    the same candidate: `n` candidates, `mean` holdout score, `delta`
    = `mean − overall_mean` (the run's mean over the whole scored pool,
    incl. the baseline — the same credit pool as 57.2). Returns the
    JSON-safe dict `{"fields": [...sorted...], "cells": {f1: {f2:
    {"n", "mean", "delta"}}}`, "marginals": {field: #candidates mutated
    it}, "overall_mean", "n_scored"}`. Pure and deterministic (G2);
    empty pool → empty shape (the renderer draws its empty state)."""
    scored = _scored(entries)
    if not scored:
        return {"fields": [], "cells": {}, "marginals": {},
                "overall_mean": None, "n_scored": 0}
    overall = sum(s for _, s in scored) / len(scored)
    marginals: dict[str, int] = {}
    pairs: dict[tuple[str, str], list[float]] = {}
    for e, score in scored:
        mut = e.get("mutation")
        fields = sorted({f for f in mut if isinstance(f, str)
                         and f in SPEC_FIELDS}
                        if isinstance(mut, list) else set())
        for f in fields:
            marginals[f] = marginals.get(f, 0) + 1
        for i, f1 in enumerate(fields):
            for f2 in fields[i + 1:]:
                pairs.setdefault((f1, f2), []).append(score)
    cells: dict[str, dict[str, dict]] = {}
    for (f1, f2) in sorted(pairs):
        ss = pairs[(f1, f2)]
        mean = sum(ss) / len(ss)
        cells.setdefault(f1, {})[f2] = {
            "n": len(ss),
            "mean": float(mean),
            "delta": float(mean - overall),
        }
    return {
        "fields": sorted(marginals),
        "cells": cells,
        "marginals": marginals,
        "overall_mean": float(overall),
        "n_scored": len(scored),
    }


# --- 60.4 the objective-weight reslice --------------------------------------

def _minmax(x: float, lo: float, hi: float) -> float:
    """Pool-relative min-max of `x` in `[lo, hi]`; a degenerate pool
    (`hi == lo`) → `1.0` (every candidate is best on that axis)."""
    if hi == lo:
        return 1.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def _weight(name: str, w) -> float:
    if isinstance(w, bool) or not isinstance(w, (int, float)) \
            or not math.isfinite(w) or w < 0:
        raise ValueError(
            f"objective weight {name!r} must be a finite number >= 0, "
            f"got {w!r} (SPEC.md 60.4)")
    return float(w)


def weighted_reslice(entries, w_score, w_train, w_size=0.0,
                     target=0.75, state_dim=None, n_out=None,
                     grid=None) -> dict:
    """SPEC.md 60.4: re-weight score vs train-time vs model-size and
    re-slice which **logged** candidates would pass — the §40.2
    counterfactual re-gater, made interactive (zero retraining).

    The pool is the 57.2 scored rule (baseline + experiment rows with a
    finite `holdout_score` and a dict `spec`). Each candidate gets a
    **composite** in `[0, 1]` — the weighted average of its
    *pool-relative* min-max positions: the score axis (higher better),
    the train axis (`1 − minmax(train_seconds)`; a missing train
    drops the axis for that candidate), and the size axis
    (`1 − minmax(spec_n_params(spec, state_dim, n_out, grid))`, 58.2;
    `None` for data-dependent families / missing geometry — the axis
    drops for that candidate). A candidate's composite renormalizes
    over the axes it has, so `w_size > 0` never punishes an
    unsized family. **pass** = `composite >= target`.

    Validation (loud, construction-time): every weight is a finite
    number `>= 0` (at least one strictly positive), `target` a finite
    number in `[0, 1]`. Returns `{"weights", "target", "pool",
    "passing", "pass", "final", "candidates"}` — `final` is the
    passing candidate with the highest composite (ties: lower
    `train_seconds`, then log order — the 40.2 convention), `None`
    when none pass; each candidate carries `cand` (1-based log
    position), `spec_hash`, `score`, `train`, `size`, the three
    normalized axes (`n_score` / `n_train` / `n_size`, `None` = axis
    absent), `composite`, and `pass`. Pure and deterministic (G2);
    an empty pool returns the zero shape (`pass: False`)."""
    ws = _weight("score", w_score)
    wt = _weight("train", w_train)
    wz = _weight("size", w_size)
    if ws == 0.0 and wt == 0.0 and wz == 0.0:
        raise ValueError(
            "at least one objective weight must be strictly positive "
            "(SPEC.md 60.4)")
    t = _num(target)
    if t is None or not (0.0 <= t <= 1.0):
        raise ValueError(
            f"the composite target must be a finite number in [0, 1], "
            f"got {target!r} (SPEC.md 60.4)")
    scored = _scored(entries)
    if not scored:
        return {"weights": {"score": ws, "train": wt, "size": wz},
                "target": float(t), "pool": 0, "passing": 0,
                "pass": False, "final": None, "candidates": []}
    scores = [s for _, s in scored]
    trains = [_num(e.get("train_seconds")) for e, _ in scored]
    trains = [x for x in trains if x is not None]
    sizes = [spec_n_params(e["spec"], state_dim, n_out, grid)
             for e, _ in scored]
    sizes = [s for s in sizes if isinstance(s, int) and s > 0]
    s_lo, s_hi = min(scores), max(scores)
    t_lo, t_hi = (min(trains), max(trains)) if trains else (0.0, 0.0)
    z_lo, z_hi = (min(sizes), max(sizes)) if sizes else (0, 0)
    cands = []
    for idx, (e, score) in enumerate(scored):
        n_score = _minmax(score, s_lo, s_hi)
        tr = _num(e.get("train_seconds"))
        n_train = None if tr is None else 1.0 - _minmax(tr, t_lo, t_hi)
        sz = spec_n_params(e["spec"], state_dim, n_out, grid)
        n_size = (None if not (isinstance(sz, int) and sz > 0)
                  else 1.0 - _minmax(float(sz), float(z_lo), float(z_hi)))
        num, den = 0.0, 0.0
        num += ws * n_score
        den += ws
        if n_train is not None and wt > 0.0:
            num += wt * n_train
            den += wt
        if n_size is not None and wz > 0.0:
            num += wz * n_size
            den += wz
        composite = num / den
        cands.append({
            "cand": idx + 1,
            "spec_hash": e.get("spec_hash"),
            "score": float(score),
            "train": tr,
            "size": sz,
            "n_score": n_score,
            "n_train": n_train,
            "n_size": n_size,
            "composite": float(composite),
            "pass": bool(composite >= t),
        })
    passing = [c for c in cands if c["pass"]]
    final = None
    if passing:
        def key(c: dict) -> tuple:
            tr = math.inf if c["train"] is None else c["train"]
            return (-c["composite"], tr, c["cand"])
        final = min(passing, key=key)
    return {
        "weights": {"score": ws, "train": wt, "size": wz},
        "target": float(t),
        "pool": len(cands),
        "passing": len(passing),
        "pass": bool(passing),
        "final": final,
        "candidates": cands,
    }
