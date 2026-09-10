"""SPEC.md 40 (v0.26, S1/S2) + 41 (v0.27, S3/S4): simulation helpers —
pure, no training.

- ``what_if`` (40.2) re-evaluates a finished run's *logged* candidates
  against a NEW objective set (the §37.2 ``evaluate`` rule) and picks the
  counterfactual final — "what would my final model have been under a
  tighter bar?" — with zero retraining.
- ``estimate_wall`` (40.1) derives a wall-time estimate from the run
  registry's same-task history for ``fit --dry-run``.
- ``project_budget`` / ``projection_points`` (41.1) — S3: budget
  projection, "will I hit 95?" — a saturating curve over the same-task
  history (41.1.2 points), ceiling or ~N-more-experiments verdict (41.1.4).
- ``trace_lines`` (41.2) — S4: the line-per-experiment decision trace;
  the *same renderer* ``run --demo`` uses (41.3) — one renderer, two
  entry points.

House rules: stdlib + numpy (a core dependency) only (``gate.evaluate``
is the one shared evaluator — no second implementation); import-cycle-free
(imports ``.gate``, ``.memory``, and the single constant ``GEN_GAP_TOL``
from ``.improver.meta_env`` — the same one home ``accounting`` uses). The
§37.2 ``fit --gate`` path still evaluates ``model`` on the saved artifact;
``what_if`` refuses to guess at ``model`` size (40.2.3), so it rejects a
``model`` objective rather than invent a second size calculator.

SPEC.md 49.1 (v0.35, advanced analysis): ``what_if_block`` — the
app's what-if half — is the honest union of the two existing CLI gate
surfaces: the score/train objectives re-gate the logged pool via
``what_if`` (40.2) while a model objective is evaluated against the
final best model's artifact via ``gate.evaluate`` (37.2).
"""
from __future__ import annotations

import math

import numpy as np

from .gate import evaluate
from .improver.meta_env import GEN_GAP_TOL  # §18.5 tolerance (0.05); one home
from .memory import (
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
    KIND_INVALID_SPEC,
    KIND_SCREEN,
)

__all__ = [
    "what_if",
    "what_if_block",
    "estimate_wall",
    "project_budget",
    "projection_points",
    "trace_lines",
]


def _finite_number(v) -> bool:
    """A finite int/float that is not a bool (JSON-safe numeric check)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def what_if(entries, objectives) -> dict:
    """SPEC.md 40.2 (v0.26, S2): re-gate the run's logged history.

    The candidate pool (40.2.2) is every scored row — ``kind`` in
    {baseline, experiment} with a finite ``holdout_score`` — each evaluated
    against ``objectives`` (the 37.2 ``evaluate`` rule) on the logged actuals
    ``score = holdout_score``, ``train = train_seconds`` (a missing actual
    fails its objective). The counterfactual **final** is the passing
    candidate with the highest ``holdout_score`` (ties: lower
    ``train_seconds``, then log order). The verdict is PASS when >= 1
    candidate passes, MISS when none.

    ``objectives`` must be §37.2 ``Objective`` objects. A ``model``
    objective is a construction error here (40.2.3) — it raises
    ``ValueError``. Pure and deterministic: ``what_if(e, o) == what_if(e, o)``.
    """
    for o in objectives:
        if o.name == "model":
            raise ValueError(
                "what-if cannot gate on 'model': a candidate's model size is "
                "only known from its trained artifact (the §37.2 `fit --gate` "
                "path evaluates it on the saved best model); use `score` and "
                "`train` here (SPEC.md 40.2.3)")
    candidates: list[dict] = []
    for idx, e in enumerate(entries):
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):
            continue
        hs = e.get("holdout_score")
        if not _finite_number(hs):
            continue
        train_raw = e.get("train_seconds")
        train = float(train_raw) if _finite_number(train_raw) else None
        res = evaluate(objectives, {"score": float(hs), "train": train})
        candidates.append({
            "cand": idx + 1,  # 1-based position in the log
            "spec_hash": e.get("spec_hash"),
            "score": float(hs),
            "train": train,
            "pass": bool(res["pass"]),
        })
    passing = [c for c in candidates if c["pass"]]
    final = None
    if passing:
        def key(c: dict) -> tuple:
            tr = math.inf if c["train"] is None else c["train"]
            return (-c["score"], tr, c["cand"])
        final = min(passing, key=key)
    return {
        "pool": len(candidates),
        "passing": len(passing),
        "pass": bool(passing),
        "final": final,
        "candidates": candidates,
    }


def what_if_block(entries, objectives, model_actual: float | None = None) -> dict:
    """SPEC.md 49.1.1–49.1.3 (v0.35): the combined what-if verdict.

    Splits the objective set into a non-model half (``score``/``train``)
    and a model half. The non-model half is the existing ``what_if``
    (40.2) — the logged candidate pool, the counterfactual final, zero
    training; the model half is ``gate.evaluate`` (37.2) against
    ``model_actual`` (the final best model's artifact size, 49.1.2 —
    per-candidate sizes are not logged, 40.2.3). The result is the
    JSON-safe dict ``{"pass", "what_if", "model"}``: a half with no
    objectives is ``None`` (vacuously true), and ``pass`` is the AND of
    the two halves. An empty objective set is a MISS: ``{"pass":
    False, "what_if": None, "model": None}``.

    A missing ``model_actual`` fails the model objective honestly (the
    37.2 rule: a missing actual fails its objective) — never guessed.
    Pure and deterministic (G2): same inputs, same dict; no training,
    no writes.
    """
    objs = tuple(objectives or ())
    if not objs:
        return {"pass": False, "what_if": None, "model": None}
    non_model = [o for o in objs if o.name != "model"]
    model = [o for o in objs if o.name == "model"]
    wf = what_if(entries, non_model) if non_model else None
    mr = evaluate(tuple(model), {"model": model_actual}) if model else None
    ok = (wf is None or bool(wf["pass"])) and (mr is None or bool(mr["pass"]))
    return {"pass": bool(ok), "what_if": wf, "model": mr}


def estimate_wall(registry_entries, task_name, max_experiments) -> dict:
    """SPEC.md 40.1.2 (v0.26, S1): a wall-time estimate for ``fit --dry-run``.

    The median of ``wall_seconds / experiments_run`` across the registry's
    finished runs of the *same* task (each with ``experiments_run > 0`` and
    ``wall_seconds > 0``) times ``max_experiments``. Returns
    ``{"estimate": <float>, "runs": N}`` when history exists, else
    ``{"estimate": None, "runs": 0}`` (the caller prints the fallback line).
    Pure and deterministic.
    """
    per_exp: list[float] = []
    for e in registry_entries:
        if e.get("task") != task_name:
            continue
        er = e.get("experiments_run")
        ws = e.get("wall_seconds")
        if _finite_number(er) and er > 0 and _finite_number(ws) and ws > 0:
            per_exp.append(float(ws) / float(er))
    if not per_exp:
        return {"estimate": None, "runs": 0}
    per_exp.sort()
    n = len(per_exp)
    median = (per_exp[n // 2] if n % 2 == 1
              else (per_exp[n // 2 - 1] + per_exp[n // 2]) / 2.0)
    return {"estimate": median * max_experiments, "runs": n}


def _positive_finite(value) -> bool:
    """A finite positive int/float that is not a bool (41.1.2 usability)."""
    return _finite_number(value) and float(value) > 0.0


def project_budget(points, target, e_current) -> dict:
    """SPEC.md 41.1.3/41.1.4 (v0.27, S3): project the budget to a target.

    Fits the Michaelis–Menten saturation curve ``score(e) = Vmax·e/(Km+e)``
    to ``points`` (``(experiments, score)`` pairs, 41.1.2) by the
    Lineweaver–Burk linearization (OLS of ``1/score`` on ``1/experiments``;
    ``Vmax = 1/b``, ``Km = a/b``). Verdict: ``Vmax <= target`` ->
    ``"ceiling"``; otherwise the curve reaches the target at
    ``e_target = target*Km/(Vmax - target)`` and the answer is
    ``max(0, ceil(e_target - e_current))`` more experiments (a 1e-9
    epsilon guards the ceiling against float noise when ``e_T`` is exact).
    Degenerate —
    < 2 usable points, a non-finite or non-positive intercept, or a
    negative ``Km`` — -> ``"insufficient"`` (the view degrades gracefully,
    41.1.4).

    Pure and deterministic: same inputs, same dict; no training, no writes.
    """
    usable = [
        (float(e), float(s))
        for e, s in (points or [])
        if _positive_finite(e) and _positive_finite(s)
    ]
    base = {"points": len(usable), "vmax": None, "km": None,
            "e_target": None, "more": None}
    if len(usable) < 2:
        return dict(base, ok=False, verdict="insufficient")
    x = np.array([1.0 / e for e, _s in usable])
    y = np.array([1.0 / s for _e, s in usable])
    coef, _res, _rank, _sv = np.linalg.lstsq(
        np.column_stack((x, np.ones_like(x))), y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if (not math.isfinite(a) or not math.isfinite(b) or b <= 0.0):
        return dict(base, ok=False, verdict="insufficient")
    vmax, km = 1.0 / b, a / b
    if (not math.isfinite(vmax) or vmax <= 0.0
            or not math.isfinite(km) or km < 0.0):
        return dict(base, ok=False, verdict="insufficient")
    if vmax <= float(target):
        return dict(base, ok=True, verdict="ceiling", vmax=vmax, km=km)
    e_target = float(target) * km / (vmax - float(target))
    more = max(0, math.ceil(e_target - float(e_current) - 1e-9))
    return dict(base, ok=True, verdict="more", vmax=vmax, km=km,
                e_target=e_target, more=int(more))


def projection_points(registry_entries, summary, run_id=None) -> list:
    """SPEC.md 41.1.2 (v0.27, S3): the curve's (experiments, score) points.

    The registry's finished same-task runs (``task == summary["task"]``,
    finite ``experiments_run > 0`` and ``final_score > 0``), plus the
    current run's own ``(experiments_run, final_best_score)`` — deduped by
    run id (41.1.2): when the current run already has a registry entry,
    the registry row is the point and the summary's is not added a second
    time. Pure.
    """
    summary = summary if isinstance(summary, dict) else {}
    task = summary.get("task")
    if task is None:
        return []
    points: list[tuple[float, float]] = []
    ids = set()
    for e in registry_entries or []:
        if not isinstance(e, dict) or e.get("task") != task:
            continue
        er, fs = e.get("experiments_run"), e.get("final_score")
        if _positive_finite(er) and _positive_finite(fs):
            points.append((float(er), float(fs)))
            ids.add(e.get("run_id"))
    er, fs = summary.get("experiments_run"), summary.get("final_best_score")
    if (run_id not in ids
            and _positive_finite(er) and _positive_finite(fs)):
        points.append((float(er), float(fs)))
    return points


def _reject_reason(entry: dict, best: float | None) -> str:
    """SPEC.md 41.2.2: the documented 39.2.2 rejection priority — score
    gate, then overfit (18.5 ``GEN_GAP_TOL``), then the CI gate."""
    score = entry.get("holdout_score")
    gap = entry.get("gen_gap")
    if best is not None and _finite_number(score) and float(score) <= best:
        return f"score gate (<= running best {best:.2f})"
    if (_finite_number(score) and _finite_number(gap)
            and float(gap) > GEN_GAP_TOL * float(score)):
        return f"overfit (gen-gap above {GEN_GAP_TOL:.0%} tolerance)"
    return "CI gate (delta-eff within z*SE)"


def trace_lines(entries) -> list:
    """SPEC.md 41.2.2 (v0.27, S4): one decision line per log entry.

    Renders ``experiments.jsonl`` in log order: the baseline seed
    champion; each experiment's mutation, holdout score, and gen-gap with
    its ACCEPTED / REJECTED (reason) decision; curriculum step-ups (they
    re-pin the running best, 39.2.2); screen rejections; invalid specs.
    The running best is reconstructed exactly as in 39.2.2 (baseline
    seeds it, an acceptance raises it, a curriculum step-up re-pins it).

    Pure and deterministic; the same renderer ``run --demo`` uses (41.3).
    """
    lines: list[str] = []
    best: float | None = None
    for i, e in enumerate(entries or [], start=1):
        kind = e.get("kind", "?")
        score = e.get("holdout_score")
        if kind == KIND_BASELINE:
            s = f"{float(score):.2f}" if _finite_number(score) else "n/a"
            lines.append(
                f"#{i} [baseline]    score {s} -> BASELINE (seed champion)")
            if _finite_number(score):
                best = float(score)
        elif kind == KIND_EXPERIMENT:
            mut = ",".join(e.get("mutation") or []) or "-"
            s = f"{float(score):.2f}" if _finite_number(score) else "n/a"
            gap = e.get("gen_gap")
            g = f"{float(gap):.4f}" if _finite_number(gap) else "n/a"
            head = (f"#{i} [experiment] mutation={mut} "
                    f"score {s} gen_gap {g} -> ")
            if e.get("accepted"):
                lines.append(head + "ACCEPTED (new best)")
                if _finite_number(score):
                    best = float(score)
            else:
                lines.append(head + f"REJECTED ({_reject_reason(e, best)})")
        elif kind == KIND_CURRICULUM:
            nb = e.get("new_baseline_score")
            s = f"{float(nb):.2f}" if _finite_number(nb) else "n/a"
            lines.append(
                f"#{i} [curriculum] step-up -> new baseline {s} "
                "(re-pins the running best)")
            if _finite_number(nb):
                best = float(nb)
        elif kind == KIND_SCREEN:
            s = f"{float(score):.2f}" if _finite_number(score) else "n/a"
            lines.append(
                f"#{i} [screen]      score {s} -> SCREEN-REJECTED "
                "(below the champion; not a full candidate)")
        elif kind == KIND_INVALID_SPEC:
            lines.append(
                f"#{i} [invalid_spec] -> REJECTED (invalid spec: "
                f"{e.get('error', '?')})")
        else:
            lines.append(f"#{i} [{kind}]")
    return lines
