"""Transfer learning: adapt a champion to similar data (SPEC.md 93.3, v0.79,
A83).

The user-facing "fine-tune" verb: a finished run's champion model is
adapted to a NEW CSV of similar rows (a later export of the same table,
a refreshed cohort, ...) and the before/after are scored on the NEW
data's own holdout — an apples-to-apples comparison (the 89.3.1 rule).
The adapted model is saved as a NEW, fully reconstructible run dir
(``summary.json`` + ``best_model.npz`` + ``adapt_report.json``) with a
registry entry whose ``parent_run`` is the champion (the 38.1.2
lineage field) — so the adapted model can itself become a future
champion (transfer chaining, 93.3.6).

Honesty rules (93.3.3/93.3.7):
* a champion that is incompatible with the new data (head / output
  count / input contract — the 80.2.3 check) fails loud with the
  reason;
* weights transfer only when the 80.3 warm-start check passes (the
  parametric families, matching shapes) — then the mode is
  ``finetune``; otherwise the champion's SPEC is retrained fresh and
  the mode is ``retrain`` (labeled as such, never silently a fresh
  init);
* G2: deterministic given (champion, data, seed); the only clock is
  the wall-time the training actually took.

Cycle-free (the ondemand pattern, 90): the champion reconstruction
lives in ``cli`` (its one home, 42), so it is imported lazily, inside
the function that needs it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ModelSpec
from .finetune import PARAMETRIC, Checkpoint, can_warm_start, validate_for_task
from .ondemand import csv_preflight  # 90.4 (A80): the pure CSV preflight
from .registry import append_entry, config_fingerprint  # 38.1
from .tasks.csv import CsvTask
from .trainer import train  # 80.3 (v0.66): the warm_start parameter

__all__ = ["plan_adapt", "adapt_model", "render_adapt_text", "ADAPT_EPS"]

# 93.3.4: the verdict margin (percentage points) — a delta within +/-
# ADAPT_EPS of the champion's own score is NEUTRAL (noise, not signal).
ADAPT_EPS = 0.5


def _champion(run_dir) -> tuple[dict, object, object, dict]:
    """(summary, task, model, best_spec dict) for a finished run — the
    ``cli._run_task_and_model`` one home (42), imported locally
    (cycle-free, the ondemand 90 pattern). A run dir that is not
    reconstructable is a ``ValueError`` (the ondemand 90.2 contract)."""
    from .cli import _run_task_and_model  # local (cycle-free)
    run_dir = Path(run_dir)
    try:
        summary, task, model = _run_task_and_model(run_dir)
    except (ValueError, FileNotFoundError):
        raise
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"champion run {run_dir} is not usable: {exc!r}") from exc
    spec = summary.get("best_spec") or {}
    if not spec.get("model_family"):
        raise ValueError(
            f"champion run {run_dir} has no best_spec.model_family — "
            "it is not a finished training run")
    return summary, task, model, dict(spec)


def _checkpoint(run_dir, model, spec_dict: dict) -> Checkpoint:
    """The champion as an 80.2 ``Checkpoint`` (so the 80.2.3/80.3
    compatibility + warm-start checks apply verbatim)."""
    return Checkpoint(
        path=str(Path(run_dir) / "best_model.npz"),
        family=str(spec_dict["model_family"]),
        model=model,
        spec=ModelSpec.from_dict(spec_dict),
        head=str(getattr(model, "head", "softmax")),
        n_out=int(getattr(model, "n_out", 2)),
    )


def _warm_ok(cp: Checkpoint, new_task) -> bool:
    """93.3.2: whether the champion's WEIGHTS can seed a model trained
    on the new data — the 80.3 warm-start check on the flat CSV path:
    the ``mlp`` family (the only parametric family whose input is flat
    features) with the same head, output count, activation,
    architecture, and input width (``state_dim * (2K + 1)`` with the
    champion's spectral order K). Everything else — non-parametric
    families (knn/tree/boost/gp/gam memorize rows: nothing to copy,
    80.3.3) and the grid/sequence families (convnet/conv1d/rnn, whose
    tasks are not CSVs) — is honestly a fresh retrain."""
    if cp.family != "mlp":
        return False
    if cp.head != getattr(new_task, "head", None):
        return False
    if cp.n_out != getattr(new_task, "n_outputs", None):
        return False
    K = int(cp.model.fourier_K)
    want = int(getattr(new_task, "state_dim", 0)) * (2 * K + 1)
    return bool(can_warm_start(cp, cp.spec, cp.n_out, cp.head, in_dim=want))


def _holdout_len(task) -> int:
    x, _y = task.holdout_rows(10**9, None)  # 28.2 protocol; clamps inside
    return int(x.shape[0])


def _per_class(task, model) -> list[dict] | None:
    """Per-class holdout accuracy for one model (softmax tasks only,
    the 28.2 applicability rule; one forward pass, G2)."""
    if getattr(task, "head", None) != "softmax":
        return None
    values = tuple(getattr(task, "class_values", ()) or ())
    if not values:
        return None
    x, y = task.holdout_rows(10**9, model)
    if x.shape[0] == 0:
        return None
    out = np.asarray(model.forward(np.atleast_2d(x)), dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    pred = out.argmax(axis=1)
    rows = []
    for i, v in enumerate(values):
        mask = (y == i)
        cnt = int(mask.sum())
        rows.append({
            "class": v,
            "n": cnt,
            "acc": (float(100.0 * (pred[mask] == i).mean()) if cnt else None),
        })
    return rows


def _weight_deltas(dst, src) -> list[dict] | None:
    """Per-layer L2 weight/bias movement (a warm start that did not
    move a layer shows 0.0 — the fine-tune left it alone). `None` when
    the two models are not layer-comparable (the honest absence)."""
    if not (hasattr(dst, "layers") and hasattr(src, "layers")):
        return None
    if len(dst.layers) != len(src.layers):
        return None
    out = []
    for (dw, db), (sw, sb) in zip(dst.layers, src.layers):
        dw, db = np.asarray(dw, dtype=np.float64), np.asarray(db, dtype=np.float64)
        sw, sb = np.asarray(sw, dtype=np.float64), np.asarray(sb, dtype=np.float64)
        if dw.shape != sw.shape or db.shape != sb.shape:
            return None
        out.append({"w_l2": float(np.linalg.norm(dw - sw)),
                    "b_l2": float(np.linalg.norm(db - sb))})
    return out


def _fresh_run_dir(runs_dir: Path, seed: int) -> Path:
    """The adapted run's dir name (the meta_env 498-504 pattern: a
    second never collides with an existing dir)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"adapt-csv-seed{seed}-{stamp}"
    out = runs_dir / base
    k = 1
    while out.exists():  # never mix two adapt runs into one directory
        k += 1
        out = runs_dir / f"{base}-{k}"
    return out


def plan_adapt(run_dir, data_path, seed: int = 7) -> dict:
    """93.3.1 (A83.A4): the DRY-RUN transfer plan — the champion's
    identity, the new data's preflight (90.4), the compatibility verdict
    (80.2.3), and the mode (``finetune`` | ``retrain``, 93.3.2). Pure
    read + one ``CsvTask`` construction (no training, no writes);
    ``ValueError`` / ``FileNotFoundError`` on a broken champion or an
    unusable CSV (the CLI exits rc 1 with the message)."""
    run_dir = Path(run_dir)
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"no such data file: {data_path}")
    summary, task, model, spec_dict = _champion(run_dir)
    if int(getattr(task, "max_steps", 1)) > 1:
        raise ValueError(
            f"the champion's task {getattr(task, 'name', '?')!r} is an episode "
            "task — adapt supports fitting tasks (csv) only (SPEC.md 93.3.4)")
    new_task = CsvTask(seed=int(seed), path=data_path)
    cp = _checkpoint(run_dir, model, spec_dict)
    ok, reason = validate_for_task(cp, new_task)
    warm = _warm_ok(cp, new_task)
    return {
        "champion_run": run_dir.name,
        "champion_family": cp.family,
        "champion_head": cp.head,
        "champion_n_out": cp.n_out,
        "champion_final_score": summary.get("final_best_score"),
        "champion_spec": dict(spec_dict),
        "new_data": str(data_path),
        "preflight": csv_preflight(data_path),
        "new_task": {
            "name": new_task.name,
            "state_dim": int(new_task.state_dim),
            "n_outputs": int(new_task.n_outputs),
            "head": new_task.head,
            "metric": getattr(new_task, "metric", None),
        },
        "compatible": bool(ok),
        "compatibility_reason": reason,
        "warm_start": bool(warm),
        "mode": "finetune" if warm else "retrain",
    }


def adapt_model(run_dir, data_path, runs_dir="runs", seed: int = 7,
                steps: int | None = None, max_seconds: float | None = 30.0,
                target: float = 95.0) -> dict:
    """93.3.2 (A83.A5): adapt the champion to the new data and score it
    before/after on the NEW data's holdout.

    Sequence (each step deterministic given the inputs, G2):
    1. reconstruct the champion (42) and the new ``CsvTask`` (22.1);
    2. the 80.2.3 compatibility check — incompatible data fails loud
       (93.3.3);
    3. **before** — the champion's official score (the 22.1 contract)
       on the new holdout;
    4. train the champion's spec on the new train split (``warm_start``
       = the champion's weights when 93.3.2 says they transfer, else a
       fresh init — the mode is labeled honestly);
    5. **after** — the same official score on the same new holdout;
    6. the verdict (IMPROVED / NEUTRAL / DEGRADED, the ADAPT_EPS
       margin), the per-class before/after, the per-layer weight
       deltas (warm starts only);
    7. artifacts — a new run dir (``summary.json`` + ``best_spec.json``
       + ``best_model.npz`` + ``adapt_report.json``) under ``runs_dir``
       and a 13-key registry entry (38.1.2) with
       ``parent_run`` = the champion (93.3.6).

    Returns the JSON-safe report dict (the one
    ``adapt_report.json`` writes)."""
    run_dir = Path(run_dir)
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"no such data file: {data_path}")
    summary, task, model, spec_dict = _champion(run_dir)
    if int(getattr(task, "max_steps", 1)) > 1:
        raise ValueError(
            f"the champion's task {getattr(task, 'name', '?')!r} is an episode "
            "task — adapt supports fitting tasks (csv) only (SPEC.md 93.3.4)")
    new_task = CsvTask(seed=int(seed), path=data_path)
    cp = _checkpoint(run_dir, model, spec_dict)
    ok, reason = validate_for_task(cp, new_task)
    if not ok:
        raise ValueError(
            f"incompatible champion for this data: {reason} (SPEC.md 93.3.3)")
    warm = _warm_ok(cp, new_task)
    mode = "finetune" if warm else "retrain"

    ho_n = _holdout_len(new_task)
    if ho_n < 2:
        raise ValueError(
            f"{data_path} has no usable holdout split — the before/after "
            "comparison needs >= 2 holdout rows (SPEC.md 93.3.3)")
    before = float(new_task.score(model, "holdout", ho_n))

    spec = cp.spec
    if steps is not None:
        d = spec.to_dict()
        d["train_steps"] = int(steps)
        spec = ModelSpec.from_dict(d)
    dataset = new_task.make_dataset()  # the train split (already standardized)
    t0 = time.perf_counter()
    res = train(
        dataset, spec, seed=int(seed),
        time_limit_seconds=(float(max_seconds) if max_seconds is not None
                            else None),
        n_out=int(new_task.n_outputs), head=str(new_task.head),
        warm_start=(cp.model if warm else None))
    new_model = res.model
    wall = time.perf_counter() - t0

    after = float(new_task.score(new_model, "holdout", ho_n))
    per_class_before = _per_class(new_task, model)
    per_class_after = _per_class(new_task, new_model)
    deltas = _weight_deltas(new_model, cp.model) if warm else None

    if after > before + ADAPT_EPS:
        verdict = "IMPROVED"
    elif after < before - ADAPT_EPS:
        verdict = "DEGRADED"
    else:
        verdict = "NEUTRAL"
    met = bool(after >= float(target))

    # --- artifacts (93.3.6): a new, reconstructible run + lineage -------
    runs_dir = Path(runs_dir)
    out_dir = _fresh_run_dir(runs_dir, int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    new_model.save(str(out_dir / "best_model.npz"))

    run_config = {
        "policy": mode,
        "parent_run": run_dir.name,
        "seed": int(seed),
        "steps": (int(steps) if steps is not None else None),
        "max_seconds": (float(max_seconds) if max_seconds is not None else None),
        "target": float(target),
    }
    summary_out = {
        "task": new_task.name,
        "seed": int(seed),
        "task_config": {"path": str(data_path)},
        "best_spec": spec.to_dict(),
        "final_best_score": after,
        "baseline_score": before,
        "target": float(target),
        "met_target": met,
        "finished_reason": "adapted",
        "policy": mode,
        "experiments_run": 1,
        "wall_seconds": round(wall, 3),
        "run_config": run_config,
        "adapt": {
            "parent_run": run_dir.name,
            "champion_final_score": summary.get("final_best_score"),
            "mode": mode,
            "warm_start": bool(warm),
            "verdict": verdict,
            "before": before,
            "after": after,
            "n_holdout": int(ho_n),
            "wall_seconds": round(wall, 3),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8")
    (out_dir / "best_spec.json").write_text(
        json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    report = {
        "parent_run": run_dir.name,
        "new_data": str(data_path),
        "seed": int(seed),
        "mode": mode,
        "warm_start": bool(warm),
        "champion_family": cp.family,
        "champion_head": cp.head,
        "champion_final_score": summary.get("final_best_score"),
        "before": before,
        "after": after,
        "delta": float(after - before),
        "verdict": verdict,
        "n_holdout": int(ho_n),
        "target": float(target),
        "met_target": met,
        "steps_requested": (int(steps) if steps is not None else None),
        "max_seconds": (float(max_seconds) if max_seconds is not None else None),
        "steps_run": int(res.steps_run),
        "wall_seconds": round(wall, 3),
        "per_class_before": per_class_before,
        "per_class_after": per_class_after,
        "weight_deltas": deltas,
        "champion_spec": dict(spec_dict),
        "run_dir": out_dir.name,
        "run_path": str(out_dir),
    }
    (out_dir / "adapt_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    entry = {
        "run_id": out_dir.name,
        "task": new_task.name,
        "seed": int(seed),
        "policy": mode,
        "final_score": after,
        "target": float(target),
        "met_target": met,
        "finished_reason": "adapted",
        "experiments_run": 1,
        "wall_seconds": round(wall, 3),
        "config_fp": config_fingerprint(run_config),
        "parent_run": run_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_entry(runs_dir, entry)
    return report


def render_adapt_text(r: dict) -> str:
    """93.3.5: the CLI narrative of one adapt report — deterministic,
    ASCII (the house console rule)."""
    mode_line = r["mode"]
    if r.get("warm_start"):
        mode_line += " (warm start from the champion's weights)"
    else:
        mode_line += " (weights cannot transfer - the champion's spec, fresh init)"
    lines = [
        f"champion : {r['parent_run']} ({r.get('champion_family', '?')}, "
        f"head={r.get('champion_head', '?')})",
        f"data     : {r['new_data']} (holdout n={r['n_holdout']})",
        f"mode     : {mode_line}",
        f"before   : {r['before']:.1f} (the champion on the new holdout)",
        f"after    : {r['after']:.1f} (the adapted model on the same holdout)",
        f"verdict  : {r['verdict']} ({r['delta']:+.1f})",
        f"target   : {r['target']:.1f} -> "
        f"{'met' if r.get('met_target') else 'not met'}",
        f"run dir  : {r['run_path']}",
    ]
    return "\n".join(lines)
