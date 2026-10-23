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

from .config import ModelSpec, SpecError
from .finetune import PARAMETRIC, Checkpoint, can_warm_start, validate_for_task
from .ondemand import csv_preflight  # 90.4 (A80): the pure CSV preflight
from .registry import append_entry, config_fingerprint  # 38.1
from .tasks.csv import CsvTask
from .trainer import train  # 80.3 (v0.66): the warm_start parameter

__all__ = ["plan_adapt", "adapt_model", "cross_modal_adapt",
           "render_adapt_text", "ADAPT_EPS"]

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


def _eval_n(task) -> int:
    """The sample size for one task's `score(model, split, n)` (SPEC.md 99,
    v0.85 B4/C2): the holdout length for file-based tasks (the 22.2
    protocol), and the task's `default_dataset_size` (else 256) for
    synthetic tasks, which score on fresh seed-derived points (the
    ParityTask / SequenceMotifV1 pattern)."""
    fn = getattr(task, "holdout_rows", None)
    if fn is not None:
        try:
            x, _y = fn(10**9, None)
            n = int(x.shape[0])
            if n >= 2:
                return n
        except Exception:  # fall through to the synthetic rule
            pass
    n = getattr(task, "default_dataset_size", None)
    if isinstance(n, int) and not isinstance(n, bool) and n >= 2:
        return int(n)
    return 256


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
                target: float = 95.0,
                forget_tol: float | None = None) -> dict:
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

    ``forget_tol`` (SPEC.md 99, v0.85 B4; ``None`` = off, the exact
    legacy path): the forgetting guard — after adapting, the model is
    re-scored on the CHAMPION task's own holdout and the drop vs the
    champion's recorded final score is the forgetting; ``ok`` is
    ``drop <= forget_tol``. An incompatible original task (the model can
    no longer be scored there) degrades to an honest measured-None
    report, never an exception.

    Returns the JSON-safe report dict (the one
    ``adapt_report.json`` writes)."""
    if forget_tol is not None and (
            isinstance(forget_tol, bool)
            or not isinstance(forget_tol, (int, float))
            or not (0.0 <= float(forget_tol) < float("inf"))):
        raise ValueError(
            f"forget_tol must be a finite number >= 0 (or None = off), "
            f"got {forget_tol!r} (SPEC.md 99)")
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

    # SPEC.md 99 (v0.85) B4: the forgetting guard — did adapting cost the
    # champion its ORIGINAL task? One extra scoring pass on the original
    # holdout (opt-in; None = off, zero legacy cost). The drop vs the
    # champion's recorded final score is the forgetting; the adapted model
    # that can no longer be scored on the original task degrades to an
    # honest measured-None report (never an exception).
    forgetting = None
    if forget_tol is not None:
        orig_before = summary.get("final_best_score")
        orig_after = None
        try:
            orig_after = float(task.score(new_model, "holdout", _eval_n(task)))
        except Exception:  # measured absence — reported, not fatal
            orig_after = None
        drop = None
        if (isinstance(orig_before, (int, float))
                and not isinstance(orig_before, bool)
                and orig_after is not None):
            drop = float(orig_before) - orig_after
        ok = (drop <= float(forget_tol)) if drop is not None else None
        forgetting = {
            "forget_tol": float(forget_tol),
            "orig_task": getattr(task, "name", None),
            "orig_before": (float(orig_before)
                            if isinstance(orig_before, (int, float))
                            and not isinstance(orig_before, bool) else None),
            "orig_after": orig_after,
            "drop": drop,
            "ok": ok,
        }

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
            "forgetting": forgetting,  # SPEC.md 99 (v0.85) B4
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
        "forgetting": forgetting,  # SPEC.md 99 (v0.85) B4
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


def cross_modal_adapt(src_run_dir, target_task, seed: int = 7,
                      steps: int | None = None,
                      max_seconds: float | None = 30.0,
                      runs_dir="runs") -> dict:
    """SPEC.md 99 (v0.85) C2 — cross-modality warm-start adaptation.

    The champion of a finished run is adapted to a task of a DIFFERENT
    modality (e.g. an image champion onto an audio task, or a flat
    champion onto a sequence task): the target family is derived from
    the target task's capabilities (grid -> convnet, sequence -> conv1d,
    else mlp), the champion's spec is re-expressed for that family
    (a champion whose architecture cannot be expressed there falls back
    to the mlp surface), and ``train(..., warm_start=champion,
    cross_modal=True)`` seeds it with the partial shape-mapped transfer
    (the 99 C2 fallback; the strict 80.3 path is unchanged). Before/after
    are the champion's and the adapted model's official scores on the
    TARGET holdout (each an honest ``None`` when the model cannot be
    scored there — a modality mismatch, never an exception).

    Artifacts mirror ``adapt_model``: a fresh run dir (summary.json +
    best_spec.json + best_model.npz + warmstart_report.json) and a
    registry entry with ``parent_run`` = the source run (93.3.6
    lineage). G2: deterministic given (champion, target, seed); the only
    clock is wall time."""
    src_run_dir = Path(src_run_dir)
    summary, src_task, src_model, spec_dict = _champion(src_run_dir)
    target_name = getattr(target_task, "name", "<task>")

    # --- target family (capabilities -> family, the 25.3/83.1 rule) -------
    caps = frozenset(getattr(target_task, "capabilities", ()) or ())
    if getattr(target_task, "grid_capable", False):
        preferred = "convnet"
    elif "sequence" in caps:
        preferred = "conv1d"
    else:
        preferred = "mlp"
    d = dict(spec_dict)
    if steps is not None:
        d["train_steps"] = int(steps)
    d["model_family"] = preferred
    try:
        target_spec = ModelSpec.from_dict(d)
        target_family = preferred
    except SpecError:
        # the champion's architecture cannot be expressed in the
        # preferred family — the mlp surface always can (loudly so when
        # even that fails, e.g. a depth-3 champion)
        d["model_family"] = "mlp"
        target_spec = ModelSpec.from_dict(d)
        target_family = "mlp"

    # --- before: the champion on the target holdout (honest None) --------
    before = None
    try:
        before = float(target_task.score(src_model, "holdout", _eval_n(target_task)))
    except Exception:  # measured absence — reported, not fatal
        before = None

    # --- train the target (cross-modal warm start) -------------------------
    if target_family == "convnet":
        gd = getattr(target_task, "grid_dataset", None)
        if gd is None:
            raise ValueError(
                f"target task {target_name!r} is grid_capable but exposes "
                "no grid_dataset() (SPEC.md 99)")
        size = int(getattr(target_task, "default_dataset_size", 60) or 60)
        dataset = gd(size)
    else:
        dataset = target_task.make_dataset()
    t0 = time.perf_counter()
    res = train(
        dataset, target_spec, seed=int(seed),
        time_limit_seconds=(float(max_seconds) if max_seconds is not None
                            else None),
        n_out=int(target_task.n_outputs), head=str(target_task.head),
        warm_start=src_model, cross_modal=True)
    new_model = res.model
    wall = time.perf_counter() - t0

    after = None
    try:
        after = float(target_task.score(new_model, "holdout", _eval_n(target_task)))
    except Exception:  # measured absence — reported, not fatal
        after = None

    warm_report = getattr(new_model, "xmodal_report", None)
    if warm_report is None:
        # the strict 80.3 path succeeded — every slot matched and copied
        # (copy_weights checks all shapes before any mutation)
        warm_report = {"mode": "strict-copy",
                       "copied": len(getattr(new_model, "layers", ())),
                       "reshaped": 0, "kept_init": 0}

    # --- artifacts (93.3.6 lineage, the adapt_model pattern) --------------
    runs_dir = Path(runs_dir)
    out_dir = _fresh_run_dir(runs_dir, int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    new_model.save(str(out_dir / "best_model.npz"))

    run_config = {
        "policy": f"cross_modal_{target_family}",
        "parent_run": src_run_dir.name,
        "seed": int(seed),
        "steps": (int(steps) if steps is not None else None),
        "max_seconds": (float(max_seconds) if max_seconds is not None else None),
    }
    cross_block = {
        "parent_run": src_run_dir.name,
        "src_task": getattr(src_task, "name", None),
        "src_family": str(spec_dict.get("model_family")),
        "target_task": target_name,
        "target_family": target_family,
        "before": before,
        "after": after,
        "warm_report": warm_report,
        "wall_seconds": round(wall, 3),
    }
    summary_out = {
        "task": target_name,
        "seed": int(seed),
        "best_spec": target_spec.to_dict(),
        "final_best_score": after,
        "baseline_score": before,
        "finished_reason": "cross_modal_adapted",
        "policy": f"cross_modal_{target_family}",
        "experiments_run": 1,
        "wall_seconds": round(wall, 3),
        "run_config": run_config,
        "cross_modal": cross_block,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8")
    (out_dir / "best_spec.json").write_text(
        json.dumps(target_spec.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "warmstart_report.json").write_text(
        json.dumps(warm_report, indent=2), encoding="utf-8")

    report = {
        "parent_run": src_run_dir.name,
        "src_task": getattr(src_task, "name", None),
        "src_family": str(spec_dict.get("model_family")),
        "target_task": target_name,
        "target_family": target_family,
        "seed": int(seed),
        "warm_start": True,
        "warm_report": warm_report,
        "before": before,
        "after": after,
        "steps_run": int(res.steps_run),
        "wall_seconds": round(wall, 3),
        "run_dir": out_dir.name,
        "run_path": str(out_dir),
    }
    entry = {
        "run_id": out_dir.name,
        "task": target_name,
        "seed": int(seed),
        "policy": f"cross_modal_{target_family}",
        "final_score": after,
        "target": None,
        "met_target": None,
        "finished_reason": "cross_modal_adapted",
        "experiments_run": 1,
        "wall_seconds": round(wall, 3),
        "config_fp": config_fingerprint(run_config),
        "parent_run": src_run_dir.name,
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
    forgetting_line = None
    if r.get("forgetting") is not None:
        f = r["forgetting"]
        if f.get("drop") is None:
            forgetting_line = "forget   : n/a (the original task could not be re-scored)"
        else:
            status = "kept" if f.get("ok") else "FORGOT"
            forgetting_line = (f"forget   : {status} (drop {f['drop']:.1f} "
                               f"vs tol {f['forget_tol']:.1f} on the "
                               f"original task)")
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
    ]
    if forgetting_line is not None:
        lines.append(forgetting_line)
    lines.append(f"run dir  : {r['run_path']}")
    return "\n".join(lines)
