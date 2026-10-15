"""Portable run export (SPEC.md 92.10, B3, v0.78) — bundle a finished run
into one self-contained zip a colleague can run with *numpy only*.

`export_run(run_dir, out_path)` writes a deterministic zip:

- `best_model.npz` — the run's champion weights (copied byte-for-byte);
- `model_config.json` — family, head, state dim, feature names, the
  task's train-only standardization stats, and the class values, so the
  standalone predictor applies exactly the same preprocessing the task
  used at fit time (the 42.1.3 rule);
- `predict.py` — a standalone, pure-numpy predictor for the four
  plain-array families (mlp / knn / tree / boost) — it re-implements
  each family's `forward` on the saved npz, so no autorefine import is
  needed;
- `README.md` — what this is, how to predict, and (for non-portable
  families such as convnet/gp/gam) the note that those checkpoints still
  need the autorefine library.

The zip is deterministic: fixed member order and a fixed 1980-01-01
timestamp, so two exports of the same run are byte-identical (G2).
Cycle-free: imports only stdlib + numpy + the task registry (never the
cli), so `cli` may import this module at top level.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .tasks import TASKS
from .tasks.cartpole import CartPoleV1
from .tasks.parity import ParityTask
from .tasks.sine import SineRegressionV1

__all__ = ["export_run", "PORTABLE_FAMILIES"]

# SPEC.md 92.10: families whose checkpoints are plain arrays and whose
# forward is re-implementable in the standalone predict.py.
PORTABLE_FAMILIES = ("mlp", "knn", "tree", "boost")

# deterministic zip timestamp (the minimum the zip format allows)
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _reconstruct_task(summary: dict):
    """The task a run trained (the cli `_task_from_summary` rule, kept
    here so this module never imports the cli)."""
    task_name = summary.get("task")
    seed = int(summary["seed"])
    cfg = summary.get("task_config") or {}
    if task_name in TASKS:
        return TASKS[task_name](seed=seed, **cfg) if cfg \
            else TASKS[task_name](seed=seed)
    if summary.get("curriculum"):  # SPEC.md 20.1: a curriculum level
        lvl = summary["curriculum"]["levels"][-1]
        if "n_bits" in lvl and "p_flip" in lvl:
            return ParityTask(seed=seed, n_bits=lvl["n_bits"], p_flip=lvl["p_flip"])
        if "noise" in lvl and "freq_scale" in lvl and "amplitude" in lvl:
            return SineRegressionV1(
                seed, noise=lvl["noise"], freq_scale=lvl["freq_scale"],
                amplitude=lvl["amplitude"])
        if "ic_scale" in lvl:
            return CartPoleV1(seed, ic_scale=lvl["ic_scale"])
    return None


def _config_dict(summary: dict, family: str, task) -> dict:
    cfg: dict = {
        "model_family": family,
        "task": summary.get("task"),
        "head": getattr(task, "head", None),
        "state_dim": int(getattr(task, "state_dim", 0)) or None,
        "feature_names": list(getattr(task, "feature_names", None) or []),
        "class_values": list(getattr(task, "class_values", None) or []),
        "portable": family in PORTABLE_FAMILIES,
        "autorefine_version": _version(),
    }
    mean = getattr(task, "feature_mean", None)
    std = getattr(task, "feature_std", None)
    if mean is not None and std is not None:
        cfg["feature_mean"] = [float(v) for v in mean]
        cfg["feature_std"] = [float(v) for v in std]
    return cfg


def _version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _add(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def export_run(run_dir: Path, out_path: Path) -> dict:
    """Bundle `run_dir`'s best model into a portable zip at `out_path`.

    Returns `{"portable": bool, "family": str, "out": str, "files": [...]}`
    for the CLI/tests. Raises ValueError on a broken run dir (missing
    summary.json or best_model.npz) or an unknown task with a portable
    family (its preprocessing stats are unrecoverable).
    """
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"no summary.json in run dir {str(run_dir)!r}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_path = run_dir / "best_model.npz"
    if not model_path.is_file():
        raise ValueError(f"no best_model.npz in run dir {str(run_dir)!r}")
    family = (summary.get("best_spec") or {}).get("model_family", "mlp")
    task = _reconstruct_task(summary)
    portable = family in PORTABLE_FAMILIES
    if portable and task is None:
        raise ValueError(
            f"cannot export run {str(run_dir)!r}: its task "
            f"{summary.get('task')!r} is not reconstructible, so its "
            "preprocessing stats are unknown to a standalone predictor")
    config = _config_dict(summary, family, task) if task is not None \
        else _config_dict(summary, family, None)

    files = ["best_model.npz", "model_config.json"]
    if portable:
        files.append("predict.py")
    files.append("README.md")

    readme = _readme_text(summary, family, portable, task)

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _add(zf, "best_model.npz", model_path.read_bytes())
        _add(zf, "model_config.json",
             json.dumps(config, indent=2, sort_keys=True,
                        allow_nan=False).encode("utf-8"))
        if portable:
            _add(zf, "predict.py", _PREDICT_TEMPLATE.encode("utf-8"))
        _add(zf, "README.md", readme.encode("utf-8"))
    return {
        "portable": portable,
        "family": family,
        "out": str(out_path),
        "files": files,
    }


def _readme_text(summary: dict, family: str, portable: bool, task) -> str:
    head = getattr(task, "head", None)
    lines = [
        "# AutoRefine portable model export",
        "",
        f"- task: {summary.get('task')}",
        f"- model family: {family}",
        f"- head: {head}",
        f"- final best score: {summary.get('final_best_score')}",
        f"- autorefine version: {_version()}",
        "",
        "## Contents",
        "",
        "- `best_model.npz` - the champion model (plain numpy arrays)",
        "- `model_config.json` - the model/task metadata the predictor needs",
        "- `predict.py` - standalone numpy-only predictor"
        if portable else
        "- `predict.py` is NOT included: this family's forward pass needs "
        "the autorefine library",
        "",
    ]
    if portable:
        lines += [
            "## Usage",
            "",
            "    pip install numpy",
            "    python predict.py new_data.csv     # one JSON line per row",
            '    python predict.py --row "1.0, 2.5, 0"',
            "",
            "Each output line is `\"prediction\"` (the model's class/value) "
            "and `\"probabilities\"` (softmax-head models).",
            "",
            "CSV rows are read with the same header/positional rules the "
            "run used at fit time; if the run had a label column, keep it "
            "in the header - extra columns are ignored. New rows are "
            "standardized with the run's own train-time stats, exactly as "
            "the model was fitted.",
            "",
            "Media-task note: predicting raw media *files* (images/audio) "
            "still needs the autorefine library; `predict.py` scores "
            "feature vectors (CSV rows) only.",
            "",
        ]
    else:
        lines += [
            "## Usage",
            "",
            f"This {family} checkpoint loads through autorefine's "
            "`autorefine predict --run` / `autorefine eval` verbs instead "
            "of a standalone script.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The standalone predictor (written verbatim into the export zip). Pure
# numpy; it reads best_model.npz + model_config.json next to itself.
# ---------------------------------------------------------------------------
_PREDICT_TEMPLATE = '''\
#!/usr/bin/env python3
"""Standalone predictor exported from an AutoRefine run. Pure numpy.

Usage:
    python predict.py new_data.csv       # one JSON line per CSV row
    python predict.py --row "1.0,2.5,0"  # a single positional row
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "model_config.json").read_text(encoding="utf-8"))
VOTE_EPS = 1e-6


def _finite(v):
    f = float(v)
    if not np.isfinite(f):
        raise ValueError(f"row value {v!r} is not a finite number")
    return f


def _fourier(x, K, scale):
    if K <= 0:
        return x
    x_t = x / scale
    cols = [x]
    for k in range(1, K + 1):
        cols.append(np.cos(k * x_t))
        cols.append(np.sin(k * x_t))
    return np.column_stack(cols)


def _tree_predict(z, keys, x):
    feat = z[keys[0]].astype(np.int64)
    thr = z[keys[1]].astype(np.float64)
    left = z[keys[2]].astype(np.int64)
    right = z[keys[3]].astype(np.int64)
    leafv = z[keys[4]].astype(np.float64)
    n = x.shape[0]
    out = np.zeros((n, leafv.shape[1]))
    idx = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    while not done.all():
        active = np.flatnonzero(~done)
        node = idx[active]
        is_leaf = left[node] < 0
        la = active[is_leaf]
        out[la] = leafv[node[is_leaf]]
        done[la] = True
        ia = active[~is_leaf]
        if ia.size:
            n2 = idx[ia]
            idx[ia] = np.where(x[ia, feat[n2]] < thr[n2], left[n2], right[n2])
    return out


def forward(x):
    fam = CFG["model_family"]
    with np.load(HERE / "best_model.npz", allow_pickle=False) as z:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if fam == "mlp":
            K = int(z["fourier_K"]) if "fourier_K" in z.files else 0
            if K > 0:
                x = _fourier(x, K, z["fourier_scale"].astype(np.float64))
            n_layers = int(z["n_layers"])
            act = str(z["activation"])
            f = np.tanh if act == "tanh" else (lambda v: np.maximum(v, 0))
            h = x
            for i in range(n_layers - 1):
                h = f(h @ z[f"layer{i}_w"] + z[f"layer{i}_b"])
            return h @ z[f"layer{n_layers-1}_w"] + z[f"layer{n_layers-1}_b"]
        if fam == "knn":
            X = z["X"].astype(np.float64)
            k = min(int(z["k"]), X.shape[0])
            head = str(z["head"])
            n_out = int(z["n_out"])
            K = int(z["fourier_K"]) if "fourier_K" in z.files else 0
            if K > 0:
                x = _fourier(x, K, z["fourier_scale"].astype(np.float64))
            d2 = (x * x).sum(axis=1)[:, None] \
                + (X * X).sum(axis=1)[None, :] - 2.0 * (x @ X.T)
            d2 = np.maximum(d2, 0.0)
            order = np.argsort(d2, axis=1, kind="stable")[:, :k]
            if head == "mse":
                near = z["y"][order]
                return np.asarray(near.mean(axis=1), dtype=np.float64) \
                    .reshape(len(order), n_out)
            labels = np.asarray(z["y"], dtype=np.int64).reshape(-1)
            near_l = labels[order]
            votes = np.zeros((len(order), n_out), dtype=np.float64)
            np.add.at(votes, (np.arange(len(order))[:, None], near_l), 1.0)
            return votes / k + VOTE_EPS
        if fam == "tree":
            n_out = int(z["n_out"])
            acc = np.zeros((x.shape[0], n_out))
            for i in range(int(z["n_trees"])):
                acc += _tree_predict(
                    z, (f"t{i}_feature", f"t{i}_threshold", f"t{i}_left",
                        f"t{i}_right", f"t{i}_leaf_value"), x)
            return acc / int(z["n_trees"])
        if fam == "boost":
            n_out = int(z["n_out"])
            shrink = float(z["shrink"])
            out = np.tile(z["base"].astype(np.float64), (x.shape[0], 1))
            for t in range(int(z["n_rounds"])):
                for c in range(n_out):
                    tv = _tree_predict(
                        z, (f"t{t}c{c}_feature", f"t{t}c{c}_threshold",
                            f"t{t}c{c}_left", f"t{t}c{c}_right",
                            f"t{t}c{c}_leaf_value"), x).reshape(x.shape[0])
                    out[:, c] += shrink * tv
            return out
    raise ValueError(f"unknown model_family {fam!r}")


def standardize(raw):
    raw = np.asarray(raw, dtype=np.float64).ravel()
    mean = CFG.get("feature_mean")
    std = CFG.get("feature_std")
    if mean is None or std is None:
        return raw
    return (raw - np.asarray(mean, dtype=np.float64)) \
        / np.asarray(std, dtype=np.float64)


def decode(out_row):
    out_row = np.asarray(out_row, dtype=np.float64).ravel()
    if CFG.get("head") == "softmax":
        idx = int(out_row.argmax())
        values = CFG.get("class_values")
        pred = values[idx] if values else idx
        z = out_row - out_row.max()
        e = np.exp(z)
        p = e / e.sum()
        return {"prediction": pred,
                "probabilities": [float(v) for v in p]}
    return {"prediction": float(out_row[0]), "probabilities": None}


def _cell_is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def csv_rows(path):
    """CSV of NEW rows -> raw feature vectors (the 42.1.1 rule: header
    cells match feature names case-insensitively, extra columns such as a
    label column are OK; no header -> positional, exactly state_dim)."""
    state_dim = int(CFG.get("state_dim") or 0)
    names = list(CFG.get("feature_names") or [])
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = [[(c or "").strip() for c in raw] for raw in reader
                if any((c or "").strip() for c in raw)]
    if not rows:
        raise ValueError(f"CSV {path} has no rows")
    first = rows[0]
    if any(not _cell_is_number(c) for c in first):
        header, data = first, rows[1:]
    else:
        header, data = None, rows
    if not data:
        raise ValueError(f"CSV {path} has a header but no data rows")
    out = []
    for r in data:
        if header is not None:
            if names:
                idx = {h.lower(): j for j, h in enumerate(header)}
                vec = []
                for n in names:
                    j = idx.get(n.lower())
                    if j is None:
                        raise ValueError(
                            f"CSV {path}: header {header} is missing "
                            f"feature column {n!r}")
                    if j >= len(r) or r[j] == "":
                        raise ValueError(
                            f"CSV {path}: row {r} has no value for "
                            f"column {n!r}")
                    vec.append(_finite(r[j]))
            else:
                if len(header) != state_dim:
                    raise ValueError(
                        f"CSV {path}: the header must have exactly "
                        f"{state_dim} column(s), got {header}")
                vec = [_finite(r[j]) for j in range(state_dim)]
        else:
            if len(r) != state_dim:
                raise ValueError(
                    f"CSV {path}: expected {state_dim} cell(s) per row, "
                    f"got {len(r)}")
            vec = [_finite(c) for c in r]
        out.append(np.array(vec, dtype=np.float64))
    return out


def main(argv):
    if argv and argv[0] == "--row":
        if len(argv) != 2:
            print('usage: python predict.py --row "v1, v2, ..."',
                  file=sys.stderr)
            return 2
        row = [_finite(v) for v in argv[1].split(",") if v.strip() != ""]
        feats = standardize(row)
        out = forward(feats[None, :])
        print(json.dumps(decode(out[0])))
        return 0
    if argv and not argv[0].startswith("--"):
        for raw in csv_rows(argv[0]):
            feats = standardize(raw)
            out = forward(feats[None, :])
            print(json.dumps(decode(out[0])))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''
