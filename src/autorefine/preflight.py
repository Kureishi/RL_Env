"""Data-health preflight: pure stats over a constructed task (SPEC.md 43.1).

A leaf in the `predict` style (SPEC.md 42.1): duck-typed task in, dict out;
stdlib + numpy only; no core imports, so it stays import-cycle free.
`data_health(task, target=None)` computes, from the *already-built* probe
task the `fit --dry-run` plan uses (40.1 — no second data load, no
training), the class balance + headroom note (43.1.1), the constant /
near-constant columns, the per-column missing-value counts, and the
per-feature ranges. `format_health(health)` renders the human `health :`
block the plan prints (43.1.2); the CLI stays a thin printer.

Bad health is a *warning in the plan*, never an error (43.1.3): both
functions raise only on a genuinely malformed task object, which the
probe constructor would have rejected first (rc 1, 40.1.3).
"""
from __future__ import annotations

import numpy as np

# SPEC.md 43.1.1: a feature whose single most frequent value covers at
# least this share of its non-empty values is near-constant (99%).
_NEAR_CONSTANT_SHARE = 0.99

# SPEC.md 43.2.2: the smoke-train wall below which the check is `ok`
# (a slow-but-finished train is `warn`, an error is `FAIL`).
_SMOKE_WALL_SECONDS = 1.0


def data_health(task, target: float | None = None) -> dict:
    """SPEC.md 43.1.1 (v0.29): one pure pass over the probe task.

    csv tasks (a `_read_csv` + `feature_names` pair) report the full
    block; media tasks (no raw rows) and any other task degrade to the
    media shape (43.1.1) — item count, head, and the class balance from
    the task's own label array when available. `target` (the run's
    `--target`) enables the softmax headroom note; it is ignored for an
    mse head.
    """
    if hasattr(task, "_read_csv") and getattr(task, "feature_names", None):
        return _csv_health(task, target)
    return _media_health(task, target)


def _class_balance(labels, label_fn) -> tuple[list, float | None]:
    """(per-class rows in descending-count order, ties by label; majority
    share) — the shared shape of the csv and media branches."""
    if labels is None:
        return None, None
    counts: dict = {}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    total = len(labels)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], label_fn(kv[0])))
    balance = [{"label": label_fn(k), "count": int(c), "share": c / total}
               for k, c in ordered]
    return balance, (balance[0]["share"] if balance else None)


def _headroom_note(majority_share: float, target: float) -> str:
    """SPEC.md 43.1.1: a constant majority-class predictor scores
    `share * 100`; state the headroom left below the target (or that the
    target is already met by the majority class alone)."""
    score = 100.0 * majority_share
    if score >= target:
        return (f"majority class alone scores {score:.1f} — a constant "
                f"predictor already meets target {target:g}")
    return (f"majority class alone scores {score:.1f} — "
            f"{target - score:.1f} points of headroom below target "
            f"{target:g}")


def _csv_health(task, target) -> dict:
    """SPEC.md 43.1.1: the csv block — raw rows only (the task already
    holds the parsed arrays; nothing is re-learned or re-split)."""
    names, rows = task._read_csv()
    label_name = task.label_name
    n_rows = len(rows)

    # missing values: every column of the raw file (43.1.1), with the
    # role the task gave it (a column with an empty value is dropped
    # from the features, 22.1)
    missing = []
    for c in names:
        k = sum(1 for r in rows if r.get(c, "") == "")
        if k:
            role = ("label" if c == label_name
                    else "feature" if c in task.feature_names else "ignored")
            missing.append({"column": c, "count": k, "role": role})

    # per-feature ranges + constant / near-constant detection
    features: list = []
    constant: list = []
    near_constant: list = []
    for c in task.feature_names:
        vals = [float(r[c]) for r in rows if r.get(c, "") != ""]
        lo, hi = min(vals), max(vals)
        features.append({"name": c, "min": lo, "max": hi,
                         "mean": sum(vals) / len(vals)})
        if hi == lo:
            constant.append({"name": c, "value": lo, "share": 1.0})
        else:
            counts: dict = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            top_value, top_count = max(counts.items(), key=lambda kv: kv[1])
            share = top_count / len(vals)
            if share >= _NEAR_CONSTANT_SHARE:
                near_constant.append({"name": c, "value": top_value,
                                      "share": share})

    # class balance + headroom (softmax heads only; 43.1.1)
    softmax = getattr(task, "class_values", None) is not None
    if softmax:
        balance, majority = _class_balance(
            [r[label_name] for r in rows if r.get(label_name, "") != ""],
            str)
    else:
        balance, majority = None, None
    note = (_headroom_note(majority, target)
            if softmax and majority is not None and target is not None
            else None)

    return {"kind": "csv", "n_rows": n_rows, "label": label_name,
            "head": getattr(task, "head", None),
            "balance": balance, "majority_share": majority,
            "note": note, "features": features,
            "constant": constant, "near_constant": near_constant,
            "missing": missing}


def _media_health(task, target) -> dict:
    """SPEC.md 43.1.1: the graceful media shape — no raw rows exist, so
    the block is item count + head + the class balance from the task's
    own label array (the subfolder / index.csv labels)."""
    y = getattr(task, "_y", None)
    cv = getattr(task, "class_values", None)
    if y is not None and cv is not None:
        balance, majority = _class_balance(
            [int(i) for i in np.asarray(y)],
            lambda i: str(cv[int(i)]))
    else:
        balance, majority = None, None
    n_items = (len(task._items) if hasattr(task, "_items")
               else getattr(task, "default_dataset_size", None))
    note = (f"media task: {n_items} items, state_dim {getattr(task, 'state_dim', '?')}, "
            f"{len(cv)} classes (class balance from subfolders/index)"
            if n_items is not None and cv is not None else
            f"media task: {n_items} items" if n_items is not None else
            "media task")
    return {"kind": "media", "n_rows": n_items, "label": None,
            "head": getattr(task, "head", None),
            "balance": balance, "majority_share": majority,
            "note": note, "features": [],
            "constant": [], "near_constant": [], "missing": []}


# --- the human rendering (43.1.2) ---------------------------------------------

def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def format_health(health: dict) -> list[str]:
    """SPEC.md 43.1.2 (v0.29): the plan's `health :` block — one line
    per stat, `none` for an empty list, in the plan's 2-space + 10-wide
    label style. Pure dict-in/lines-out; cp1252-safe (ASCII + em-dash
    only)."""
    head = ("  health     : "
            f"{health['n_rows']} rows, label '{health['label']}', "
            f"{len(health['features'])} feature column(s)") if health["kind"] == "csv" \
        else (f"  health     : {health['n_rows']} items "
              f"(head {health['head']})")
    lines = [head]
    if health.get("balance"):
        lines.append("  balance    : " + " | ".join(
            f"{b['label']}: {b['count']} ({100.0 * b['share']:.1f}%)"
            for b in health["balance"]))
    if health.get("note"):
        lines.append(f"  headroom   : {health['note']}")
    if health["kind"] == "csv":
        if health["features"]:
            lines.append("  features   : " + " | ".join(
                f"{f['name']} [{_fmt_num(f['min'])}, {_fmt_num(f['max'])}] "
                f"mean {_fmt_num(f['mean'])}" for f in health["features"]))
        lines.append("  constant   : "
                     + (" | ".join(f"{c['name']} (all values "
                                   f"{_fmt_num(c['value'])})"
                                   for c in health["constant"])
                      or "none"))
        lines.append("  near-const : "
                     + (" | ".join(f"{c['name']} ({100.0 * c['share']:.1f}% "
                                   f"of rows = {_fmt_num(c['value'])})"
                                   for c in health["near_constant"])
                      or "none"))
        lines.append("  missing    : "
                     + (" | ".join(f"{m['column']} ({m['count']} empty, "
                                   f"{m['role']})" for m in health["missing"])
                      or "none"))
    return lines
