"""SPEC.md 57 (v0.43): research decision surfaces — pure derivations.

The "B. Visually useful for researchers" round: four views that turn
the history already logged into decision surfaces — the spec-lineage
DAG (57.1), per-field response surfaces (57.2), the gate-decision
region inputs (57.3), and the bandit belief bars' data (57.4). All four
are *pure derivations* from data that already exists: the
`experiments.jsonl` entries (SPEC.md 16/25.4/28.1) and the update
stream's `field_stats` / `ucb` (SPEC.md 26.1/26.4). No new logged
field, no loop change, no side effects (G2).

Leaf module: it reads the log-kind registry from `memory` (35.1) and
the §18.5 tolerance + reason semantics from `accounting` (39.2.2) —
one home each. The SVG renderers live in `plotting.py` (57.5); the
app renders the block in `_render_result` (57.6).
"""
from __future__ import annotations

import math

from .memory import (
    KIND_BASELINE,
    KIND_CURRICULUM,
    KIND_EXPERIMENT,
    KIND_SCREEN,
)
from .accounting import GEN_GAP_TOL, account_run, candidate_reason
from .config import spec_n_params  # 58.2 (v0.44): the size axis

__all__ = [
    "spec_lineage",
    "field_response_stats",
    "gate_region_candidates",
    "bandit_beliefs",
    "frontier3",
    "run_verdict",       # the run verdict card (target vs final best + CI)
    "knob_signal",       # the decisive-knob ranking (signal vs noise)
    "frontier_knee",     # the efficiency knee (bang-for-buck pick)
    "rejection_anatomy", # the rejection mix + stall story
]


def _num(value) -> float | None:
    """A finite number, else None (rows may lack a field on old runs)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return float(value)
    return None


def _label(spec, h: str) -> str:
    """57.1: the node label — the candidate's model family + a 6-char
    hash prefix (ASCII, stable). A non-dict spec (old runs) degrades to
    `spec <hash6>`."""
    fam = spec.get("model_family") if isinstance(spec, dict) else None
    if not isinstance(fam, str) or not fam:
        fam = "spec"
    return f"{fam} {h[:6]}"


def spec_lineage(entries) -> dict:
    """57.1: the spec-lineage graph (a DAG, in practice a tree — every
    candidate mutates the champion in force, so each node has exactly
    one parent).

    The parent link is reconstructed from log order: the reset baseline
    seeds the champion; every accepted candidate becomes it; a
    curriculum step-up re-pins the *score* but carries the same spec
    over (20.1), so the champion — and therefore the graph — is
    untouched. `invalid_spec` rows (25.4) carry no `spec_hash` / score
    and are excluded (they never entered the search space). Returns
    `{nodes: {hash: {hash, score, accepted, label}},
    edges: [{src, dst, accepted, delta, fields}]}` — `delta` is the
    candidate score minus the champion score in force (None when
    either is missing). Pure and deterministic (G2); empty → empty
    sets."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    champion: str | None = None
    best_before: float | None = None
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind == KIND_BASELINE:
            h = e.get("spec_hash")
            if isinstance(h, str) and h:
                if h not in nodes:
                    nodes[h] = {
                        "hash": h,
                        "score": _num(e.get("holdout_score")),
                        "accepted": bool(e.get("accepted", True)),
                        "label": _label(e.get("spec"), h),
                    }
                champion = h
            score = _num(e.get("holdout_score"))
            if score is not None:
                best_before = score
        elif kind == KIND_CURRICULUM:
            # a step-up re-pins the champion's score on the new level
            new_base = _num(e.get("new_baseline_score"))
            if new_base is not None:
                best_before = new_base
        elif kind in (KIND_EXPERIMENT, KIND_SCREEN):
            h = e.get("spec_hash")
            if not (isinstance(h, str) and h):
                continue
            score = _num(e.get("holdout_score"))
            if h not in nodes:
                nodes[h] = {
                    "hash": h,
                    "score": score,
                    "accepted": bool(e.get("accepted", False)),
                    "label": _label(e.get("spec"), h),
                }
            if champion is not None:
                delta = (score - best_before) \
                    if (score is not None and best_before is not None) else None
                fields = e.get("mutation")
                edges.append({
                    "src": champion,
                    "dst": h,
                    "accepted": bool(e.get("accepted", False)),
                    "delta": delta,
                    "fields": [f for f in fields if isinstance(f, str)]
                              if isinstance(fields, list) else [],
                })
            if bool(e.get("accepted")):
                champion = h
                if score is not None:
                    best_before = score
    return {"nodes": nodes, "edges": edges}


def _fv_key(value) -> str:
    """57.2: a spec value → its response-surface key (the V2 matrix
    convention, SPEC.md 30.2): `None` → `"-"`, list/tuple joined with
    `,` (e.g. layers `[8, 16]` → `"8,16"`), else `str(v)`."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def field_response_stats(entries) -> dict[str, dict[str, dict]]:
    """57.2: per-field value → mean holdout score (partial-dependence
    style). For every *scored* entry (kind baseline/experiment with a
    finite `holdout_score` and a dict `spec`), each `(field, value)`
    pair of that candidate's FULL spec is credited with the entry's
    holdout score — the baseline row contributes too (its spec is a
    point in the same space). Screen / curriculum / invalid rows are
    not scored candidates (35.1) and are excluded. Returns
    `{field: {value: {n, mean, best}}}`, fields and values sorted
    (values numeric-first, the V2 `_fv_sort_key` order). Pure and
    deterministic (G2); empty → `{}`."""
    acc: dict[str, dict[str, list]] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):
            continue
        score = _num(e.get("holdout_score"))
        spec = e.get("spec")
        if score is None or not isinstance(spec, dict):
            continue
        for f, v in spec.items():
            if not isinstance(f, str):
                continue
            cell = acc.setdefault(f, {}).setdefault(_fv_key(v),
                                                    [0, 0.0, float("-inf")])
            cell[0] += 1
            cell[1] += score
            if score > cell[2]:
                cell[2] = score

    def _sort(k: str) -> tuple:
        try:
            return (0.0, float(k), "")
        except (TypeError, ValueError):
            return (1.0, 0.0, k)

    return {
        f: {v: {"n": acc[f][v][0],
                "mean": acc[f][v][1] / acc[f][v][0],
                "best": acc[f][v][2]}
            for v in sorted(acc[f], key=_sort)}
        for f in sorted(acc)
    }


def gate_region_candidates(entries) -> dict:
    """57.3: the gate-decision-region inputs — every scored candidate as
    a `(holdout score, gen_gap)` point with its verdict and its
    running `best_before`, plus the run's baseline score and the §18.5
    tolerance.

    The running best is reconstructed exactly like
    `accounting._rejections` (39.2.2): the baseline seeds it, a
    curriculum step-up re-pins it, an accepted candidate raises it.
    The per-point reason is `accounting.candidate_reason` (51.3.1) —
    the same score → overfit → ci priority, so the region plot and the
    report's accounting cannot drift. `gen_gap` None (old rows) is
    kept None; the renderer treats it as 0. Pure and deterministic
    (G2); no scored candidates → `candidates: []` (the baseline, if
    any, still carries `baseline_score`)."""
    baseline_score = None
    best_before: float | None = None
    cands: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind == KIND_BASELINE:
            score = _num(e.get("holdout_score"))
            if score is not None:
                baseline_score = score
                best_before = score
        elif kind == KIND_CURRICULUM:
            new_base = _num(e.get("new_baseline_score"))
            if new_base is not None:
                best_before = new_base
        elif kind in (KIND_EXPERIMENT, KIND_SCREEN):
            score = _num(e.get("holdout_score"))
            if score is None:
                continue
            accepted = bool(e.get("accepted", False))
            reason = candidate_reason(accepted, score,
                                      e.get("gen_gap"), best_before)
            cands.append({
                "score": score,
                "gen_gap": _num(e.get("gen_gap")),
                "accepted": accepted,
                "reason": reason,
                "best_before": best_before,
            })
            if accepted:
                best_before = score
    return {
        "baseline_score": baseline_score,
        "tol": float(GEN_GAP_TOL),
        "candidates": cands,
    }


# 95% two-sided normal quantile (fixed constant — G2 determinism; not a
# knob, not a knob-registry row, SPEC.md 33.2)
_Z95 = 1.959964


def bandit_beliefs(field_stats: dict | None,
                   ucb: dict | None) -> dict[str, dict]:
    """57.4: the per-field learned preference as a probability with its
    95% Wilson interval, plus the final UCB value.

    `field_stats` is the SPEC.md 26.1 shape
    (`{field: {trials, wins, win_rate}}`); `ucb` is the final
    `ucb_trace` map (SPEC.md 26.4 — None for the search policy, whose
    values are simply dropped). The Wilson interval for `p = wins/n`:
    `center = (p + z²/2n) / (1 + z²/n)`,
    `half = z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n)`, clamped to [0, 1]
    (z = 1.959964). Fields with `trials <= 0` are skipped. Returns
    `{field: {trials, wins, win_rate, lo, hi, ucb}}` sorted by field.
    Pure math (G2)."""
    out: dict[str, dict] = {}
    stats = field_stats if isinstance(field_stats, dict) else {}
    beliefs = ucb if isinstance(ucb, dict) else {}
    z2 = _Z95 * _Z95
    for f in sorted(stats):
        s = stats.get(f)
        if not isinstance(s, dict):
            continue
        n = s.get("trials", 0)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            continue
        wins = float(s.get("wins", 0.0) or 0.0)
        p = wins / n
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        half = _Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        u = beliefs.get(f)
        out[f] = {
            "trials": n,
            "wins": wins,
            "win_rate": p,
            "lo": lo,
            "hi": hi,
            "ucb": _num(u),
        }
    return out


def frontier3(entries, state_dim=None, n_out=None, grid=None) -> dict:
    """58.2: the 3-objective frontier — score × train-time × model-size.

    Points are the *scored* candidates under the 57.2 rule (kind
    baseline/experiment, a finite `holdout_score`, a dict `spec`) plus a
    finite `train_seconds` (the 2-objective `svg_pareto` needs the
    same pair). Each point's `size` is
    `config.spec_n_params(spec, state_dim, n_out, grid)` (58.2.1) —
    `None` for the data-dependent families (tree/boost/knn) and for
    convnets whose grid is missing or too small.

    3D non-dominance (58.2.2): a point A dominates B iff
    `A.score ≥ B.score` and `A.time ≤ B.time` and — only when BOTH carry
    a size — `A.size ≤ B.size`, with at least one strict inequality on
    a *comparable* axis (score and time are always comparable; a
    `None` size neither violates nor establishes dominance). The
    frontier is the set of non-dominated points. Returns
    `{"points": [{score, time, size, kind, hash, label, frontier}],
    "frontier": [point indices], "unsized": int}`. Pure and
    deterministic (G2); no scored candidates → empty points.
    """
    pts: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind not in (KIND_BASELINE, KIND_EXPERIMENT):  # 35.1 (C4)
            continue
        score = _num(e.get("holdout_score"))
        time = _num(e.get("train_seconds"))
        spec = e.get("spec")
        if score is None or time is None or not isinstance(spec, dict):
            continue
        h = e.get("spec_hash")
        h = h if isinstance(h, str) and h else ""
        pts.append({
            "score": score,
            "time": time,
            "size": spec_n_params(spec, state_dim, n_out, grid),
            "kind": kind,
            "hash": h,
            "label": _label(spec, h),
            "frontier": False,
        })

    def _dominates(a: dict, b: dict) -> bool:
        if a["score"] < b["score"] or a["time"] > b["time"]:
            return False
        both_sized = a["size"] is not None and b["size"] is not None
        if both_sized and a["size"] > b["size"]:
            return False
        strict = a["score"] > b["score"] or a["time"] < b["time"]
        if both_sized:
            strict = strict or a["size"] < b["size"]
        return strict

    frontier_idx = [
        i for i, a in enumerate(pts)
        if not any(j != i and _dominates(pts[j], a) for j in range(len(pts)))
    ]
    for i, p in enumerate(pts):
        p["frontier"] = i in frontier_idx
    return {
        "points": pts,
        "frontier": frontier_idx,
        "unsized": sum(1 for p in pts if p["size"] is None),
    }


# --- The conclusion surfaces --------------------------------------------------
# The "so what?" round: four pure derivations that turn the history
# already logged (experiments.jsonl + summary.json) into direct answers —
# did we hit target and how sure are we (run_verdict), which knobs
# actually moved the needle (knob_signal), which spec to deploy
# (frontier_knee), and why the search stalled (rejection_anatomy). The
# same house rules as the 57/58 shapers: pure, deterministic (G2), no new
# logged field, no loop change; the SVG renderers live in `plotting.py`.

# knob_signal tier thresholds — the score-impact spread (best value mean
# minus worst value mean, 0–100 holdout scale) at which a field's values
# start to matter. Fixed derivation constants (like _Z95 above): not a
# knob, not a knob-registry row, not tunable at runtime.
KNOB_SPREAD_DECISIVE = 5.0
KNOB_SPREAD_WEAK = 2.0


def run_verdict(summary, entries) -> dict:
    """The run verdict card's data — target vs final best, with the
    champion's CI and the run's bookkeeping, in one dict.

    `summary` is the `summary.json` shape (`baseline_score`,
    `final_best_score`, `finished_reason`, `improvement_factor`, and
    `run_config.target` / `run_config.z_accept` when present); `entries`
    the experiments.jsonl rows. The champion's CI is the `std` of the
    last *accepted* scored entry whose `holdout_score` equals
    `final_best_score` (the baseline row qualifies when no candidate
    improved) — None when no such row carries a `std` (old runs). `z`
    defaults to 1.0 (the gate's default z_accept). `margin = best −
    target` and `pass = margin >= 0`; both None without a target or a
    final best. Pure and deterministic (G2)."""
    summary = summary if isinstance(summary, dict) else {}
    rc = summary.get("run_config")
    rc = rc if isinstance(rc, dict) else {}
    target = _num(rc.get("target"))
    z = _num(rc.get("z_accept"))
    if z is None or z < 0.0:
        z = 1.0  # the gate's default (z_accept)
    best = _num(summary.get("final_best_score"))
    cand_std = None
    for e in entries or []:
        if not isinstance(e, dict) or not e.get("accepted"):
            continue
        if e.get("kind") not in (KIND_BASELINE, KIND_EXPERIMENT):
            continue
        s = _num(e.get("holdout_score"))
        if s is None or best is None or abs(s - best) > 1e-9:
            continue
        cand_std = _num(e.get("std"))
    margin = (best - target) if (best is not None and target is not None) \
        else None
    reason = summary.get("finished_reason")
    return {
        "target": target,
        "baseline": _num(summary.get("baseline_score")),
        "best": best,
        "std": cand_std,
        "z": z,
        "margin": margin,
        "pass": (margin >= 0.0) if margin is not None else None,
        "finished_reason": reason if isinstance(reason, str) and reason \
        else None,
        "improvement_factor": _num(summary.get("improvement_factor")),
    }


def knob_signal(entries) -> dict:
    """The decisive-knob ranking — every spec field ranked by its
    score-impact, tiered decisive / weak / noise / untried.

    Per field (the 57.2 response-surface rule — scored candidates only,
    full-spec value→mean credit): `spread = max(value means) −
    min(value means)` over the distinct values tried (0.0 with fewer
    than 2 values — an untried field cannot have an impact), plus the
    best/worst value with their means, and the field's mutation
    `trials` / `wins` (kinds=experiment rows whose `mutation` list names
    the field; a win = that candidate was accepted). Tiers by the fixed
    spread thresholds: decisive >= 5.0, weak >= 2.0, noise above 0, and
    untried (fewer than 2 values). Ordered tier → spread desc → field
    name (G2). `spread` is partial-dependence style: interactions with
    co-mutated fields confound it (the caption says so). Pure and
    deterministic (G2); empty → `fields: []`."""
    stats = field_response_stats(entries)
    trials: dict[str, int] = {}
    wins: dict[str, int] = {}
    for e in entries or []:
        if not isinstance(e, dict) or e.get("kind") != KIND_EXPERIMENT:
            continue
        mut = e.get("mutation")
        if not isinstance(mut, list):
            continue
        acc = bool(e.get("accepted"))
        for f in mut:
            if not isinstance(f, str):
                continue
            trials[f] = trials.get(f, 0) + 1
            if acc:
                wins[f] = wins.get(f, 0) + 1
    fields: list[dict] = []
    for f in sorted(stats):
        vals = stats[f]
        if not vals:
            continue
        n_values = len(vals)
        best_v = worst_v = None
        best_m = worst_m = None
        for vkey, v in vals.items():  # insertion order = sorted values
            if best_m is None or v["mean"] > best_m:
                best_m, best_v = v["mean"], vkey
            if worst_m is None or v["mean"] < worst_m:
                worst_m, worst_v = v["mean"], vkey
        spread = (best_m - worst_m) if n_values >= 2 else 0.0
        if n_values < 2:
            tier = "untried"
        elif spread >= KNOB_SPREAD_DECISIVE:
            tier = "decisive"
        elif spread >= KNOB_SPREAD_WEAK:
            tier = "weak"
        else:
            tier = "noise"
        fields.append({
            "field": f,
            "spread": spread,
            "n_values": n_values,
            "best_value": best_v,
            "best_mean": best_m,
            "worst_value": worst_v,
            "worst_mean": worst_m,
            "trials": trials.get(f, 0),
            "wins": wins.get(f, 0),
            "tier": tier,
        })
    order = {"decisive": 0, "weak": 1, "noise": 2, "untried": 3}
    fields.sort(key=lambda d: (order[d["tier"]], -d["spread"], d["field"]))
    return {
        "fields": fields,
        "thresholds": {
            "decisive": KNOB_SPREAD_DECISIVE,
            "weak": KNOB_SPREAD_WEAK,
        },
    }


def frontier_knee(entries, state_dim=None, n_out=None, grid=None) -> dict:
    """The efficiency knee — the best-quality point AND the
    bang-for-buck point on the score × train-time frontier.

    Built on `frontier3` (58.2): `best` is the point with the highest
    score (ties: lower train time, then log order); `knee` is the
    frontier point farthest from the segment joining the frontier's
    min-time endpoint to its max-score endpoint, measured in
    min-max-normalized (time, score) space (ties: lower time, then log
    order) — None with fewer than 2 frontier points (no trade-off to
    read). `delta_score = best.score − knee.score`; `time_saving_pct` is
    the knee's train-time saving vs the best, in percent (None without a
    knee or a positive best time). `points` / `unsized` pass through
    from `frontier3`. Pure and deterministic (G2); empty → empty."""
    f3 = frontier3(entries, state_dim=state_dim, n_out=n_out, grid=grid)
    pts = f3["points"]
    if not pts:
        return {
            "points": [], "best": None, "knee": None, "unsized": 0,
            "delta_score": None, "time_saving_pct": None,
        }
    best = min(range(len(pts)),
               key=lambda i: (-pts[i]["score"], pts[i]["time"], i))
    knee = None
    fidx = list(range(len(pts))) if not f3["frontier"] else f3["frontier"]
    if len(fidx) >= 2:
        a = min(fidx, key=lambda i: (pts[i]["time"], -pts[i]["score"], i))
        b = max(fidx, key=lambda i: (pts[i]["score"], -pts[i]["time"], i))
        times = [pts[i]["time"] for i in fidx]
        scores = [pts[i]["score"] for i in fidx]
        tmin, tmax = min(times), max(times)
        smin, smax = min(scores), max(scores)
        tspan = (tmax - tmin) or 1.0
        sspan = (smax - smin) or 1.0

        def norm(i: int) -> tuple[float, float]:
            return ((pts[i]["time"] - tmin) / tspan,
                    (pts[i]["score"] - smin) / sspan)

        ax, ay = norm(a)
        bx, by = norm(b)
        dx, dy = bx - ax, by - ay

        def seg_dist(i: int) -> float:
            px, py = norm(i)
            l2 = dx * dx + dy * dy
            if l2 <= 0.0:  # degenerate segment (endpoints coincide)
                return math.hypot(px - ax, py - ay)
            t = ((px - ax) * dx + (py - ay) * dy) / l2
            t = max(0.0, min(1.0, t))
            return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

        knee = min(fidx, key=lambda i: (-seg_dist(i), pts[i]["time"], i))
    delta = None
    pct = None
    if knee is not None:
        delta = pts[best]["score"] - pts[knee]["score"]
        bt = pts[best]["time"]
        if bt > 0.0:
            pct = (bt - pts[knee]["time"]) / bt * 100.0
    return {
        "points": pts,
        "best": best,
        "knee": knee,
        "unsized": f3["unsized"],
        "delta_score": delta,
        "time_saving_pct": pct,
    }


_STALL_HINTS = {
    "score": "consider more budget or a wider search space",
    "overfit": "consider a tighter overfit penalty or more regularization",
    "ci": "consider more CI blocks or a lower acceptance threshold",
}


def _stall_story(accepted: int, buckets: dict, dominant: str | None,
                 ttf: dict) -> str:
    """rejection_anatomy's one-line stall story — deterministic (G2).
    The dominant gate's hint is a fixed phrase (no numbers from the
    outside), so the sentence is a pure function of the counts."""
    total = accepted + sum(buckets.values())
    if total == 0:
        return "no scored candidates yet"
    rejected = sum(buckets.values())
    if rejected == 0:
        return "every candidate improved — signal on every try"
    if ttf.get("improved"):
        ttf_txt = f"first improvement at candidate {ttf.get('experiment_index')}"
        secs = ttf.get("seconds")
        if isinstance(secs, (int, float)) and not isinstance(secs, bool) \
                and math.isfinite(secs):
            ttf_txt += f" ({secs:.1f}s in)"
    else:
        ttf_txt = "no candidate beat the baseline"
    dom = int(buckets.get(dominant, 0)) if dominant else 0
    hint = _STALL_HINTS.get(dominant or "", "review the gate settings")
    return (f"{rejected} of {total} candidates rejected — mostly the "
            f"{dominant} gate ({dom}/{rejected}); {ttf_txt} → {hint}")


def rejection_anatomy(summary, entries) -> dict:
    """The rejection anatomy — the accepted/rejected mix by first
    failing gate, the time-to-first-improvement, the dominant gate, and
    a one-line stall story.

    The buckets are `accounting._rejections` (39.2.2 — the score →
    overfit → ci priority the gate uses, reconstructed over the same
    best-before chain) and the first-improvement timing is
    `accounting._first_improvement`, both read through the public
    `account_run`; `dominant` is the most-populated rejection bucket
    (ties: the gate priority score → overfit → ci, then None). `story`
    is the deterministic one-liner ("18 of 30 rejected — mostly the
    score gate (14/18); first improvement at candidate 3 → consider
    more budget or a wider search space"). Pure and deterministic
    (G2); no scored candidates → zeroed buckets + the empty story."""
    acc = account_run(summary, entries)
    r = acc.get("rejections") if isinstance(acc, dict) else None
    r = r if isinstance(r, dict) else {}
    ttf = acc.get("time_to_first_improvement") if isinstance(acc, dict) \
        else None
    ttf = ttf if isinstance(ttf, dict) else {}
    buckets = {k: int(r.get(k, 0) or 0) for k in ("score", "overfit", "ci")}
    accepted = int(r.get("accepted", 0) or 0)
    rejected = sum(buckets.values())
    dominant = None
    if rejected:
        dominant = max(("score", "overfit", "ci"),
                       key=lambda k: buckets[k])
    return {
        "accepted": accepted,
        "rejected": rejected,
        "by_reason": buckets,
        "dominant": dominant,
        "first_improvement": ttf,
        "story": _stall_story(accepted, buckets, dominant, ttf),
    }
