"""Streamlit app for the v0.9 dashboard (SPEC.md 23.2) — a thin rendering
layer over `autorefine.dashboard.DashboardRunner`; all run semantics live in
the core, so same-seed runs reproduce the CLI `fit` outcome (G2).

Run:  streamlit run src/autorefine/dashboard_app.py
  or: python -m autorefine dashboard            (SPEC.md 23.4 launcher)
"""
from __future__ import annotations

import csv as _csv
import json
import os
import queue as _queue
import tempfile
import threading
import time  # 55.1 (v0.41): the live freshness caption (stream mode)
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components  # 52.4 (v0.38): the keyboard JS

from autorefine import (
    __version__,
    SteeringState,          # 59.2 (v0.45): the steering rules (pin/bias/constrain)
    manual_spec,            # 59.3 (v0.45): the manual/expert-mode spec
    parameter_inspection,   # 59.1 (v0.45): the parameter inspector's data
    train_manual,           # 59.3 (v0.45): train + report one exact spec
    fingerprint_diff,       # 60.2 (v0.46): the spec fingerprint diff
    interaction_matrix,     # 60.3 (v0.46): the co-mutation joint effects
    spec_fingerprint,       # 60.2 (v0.46): the spec "DNA" bars
    whatif_preview,         # 60.1 (v0.46): the live what-if preview
    weighted_reslice,       # 60.4 (v0.46): the objective-weight re-gate
)
from autorefine.config import DEFAULT_SPEC  # 59.3 (v0.45): manual defaults
from autorefine.improver.specspace import SPEC_FIELDS  # 59 (v0.45) the registry
from autorefine.accounting import candidate_reason  # 51.3.2 (v0.37)
from autorefine.dashboard import (
    DashboardRunner,
    diff_n_summaries,  # N-run compare (SPEC.md 50.1.1, v0.36)
    diff_two_summaries,
    eta_seconds,  # 52.3.1 (v0.38): the live remaining-time estimate
    fields_seen,  # 52.2.1 (v0.38): the focus-field options
    gate_math,  # 52.1.1 (v0.38): the per-candidate gate math
    plateau_streak,  # 52.3.1 (v0.38): the live stall sentinel
    PLATEAU_HINT,  # 52.3.1 (v0.38): the display stall threshold
    running_best_curve,  # N-run curves (SPEC.md 50.1.2, v0.36)
    ucb_trace,
)
# SPEC.md 48 (v0.34): the beginner onboarding core (48.1-48.3) and the
# plain-English narration core (48.4/48.5) — the app is a thin renderer.
from autorefine.onboarding import (
    ALL_KNOBS,
    KNOB_GLOSSARY,
    preset_choices,
    preset_flags,
    preset_summary,
)
# SPEC.md 49 (v0.35): the advanced analysis core — the app is a thin
# renderer over these pure functions (49.4).
from autorefine.advanced import opposite_policy, retrain_spec
from autorefine.gate import Objective, model_size
from autorefine.dossier import build_dossier  # 58.3 (v0.44): the run dossier
from autorefine.runconfig import RunConfig, fit_recipe  # 50.1.5/50.2 (v0.36)
from autorefine.narrate import narrate_baseline, narrate_run, narrate_step
from autorefine.simulate import what_if_block
# SPEC.md 55 (v0.41): "make it feel live" — the three live-view helpers.
# The app is a thin renderer over these pure core functions (55.x).
from autorefine.live import (
    freshness_caption,  # 55.1: the live freshness caption
    list_reference_runs,  # 55.3: the reference-overlay options
    snapshot_inputs,  # 55.2: the (info, stream, best, done) resolver
    status_snapshot,  # 55.2: the mid-run status card
    stream_tick_due,  # 55.1: the non-blocking stream-mode tick decision
)
# SPEC.md 42.2.2 (v0.28): `diff_two_summaries` moved to core (the
# `autorefine compare` CLI prints it); the app keeps the name importable
# (SPEC.md 38.4 unchanged) — this *is* `dashboard.diff_two_summaries`.
from autorefine.plotting import (
    resolve_tokens,  # 56.3 (v0.42): the design-token registry (tokens expander)
    svg_action_probabilities,  # noqa: F401 (D2 view, SPEC.md 29.2)
    svg_architecture,  # C4 (28.4) + 53.3 (v0.39) the champion spec card
    svg_audio_waveform,
    svg_bandit_beliefs,  # 57.4 (v0.43) the bandit belief bars
    svg_gate_line,  # 53.1 (v0.39) the per-candidate gate number-line
    svg_gate_region,  # 57.3 (v0.43) the gate-decision region plot
    svg_live_frontier,  # 53.2 (v0.39) the live Pareto frontier
    svg_score_curve,  # 51.4.4 (v0.37): the palette/dark result re-render
    svg_field_value_matrix,  # V2 view (SPEC.md 30.2)
    svg_whatif_effect,  # 60.1 (v0.46) the what-if estimated effect
    svg_spec_fingerprint,  # 60.2 (v0.46) the spec fingerprint ("DNA")
    svg_interaction_heatmap,  # 60.3 (v0.46) the interaction heatmap
    svg_weighted_reslice,  # 60.4 (v0.46) the objective-weight reslice
    svg_field_response,  # 57.2 (v0.43) the per-field response surfaces
    svg_frontier_overlay,  # A/B view (SPEC.md 49.3.3)
    svg_loss_curves,
    svg_mutation_timeline,
    svg_run_curves,  # N-run overlay (SPEC.md 50.1.3, v0.36)
    svg_live_sparkline,  # 55.1 (v0.41) the pulsing live sparkline
    svg_reference_curve,  # 55.3 (v0.41) the reference-run overlay
    svg_score_gap_scatter,
    svg_score_strip,
    svg_time_strip,  # B2 (SPEC.md 54.2) the wall-time cost strip
    svg_seed_curves,  # V1 view (SPEC.md 30.1)
    svg_seed_variance,  # D1 view (SPEC.md 29.1)
    svg_spec_lineage,  # 57.1 (v0.43) the spec-lineage DAG
    svg_task_returns,  # noqa: F401 (D2 view, SPEC.md 29.2)
)
# SPEC.md 56.1 (v0.42): the provenance certificate — the app's Provenance
# expander (56.4) is a thin renderer over these pure core functions.
from autorefine.provenance import (
    env_provenance,
    provenance_card,
    provenance_payload,
    svg_provenance,
)
# SPEC.md 57 (v0.43): the research decision surfaces — the app is a thin
# renderer over these pure derivations (57.5); no new logged data.
from autorefine.research import (
    bandit_beliefs,  # 57.4 (v0.43) the belief bars' data (Wilson CI + UCB)
    field_response_stats,  # 57.2 (v0.43) per-field value -> mean score
    gate_region_candidates,  # 57.3 (v0.43) the (score, gap) verdict points
    spec_lineage,  # 57.1 (v0.43) the spec-lineage graph
)
from autorefine.tasks import CsvTask


def _render_past_runs(runs_dir: str) -> None:
    """SPEC.md 38.4 (v0.24, T1): the Past-runs section — the registry as a
    table (38.3 columns) plus a two-run compare (best_spec field diff +
    score delta). Inert data rendering; an empty/missing registry shows
    the empty-state caption (38.4.2)."""
    from autorefine.registry import gate_label, load_registry
    st.subheader("Past runs")
    entries = load_registry(runs_dir)
    if not entries:
        st.caption(f"No finished runs yet in `{runs_dir}` — the registry is "
                   f"appended when a run finishes (SPEC.md 38.1).")
        return
    rows = [
        {"run_id": e.get("run_id"), "task": e.get("task"), "seed": e.get("seed"),
         "policy": e.get("policy") or "—", "score": e.get("final_score"),
         "target": e.get("target") if e.get("target") is not None else "—",
         "gate": gate_label(e.get("met_target")),
         "wall_s": e.get("wall_seconds"), "exps": e.get("experiments_run"),
         "parent": e.get("parent_run") or "—"}
        for e in entries
    ]
    st.dataframe(rows, width="stretch")
    with st.expander("Compare two runs (best_spec diff)"):
        if len(entries) < 2:
            st.caption("Finish at least two runs to compare them here.")
            return
        ids = [str(e.get("run_id")) for e in entries]
        c1, c2 = st.columns(2)
        run_a = c1.selectbox("Run A", ids, key="past_run_a")
        run_b = c2.selectbox("Run B", ids,
                             index=1 if len(ids) > 1 else 0, key="past_run_b")

        def _summary_of(run_id: str) -> dict | None:
            p = Path(runs_dir) / run_id / "summary.json"
            if not p.is_file():
                return None
            return json.loads(p.read_text(encoding="utf-8"))

        sa, sb = _summary_of(run_a), _summary_of(run_b)
        if sa is None or sb is None:
            st.warning("One or both run dirs have no summary.json — nothing "
                       "to diff.")
            return
        d = diff_two_summaries(sa, sb)
        if d["score_delta"] is not None:
            st.write(f"score A **{d['score_a']:.2f}** vs B **{d['score_b']:.2f}** "
                     f"(Δ {d['score_delta']:+.2f})")
        if d["spec_diff"]:
            st.dataframe(d["spec_diff"], width="stretch")
        else:
            st.caption("`best_spec` is identical — the runs differ only in "
                       "non-spec settings (compare the recipes: each run's "
                       "`run_config.json`).")

    # SPEC.md 50.1.5 (v0.36): the N-run compare — generalises the two-run
    # diff above (38.4) to 2-3 runs: overlaid running-best curves (50.1.3),
    # the per-run gate rows + canonical recipes (50.1.1/37.1), and the
    # best_spec union table. Inert data rendering (50.3.1).
    with st.expander("Compare 2–3 runs — curves, recipes, gates (SPEC.md 50.1)"):
        sel = st.multiselect("Runs (pick 2–3)", ids, key="nrun_sel")
        if len(sel) > 3:
            st.caption("Comparing the first three selected runs (at most "
                       "three, SPEC.md 50.1.5).")
            sel = sel[:3]
        if len(sel) < 2:
            st.caption("Pick two or three runs to compare (SPEC.md 50.1.5).")
            return
        tagged = []
        for rid in sel:  # 50.1.4 pattern: tag run_id = dir name (summaries
            p = Path(runs_dir) / rid / "summary.json"  # don't carry it, 50.1.1)
            if not p.is_file():
                st.warning(f"run {rid!r} has no summary.json — nothing to "
                           f"compare (SPEC.md 50.1.5).")
                return
            s = json.loads(p.read_text(encoding="utf-8"))
            s["run_id"] = rid
            tagged.append(s)
        d = diff_n_summaries(tagged)
        # curves (50.1.2): each run's scored log rows; a missing
        # experiments.jsonl yields [] (the run drops out of the overlay)
        named = []
        for rid in sel:
            ep = Path(runs_dir) / rid / "experiments.jsonl"
            rows = ([json.loads(l) for l in
                     ep.read_text(encoding="utf-8").splitlines()
                     if l.strip()] if ep.is_file() else [])
            named.append((rid, running_best_curve(rows)))
        st.markdown(svg_run_curves(named), unsafe_allow_html=True)
        st.dataframe([
            {"run": r["run_id"], "final": r["final_score"],
             "target": r["target"],
             "gate": {True: "PASS", False: "MISS"}.get(r["met_target"], "—"),
             "exps": r["experiments_run"], "wall_s": r["wall_seconds"]}
            for r in d["runs"]
        ], width="stretch")
        if d["best_run_id"] is not None:
            st.caption(f"best: {d['best_run_id']} "
                       f"(score span {d['score_span']}) (SPEC.md 50.1.1)")
        for rid in sel:  # the canonical recipe per run (37.1.4)
            cp = Path(runs_dir) / rid / "run_config.json"
            try:
                cfg = RunConfig.from_dict(
                    json.loads(cp.read_text(encoding="utf-8")))
                recipe = " ".join(fit_recipe(cfg))
            except Exception:
                recipe = "n/a (no run_config.json — pre-v0.23 run)"
            st.caption(f"recipe — {rid}")
            st.code(recipe)
        if d["spec_fields"]:
            st.dataframe([
                {"field": f["field"],
                 **{f"run {i + 1}": _spec_v(v)
                    for i, v in enumerate(f["values"])}}
                for f in d["spec_fields"]
            ], width="stretch")
        else:
            st.caption("`best_spec` is identical across the runs — the runs "
                       "differ only in non-spec settings (see the recipes).")


def _launcher_runs_dir() -> str:
    """`--runs-dir` forwarded by the SPEC.md 23.4 launcher (STREAMLIT_ARGS)."""
    toks = os.environ.get("STREAMLIT_ARGS", "").split()
    for i, t in enumerate(toks):
        if t in ("--runs-dir", "--runs-dir=") and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith("--runs-dir="):
            return t.split("=", 1)[1]
    return "runs"


def _spec_v(v) -> str:
    """Spec value to chip text (SPEC.md 26.2): None renders as an em dash,
    list values are joined (e.g. hidden-layer sizes)."""
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def _inspector_field(r: dict) -> None:
    """SPEC.md 59.1: one parameter-inspector row — a compact, read-only
    block. Architecture rows (the `SPEC_FIELDS` registry, 36.2) carry the
    families, the current / baseline / best-seen values, the measured Δ,
    the win-rate (a progress bar), and the 95% CI; loop rows (the `KNOBS`
    registry, 33.2) carry the knob's CLI flags + app widget. Display-only
    (no interactive widget — AppTest-safe, 59.4)."""
    cur, base = r.get("current"), r.get("baseline")
    if r["layer"] == "architecture":
        fams = ", ".join(r.get("families") or []) or "—"
        st.markdown(
            f"**{r['name']}** · `{r.get('kind') or '—'}` · {fams} · "
            f"§{r.get('spec_ref')}")
        delta = r.get("delta")
        st.caption(
            f"current `{_spec_v(cur)}` · baseline `{_spec_v(base)}` · "
            f"best-seen `{_spec_v(r.get('best_seen'))}`"
            + (f" (Δ {float(delta):+.2f})" if delta is not None else ""))
        wr = r.get("win_rate")
        if wr is not None:  # a bandit-policy run (field_stats present)
            lo, hi = r.get("ci_lo"), r.get("ci_hi")
            ci = (f" · 95% CI [{lo:.2f}–{hi:.2f}]"
                  if lo is not None and hi is not None else "")
            st.progress(float(wr), text=f"win-rate {float(wr):.0%}{ci}")
        space = r.get("space")
        if space:
            st.caption("space: " + " · ".join(_spec_v(v) for v in space))
    else:  # loop (KNOBS, 33.2) — no win-rate (not bandit arms)
        cli = (", ".join("--" + c.replace("_", "-")
                         for c in (r.get("cli") or [])) or "—")
        st.markdown(f"**{r['name']}** · §{r.get('spec_ref')}")
        st.caption(
            f"current `{_spec_v(cur)}` · baseline `{_spec_v(base)}` · "
            f"cli {cli} · widget `{r.get('app_widget') or '—'}`")


def _drill_body(u: dict, gm: dict, reason: str,
                target: float | None = None) -> None:
    """SPEC.md 52.1.2 (v0.38): the drill-down body — the per-candidate
    gate-math table (52.1.1: the score gate 39.2.2, the overfit gate
    18.5, the CI gate 18.6), the spec-diff line (26.2), and the
    candidate's train/holdout loss curves (28.1, the ready-made
    `svg_loss_curves`). Shared by the live drill (the latest candidate,
    52.1.2) and the Experiments tab (every candidate, 52.1.2). An
    unscored (dup) candidate renders the honest caption instead of the
    numbers (52.1.1: every output `None` when its input is not
    finite).

    SPEC.md 53.1 (v0.39): the body now leads with the gate number-line
    — the same 52.1.1 gate-math numbers + the 51.3.2 reason word + the
    target, rendered as a picture (best/target markers, the CI band,
    the candidate's dot, the margin annotation)."""
    st.markdown(svg_gate_line(
        gm.get("candidate"), gm.get("best_before"),
        target, gm.get("z_se") or 0.0,
        bool(gm.get("accepted", False)), reason), unsafe_allow_html=True)

    def _fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    if gm.get("candidate") is None:
        st.caption("unscored (free duplicate, R3) — the gate math is "
                   "undefined for this step (SPEC.md 52.1.1)")
    else:
        st.dataframe([
            {"metric": "candidate score", "value": _fmt(gm.get("candidate")),
             "gate": "the holdout score (SPEC.md 18.3)"},
            {"metric": "best before", "value": _fmt(gm.get("best_before")),
             "gate": "the running best at this step (39.2.2)"},
            {"metric": "delta (candidate - best)",
             "value": _fmt(gm.get("delta")),
             "gate": "the score gate — 39.2.2 priority 1"},
            {"metric": "gen gap (holdout - generation)",
             "value": _fmt(gm.get("gen_gap")), "gate": "SPEC.md 18.5"},
            {"metric": "gen tolerance (5% of score)",
             "value": _fmt(gm.get("gen_tol")),
             "gate": "the overfit gate — SPEC.md 18.5"},
            {"metric": "SE (block std)", "value": _fmt(gm.get("se")),
             "gate": "SPEC.md 18.3/18.6 (52.1.1)"},
            {"metric": "z (the env's z)", "value": _fmt(gm.get("z")),
             "gate": "SPEC.md 18.6 (0 = legacy)"},
            {"metric": "z * SE threshold", "value": _fmt(gm.get("z_se")),
             "gate": "accept iff delta > max(0, z*SE) — SPEC.md 18.6"},
        ], width="stretch")
        st.caption(f"verdict: **{reason}** (the 51.3.2 reason column)")
    diffs = u.get("spec_diff") or []
    if diffs:
        st.caption("spec: " + " · ".join(
            f"{d['field']}: {_spec_v(d['old'])} → {_spec_v(d['new'])}"
            for d in diffs))
    else:
        st.caption("spec: unchanged this step (no field mutation)")
    st.markdown(svg_loss_curves(u.get("loss_history") or []),
                unsafe_allow_html=True)


def _palette_kwargs() -> dict:
    """SPEC.md 51.4.4 (v0.37): the Setup-tab `cb_palette` checkbox → kwargs
    for the result view's multi-series SVG re-render. Off (default) → `{}`
    (the stored pre-v0.37 SVGs render byte-identically, G2). On →
    `palette="okabe"` + the auto-detected dark mode (`theme.base == "dark"`).
    A view preference, not a run knob (51.4.4: `ALL_KNOBS` unchanged)."""
    if not st.session_state.get("cb_palette", False):
        return {}
    try:
        dark = st.get_option("theme.base") == "dark"
    except Exception:  # pragma: no cover — the theme option is always present
        dark = False
    return {"palette": "okabe", "dark": bool(dark)}


def _pal_svg(pal: dict, stored: str, recompute) -> str:
    """SPEC.md 51.4.4 (v0.37): the palette-on/off SVG choice. Off → the
    stored pre-v0.37 SVG (byte-identical, G2). On → re-render the
    multi-series SVG from the stored data (a deterministic function of the
    same inputs); a run dir that lacks the source falls back to `stored`."""
    if not pal:
        return stored
    try:
        return recompute()
    except Exception:
        return stored  # an older run dir may lack the re-render source



def _resolve_csv(upload, path_str: str) -> str | None:
    """Uploaded bytes → session temp file; otherwise the sidebar path.

    v0.10 (SPEC.md 24.5): the sidebar path may be a directory of labelled
    images/audio as well as a CSV file; uploads stay CSV.
    """
    if upload is not None:
        dest = Path(tempfile.gettempdir()) / "autorefine_dashboard"
        dest.mkdir(parents=True, exist_ok=True)
        name = Path(upload.name).name or "upload.csv"
        out = dest / name
        out.write_bytes(upload.getvalue())
        return str(out)
    p = path_str.strip()
    if p and (Path(p).is_file() or Path(p).is_dir()):
        return p
    return None


# SPEC.md 32.1: the preview computation is cached on (path, mtime_ns, size)
# — the upload flow reuses one temp dir, so a changed file re-probes;
# same file + same stamp is served from the cache
@st.cache_data
def _preview_data(csv_path: str, stamp: int, size: int) -> dict:
    """SPEC.md 32.1: pure preview computation (no `st.*` calls) —
    inferred head/splits caption + first rows; returns
    {"kind", "caption", "header", "rows"} or {"error": msg}.
    v0.10 (SPEC.md 24.5): media directories preview their items."""
    from autorefine.tasks import TASKS, detect_modality
    p = Path(csv_path)
    if not p.is_file() and not p.is_dir():
        return {"error": f"no such file or directory: {p}"}
    if p.is_file():
        cls = CsvTask
    else:
        m = detect_modality(p)
        if m is None or m == "mixed":
            return {"error": (f"Directory {p} holds no image/audio items "
                             f"(or is mixed) — one subfolder per class, "
                             f"or an index.csv (SPEC.md 24.2/24.5)")}
        cls = TASKS[m]
    try:
        probe = cls(seed=0, path=csv_path)
    except ValueError as exc:
        return {"error": f"data not usable as a task: {exc}"}
    # SPEC.md 36.1.5 (v0.22, G1): the caption reads the task's declared
    # `metric` instead of guessing from the head
    metric_txt = (f"accuracy ({probe.n_outputs} classes: "
                  f"{probe.class_values})" if probe.metric == "accuracy"
                  else f"{probe.metric} (regression)")
    caption = (
        f"label **{probe.label_name}** · metric **{metric_txt}** · "
        f"features {', '.join(probe.feature_names)} · "
        f"items {len(probe._x_tr)}/{len(probe._x_ho)}/{len(probe._x_ge)} "
        f"(train/holdout/gen)"
    )
    if p.is_file():
        # simple deterministic preview: header + first 5 data rows
        with p.open(encoding="utf-8", newline="") as f:
            reader = _csv.reader(f)
            header = next(reader)
            data = [next(reader, None) for _ in range(5)]
            data = [r for r in data if r]
        return {"kind": "csv", "caption": caption,
                "header": header, "rows": data}
    rows = []
    for sub in sorted(p.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                if f.is_file():
                    rows.append({"class": sub.name, "file": f.name})
                    if len(rows) >= 5:
                        break
        if len(rows) >= 5:
            break
    return {"kind": "media", "caption": caption,
            "header": None, "rows": rows}


def _preview(csv_path: str) -> None:
    """Inferred head/splits + first rows (SPEC.md 23.2 data preview) —
    the thin renderer over the cached `_preview_data` (SPEC.md 32.1);
    v0.10 (SPEC.md 24.5): media directories preview their items."""
    p = Path(csv_path)
    if not p.exists():
        st.warning(f"no such file or directory: {p}")
        return
    stat = p.stat()
    data = _preview_data(csv_path, stat.st_mtime_ns, stat.st_size)
    if "error" in data:
        st.warning(data["error"])
        return
    st.caption(data["caption"])
    import pandas as pd  # a streamlit dependency, app-only
    if not data["rows"]:
        return
    if data["kind"] == "csv":
        st.dataframe(pd.DataFrame(data["rows"], columns=data["header"]).head(5),
                     width="stretch")
    else:
        st.dataframe(pd.DataFrame(data["rows"]).head(5), width="stretch")


def _worker_loop(runner: DashboardRunner, record: dict) -> None:
    """SPEC.md 51.2.3 (v0.37): the daemon worker — drives the runner and
    posts ``info`` / ``update`` / ``result`` / ``error`` / ``end`` messages
    onto the shared ``record``.

    The main thread drains those messages synchronously (the AppTest
    contract: a run still completes within one script run). This thread
    only ever touches the runner + the record — never ``st.session_state``
    (Streamlit's session is not thread-safe); the drain does the rendering.
    A Stop request (``record["stop"]["flag"]``) is honored *between*
    experiments: it calls ``request_stop()`` before the next ``next()``
    (51.2.3), never inside one.
    """
    msgs = record["messages"]
    q = record["queue"]

    def post(kind: str, payload) -> None:
        msgs.append((kind, payload))
        record["last_msg_at"] = time.monotonic()  # 55.1: freshness (stream mode)
        q.put(True)  # a wakeup token; the payload itself lives in `msgs`

    try:
        info = runner.start()
        post("info", info)
        while not runner.done:
            if record["stop"]["flag"]:  # 51.2.3: honor Stop between experiments
                runner.request_stop()
            u = runner.next()
            post("update", u)
            if u.get("done"):
                break
        res = runner.finish()
        post("result", res)
    except Exception as exc:  # start() ValueError + any runtime error (51.2.3)
        post("error", exc)
    finally:
        post("end", None)


def _drain_live(record: dict, narrate: bool = False,
                stream_mode: bool = False) -> None:
    """SPEC.md 51.2.3 (v0.37): the synchronous drain — render the existing
    live UI (Task line, table, curve, decision views, progress) on the main
    thread from the worker's messages. A run still completes within one
    script run (the AppTest contract, 51.1.2).

    ``stream_mode`` (SPEC.md 55.1, v0.41) is the opt-in **live** path: when
    set, the drain is *non-blocking* — it renders the messages available so
    far and returns as soon as the worker has caught up (the app then
    ``st.rerun()``s and re-drains, so the page *ticks*), and it shows the
    live freshness caption (55.1) + the pulsing best-score sparkline
    (55.1.2). Default (``False``) is the byte-identical synchronous drain —
    every AppTest runs this path (55.4).

    Re-runnable: it replays ``record["messages"]`` from the start, so if a
    script re-run **preempts** a live drain, the new run reattaches and
    re-renders from the accumulated messages (51.2.3/51.2.4) — the worker
    still owns the runner and writes the full artifact set.

    The live table carries the 51.3.2 **reason** column (accepted /
    score / overfit / ci / dup) via ``candidate_reason`` — the running best
    is tracked (the baseline seeds it; each update's ``best_score``
    succeeds it), so the verdict is correct for every row.
    """
    runner = record["runner"]
    msgs = record["messages"]
    q = record["queue"]
    thread = record["thread"]
    idx = 0

    import pandas as pd  # a streamlit dependency, app-only

    info = None
    rows: list[dict] = []
    best_series: list[float] = []
    best_before = None
    stream: list[dict] = []
    table = chart = bar = note = narr = None
    vbars = vmatrix = vtimeline = vucb = vstrip = vscatter = None
    vfresh = vspark = None  # 55.1 (v0.41): the live freshness + sparkline
    # 56.5 (v0.42): the reduced-motion opt-out (A.6) — read once here; the
    # Setup tab renders before the drain, so it is set by the time either
    # sparkline site runs (default off → byte-identical, 56.6)
    reduced_motion = st.session_state.get("reduce_motion", False)

    while True:
        if idx >= len(msgs):
            if not thread.is_alive():
                record["drained"] = True
                break
            if stream_mode:  # 55.1: non-blocking — the app ticks and re-drains
                break
            try:
                q.get(timeout=300)
            except _queue.Empty:
                st.error("run stalled — no worker progress; press Stop or "
                         "refresh (SPEC.md 51.2.4)")
                record["drained"] = True
                break
        kind, payload = msgs[idx]
        idx += 1

        if kind == "info":
            info = payload
            st.subheader("Task")
            st.caption(
                f"label **{info['label']}** · head **{info['head']}** · "
                f"rows {info['rows']['train']}/{info['rows']['holdout']}/{info['rows']['gen']} "
                f"· baseline **{info['baseline_score']:.2f}** vs target **{info['target']:.1f}** "
                f"· {info['policy']}, seed {info['seed']}"
            )
            if narrate:  # SPEC.md 48.5.2: the friendly first line (off by default)
                st.caption(narrate_baseline(info))
            # all table cells are strings (st.dataframe → Arrow; mixed types
            # break it); the reason column is the 51.3.2 addition
            rows = [{
                "#": 0, "accepted": "—", "reason": "—",
                "candidate": f"{info['baseline_score']:.2f}", "gen gap": "—",
                "mutation": "baseline", "best": f"{info['baseline_score']:.2f}",
                "diff": "—",
            }]
            best_series = [round(info["baseline_score"], 2)]
            best_before = info["baseline_score"]
            rows_meta = [None]  # 52.2.2: per-row mutations (baseline: none)
            table = st.empty()
            chart = st.empty()
            bar = st.progress(0.0, text="starting…")
            note = st.empty()
            narr = st.empty() if narrate else None  # SPEC.md 48.5.1 placeholder
            # decision-view placeholders, live (SPEC.md 26.5): D1 win-rate bars,
            # V2 field x value matrix, D3 mutation timeline, D4 UCB trace
            vbars = st.empty()
            vmatrix = st.empty()  # V2 (SPEC.md 30.2)
            vtimeline = st.empty()
            vucb = st.empty() if runner.policy_name == "bandit" else None
            vstrip = st.empty()  # G1 (SPEC.md 27.1)
            vscatter = st.empty()  # G2 (SPEC.md 27.2)
            vfrontier = st.empty()  # 53.2 (v0.39) the live Pareto frontier
            vchamp = st.empty()    # 53.3 (v0.39) the live champion spec card
            vtime = st.empty()  # B2 (SPEC.md 54.2) the wall-time cost strip
            # live-interaction placeholders (SPEC.md 52, v0.38): the
            # ETA/plateau caption (52.3) + the latest-candidate drill-down
            # (52.1); the focus selectbox (52.2) is created exactly once,
            # at the first update that introduces any mutation field
            meta = st.empty()
            drill = st.empty()
            focus_ph = None
            if stream_mode:  # 55.1 (v0.41): the live freshness + sparkline
                vfresh = st.empty()
                vspark = st.empty()
                vfresh.caption(freshness_caption(0, 0.0, done=False))
                vspark.markdown(
                    svg_live_sparkline(best_series, info.get("target"),
                                       live=True, reduced_motion=reduced_motion),
                    unsafe_allow_html=True)
            continue

        if kind == "update":
            u = payload
            budget = info["budget_experiments"]
            spent = budget - int(u["experiments_left"])
            is_dup = not isinstance(u["candidate_score"], (int, float))
            # SPEC.md 51.3.2 (v0.37): the visible reason column — the
            # per-candidate gate verdict (accepted/score/overfit/ci/dup);
            # a Stop-honored final step (51.2.1) shows `stopped` instead
            # (candidate_reason covers scored candidates only)
            reason = ("stopped" if u.get("reason") == "stopped"
                      else candidate_reason(u["accepted"],
                                            u["candidate_score"],
                                            u["gen_gap"], best_before))
            rows.append({
                "#": u["index"],
                "accepted": "yes" if u["accepted"] else "no",
                "reason": reason,
                "candidate": (f"{u['candidate_score']:.2f}"
                              if isinstance(u["candidate_score"], (int, float)) else "dup"),
                "gen gap": (f"{u['gen_gap']:.2f}"
                            if isinstance(u["gen_gap"], (int, float)) else "—"),
                "mutation": ", ".join(u["mutation"]) or (u["reason"] or "—"),
                "best": f"{u['best_score']:.2f}",
                "diff": (" · ".join(
                    f"{d['field']}: {_spec_v(d['old'])} → {_spec_v(d['new'])}"
                    for d in u.get("spec_diff") or []
                ) or (u["reason"] or "—")),
            })
            rows_meta.append(list(u.get("mutation") or []))  # 52.2.2
            best_series.append(round(u["best_score"], 2))
            # the bar tracks the *budget*, not the stream-row count: spent =
            # budget − experiments_left, so free duplicate rejections (R3) do
            # not inflate the counter (SPEC.md 23.2)
            frac = max(0.0, min(1.0, spent / max(1, budget)))
            bar.progress(frac, text=(
                f"experiment {spent} of {budget}"
                + ("  ·  dup step (free, R3)" if is_dup else "")))
            # 52.2.2: the focus filter over the live table — keep only
            # the rows whose update mutated the field (the baseline row
            # mutated nothing and drops out when a field is focused)
            focus = st.session_state.get("focus_field") or "All"
            shown = ([r for r, m in zip(rows, rows_meta) if m and focus in m]
                     if focus != "All" else rows)
            table.dataframe(pd.DataFrame(shown), width="stretch")
            chart.line_chart(pd.DataFrame({"best score": best_series}))
            # decision views, live (SPEC.md 26.5) — pure over the stream so far
            stream.append(u)
            # 52.2.2: the focus-field selectbox — exactly once per script
            # run, at the first update that introduces any mutation field
            # (a second key="focus_field" mid-drain would raise; the
            # position is deterministic for a given message set)
            if focus_ph is None and fields_seen(stream):
                focus_ph = st.selectbox(
                    "Focus field (filter the table + timeline)",
                    ("All", *fields_seen(stream)), key="focus_field",
                    help="Keep only the candidates that mutated this "
                         "field — the live table and the mutation "
                         "timeline both filter (SPEC.md 52.2.2).")
            if u.get("field_stats"):  # D1: per-field win-rate bars (SPEC.md 26.1)
                vbars.bar_chart(
                    pd.DataFrame.from_dict(u["field_stats"], orient="index")
                    .sort_index())
            if u.get("field_value_stats"):  # V2 (SPEC.md 30.2)
                vmatrix.markdown(
                    svg_field_value_matrix(u["field_value_stats"]),
                    unsafe_allow_html=True)
            vtimeline.markdown(  # D3 (SPEC.md 26.3) + the 52.2.2 focus filter
                svg_mutation_timeline(
                    stream, field=None if focus == "All" else focus),
                unsafe_allow_html=True)
            vstrip.markdown(svg_score_strip(stream), unsafe_allow_html=True)
            vscatter.markdown(svg_score_gap_scatter(stream), unsafe_allow_html=True)
            vfrontier.markdown(svg_live_frontier(stream), unsafe_allow_html=True)
            vtime.markdown(  # B2 (SPEC.md 54.2): the wall-time cost strip
                svg_time_strip(stream, info.get("baseline_train_seconds")),
                unsafe_allow_html=True)
            if u["accepted"]:
                # 53.3 (v0.39): the champion card re-renders on every
                # acceptance, the just-mutated field (the mutation's
                # first field) highlighted (53.3.2)
                hl = (u.get("mutation") or [None])[0]
                vchamp.markdown(svg_architecture(
                    runner.env.best_spec.to_dict(),
                    runner.env.task.state_dim,
                    runner.env.task.n_outputs,
                    highlight=hl if isinstance(hl, str) else None),
                    unsafe_allow_html=True)
            if vucb is not None and u.get("ucb"):  # D4: bandit UCB (SPEC.md 26.4)
                trace = ucb_trace(stream, alpha=runner.policy.alpha)
                vucb.line_chart(
                    pd.DataFrame({f: trace[f] for f in sorted(trace)}
                                 ).astype("float64"))
            if narr is not None:
                # SPEC.md 48.5.1: the plain-English line replaces the terse
                # caption (the table + charts above still show every detail)
                narr.caption(narrate_step(u))
            else:
                # per-step caption, including the final step (SPEC.md 23.2)
                score_txt = (f"score {u['candidate_score']:.2f}"
                             if isinstance(u["candidate_score"], (int, float))
                             else "duplicate")
                kindtxt = ("dup step (free, R3)" if is_dup
                           else f"experiment {spent} of {budget}")
                diff_txt = " · ".join(
                    f"{d['field']}: {_spec_v(d['old'])} → {_spec_v(d['new'])}"
                    for d in u.get("spec_diff") or []
                ) or (", ".join(u["mutation"]) or "-")
                note.caption(
                    f"{kindtxt}: {'accepted +' if u['accepted'] else 'rejected '} "
                    f"{score_txt} · {diff_txt}"
                )
            # 52.3.1: the live ETA + the stall sentinel (the 31.1
            # patience semantics, display-only — the env's opt-in
            # `stall_patience` gate stays off unless explicitly set)
            streak = plateau_streak(stream)
            left = int(u.get("experiments_left") or 0)
            eta = eta_seconds(stream, left)
            eta_txt = f"≈ {eta:.1f}s" if eta is not None else "—"
            meta.caption(
                f"ETA {eta_txt} · {left} experiment(s) left"
                + (f"  ·  plateau: {streak} non-improving scored "
                   "experiments in a row — Stop would be honest "
                   "(SPEC.md 31.1)" if streak >= PLATEAU_HINT else ""))
            # 52.1.2: the live drill-down — the latest candidate's gate
            # math (52.1.1, over its CI-gate inputs) + loss curves
            gm = gate_math(u, best_before, runner.env.z_accept)
            with drill.expander(
                    f"candidate #{u['index']} — {reason} · gate math + "
                    "curves (SPEC.md 52.1)", expanded=False):
                _drill_body(u, gm, reason, info["target"])
            if stream_mode and vfresh is not None:  # 55.1: re-render live
                _el = (time.monotonic()
                       - record.get("last_msg_at", time.monotonic()))
                vfresh.caption(freshness_caption(len(stream), _el,
                                                 done=False))
                vspark.markdown(
                    svg_live_sparkline(best_series, info.get("target"),
                                       live=True, reduced_motion=reduced_motion),
                    unsafe_allow_html=True)
            best_before = u["best_score"]  # the running best for the next row
            continue

        if kind == "result":
            res = payload
            # Persistence (SPEC.md 23.2): store the finished result so the
            # Results/Experiments tabs (51.1.1) and any later re-render show
            # it. The result view itself renders in the Results tab (51.1.2).
            st.session_state["result"] = {
                "info": info, "rows": rows, "best_series": best_series,
                "res": res,
                # SPEC.md 52 (v0.38): the Experiments-tab drill-down —
                # the full update stream (each carries `se`/
                # `effective_score`, 52.1.1) + the env's z (18.6)
                "stream": stream,
                "z": runner.env.z_accept,
                # 53.3 (v0.39): the champion card's state_dim/n_out (the
                # Experiments tab re-renders the architecture SVG from
                # the stored stream)
                "state_dim": runner.env.task.state_dim,
                "n_out": runner.env.task.n_outputs,
            }
            continue

        if kind == "error":
            st.error(str(payload))
            # 56.4 (v0.42): the friendly next-step hint (A.4) — point the
            # user at the onboarding commands rather than a raw traceback
            st.caption(
                "Next steps: `autorefine doctor` checks versions + a smoke "
                "train, and `autorefine fit --dry-run` validates your data "
                "before a full run (SPEC.md 56.4).")
            continue

        if kind == "end":
            record["drained"] = True
            break


def _run(csv_path: str, label: str, target: float, policy: str, seed: int,
         experiments: int, max_train: float, runs_dir: str,
         quality: str = "v04", narrate: bool = False,
         initial_stop: bool = False,
         stream_mode: bool = False,
         steering: "SteeringState | None" = None) -> None:
    """Start the live loop (SPEC.md 51.2.3): build the runner, launch the
    daemon worker, then drain its messages synchronously into the live UI.

    ``narrate`` (default ``False``) swaps the terse per-step caption for a
    plain-English line (48.5.3). ``initial_stop`` (51.2.3) seeds the worker's
    stop flag — a Stop pressed alongside Run aborts after the first step.
    ``stream_mode`` (SPEC.md 55.1, v0.41) is the opt-in live path: the drain
    is non-blocking and the app ``st.rerun()``s to tick the page; the default
    (``False``) is the byte-identical synchronous drain (55.4).
    """
    runner = DashboardRunner(
        csv_path=csv_path, label=label or None, target=target, policy=policy,
        seed=seed, experiments=experiments, max_train_seconds=max_train,
        runs_dir=runs_dir, search_quality=quality,
        steering=steering,  # SPEC.md 59.2 (v0.45): the steering rules (None = off)
    )
    record = {
        "thread": None,
        "stop": {"flag": bool(initial_stop)},
        "queue": _queue.Queue(),
        "messages": [],
        "drained": False,
        "runner": runner,
    }
    st.session_state["_worker"] = record
    record["thread"] = threading.Thread(
        target=_worker_loop, args=(runner, record), daemon=True)
    record["thread"].start()
    _drain_live(record, narrate, stream_mode=stream_mode)
    if stream_mode and stream_tick_due(record):  # 55.1: tick the page
        st.rerun()


def _render_result(res: dict) -> None:
    """Result section (verdict, metrics, SVG plots, artifacts) — shared by
    the live run and the restored view (SPEC.md 23.2)."""
    st.divider()
    st.subheader("Result")
    if res["verdict"] == "PASS":
        st.success(
            f"PASS: final **{res['final_best_score']:.2f}** ≥ target "
            f"{res['target']:.1f} on held-out data (the loop never trained on "
            f"these points; gen_gap is the overfit guard) — SPEC.md 22.1"
        )
    else:
        st.error(
            f"MISS: final **{res['final_best_score']:.2f}** < target "
            f"{res['target']:.1f} — raise the budget (experiments / train "
            f"seconds) or extend the spec space (README 'Extending')"
        )
    # SPEC.md 51.2.3 (v0.37): a Stop-honored run finishes early — a neutral
    # note next to the honest verdict (a stopped run is not a cheating run,
    # 51.2.4). Only present when the loop ended with `stopped`.
    if res.get("finished_reason") == "stopped":
        st.warning(
            "Stopped by user — the run ended early, between experiments; the "
            "verdict and artifacts reflect the partial run so far "
            "(SPEC.md 51.2.3/51.2.4).")
    # SPEC.md 48.4 (v0.34): the plain-English "what happened" narrative —
    # always shown here (the live narrate toggle, 48.5, is separate); pure
    # over the stored result, so the restored view renders the same block.
    st.markdown(narrate_run(res))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("baseline → final",
              f"{res['baseline_score']:.2f}",
              f"{res['final_best_score']:.2f}")
    factor = res["improvement_factor"]
    m2.metric("improvement", f"{factor:.2f}×" if factor else "—")
    m3.metric("experiments", res["experiments_run"])
    wall = res["wall_seconds"]
    m4.metric("wall time", f"{wall:.1f}s" if wall is not None else "—")

    # v0.23 (SPEC.md 37): the canonical recipe + the objective-set gate rows.
    # Both via .get() — the restored view (SPEC.md 23.2) must re-render a
    # result either with or without them (older payloads lack both keys).
    if res.get("recipe"):  # 37.1.4: copy-pasteable `fit` re-run
        st.caption("Reproduce (copy-paste) — SPEC.md 37.1.4")
        st.code(" ".join(res["recipe"]))
    if res.get("gate"):  # 37.2: the per-objective acceptance rows
        gate = res["gate"]
        st.caption(
            f"gate: {len(gate['objectives'])} objective(s) (SPEC.md 37.2)")
        st.dataframe([{
            "objective": r["name"],
            "op": r["op"],
            "threshold": r["threshold"],
            "actual": ("n/a" if r["actual"] is None else round(r["actual"], 6)),
            "result": "PASS" if r["pass"] else "MISS",
        } for r in gate["objectives"]], width="stretch")

    # SPEC.md 50.2 (v0.36): one-click exports — the GUI-for-exploration ->
    # CLI-for-CI bridge, next to the copy-paste recipe above. The app writes
    # nothing into the runs tree (23.1/49.2.1): the share bundle is
    # in-memory + a download button; `autorefine share` stays the
    # file-producing surface (47.4).
    rd = Path(res["run_dir"])
    rc_path = rd / "run_config.json"
    if rc_path.is_file():
        st.download_button("Download run_config.json",
                           data=rc_path.read_bytes(),  # 50.2.2
                           file_name="run_config.json",
                           mime="application/json", key="dl_runconfig")
    else:
        st.caption("no run_config.json in this run dir — nothing to "
                   "download (pre-v0.23 run, SPEC.md 50.2.2)")
    if st.button("Build share bundle", key="build_share"):  # 50.2.3
        try:
            from autorefine.cli import _share_report_html  # 47.4.2: cli
            from autorefine.sharing import (  # owns the HTML regeneration
                share_payload, zip_bundle_bytes)
            payload = share_payload(rd, _share_report_html(rd))
            bundle = zip_bundle_bytes(payload)
            st.session_state["share_bundle_" + str(rd)] = (
                bundle, [[n, len(payload[n])] for n in sorted(payload)])
        except Exception as exc:  # local, friendly (50.2.3)
            st.error(f"share bundle failed: {exc} (SPEC.md 50.2.3)")
    stored = st.session_state.get("share_bundle_" + str(rd))
    if stored:
        bundle, listing = stored
        st.dataframe({"entry": [r[0] for r in listing],
                      "bytes": [r[1] for r in listing]}, width="stretch")
        st.download_button("Download share bundle (.zip)", data=bundle,
                           file_name=f"{rd.name}-share.zip",
                           mime="application/zip", key="dl_share")

    # SPEC.md 56.1/56.4 (v0.42): the Provenance (certificate) expander (A.1)
    # — the "what produced this result" block, rendered at view time from the
    # run dir (summary.json + run_config.json). Nothing is written into the
    # runs tree (the 47.4 file-producing surface stays `autorefine share`).
    with st.expander("Provenance (certificate)", expanded=False):
        try:
            _sd = Path(res["run_dir"])
            _summary = {}
            _sjson = _sd / "summary.json"
            if _sjson.is_file():
                _summary = json.loads(_sjson.read_text(encoding="utf-8"))
            _rc = {}
            _rcjson = _sd / "run_config.json"
            if _rcjson.is_file():
                _rc = json.loads(_rcjson.read_text(encoding="utf-8"))
            _payload = provenance_payload(
                env_provenance(),
                seed=_summary.get("seed"), task=_summary.get("task"),
                target=_summary.get("target", _rc.get("target")),
                run_config=_rc or None)
            st.markdown(svg_provenance(_payload, **_palette_kwargs()),
                        unsafe_allow_html=True)
            st.code(provenance_card(_payload), language="text")
        except Exception as exc:  # 56.4: a friendly error, not a page crash
            st.error(f"provenance card unavailable: {exc} (SPEC.md 56.4)")

    # SPEC.md 36.2 (v0.22, G2): the app's spec surface reads the field
    # registry — one table for the field list, not a per-surface literal
    from autorefine.improver.specspace import SPEC_FIELD_NAMES
    st.caption(
        f"spec space: {len(SPEC_FIELD_NAMES)} fields — "
        f"{', '.join(SPEC_FIELD_NAMES)} (SPEC.md 36.2)"
    )

    st.subheader("Plots")
    # SPEC.md 51.4.4 (v0.37): the colorblind-safe re-render — when the Setup
    # tab's `cb_palette` checkbox is on, the multi-series SVGs re-render from
    # the stored data with `palette="okabe"` + the auto-detected dark mode;
    # off (default) they render the stored pre-v0.37 SVGs byte-identically.
    pal = _palette_kwargs()
    st.markdown(
        _pal_svg(pal, res["svg_score"],
                 lambda: svg_score_curve(_load_entries(res["run_dir"]), **pal)),
        unsafe_allow_html=True)
    st.markdown(res["svg_pareto"], unsafe_allow_html=True)

    # decision views, final state (SPEC.md 26.5) — pure over the stored
    # update stream, so the restored view re-renders them with no new state
    st.subheader("Decision views")
    import pandas as pd  # a streamlit dependency, app-only
    fs = res.get("field_stats")
    if fs:  # D1: per-field win-rate bars (SPEC.md 26.1)
        st.bar_chart(pd.DataFrame.from_dict(fs, orient="index").sort_index())
    if res.get("field_value_svg"):  # V2: field x value matrix (SPEC.md 30.2)
        st.markdown(res["field_value_svg"], unsafe_allow_html=True)
    if res.get("family_bars_svg"):  # V5: model-family bars (SPEC.md 30.5)
        st.markdown(res["family_bars_svg"], unsafe_allow_html=True)
    if res.get("timeline_svg"):  # D3: mutation timeline (SPEC.md 26.3)
        st.markdown(res["timeline_svg"], unsafe_allow_html=True)
    ucb = res.get("ucb_trace")
    if ucb:  # D4: bandit UCB trace (SPEC.md 26.4); None for the search policy
        st.line_chart(pd.DataFrame({f: ucb[f] for f in sorted(ucb)})
                      .astype("float64"))
    # acceptance-gate views (SPEC.md 27): G1 + G2 always; G3 only when the
    # run has a ladder (finish() returns None otherwise, SPEC.md 27.3)
    if res.get("strip_svg"):  # G1: candidate score strip (SPEC.md 27.1)
        st.markdown(
            _pal_svg(pal, res["strip_svg"],
                     lambda: svg_score_strip(res["updates"], **pal)),
            unsafe_allow_html=True)
    if res.get("scatter_svg"):  # G2: score vs gen-gap (SPEC.md 27.2)
        st.markdown(
            _pal_svg(pal, res["scatter_svg"],
                     lambda: svg_score_gap_scatter(res["updates"], **pal)),
            unsafe_allow_html=True)
    if res.get("time_strip_svg"):  # B2 (SPEC.md 54.2): wall-time cost strip
        st.markdown(
            _pal_svg(pal, res["time_strip_svg"],
                     lambda: svg_time_strip(res["updates"],
                                            res.get("baseline_train_seconds"),
                                            **pal)),
            unsafe_allow_html=True)
    if res.get("ladder_svg"):  # G3: curriculum ladder (SPEC.md 27.3)
        st.markdown(res["ladder_svg"], unsafe_allow_html=True)

    # SPEC.md 57 (v0.43): the research decision surfaces — pure derivations
    # over the run's experiments.jsonl (57.5); no new logged data, no new
    # widget (AppTest-safe), and outside the byte-identity-pinned report.
    st.subheader("Research views")
    try:
        _ren = _load_entries(res["run_dir"])
    except (ValueError, OSError) as exc:  # 49.4.3: a friendly error, not a crash
        st.error(f"research views unavailable: {exc} (SPEC.md 57.5)")
        _ren = []
    st.markdown(svg_spec_lineage(spec_lineage(_ren), **pal),
                unsafe_allow_html=True)
    st.markdown(svg_field_response(field_response_stats(_ren), **pal),
                unsafe_allow_html=True)
    st.markdown(svg_gate_region(gate_region_candidates(_ren), **pal),
                unsafe_allow_html=True)
    _fs = res.get("field_stats")
    if isinstance(_fs, dict) and _fs:
        _ups = res.get("updates") or []
        _ucb = (_ups[-1].get("ucb") or {}) if _ups else {}
        st.markdown(svg_bandit_beliefs(bandit_beliefs(_fs, _ucb), **pal),
                    unsafe_allow_html=True)
    else:
        st.caption("bandit belief bars need a bandit-policy run (field_stats) "
                   "— SPEC.md 57.4")

    # SPEC.md 59.1 (v0.45): the parameter inspector — the two real layers
    # (architecture `SPEC_FIELDS` + loop `KNOBS`) as one read-only block
    # each. Pure derivation over the run's data (59.1); display-only
    # (AppTest-safe); a friendly caption on failure (the 49.4.3 pattern),
    # never a page crash.
    st.subheader("Parameter inspector (SPEC.md 59.1)")
    try:
        _ins_entries = _ren  # the entries loaded in the Research views above
        _ins_best = res.get("best_spec")
        _ins_cfg = (res.get("summary") or {}).get("run_config")
        _ins_fs = res.get("field_stats")
        _ups_i = res.get("updates") or []
        _ins_ucb = (_ups_i[-1].get("ucb") or {}) if _ups_i else {}
        _rows = parameter_inspection(_ins_entries, _ins_best, _ins_cfg,
                                     _ins_fs, _ins_ucb)
    except (ValueError, TypeError, KeyError) as exc:  # 49.4.3
        st.error(f"parameter inspector unavailable: {exc} (SPEC.md 59.1)")
    else:
        _ca, _cb = st.columns(2)
        with _ca:
            st.markdown("**Architecture** (the `SPEC_FIELDS` registry, 36.2)")
            for r in _rows:
                if r["layer"] == "architecture":
                    _inspector_field(r)
        with _cb:
            st.markdown("**Loop** (the `KNOBS` registry, 33.2)")
            for r in _rows:
                if r["layer"] == "loop":
                    _inspector_field(r)

    # SPEC.md 60 (v0.46): what-if & comparison — the spec space *steered*,
    # not just read: the live what-if preview (60.1), the spec fingerprint
    # "DNA" (60.2), the interaction heatmap (60.3), and the objective-
    # weight reslice (60.4). Pure derivations over the run's data (60.5);
    # display-only (AppTest-safe); a friendly caption on failure (the
    # 49.4.3 pattern), never a page crash.
    st.subheader("What-if & comparison (SPEC.md 60)")
    try:
        # 60.1 — the live what-if preview: one field, one value from its
        # registry space, zero retraining
        _wf_field = st.selectbox("What-if field", list(SPEC_FIELDS),
                                 key="wf_field", index=3)
        _wf_space = list(SPEC_FIELDS[_wf_field].space)
        _wf_value = st.selectbox(
            f"what-if value ({_wf_field})", _wf_space, key="wf_value",
            index=min(2, len(_wf_space) - 1),
            format_func=lambda v: ", ".join(str(x) for x in v)
            if isinstance(v, (list, tuple)) else str(v))
        try:
            _wf = whatif_preview(_ren, res.get("best_spec"), _wf_field,
                                 _wf_value)
        except ValueError as exc:  # 49.4.3: a friendly error, not a crash
            st.error(f"what-if preview: {exc} (SPEC.md 60.1)")
        else:
            _wfl, _wfr = st.columns(2)
            with _wfl:
                st.markdown("**Would-be candidate**")
                if _wf["valid"]:
                    st.markdown(svg_architecture(
                        _wf["candidate"], res.get("state_dim"),
                        res.get("n_out"), highlight=_wf_field),
                        unsafe_allow_html=True)
                else:
                    st.error(f"invalid combination: {_wf['error']} "
                             f"(SPEC.md 60.1)")
            with _wfr:
                st.markdown("**Estimated effect (logged surface, 57.2)**")
                st.markdown(svg_whatif_effect(_wf, **pal),
                            unsafe_allow_html=True)
        # 60.2 — the spec fingerprint ("DNA"); optional A-vs-B compare
        _fp_a = (res.get("best_spec") if isinstance(res.get("best_spec"),
                                                    dict)
                 else DEFAULT_SPEC.to_dict())
        if st.checkbox("Compare champion vs baseline fingerprint",
                       value=False, key="fp_compare"):
            st.markdown(svg_spec_fingerprint(_fp_a, DEFAULT_SPEC.to_dict(),
                                             **pal),
                        unsafe_allow_html=True)
        else:
            st.markdown(svg_spec_fingerprint(_fp_a, **pal),
                        unsafe_allow_html=True)
        # 60.3 — the interaction heatmap (co-mutated field pairs)
        st.markdown(svg_interaction_heatmap(interaction_matrix(_ren), **pal),
                    unsafe_allow_html=True)
        # 60.4 — the objective-weight reslice (interactive re-gater)
        _wsc = st.slider("Objective weight: score", 0.0, 1.0, 0.5, 0.05,
                         key="wobj_score")
        _wtc = st.slider("Objective weight: train-time", 0.0, 1.0, 0.25, 0.05,
                         key="wobj_train")
        _wzc = st.slider("Objective weight: model-size", 0.0, 1.0, 0.25, 0.05,
                         key="wobj_size")
        _wtg = st.slider("Composite target", 0.5, 1.0, 0.75, 0.01,
                         key="wobj_target")
        try:
            _wr2 = weighted_reslice(_ren, _wsc, _wtc, _wzc, _wtg,
                                    state_dim=res.get("state_dim"),
                                    n_out=res.get("n_out"),
                                    grid=res.get("grid"))
        except ValueError as exc:  # 49.4.3
            st.error(f"objective reslice: {exc} (SPEC.md 60.4)")
        else:
            st.markdown(svg_weighted_reslice(_wr2, **pal),
                        unsafe_allow_html=True)
            if _wr2["pass"]:
                st.success(f"{_wr2['passing']}/{_wr2['pool']} logged "
                           f"candidate(s) pass the weighted gate; "
                           f"counterfactual final = candidate "
                           f"{_wr2['final']['cand']} (SPEC.md 60.4)")
            else:
                st.warning("no logged candidate passes the weighted gate "
                           "(SPEC.md 60.4)")
    except (ValueError, TypeError, KeyError) as exc:  # 49.4.3
        st.error(f"what-if & comparison unavailable: {exc} (SPEC.md 60)")

    # learning views (SPEC.md 28): computed once in finish(), re-rendered
    # here from the stored result — the restored view shows the same views
    if any(res.get(k) for k in ("loss_curves", "per_class_svg",
                                "confusion_svg", "difficulty_svg",
                                "boundary_svg", "error_gallery",
                                "arch_svg")):
        st.subheader("Learning views")
        curves = res.get("loss_curves") or {}
        if curves:  # C1 (SPEC.md 28.1): pick an experiment's train/holdout curves
            pick = st.selectbox("Training curves — experiment",
                                list(curves), key="curve_pick")
            st.markdown(svg_loss_curves(curves[pick]),
                        unsafe_allow_html=True)
        if res.get("per_class_svg"):  # C2 (SPEC.md 28.2)
            st.markdown(res["per_class_svg"], unsafe_allow_html=True)
        if res.get("confusion_svg"):  # C2 (SPEC.md 28.2)
            st.markdown(res["confusion_svg"], unsafe_allow_html=True)
        if res.get("difficulty_svg"):  # 58.1 (v0.44): easiest -> hardest
            st.markdown(res["difficulty_svg"], unsafe_allow_html=True)
        if res.get("boundary_svg"):  # V3 (SPEC.md 30.3): where it fails on a 2-D plane
            st.markdown(res["boundary_svg"], unsafe_allow_html=True)
        gallery = res.get("error_gallery")
        if gallery:  # C3 (SPEC.md 28.3): misclassified holdout items
            _render_gallery(gallery)
        if res.get("arch_svg"):  # C4 (SPEC.md 28.4): what we ended up building
            st.markdown(res["arch_svg"], unsafe_allow_html=True)

    # SPEC.md 49 (v0.35): the advanced analysis section — three panels,
    # every action behind a button (inert until pressed, 49.4.2); shared
    # by the live and restored views (49.4.1).
    _render_advanced(res)

    st.subheader("Artifacts")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button("report.html", res["report_html"], mime="text/html",
                       file_name="report.html")
    c2.download_button("best_spec.json", res["artifacts"]["best_spec.json"],
                       mime="application/json", file_name="best_spec.json")
    c3.download_button("best_model.npz", res["artifacts"]["best_model.npz"],
                       mime="application/octet-stream", file_name="best_model.npz")
    c4.download_button("experiments.jsonl", res["artifacts"]["experiments.jsonl"],
                       mime="application/jsonlines", file_name="experiments.jsonl")
    # 58.3 (v0.44): the run dossier — the whole story of this run, one
    # self-contained HTML file; a build failure is a friendly caption
    # (the 49.4.3 pattern), never a page crash
    try:
        dossier = build_dossier(
            res["run_dir"],
            state_dim=res.get("state_dim"),
            n_out=res.get("n_out"),
            grid=res.get("grid"),
            difficulty=res.get("difficulty"),
            gallery=res.get("error_gallery"),
            per_class_svg=res.get("per_class_svg"),
            confusion_svg=res.get("confusion_svg"),
        ).encode("utf-8")
    except (ValueError, FileNotFoundError, OSError) as exc:
        st.caption(f"dossier unavailable: {exc} (SPEC.md 58.3/49.4.3)")
    else:
        c5.download_button("dossier.html", dossier, mime="text/html",
                           file_name="dossier.html")
    st.caption(
        f"run dir: `{res['run_dir']}` — or CLI: "
        f"`python -m autorefine report --run {res['run_dir']} --html`"
    )


def _load_entries(run_dir: str) -> list:
    """SPEC.md 49.1.4 (v0.35): the run's experiments.jsonl rows — the
    what-if pool (40.2). A missing file is a friendly error, not a page
    crash (49.4.3)."""
    p = Path(run_dir) / "experiments.jsonl"
    if not p.is_file():
        raise ValueError(f"no experiments.jsonl in {run_dir} — not a finished run")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _ab_rerun(res: dict) -> dict:
    """SPEC.md 49.3.2 (v0.35): the policy A/B re-run — the run's own
    run_config (37.1: task path/label/split, seed, target, budget, the
    ci_blocks > 0 ⇒ v04 quality rule) with the opposite policy. The
    second run is a *normal* run (23.1): its own timestamped run dir,
    artifacts, and registry entry (38.1), in the same runs dir.
    A missing data path or a non-bandit/search policy is a ValueError
    (49.3.1/49.3.2)."""
    cfg = (res.get("summary") or {}).get("run_config") or {}
    tc = cfg.get("task_config") or {}
    path = tc.get("path")
    if not path:
        raise ValueError("the run's data path is not in its run_config — "
                         "cannot re-run (SPEC.md 49.3.2)")
    policy = opposite_policy(cfg.get("policy"))  # ValueError: rl/unknown
    runner = DashboardRunner(
        csv_path=path,
        label=tc.get("label"),
        split_frac=float(tc.get("split_frac", 0.2)),
        target=float(cfg.get("target", res.get("target", 95.0))),
        policy=policy,
        seed=int(cfg.get("seed", 7)),
        experiments=int(cfg.get("max_experiments", 30)),
        max_seconds=float(cfg.get("max_wall_seconds", 900.0)),
        max_train_seconds=float(cfg.get("max_train_seconds", 30.0)),
        runs_dir=str(Path(res["run_dir"]).parent),
        search_quality="v04" if int(cfg.get("ci_blocks", 0)) > 0 else "legacy",
    )
    runner.start()
    return runner.run_all()


def _render_advanced(res: dict) -> None:
    """SPEC.md 49.4 (v0.35): the "Advanced analysis" section of the result
    view — the three interactive panels (49.1.4 what-if re-gating,
    49.2.3 editable best-spec re-scoring, 49.3.4 policy A/B). Every
    action is behind a button with a unique key= (49.4.2); until pressed,
    the section is captions + inputs only. Each panel fails locally and
    friendly (49.4.3). Shared by the live and restored views (49.4.1).
    """
    st.subheader("Advanced analysis")
    run_dir = res.get("run_dir")

    # --- 49.1.4: interactive what-if re-gating (zero training) -------------
    with st.expander("What-if re-gating — move the bars, zero training (49.1)"):
        st.caption(
            "Re-score this run's logged history against a new objective set — "
            "the app form of `report --what-if` (SPEC.md 40.2). The `model` "
            "objective is evaluated on the final best model's artifact "
            "(SPEC.md 49.1.2). Nothing is trained or written.")
        c1, c2 = st.columns(2)
        wi_target = c1.number_input(
            "score >= target", value=float(res.get("target", 95.0)),
            min_value=0.0, step=1.0, key="wi_target")
        wi_train = c2.number_input(
            "train <= seconds", value=30.0, min_value=0.1, step=5.0,
            key="wi_train")
        wi_model = st.number_input(
            "model <= values", value=100000, min_value=1, step=10000,
            key="wi_model")
        cb_score = st.checkbox("score >= target", value=True, key="wi_cb_score")
        cb_train = st.checkbox("train <= seconds", value=True, key="wi_cb_train")
        cb_model = st.checkbox(
            "model <= values (final artifact)", value=False, key="wi_cb_model")
        if st.button("Re-score the logged history", key="wi_button",
                     type="primary"):
            objectives = []
            if cb_score:
                objectives.append(Objective("score", ">=", float(wi_target)))
            if cb_train:
                objectives.append(Objective("train", "<=", float(wi_train)))
            if cb_model:
                objectives.append(Objective("model", "<=", float(wi_model)))
            try:
                entries = _load_entries(run_dir)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
                entries = None
            if entries is not None:
                model_actual = None
                if cb_model:
                    try:  # 49.1.2: the artifact size; failure -> honest MISS
                        model_actual = int(model_size(Path(run_dir) / "best_model.npz"))
                    except Exception:
                        model_actual = None
                block = what_if_block(entries, objectives, model_actual)
                if block["pass"]:
                    st.success(
                        "PASS — under this bar the run's logged history still "
                        "contains a passing candidate (SPEC.md 49.1.1).")
                else:
                    st.error(
                        "MISS — no logged candidate meets every selected "
                        "objective (SPEC.md 49.1.3).")
                wf = block.get("what_if")
                if wf:
                    final = wf.get("final") or {}
                    st.dataframe([{
                        "scored pool": wf["pool"],
                        "passing": wf["passing"],
                        "counterfactual final": final.get("cand", "—"),
                        "final score": final.get("score", "—"),
                        "final spec": (str(final.get("spec_hash") or "")[:8] or "—"),
                    }])
                mr = block.get("model")
                if mr:
                    st.dataframe([{
                        "objective": r["name"],
                        "op": r["op"],
                        "threshold": r["threshold"],
                        "actual": "n/a" if r["actual"] is None else r["actual"],
                        "result": "PASS" if r["pass"] else "MISS",
                    } for r in mr["objectives"]])

    # --- 49.2.3: editable best-spec re-scoring (one training pass) ---------
    with st.expander("Edit the best spec — one extra training pass (49.2)"):
        st.caption(
            "Nudge a field (hidden width, optimizer, knn_k, …) and re-evaluate "
            "with this run's own seed — a controlled comparison with exactly "
            "one variable (SPEC.md 49.2.2). One training pass per press; "
            "the run dir is never written (49.2.1).")
        spec_text = st.text_area(
            "best spec", json.dumps(res.get("best_spec") or {}, indent=2),
            height=240, key="spec_editor")
        if st.button("Re-evaluate this spec", key="spec_button", type="primary"):
            try:
                spec = json.loads(spec_text)
            except json.JSONDecodeError as exc:
                st.error(f"not valid JSON: {exc} (SPEC.md 49.2.3)")
            else:
                try:
                    out = retrain_spec(run_dir, spec)
                except ValueError as exc:  # SpecError is a ValueError
                    st.error(str(exc))
                else:
                    st.dataframe([{
                        "score": round(out["score"], 4),
                        "gen score": round(out["gen_score"], 4),
                        "gen gap": round(out["gen_gap"], 4),
                        "train s": round(out["train_seconds"], 3),
                        "final loss": round(out["final_loss"], 6),
                        "time capped": out["time_capped"],
                    }])
                    final = res.get("final_best_score")
                    if isinstance(final, (int, float)):
                        st.caption(
                            f"vs this run's final {float(final):.2f}: "
                            f"{out['score'] - float(final):+.2f} (SPEC.md 49.2.3)")

    # --- 49.3.4: policy A/B (one full budget of the other policy) ----------
    with st.expander("Policy A/B — re-run with the other policy (49.3)"):
        cur = ((res.get("summary") or {}).get("run_config") or {}).get("policy")
        if cur not in ("bandit", "search"):
            st.caption(
                "A/B is available for bandit and search runs only — `rl` "
                "stays CLI-only (SPEC.md 23.1/49.3.1).")
        else:
            other = opposite_policy(cur)
            st.caption(
                f"This run used **{cur}**. Pressing the button re-runs the "
                f"run's *exact* budget (its own run_config, SPEC.md 49.3.2) "
                f"with **{other}** — one full training budget, in this tab; "
                f"the second run gets its own run dir + registry entry.")
            if st.button(f"Re-run with {other}", key="ab_button", type="primary"):
                with st.spinner(f"running the {other} policy… (a full budget)"):
                    try:
                        res2 = _ab_rerun(res)
                    except ValueError as exc:
                        st.error(str(exc))
                        res2 = None
                if res2 is not None:
                    a_pts = (res.get("summary") or {}).get("pareto_frontier") or []
                    b_pts = (res2.get("summary") or {}).get("pareto_frontier") or []
                    st.markdown(svg_frontier_overlay(
                        [(cur, a_pts), (other, b_pts)], target=res.get("target")),
                        unsafe_allow_html=True)
                    fa, fb = res.get("final_best_score"), res2.get("final_best_score")
                    if isinstance(fa, (int, float)) and isinstance(fb, (int, float)):
                        winner = cur if fa >= fb else other
                        st.caption(
                            f"final scores — {cur}: {fa:.2f} vs {other}: {fb:.2f} "
                            f"→ **{winner}** wins by {abs(fa - fb):.2f}")
                    st.caption(f"the {other} run: `{res2['run_dir']}` (SPEC.md 49.3.4)")

    # --- 59.3 (v0.45): manual/expert mode — set the exact spec, train once --
    with st.expander("Manual mode — train this exact spec (59.3)"):
        st.caption(
            "Override the search entirely (SPEC.md 59.3): set each field to "
            "the exact value you want, then train one pass — the loop "
            "validates + reports, it does not discover. Starts from this "
            "run's best spec; override any field below. Nothing is written "
            "to the run dir (49.2.1).")
        _rcfg = (res.get("summary") or {}).get("run_config") or {}
        _mseed = int(_rcfg.get("seed", 7))
        _mtask = _rcfg.get("task")
        _mtaskcfg = _rcfg.get("task_config") or {}
        st.caption(
            f"seed **{_mseed}** · task **{_mtask or '—'}** (this run's "
            f"run_config, SPEC.md 59.3.2)")
        _mdefaults = dict(DEFAULT_SPEC.to_dict())
        _mdefaults.update(res.get("best_spec") or {})
        _mvals = {}
        for _fname in SPEC_FIELDS:
            _fsp = list(SPEC_FIELDS[_fname].space)
            _flab = [_spec_v(v) for v in _fsp]
            _dv = _spec_v(_mdefaults.get(_fname))
            _idx = _flab.index(_dv) if _dv in _flab else 0
            _pick = st.selectbox(_fname, _flab, index=_idx,
                                 key=f"manual_{_fname}")
            _mvals[_fname] = _fsp[_flab.index(_pick)]
        if st.button("Train this exact spec", key="manual_button",
                     type="primary"):
            try:
                manual_spec(_mvals)  # validate early (loud per-field / combo)
            except ValueError as exc:  # SpecError is a ValueError
                st.error(str(exc))
            else:
                try:
                    _mout = train_manual(
                        _mtask or "cartpole-v1", _mvals, seed=_mseed,
                        task_config=_mtaskcfg or None)
                except (ValueError, OSError) as exc:  # unknown task / no data
                    st.error(str(exc))
                else:
                    st.dataframe([{
                        "score": round(_mout["score"], 4),
                        "gen score": round(_mout["gen_score"], 4),
                        "gen gap": round(_mout["gen_gap"], 4),
                        "train s": round(_mout["train_seconds"], 3),
                        "final loss": round(_mout["final_loss"], 6),
                        "time capped": _mout["time_capped"],
                    }])
                    _mbest = res.get("final_best_score")
                    if isinstance(_mbest, (int, float)):
                        st.caption(
                            f"vs this run's final {float(_mbest):.2f}: "
                            f"{_mout['score'] - float(_mbest):+.2f} "
                            f"(SPEC.md 59.3.2)")


def _render_gallery(items: list[dict]) -> None:
    """C3 (SPEC.md 28.3): the misclassified holdout items — an image
    thumbnail (Pillow imported lazily inside plotting) or a hand-rolled
    audio waveform SVG, each captioned `file — true X -> predicted Y`."""
    from autorefine.plotting import _image_data_uri
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            caption = (f"{it.get('file', '')} — true {it.get('label')} -> "
                       f"predicted {it.get('predicted')}")
            path = it.get("path")
            if path and Path(path).exists():
                uri = _image_data_uri(path)
                if uri:
                    st.markdown(
                        f'<img src="{uri}" style="width:96px;'
                        f'height:96px;image-rendering:pixelated"/>',
                        unsafe_allow_html=True)
                    st.caption(caption)
                    continue
            wf = it.get("waveform")
            if wf:
                st.markdown(svg_audio_waveform(wf, it.get("sample_rate"),
                                               title=str(it.get("file", ""))),
                            unsafe_allow_html=True)
            st.caption(caption)


def _render_experiments(payload: dict) -> None:
    """SPEC.md 51.1.1 (v0.37): the Experiments tab — the finished run's
    per-candidate rows (including the 51.3.2 reason column, carried from
    the live table by the stored rows) + the best-score curve. Renders the
    live or the restored run (23.2 persistence); the verdict/plots/artifacts
    live in the Results tab's `_render_result` (51.1.1).

    SPEC.md 52 (v0.38): each candidate also gets a drill-down expander
    (its gate math + loss curves, 52.1) over the stored stream (52.1.2),
    and the table honors the Run tab's `focus_field` filter (52.2: one
    widget, two views)."""
    import pandas as pd  # a streamlit dependency, app-only
    info = payload["info"]
    stream = payload.get("stream") or []
    z = payload.get("z", 0.0)
    st.subheader("Experiments")
    st.caption(
        f"label **{info['label']}** · head **{info['head']}** · "
        f"rows {info['rows']['train']}/{info['rows']['holdout']}/{info['rows']['gen']} "
        f"· baseline **{info['baseline_score']:.2f}** vs target **{info['target']:.1f}** "
        f"· {info['policy']}, seed {info['seed']}"
    )
    # 52.2.2: the focus filter — the Run tab's selectbox owns the widget;
    # here the table keeps only the candidates that mutated the field
    # (the baseline row, which mutated nothing, drops out)
    focus = st.session_state.get("focus_field") or "All"
    rows = payload["rows"]
    if focus != "All" and stream:
        keep = {u.get("index") for u in stream
                if isinstance(u.get("mutation"), list) and focus in u["mutation"]}
        rows = [r for r in rows if r.get("#") in keep]
    st.dataframe(pd.DataFrame(rows), width="stretch")
    st.line_chart(pd.DataFrame({"best score": payload["best_series"]}))
    # 52.1.2: the per-candidate drill-downs — the running best is replayed
    # exactly as the live loop does (the baseline seeds it; each update's
    # `best_score` succeeds it), so the gate math is row-correct (52.1.1)
    if stream:
        st.subheader("Drill-downs — gate math + curves per candidate")
        best_before = info.get("baseline_score")
        for u in stream:
            if not isinstance(u, dict):
                continue
            gm = gate_math(u, best_before, z)
            reason = ("stopped" if u.get("reason") == "stopped"
                      else candidate_reason(u.get("accepted"),
                                           u.get("candidate_score"),
                                           u.get("gen_gap"), best_before))
            with st.expander(f"candidate #{u.get('index')} — {reason} · "
                             "gate math + curves (SPEC.md 52.1)",
                             expanded=False):
                _drill_body(u, gm, reason, info.get("target"))
            if u.get("best_score") is not None:
                best_before = u["best_score"]
    # 53.2/53.3 (v0.39): the decision views over the stored stream —
    # the live Pareto frontier (53.2) + the champion spec card with the
    # last acceptance's first mutated field highlighted (53.3)
    if stream:
        st.subheader("Decision views (SPEC.md 53.2)")
        st.markdown(svg_live_frontier(stream), unsafe_allow_html=True)
    best_spec = (payload.get("res") or {}).get("best_spec")
    if best_spec:
        last_hl = None
        for u in reversed(stream):
            if u.get("accepted") and u.get("mutation"):
                last_hl = u["mutation"][0]
                break
        st.subheader("Champion spec (SPEC.md 53.3)")
        st.markdown(svg_architecture(
            best_spec, payload.get("state_dim", 1), payload.get("n_out", 1),
            highlight=last_hl if isinstance(last_hl, str) else None),
            unsafe_allow_html=True)
    # B2 (SPEC.md 54.2): the wall-time cost strip over the stored stream
    if stream:
        st.subheader("Wall-time cost strip (SPEC.md 54.2)")
        st.markdown(svg_time_strip(
            stream, (payload.get("res") or {}).get("baseline_train_seconds")),
            unsafe_allow_html=True)


def _render_optin_views(path, label: str, target: float, policy: str,
                        seed: int, experiments: int, max_train: float,
                        runs_dir: str, quality: str) -> None:
    """D1/D2 (SPEC.md 29): the opt-in multi-run / policy views — each renders
    only when the user presses its button (inert otherwise; the §23.2 run/
    clear flow and persistence are unchanged).

    - **Seed variance** (D1, SPEC.md 29.1): a `DashboardRunner.seed_sweep`
      over N seeds → `svg_seed_variance`. Needs a data path.
    - **RL policy view (precomputed)** (D2, SPEC.md 29.2): renders the
      `action_probabilities.svg` + `task_returns.svg` a `policy-report` run
      already wrote. **Never runs RL live** (keeps RL CLI-only, SPEC.md 23.1).
    """
    st.divider()
    st.subheader("Multi-run / policy views (v0.15)")

    # --- D1: seed-variance box plot (SPEC.md 29.1) ---------------------------
    st.subheader("Seed variance — is the improvement real?")
    if path is None:
        st.caption("Set a data path (or upload a CSV) to run a seed sweep.")
    else:
        var_n = st.number_input("Number of seeds", min_value=1, max_value=25,
                                value=3, step=1, key="var_seeds")
        if st.button("Run the seed sweep", key="var_button"):
            var_runner = DashboardRunner(
                csv_path=path, label=label.strip() or None, target=float(target),
                policy=policy, seed=int(seed), experiments=int(experiments),
                max_train_seconds=float(max_train),
                runs_dir=runs_dir.strip() or "runs", search_quality=quality,
            )
            seeds = list(range(int(seed), int(seed) + int(var_n)))
            # SPEC.md 32.1: stream the per-step `on_update` (the SPEC.md 31.2
            # serial callback path — the app keeps workers=1) into a
            # progress bar; the denominator is an upper bound (free duplicate
            # rejections, R3, may exceed it), so the fraction is clamped
            total_steps = max(1, len(seeds) * max(1, int(experiments)))
            bar = st.progress(0.0, text=f"seed sweep: 0 of ≤{total_steps} steps")
            steps = {"n": 0}

            def _on_update(_update: dict) -> None:
                steps["n"] += 1
                bar.progress(min(1.0, steps["n"] / total_steps),
                             text=f"seed sweep: step {steps['n']} "
                                  f"(of ≤{total_steps})")

            try:
                sweep = var_runner.seed_sweep(seeds, on_update=_on_update)
            except ValueError as exc:
                st.error(str(exc))
            else:
                bar.progress(1.0, text="seed sweep: complete")  # SPEC.md 32.1
                passing = sum(1 for s in sweep if s["pass"])
                beats = sum(1 for s in sweep if s["final"] > s["baseline"])
                st.caption(f"{len(sweep)} seeds · {passing} pass target "
                           f"{float(target):.1f} · {beats} beat their own baseline")
                st.markdown(svg_seed_variance(sweep, target=float(target)),
                            unsafe_allow_html=True)
                # V1 (SPEC.md 30.1): per-seed curves — when did they diverge?
                st.markdown(svg_seed_curves(sweep, target=float(target)),
                            unsafe_allow_html=True)

    # --- D2: RL policy view, precomputed (SPEC.md 29.2) ----------------------
    st.subheader("RL policy view (precomputed — never runs RL live)")
    st.caption("Point at a `autorefine policy-report` output directory; this "
               "renders its precomputed SVGs (RL stays CLI-only, SPEC.md 23.1).")
    pol_dir = st.text_input("policy-report output dir", value="", key="pol_dir")
    if st.button("Load the policy view", key="pol_button") and pol_dir.strip():
        pd_ = Path(pol_dir.strip())
        # v0.16 (SPEC.md 30.6): render whichever of the three precomputed
        # SVGs exist — a v0.15-era two-file directory still renders
        cands = [pd_ / name for name in
                 ("action_probabilities.svg", "task_returns.svg",
                  "policy_trace_curve.svg")]
        present = [p for p in cands if p.exists()]
        if present:
            for p in present:
                st.markdown(p.read_text(encoding="utf-8"), unsafe_allow_html=True)
        else:
            st.warning(f"{pd_} has no action_probabilities.svg / "
                       f"task_returns.svg — run `autorefine policy-report "
                       f"--runs-dir {pd_}` first.")


# SPEC.md 52.4 (v0.38): the keyboard shortcuts (S = Stop, R = Run) — a
# zero-height same-origin (srcdoc) iframe; its JS listens for keydown on
# the parent document and DOM-clicks the matching existing button (the
# 51.2.3 Stop / the run_button) — no new buttons, no new state. Guards:
# skip while typing in an input/textarea/select, and while meta/ctrl/alt
# are held (browser shortcuts stay intact).
_KEYBOARD_HTML = (
    "<script>"
    "(function () {"
    "  function findButton(snippet) {"
    "    var doc = window.parent.document;"
    "    var bs = doc.querySelectorAll('button');"
    "    for (var i = 0; i < bs.length; i++) {"
    "      if ((bs[i].textContent || '').indexOf(snippet) !== -1) {"
    "        return bs[i];"
    "      }"
    "    }"
    "    return null;"
    "  }"
    "  window.parent.document.addEventListener('keydown', function (ev) {"
    "    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;"
    "    var t = ev.target;"
    "    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||"
    "             t.tagName === 'SELECT')) return;"
    "    var key = (ev.key || '').toLowerCase();"
    "    if (key === 's') {"
    "      var b = findButton('Stop the run');"
    "      if (b) { b.click(); ev.preventDefault(); }"
    "    } else if (key === 'r') {"
    "      var b = findButton('Run the improvement loop');"
    "      if (b) { b.click(); ev.preventDefault(); }"
    "    }"
    "  }, true);"
    "})();"
    "</script>"
)


def main() -> None:
    st.set_page_config(page_title="AutoRefine dashboard", page_icon=":gear:",
                       layout="wide")
    st.title("AutoRefine — autonomous model improvement")
    st.caption(
        f"v{__version__} (SPEC.md 23/24): upload a CSV (or point at a "
        f"directory of labelled images/audio), watch every experiment, get a "
        f"gated model + plots. `rl` policy stays CLI-only "
        f"(`autorefine fit --policy rl`)."
    )

    side = st.sidebar
    # SPEC.md 48 (v0.34): the guided two-step setup (48.1) — presets (48.3),
    # the glossary (48.2), the beginner knobs (data/label/target) top-level,
    # the advanced knobs under a closed "Advanced" expander (48.1.2), and the
    # opt-in narrate toggle (48.5). Defaults are unchanged: no preset
    # selected renders the pre-v0.34 sidebar byte-identically, and narrate
    # off renders the pre-48.5 loop byte-identically (48.5.3). A preset is
    # applied when the selection CHANGES (48.3.3): it fills the knobs it
    # defines — including the Advanced ones — and a later re-selection of
    # the same preset does not clobber the user's manual edits.
    preset_pairs = preset_choices()  # 48.3.2: stable (name, label) order
    preset_labels = ["(none)"] + [lab for _name, lab in preset_pairs]
    label_to_name = {lab: name for name, lab in preset_pairs}
    pi = side.selectbox("Preset (optional)", preset_labels, index=0,
                        key="preset",
                        help="A one-click bundle: picking it fills in the "
                             "settings it defines — feel free to override "
                             "any of them afterwards (SPEC.md 48.3).")
    if pi != "(none)":  # "(none)" applies nothing; selectbox returns the label
        preset_name = label_to_name[pi]
        if st.session_state.get("_preset_applied") != preset_name:  # 48.3.3
            # Apply on selection change (before this run's widgets are
            # instantiated, which Streamlit allows); re-selecting the same
            # preset keeps the user's manual edits.
            for knob, value in preset_flags(preset_name).items():
                st.session_state[knob] = value
            st.session_state["_preset_applied"] = preset_name
        st.caption(preset_summary(preset_name))

    with side.expander("What each setting does", expanded=False):  # 48.2
        for knob in ALL_KNOBS:
            st.markdown(f"**{knob}** — {KNOB_GLOSSARY[knob]}")

    side.header("Data")
    upload = side.file_uploader("CSV file (header + rows)", type=["csv"],
                                key="csv_upload", help=KNOB_GLOSSARY["data"])
    csv_path = side.text_input("…or a CSV path / a directory of labelled "
                               "images or audio (v0.10)", value="", key="csv_path",
                               help=KNOB_GLOSSARY["data"])
    label = side.text_input("Label column (blank = auto-detect)", value="",
                            key="label", help=KNOB_GLOSSARY["label"])
    target = side.number_input("Target score (0–100)", min_value=0.0, max_value=100.0,
                               value=95.0, step=0.5, key="target",
                               help=KNOB_GLOSSARY["target"])

    with side.expander("Advanced", expanded=False):  # 48.1.2
        policy = side.selectbox("Policy (improver)", ("bandit", "search"),
                                index=0, key="policy",
                                help=KNOB_GLOSSARY["policy"])
        seed = side.number_input("Seed", min_value=0, max_value=1_000_000, value=7,
                                 key="seed", help=KNOB_GLOSSARY["seed"])
        experiments = side.number_input("Experiments (budget)", min_value=1,
                                        max_value=500, value=30, step=1,
                                        key="experiments",
                                        help=KNOB_GLOSSARY["experiments"])
        max_train = side.number_input("Max train seconds per spec", min_value=1.0,
                                      max_value=600.0, value=30.0, step=5.0,
                                      key="max_train",
                                      help=KNOB_GLOSSARY["max_train"])
        quality = side.selectbox("Search quality", ("v04", "legacy"), index=0,
                                 key="quality", help=KNOB_GLOSSARY["quality"])
        runs_dir = side.text_input("Runs dir", value=_launcher_runs_dir(),
                                   key="runs_dir", help=KNOB_GLOSSARY["runs_dir"])

    narrate = side.checkbox("Narrate the loop (plain English)", value=False,
                            key="narrate",
                            help="A friendly line per experiment instead of the "
                                 "terse caption (off by default, SPEC.md 48.5).")

    live_mode = side.checkbox("Live dashboard (tick mode)", value=False,
                              key="live_mode",
                              help="Tick the page as experiments land instead "
                                   "of one synchronous block (off by default; "
                                   "the default path stays byte-identical — "
                                   "SPEC.md 55.1).")

    path = _resolve_csv(upload, csv_path)
    result = st.session_state.get("result")
    if path is None and result is None:
        # the idle screen (SPEC.md 23.2) — unchanged by the 51.1 tabs
        st.info("Upload a CSV (or give a path — a CSV file, or a directory "
                "of labelled images/audio, v0.10) to begin. Labels are the "
                "column (label/target/y/class, else last) or the subfolder "
                "name / index.csv; the head is inferred: 2–50 integer "
                "classes → accuracy, otherwise R² (SPEC.md 22.1/24.2).")
        st.stop()

    # SPEC.md 51.1.1 (v0.37): the five-tab structure — the same functions in
    # the same order per interaction as the pre-v0.37 scroll (51.1.2); every
    # widget keeps its exact `key=` (the app tests are key-based).
    t_setup, t_run, t_results, t_compare, t_exps = st.tabs(
        ["Setup", "Run", "Results", "Compare", "Experiments"])

    with t_setup:  # 51.1.1: the data preview, the settings summary, 51.4.4
        if path is not None:
            st.subheader("Data")
            _preview(path)
        else:  # 56.4 (v0.42): the explicit empty state (A.4)
            st.info(
                "No data loaded — upload a CSV or set a path in the sidebar "
                "to begin. A finished run's results are still viewable in "
                "the Results tab (SPEC.md 56.4).")
        st.caption(f"policy **{policy}** · seed **{seed}** · experiments "
                   f"**{experiments}** · max train **{max_train:g}s** · "
                   f"quality **{quality}** · target **{target:g}** "
                   f"(SPEC.md 51.1.1)")
        # 51.4.4: a view preference, not a run knob (the 48.1 knob
        # invariant holds — `ALL_KNOBS` is untouched)
        st.checkbox("Colorblind-safe palette (Okabe-Ito)", value=False,
                    key="cb_palette",
                    help="Re-renders the result plots colorblind-safe + "
                         "dark-mode-aware (a view preference, not a run "
                         "setting — SPEC.md 51.4.4).")
        # 56.5 (v0.42): the reduced-motion opt-out (A.6) — a view preference
        # for the live sparkline's pulsing halo, not a run knob (51.4.4)
        st.checkbox("Reduce motion", value=False, key="reduce_motion",
                    help="Render the live best-score sparkline as a static "
                         "dot instead of the pulsing halo (accessibility "
                         "opt-out — SPEC.md 56.5).")
        # 56.3 (v0.42): the design-token registry (A.3) — the single source
        # of truth for the SVG color/shape tokens, rendered as a table
        with st.expander("Design tokens", expanded=False):
            st.caption(
                "The SVG design-token registry (SPEC.md 56.3) — one row per "
                "token; the `okabe`/`dark` overrides are applied by the "
                "active theme.")
            _tok = resolve_tokens()
            st.dataframe(
                [{"token": k, "value": v} for k, v in _tok.items()],
                width="stretch")
        # SPEC.md 59.2 (v0.45): the human-in-the-loop steering verbs —
        # pin (freeze a field), bias (redirect a mutation), constrain
        # (allowed set). Opt-in: an empty rule list is the pre-v0.45
        # byte-identical path (59.2). A Setup-side configuration, not a
        # run knob (the 48.1 knob invariant — `ALL_KNOBS` is untouched).
        with st.expander("Steer the search — pin / bias / constrain (59.2)",
                         expanded=False):
            st.caption(
                "Human-in-the-loop over the spec space (SPEC.md 59.2): "
                "**pin** a field to a value (the improver won't mutate it), "
                "**bias** a mutation of a field toward a value, or "
                "**constrain** a field to an allowed set (a candidate "
                "outside it is rejected as `invalid_spec`). All opt-in — "
                "leave empty for the default loop.")
            _rules = st.session_state.get("steering_rules") or {
                "pins": [], "biases": [], "constraints": []}
            _verb = st.selectbox("Verb", ("pin", "bias", "constrain"),
                                 key="stg_verb")
            _f = st.selectbox("Field", list(SPEC_FIELDS), key="stg_field")
            _fsp = list(SPEC_FIELDS[_f].space)
            _flab = [_spec_v(v) for v in _fsp]
            if _verb == "constrain":
                _sel = st.multiselect(
                    "Allowed values", _flab, default=_flab[:1],
                    key=f"stg_vals_{_f}")
                _val = [_fsp[_flab.index(x)] for x in _sel] or _fsp[:1]
            else:
                _sel = st.selectbox("Value", _flab, key=f"stg_val_{_f}")
                _val = _fsp[_flab.index(_sel)]
            if st.button("Add rule", key="stg_add"):
                if _verb == "pin":
                    _rules["pins"] = [[ff, vv] for ff, vv in _rules["pins"]
                                      if ff != _f] + [[_f, _val]]
                elif _verb == "bias":
                    _rules["biases"] = [[ff, vv] for ff, vv in _rules["biases"]
                                        if ff != _f] + [[_f, _val]]
                else:
                    _rules["constraints"] = \
                        [[ff, vvs] for ff, vvs in _rules["constraints"]
                         if ff != _f] + [[_f, list(_val)]]
                st.session_state["steering_rules"] = _rules
                st.rerun()
            _n = (len(_rules["pins"]) + len(_rules["biases"])
                  + len(_rules["constraints"]))
            if _n:
                st.markdown(f"**{_n} active rule(s)**")
                for ff, vv in _rules["pins"]:
                    st.caption(f"pin `{ff}` = {_spec_v(vv)}")
                for ff, vv in _rules["biases"]:
                    st.caption(f"bias `{ff}` → {_spec_v(vv)}")
                for ff, vvs in _rules["constraints"]:
                    st.caption(
                        f"constrain `{ff}` ∈ "
                        + ", ".join(_spec_v(x) for x in vvs))
                if st.button("Clear all steering rules", key="stg_clear"):
                    st.session_state["steering_rules"] = {
                        "pins": [], "biases": [], "constraints": []}
                    st.rerun()
            else:
                st.caption("No rules — the default loop runs unchanged.")

    with t_run:  # 51.1.1: the Run button, the live loop, the Stop button
        stop_pressed = st.button(
            "Stop the run (abort after the current experiment)",
            key="stop_button",
            help="Honored between experiments, never inside one — a "
                 "partial, honest run with full artifacts (SPEC.md 51.2.3).")
        rec = st.session_state.get("_worker")
        live = (rec is not None and not rec["drained"]
                and rec.get("thread") is not None and rec["thread"].is_alive())
        if live:  # 51.2.3 preemption-safe reattach: drain the surviving run
            # 56.4 (v0.42): the explicit loading state (A.4) — "it's running"
            st.info("RUNNING — the improvement loop is in progress "
                    "(SPEC.md 56.4).")
            if stop_pressed:
                rec["stop"]["flag"] = True
            _drain_live(rec, narrate, stream_mode=live_mode)
            if live_mode and stream_tick_due(rec):  # 55.1: keep ticking
                st.rerun()
        elif st.button("Run the improvement loop", type="primary",
                       key="run_button"):
            if path is None:
                st.error("Choose a CSV (upload or path) to start a new run.")
            else:
                # SPEC.md 59.2 (v0.45): build the steering rules (None = off,
                # the pre-v0.45 path); an invalid rule is a friendly stop.
                _stg_rules = st.session_state.get("steering_rules") or {}
                _stg = None
                if (_stg_rules.get("pins") or _stg_rules.get("biases")
                        or _stg_rules.get("constraints")):
                    try:
                        _stg = SteeringState.from_dict(_stg_rules)
                    except ValueError as exc:
                        st.error(f"invalid steering rule: {exc} (SPEC.md 59.2)")
                        st.stop()
                _run(path, label.strip() or None, float(target), policy,
                     int(seed), int(experiments), float(max_train),
                     runs_dir.strip() or "runs", quality,
                     narrate=narrate, initial_stop=stop_pressed,
                     stream_mode=live_mode, steering=_stg)
        elif result is not None:
            # widget-triggered re-run (download click, sidebar change) — the
            # last completed result stays on screen (SPEC.md 23.2 persistence)
            if st.button("New run (clear result)", key="clear_button"):
                st.session_state.pop("result", None)
                st.session_state.pop("_worker", None)
                st.rerun()
        else:
            st.caption("Press **Run** — every experiment is shown as it "
                       "happens (live table + curve), then the verdict, "
                       "plots, and downloads (SPEC.md 23.2).")
        # SPEC.md 55.2/55.3 (v0.41): the "feel live" views — the mid-run
        # status snapshot (55.2) + the reference-run overlay (55.3). Gated
        # on a run existing (live or finished); a no-op before that (55.4).
        result = st.session_state.get("result")  # re-read (the run just
        # finished; the sidebar's `result` was captured before it, 51.1.2)
        if result is not None or live:
            sinfo, sstream, sseries, sdone = snapshot_inputs(rec, result)
            if st.button("Copy status (how's it going?)",
                         key="copy_status",
                         help="A self-contained text card: task, best, "
                              "budget, ETA, last 3 decisions (SPEC.md 55.2)."):
                st.code(status_snapshot(sinfo, sstream, sseries,
                                        done=sdone), language="text")
            _runs = runs_dir.strip() or "runs"
            _refs = list_reference_runs(_runs)
            _ref_labels = ["(none)"] + [name for name, _c in _refs]
            _ref_i = st.selectbox("Reference run (overlay beneath the "
                                  "current curve)", _ref_labels, index=0,
                                  key="ref_run",
                                  help="Draw a past run's best-score curve "
                                       "faintly beneath the current one — "
                                       "'am I beating last time?' "
                                       "(SPEC.md 55.3).")
            if _ref_i != "(none)" and sseries:
                _ref_curve = dict(_refs)[_ref_i]
                st.markdown(
                    svg_reference_curve(sseries, _ref_curve,
                                        target=(sinfo or {}).get("target")),
                    unsafe_allow_html=True)
        # 52.4 (v0.38): the keyboard shortcuts (S = Stop, R = Run) — the
        # zero-height iframe renders once per script run in the Run tab;
        # inert while idle (the idle screen st.stop()s before the tabs)
        components.html(_KEYBOARD_HTML, height=0)

    with t_results:  # 51.1.1: the result view for the live or restored run
        result = st.session_state.get("result")  # re-read fresh (51.1.2)
        if result is not None:
            _render_result(result["res"])
        else:
            st.caption("Finish a run first — the verdict, plots, and "
                       "downloads appear here (SPEC.md 51.1.1).")

    with t_compare:  # 51.1.1: seed-sweep / RL views (29.1/29.2) + past runs
        _render_optin_views(path, label, float(target), policy, int(seed),
                            int(experiments), float(max_train),
                            runs_dir.strip() or "runs", quality)
        _render_past_runs(runs_dir.strip() or "runs")

    with t_exps:  # 51.1.1: the per-candidate rows (51.3 reason column) + curve
        result = st.session_state.get("result")
        if result is not None:
            _render_experiments(result)
        else:
            st.caption("Finish a run first — the per-candidate table (with "
                       "the reason column) and the best-score curve appear "
                       "here (SPEC.md 51.1.1).")


main()
