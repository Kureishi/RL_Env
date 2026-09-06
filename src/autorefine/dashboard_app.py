"""Streamlit app for the v0.9 dashboard (SPEC.md 23.2) — a thin rendering
layer over `autorefine.dashboard.DashboardRunner`; all run semantics live in
the core, so same-seed runs reproduce the CLI `fit` outcome (G2).

Run:  streamlit run src/autorefine/dashboard_app.py
  or: python -m autorefine dashboard            (SPEC.md 23.4 launcher)
"""
from __future__ import annotations

import csv as _csv
import os
import tempfile
from pathlib import Path

import streamlit as st

from autorefine import __version__
from autorefine.dashboard import DashboardRunner, ucb_trace
from autorefine.plotting import (
    svg_action_probabilities,  # noqa: F401 (D2 view, SPEC.md 29.2)
    svg_audio_waveform,
    svg_field_value_matrix,  # V2 view (SPEC.md 30.2)
    svg_loss_curves,
    svg_mutation_timeline,
    svg_score_gap_scatter,
    svg_score_strip,
    svg_seed_curves,  # V1 view (SPEC.md 30.1)
    svg_seed_variance,  # D1 view (SPEC.md 29.1)
    svg_task_returns,  # noqa: F401 (D2 view, SPEC.md 29.2)
)
from autorefine.tasks import CsvTask


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


def _run(csv_path: str, label: str, target: float, policy: str, seed: int,
         experiments: int, max_train: float, runs_dir: str,
         quality: str = "v04") -> None:
    """The live loop: one placeholder table + chart updated per step."""
    runner = DashboardRunner(
        csv_path=csv_path, label=label or None, target=target, policy=policy,
        seed=seed, experiments=experiments, max_train_seconds=max_train,
        runs_dir=runs_dir, search_quality=quality,
    )
    try:
        info = runner.start()
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("Task")
    st.caption(
        f"label **{info['label']}** · head **{info['head']}** · "
        f"rows {info['rows']['train']}/{info['rows']['holdout']}/{info['rows']['gen']} "
        f"· baseline **{info['baseline_score']:.2f}** vs target **{info['target']:.1f}** "
        f"· {info['policy']}, seed {info['seed']}"
    )

    import pandas as pd  # a streamlit dependency, app-only

    # all table cells are strings (st.dataframe → Arrow; mixed types break it)
    rows = [{
        "#": 0, "accepted": "—",
        "candidate": f"{info['baseline_score']:.2f}", "gen gap": "—",
        "mutation": "baseline", "best": f"{info['baseline_score']:.2f}",
        "diff": "—",
    }]
    best_series = [round(info["baseline_score"], 2)]
    table = st.empty()
    chart = st.empty()
    bar = st.progress(0.0, text="starting…")
    note = st.empty()
    # decision-view placeholders, live (SPEC.md 26.5): D1 win-rate bars,
    # V2 field x value matrix, D3 mutation timeline, D4 UCB trace (bandit only)
    vbars = st.empty()
    vmatrix = st.empty()  # V2 (SPEC.md 30.2)
    vtimeline = st.empty()
    vucb = st.empty() if policy == "bandit" else None
    # G1 + G2 gate views, live (SPEC.md 27.4): pure over the stream so far
    vstrip = st.empty()
    vscatter = st.empty()
    stream: list[dict] = []          # updates so far: the views' only input

    while not runner.done:  # runner state machine (SPEC.md 23.1)
        u = runner.next()
        rows.append({
            "#": u["index"],
            "accepted": "yes" if u["accepted"] else "no",
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
        best_series.append(round(u["best_score"], 2))
        # the bar tracks the *budget*, not the stream-row count: spent =
        # budget − experiments_left, so free duplicate rejections (R3) do not
        # inflate the counter (SPEC.md 23.2)
        budget = info["budget_experiments"]
        spent = budget - int(u["experiments_left"])
        is_dup = not isinstance(u["candidate_score"], (int, float))
        frac = max(0.0, min(1.0, spent / max(1, budget)))
        bar.progress(frac, text=(
            f"experiment {spent} of {budget}"
            + ("  ·  dup step (free, R3)" if is_dup else "")))
        table.dataframe(pd.DataFrame(rows), width="stretch")
        chart.line_chart(pd.DataFrame({"best score": best_series}))
        # decision views, live (SPEC.md 26.5) — pure over the stream so far
        stream.append(u)
        if u.get("field_stats"):  # D1: per-field win-rate bars (SPEC.md 26.1)
            vbars.bar_chart(
                pd.DataFrame.from_dict(u["field_stats"], orient="index")
                .sort_index())
        if u.get("field_value_stats"):  # V2: field x value matrix (SPEC.md 30.2)
            vmatrix.markdown(
                svg_field_value_matrix(u["field_value_stats"]),
                unsafe_allow_html=True)
        vtimeline.markdown(  # D3: mutation timeline (SPEC.md 26.3)
            svg_mutation_timeline(stream), unsafe_allow_html=True)
        # G1 candidate score strip + G2 score vs gen-gap scatter (SPEC.md 27)
        vstrip.markdown(svg_score_strip(stream), unsafe_allow_html=True)
        vscatter.markdown(svg_score_gap_scatter(stream), unsafe_allow_html=True)
        if vucb is not None and u.get("ucb"):  # D4: bandit UCB (SPEC.md 26.4)
            # the same pure function as the result view: aligned per-field
            # series with None gaps for the steps before a field was credited
            trace = ucb_trace(stream, alpha=runner.policy.alpha)
            vucb.line_chart(
                pd.DataFrame({f: trace[f] for f in sorted(trace)}
                             ).astype("float64"))
        # per-step caption, including the final step (SPEC.md 23.2) — rendered
        # before the break so the last row is not skipped; D2 spec-diff chips
        # (SPEC.md 26.2) replace the plain field list
        score_txt = (f"score {u['candidate_score']:.2f}"
                     if isinstance(u["candidate_score"], (int, float)) else "duplicate")
        kind = ("dup step (free, R3)" if is_dup
                else f"experiment {spent} of {budget}")
        diff_txt = " · ".join(
            f"{d['field']}: {_spec_v(d['old'])} → {_spec_v(d['new'])}"
            for d in u.get("spec_diff") or []
        ) or (", ".join(u["mutation"]) or "-")
        note.caption(
            f"{kind}: {'accepted +' if u['accepted'] else 'rejected '} "
            f"{score_txt} · {diff_txt}"
        )
        if u["done"]:
            break

    res = runner.finish()
    # Persistence (SPEC.md 23.2): a widget-triggered re-run (download click,
    # sidebar change) must not lose the run — store the finished result so
    # `main()` re-renders it on the next script run
    st.session_state["result"] = {
        "info": info, "rows": rows, "best_series": best_series, "res": res,
    }
    _render_result(res)


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

    # SPEC.md 36.2 (v0.22, G2): the app's spec surface reads the field
    # registry — one table for the field list, not a per-surface literal
    from autorefine.improver.specspace import SPEC_FIELD_NAMES
    st.caption(
        f"spec space: {len(SPEC_FIELD_NAMES)} fields — "
        f"{', '.join(SPEC_FIELD_NAMES)} (SPEC.md 36.2)"
    )

    st.subheader("Plots")
    st.markdown(res["svg_score"], unsafe_allow_html=True)
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
        st.markdown(res["strip_svg"], unsafe_allow_html=True)
    if res.get("scatter_svg"):  # G2: score vs gen-gap (SPEC.md 27.2)
        st.markdown(res["scatter_svg"], unsafe_allow_html=True)
    if res.get("ladder_svg"):  # G3: curriculum ladder (SPEC.md 27.3)
        st.markdown(res["ladder_svg"], unsafe_allow_html=True)

    # learning views (SPEC.md 28): computed once in finish(), re-rendered
    # here from the stored result — the restored view shows the same views
    if any(res.get(k) for k in ("loss_curves", "per_class_svg",
                                "confusion_svg", "boundary_svg",
                                "error_gallery", "arch_svg")):
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
        if res.get("boundary_svg"):  # V3 (SPEC.md 30.3): where it fails on a 2-D plane
            st.markdown(res["boundary_svg"], unsafe_allow_html=True)
        gallery = res.get("error_gallery")
        if gallery:  # C3 (SPEC.md 28.3): misclassified holdout items
            _render_gallery(gallery)
        if res.get("arch_svg"):  # C4 (SPEC.md 28.4): what we ended up building
            st.markdown(res["arch_svg"], unsafe_allow_html=True)

    st.subheader("Artifacts")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("report.html", res["report_html"], mime="text/html",
                       file_name="report.html")
    c2.download_button("best_spec.json", res["artifacts"]["best_spec.json"],
                       mime="application/json", file_name="best_spec.json")
    c3.download_button("best_model.npz", res["artifacts"]["best_model.npz"],
                       mime="application/octet-stream", file_name="best_model.npz")
    c4.download_button("experiments.jsonl", res["artifacts"]["experiments.jsonl"],
                       mime="application/jsonlines", file_name="experiments.jsonl")
    st.caption(
        f"run dir: `{res['run_dir']}` — or CLI: "
        f"`python -m autorefine report --run {res['run_dir']} --html`"
    )


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


def _render_restored(payload: dict) -> None:
    """Re-show the last completed run after a widget re-run (download
    click, sidebar change) — SPEC.md 23.2 persistence."""
    import pandas as pd  # a streamlit dependency, app-only
    info = payload["info"]
    st.subheader("Task")
    st.caption(
        f"label **{info['label']}** · head **{info['head']}** · "
        f"rows {info['rows']['train']}/{info['rows']['holdout']}/{info['rows']['gen']} "
        f"· baseline **{info['baseline_score']:.2f}** vs target **{info['target']:.1f}** "
        f"· {info['policy']}, seed {info['seed']} · last completed run"
    )
    st.dataframe(pd.DataFrame(payload["rows"]), width="stretch")
    st.line_chart(pd.DataFrame({"best score": payload["best_series"]}))
    _render_result(payload["res"])


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
    side.header("Data")
    upload = side.file_uploader("CSV file (header + rows)", type=["csv"], key="csv_upload")
    csv_path = side.text_input("…or a CSV path / a directory of labelled "
                               "images or audio (v0.10)", value="", key="csv_path")
    label = side.text_input("Label column (blank = auto-detect)", value="", key="label")
    target = side.number_input("Target score (0–100)", min_value=0.0, max_value=100.0,
                               value=95.0, step=0.5, key="target")
    side.header("Search")
    policy = side.selectbox("Policy (improver)", ("bandit", "search"),
                            index=0, key="policy",
                            help="UCB field-bandit (default) or v1 search; "
                                 "rl is CLI-only (SPEC.md 23.1)")
    seed = side.number_input("Seed", min_value=0, max_value=1_000_000, value=7,
                             key="seed")
    experiments = side.number_input("Experiments (budget)", min_value=1,
                                    max_value=500, value=30, step=1, key="experiments")
    max_train = side.number_input("Max train seconds per spec", min_value=1.0,
                                  max_value=600.0, value=30.0, step=5.0,
                                  key="max_train")
    quality = side.selectbox("Search quality", ("v04", "legacy"), index=0,
                             key="quality",
                             help="v0.4 CI/efficiency acceptance (default, the"
                                 " CLI rule) or the legacy v0.3 strict-score rule")
    runs_dir = side.text_input("Runs dir", value=_launcher_runs_dir(), key="runs_dir")

    path = _resolve_csv(upload, csv_path)
    result = st.session_state.get("result")
    if path is None and result is None:
        st.info("Upload a CSV (or give a path — a CSV file, or a directory "
                "of labelled images/audio, v0.10) to begin. Labels are the "
                "column (label/target/y/class, else last) or the subfolder "
                "name / index.csv; the head is inferred: 2–50 integer "
                "classes → accuracy, otherwise R² (SPEC.md 22.1/24.2).")
        st.stop()

    if path is not None:
        st.subheader("Data")
        _preview(path)

    if st.button("Run the improvement loop", type="primary", key="run_button"):
        if path is None:
            st.error("Choose a CSV (upload or path) to start a new run.")
        else:
            _run(path, label.strip() or None, float(target), policy, int(seed),
                 int(experiments), float(max_train), runs_dir.strip() or "runs",
                 quality)
    elif result is not None:
        # widget-triggered re-run (download click, sidebar change) — the
        # last completed result stays on screen (SPEC.md 23.2 persistence).
        # The clear button is checked *before* rendering: when pressed, the
        # stored result is dropped and the script re-runs into the idle view
        # without ever rendering the stale result in that pass.
        if st.button("New run (clear result)", key="clear_button"):
            st.session_state.pop("result", None)
            st.rerun()
        _render_restored(result)
    else:
        st.caption("Press **Run** — every experiment is shown as it happens "
                   "(live table + curve), then the verdict, plots, and "
                   "downloads (SPEC.md 23.2).")

    # D1/D2 (SPEC.md 29): opt-in multi-run / policy views (inert until pressed)
    _render_optin_views(path, label, float(target), policy, int(seed),
                        int(experiments), float(max_train),
                        runs_dir.strip() or "runs", quality)


main()
