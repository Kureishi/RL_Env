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
from autorefine.dashboard import DashboardRunner
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


def _preview(csv_path: str) -> None:
    """Inferred head/splits + first rows (SPEC.md 23.2 data preview);
    v0.10 (SPEC.md 24.5): media directories preview their items."""
    from autorefine.tasks import TASKS, detect_modality
    p = Path(csv_path)
    if p.is_file():
        cls = CsvTask
    elif p.is_dir():
        m = detect_modality(p)
        if m is None or m == "mixed":
            st.warning(f"Directory {p} holds no image/audio items (or is "
                       f"mixed) — one subfolder per class, or an index.csv "
                       f"(SPEC.md 24.2/24.5)")
            return
        cls = TASKS[m]
    else:
        st.warning(f"no such file or directory: {p}")
        return
    try:
        probe = cls(seed=0, path=csv_path)
    except ValueError as exc:
        st.warning(f"data not usable as a task: {exc}")
        return
    head_txt = (f"softmax ({probe.n_outputs} classes: "
                f"{probe.class_values})" if probe.head == "softmax"
                else "mse (regression)")
    st.caption(
        f"label **{probe.label_name}** · head **{head_txt}** · "
        f"features {', '.join(probe.feature_names)} · "
        f"items {len(probe._x_tr)}/{len(probe._x_ho)}/{len(probe._x_ge)} "
        f"(train/holdout/gen)"
    )
    import pandas as pd  # a streamlit dependency, app-only
    if p.is_file():
        # simple deterministic preview: header + first 5 data rows
        with p.open(encoding="utf-8", newline="") as f:
            reader = _csv.reader(f)
            header = next(reader)
            data = [next(reader, None) for _ in range(5)]
            data = [r for r in data if r]
        if data:
            st.dataframe(pd.DataFrame(data, columns=header).head(5),
                         width="stretch")
    else:
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
        if rows:
            st.dataframe(pd.DataFrame(rows).head(5), width="stretch")


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
    }]
    best_series = [round(info["baseline_score"], 2)]
    table = st.empty()
    chart = st.empty()
    bar = st.progress(0.0, text="starting…")
    note = st.empty()

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
        # per-step caption, including the final step (SPEC.md 23.2) — rendered
        # before the break so the last row is not skipped
        score_txt = (f"score {u['candidate_score']:.2f}"
                     if isinstance(u["candidate_score"], (int, float)) else "duplicate")
        kind = ("dup step (free, R3)" if is_dup
                else f"experiment {spent} of {budget}")
        note.caption(
            f"{kind}: {'accepted +' if u['accepted'] else 'rejected '} "
            f"{score_txt} · mutation: {', '.join(u['mutation']) or '-'}"
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

    st.subheader("Plots")
    st.markdown(res["svg_score"], unsafe_allow_html=True)
    st.markdown(res["svg_pareto"], unsafe_allow_html=True)

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


main()
