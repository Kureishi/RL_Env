"""The run dossier (SPEC.md 58.3, v0.44) — the "whole story of this run".

One scrollable, self-contained HTML artifact (inline ``<style>``, no
external assets, 22.2) combining every decision surface of a finished
run: the recipe (37.1), the provenance certificate (56.1), the frontier
(21.2 + the 3-objective view, 58.2), the spec lineage (57.1), the
per-field sensitivity (57.2), and the error-analysis pack (28.2 / 28.3 /
58.1).

House rules: core module — stdlib only, streamlit-free (23.1), imports
only pure leaf modules (``memory`` / ``plotting`` / ``provenance`` /
``research`` / ``runconfig``). Pure over the run dir: no training, no
RNG, no timestamps — a re-render of the same inputs is byte-identical
(G2). All dynamic text is HTML-escaped. A run dir without ``summary.json``
is a ``ValueError`` (the ``_run_task_and_model`` contract, SPEC.md 42).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .memory import KIND_BASELINE, KIND_EXPERIMENT, RunMemory
from .plotting import (
    _image_data_uri,
    svg_audio_waveform,
    svg_difficulty_ranking,
    svg_field_response,
    svg_frontier3,
    svg_pareto,
    svg_spec_lineage,
)
from .provenance import env_provenance, provenance_payload, svg_provenance
from .research import field_response_stats, frontier3, spec_lineage
from .runconfig import RunConfig, format_recipe


def _fmt(v) -> str:
    """A table-cell value → its short ASCII form (None reads as `—`)."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _pareto_points(entries, summary: dict) -> list[dict]:
    """58.3 (the 21.2 rule, the `dashboard.finish()` contract): the
    summary's `pareto_frontier` when present, else the scored
    baseline/experiment rows (35.1 kind registry)."""
    pts = summary.get("pareto_frontier")
    if pts:
        return [p for p in pts if isinstance(p, dict)
                and isinstance(p.get("score"), (int, float))
                and not isinstance(p.get("score"), bool)]
    out: list[dict] = []
    for e in entries:
        if (e.get("kind") in (KIND_BASELINE, KIND_EXPERIMENT)
                and isinstance(e.get("holdout_score"), (int, float))
                and not isinstance(e.get("holdout_score"), bool)):
            out.append({"score": e["holdout_score"],
                        "train_seconds": e.get("train_seconds", 0.0)})
    return out


def _load_run_config(run_dir: Path, summary: dict) -> dict | None:
    """58.3: the run's canonical recipe — `run_config.json` (37.1,
    written at reset), falling back to the summary's `run_config`
    mirror; None when the run predates v0.23."""
    rc_path = run_dir / "run_config.json"
    if rc_path.is_file():
        try:
            d = json.loads(rc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            d = None
        if isinstance(d, dict):
            return d
    mirror = summary.get("run_config")
    return mirror if isinstance(mirror, dict) else None


def _gallery_items(gallery) -> list[dict]:
    """58.3 (the 28.3 shape): the misclassified holdout items as dicts
    (file / path / label / predicted / waveform / sample_rate)."""
    return [it for it in (gallery or []) if isinstance(it, dict)]


def build_dossier(run_dir, state_dim=None, n_out=None, grid=None,
                  difficulty=None, gallery=None,
                  per_class_svg=None, confusion_svg=None) -> str:
    """SPEC.md 58.3 (v0.44): the run dossier — one scrollable,
    self-contained HTML document (inline ``<style>``, no external
    assets, 22.2; no timestamps — a re-render is byte-identical, G2)
    combining the whole story of a finished run:

    1. **Summary** — the run's headline numbers (the report's table).
    2. **Recipe** — the canonical `run_config.json` (37.1) as the
       copy-pasteable `format_recipe` line plus the full JSON; the
       summary's `run_config` mirror is the fallback; a caption when
       the run predates the config artifact.
    3. **Provenance** — the 56.1 certificate (env facts + the run's
       identity + the run-config hash) as the SVG card.
    4. **Frontier** — the 2-objective `svg_pareto` (21.2) + the
       3-objective `svg_frontier3` (58.2), whose size axis comes from
       `config.spec_n_params` with the task's `state_dim` / `n_out` /
       `feature_grid` (None args degrade every size to the
       data-dependent case).
    5. **Spec lineage** — the 57.1 mutation DAG.
    6. **Sensitivity** — the 57.2 per-field response surfaces.
    7. **Error analysis** — the per-class bars + confusion matrix
       (28.2, pre-rendered by the caller — the dossier never trains),
       the 58.1 per-input difficulty ranking (the `difficulty` dict,
       the `diagnostics.holdout_difficulty` shape), and the 28.3 media
       error gallery (data-URI thumbnails / waveform SVGs). The
       section degrades to a caption when the run carries no
       classification data.

    `run_dir` is a finished run dir (``summary.json`` +
    ``experiments.jsonl``); a missing summary is a `ValueError` (the
    `_run_task_and_model` contract, 42). All dynamic text is
    HTML-escaped; the pre-rendered SVGs are inlined as-is (they are
    valid XML by their own contract). Pure over the run dir (G2).
    """
    run_dir = Path(run_dir)
    try:
        summary = RunMemory(run_dir).load_summary()
    except FileNotFoundError:
        raise ValueError(f"no summary.json in run dir {str(run_dir)!r} "
                         f"(SPEC.md 58.3/42)")
    try:
        entries = RunMemory(run_dir).load_experiments()
    except FileNotFoundError:
        entries = []
    entries = [e for e in entries if isinstance(e, dict)]
    rc_dict = _load_run_config(run_dir, summary)
    recipe_line: str | None = None
    if isinstance(rc_dict, dict):
        try:
            recipe_line = format_recipe(RunConfig.from_dict(rc_dict))
        except (ValueError, TypeError, KeyError):
            recipe_line = None  # a non-canonical mirror: show the JSON only
    target = rc_dict.get("target") if isinstance(rc_dict, dict) else None

    esc = html.escape
    L: list[str] = []
    a = L.append
    task = esc(str(summary.get("task", "?")))
    a("<!DOCTYPE html>")
    a('<html lang="en"><head><meta charset="utf-8"/>')
    a(f"<title>AutoRefine dossier — {task}</title>")
    a("<style>")
    a("body{font-family:ui-monospace,Consolas,monospace;margin:24px;"
      "color:#1f2933;}")
    a("h1{font-size:20px;} h2{font-size:15px;margin-top:28px;}")
    a("table{border-collapse:collapse;margin:8px 0 4px;}")
    a("td,th{border:1px solid #cbd2d9;padding:4px 10px;text-align:left;"
      "font-size:13px;}")
    a("th{background:#f0f4f8;}")
    a("pre{background:#f0f4f8;border:1px solid #cbd2d9;padding:10px;"
      "overflow:auto;}")
    a(".dim{color:#6b7280;font-size:13px;}")
    a(".foot{color:#6b7280;font-size:12px;margin-top:32px;}")
    a("</style></head><body>")
    a(f"<h1>AutoRefine dossier — {task} (seed {_fmt(summary.get('seed'))})</h1>")

    a("<h2>Summary</h2>")
    a("<table>")
    for key in ("finished_reason", "baseline_score", "final_best_score",
                "improvement_factor", "experiments_run", "wall_seconds"):
        a(f"<tr><th>{key}</th><td>{esc(_fmt(summary.get(key)))}</td></tr>")
    a("</table>")

    # 1. the recipe (37.1) — the copy-pasteable line + the full JSON
    a("<h2>Recipe (run_config)</h2>")
    if recipe_line:
        a(f"<pre>{esc(recipe_line)}</pre>")
    if isinstance(rc_dict, dict):
        a(f"<pre>{esc(json.dumps(rc_dict, indent=2, sort_keys=True))}</pre>")
    elif recipe_line is None:
        a('<p class="dim">no run_config.json in this run (pre-v0.23)</p>')

    # 2. the provenance certificate (56.1)
    a("<h2>Provenance</h2>")
    a(svg_provenance(provenance_payload(
        env_provenance(),
        seed=summary.get("seed"),
        task=summary.get("task"),
        target=target,
        run_config=rc_dict,
    )))

    # 3. the frontier — 2-objective (21.2) + 3-objective (58.2)
    a("<h2>Frontier</h2>")
    a(svg_pareto(_pareto_points(entries, summary)))
    a(svg_frontier3(frontier3(entries, state_dim, n_out, grid)))

    # 4. the spec lineage (57.1)
    a("<h2>Spec lineage</h2>")
    a(svg_spec_lineage(spec_lineage(entries)))

    # 5. the per-field sensitivity (57.2)
    a("<h2>Sensitivity (per-field response)</h2>")
    a(svg_field_response(field_response_stats(entries)))

    # 6. the error-analysis pack (28.2 / 28.3 / 58.1)
    a("<h2>Error analysis</h2>")
    has_classification = bool(per_class_svg) or bool(confusion_svg)
    if per_class_svg:
        a(per_class_svg)
    if confusion_svg:
        a(confusion_svg)
    if isinstance(difficulty, dict):
        a(svg_difficulty_ranking(difficulty))
    items = _gallery_items(gallery)
    if items:
        a("<ul>")
        for it in items:
            uri = _image_data_uri(it.get("path"))
            cap = (f"{esc(str(it.get('file', '?')))} — true "
                   f"{esc(str(it.get('label', '?')))} -> predicted "
                   f"{esc(str(it.get('predicted', '?')))}")
            a("<li>")
            if uri:  # image thumbnail (PIL lazy; self-contained data URI)
                a(f'<img src="{uri}" alt="{esc(str(it.get("file", "?")))}" '
                  f'style="max-width:180px;max-height:180px;">')
            a(f"<div>{cap}</div>")
            if isinstance(it.get("waveform"), (list, tuple)):
                a(svg_audio_waveform(it["waveform"], it.get("sample_rate"),
                                     title=str(it.get("file", "audio"))))
            a("</li>")
        a("</ul>")
    if not (has_classification or isinstance(difficulty, dict) or items):
        a('<p class="dim">no holdout classification data for this run '
          '(episode / mse task, or no holdout split)</p>')

    a('<p class="foot">Generated by autorefine — self-contained, no '
      'external assets (SPEC.md 22.2); a re-render is byte-identical '
      '(G2).</p>')
    a("</body></html>")
    return "\n".join(L) + "\n"
