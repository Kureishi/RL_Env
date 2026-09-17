"""v0.49 — "B. Reporting for a wider range of audiences" (SPEC.md 63, A53, M52).

Covers the A53 items as implemented — three more readers / formats of the
*same already-logged* data, all additive / opt-in (A1–A52 stay green; the
default and ``technical`` report paths are byte-identical, 63.5):

- 63.1 **B2 formats** — ``REPORT_FORMATS == ("md","txt","pdf")``;
  ``build_report_doc`` on a synthetic summary/entries carries the headline
  order, the int ``win_rate`` rollup (hand-computed trials/wins), the
  per-entry log, and the best spec; ``render_report_md`` has the ``#``
  headings + ``|`` tables + the fenced spec; ``render_report_txt`` has no
  pipes/fences; all three renderers are byte-stable (G2), including two
  independent PDF builds; ``render_report`` dispatches and rejects ``pdf``
  / unknown formats.
- 63.2 **B3 user guide** — ``user_guide_view`` over a fake softmax task +
  model: ``what_it_predicts`` (hand-computed from ``head`` /
  ``class_values``), the examples with hand-computed softmax
  max-probabilities (``1/(1+2e^-3)``), ``confidence``
  (``mean_max_prob`` / ``frac_at_least_0.8``), the ``when_to_distrust`` +
  ``known_limitations`` lines, and the ``how_to_read`` margin; the mse-head
  path (tolerance-correct examples, no confidence); and the three
  degradations (no ``holdout_rows`` / ``task=None`` / ``model=None``).
- 63.3 **B4 decision** — ``decision_view`` across the verdict branches:
  GO → ``ship``; below-target with ``proj=more`` → ``more_budget`` (the
  asymptote); ``ceiling`` → ``relax_or_expand``; ``insufficient`` →
  ``collect_history``; absent → ``investigate``; ``target_met`` (met +
  margin), the seed-variance block, the top-3 ``failure_modes`` cap with
  the rejection buckets (hand-traced through the 39.2.2 priority), the
  targeted hints, and the ``render_decision`` byte-stability.
- **CLI** — a real tiny parity run → ``--format md/txt`` rc 0,
  ``--format pdf`` rc 0 + ``report.pdf`` (``%PDF``, byte-identical on
  re-run, the G2 CLI pin); ``--user`` rc 0 + the no-split note (parity has
  no ``holdout_rows``); ``--decision`` rc 0 + verdict / next-step phrases;
  mutual exclusion rc 1 (``--format md --json``, ``--user --json``,
  ``--decision --what-if``, ``--user --format md``,
  ``--decision --audience exec``); the default report stays byte-identical
  to ``--audience technical``.
- **exports (63.6.5)** — the ten new top-level names are in ``__all__``
  (33.1); the version is ``0.49.0`` in both sources.

House rules (A53): no cross-test imports (all fixtures synthesized here);
the pure core is tested by hand-computation; ``dashboard_app`` is never
imported.
"""
from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    Budget,
    SearchPolicy,
    REPORT_FORMATS,
    build_report_doc,
    decision_view,
    render_decision,
    render_report,
    render_report_md,
    render_report_pdf,
    render_report_txt,
    render_user_guide,
    user_guide_view,
)
from autorefine.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]

# A53: the ten new top-level exports (63.6.5).
NEW_EXPORTS = (
    "REPORT_FORMATS", "build_report_doc", "render_report_md",
    "render_report_txt", "render_report_pdf", "render_report",
    "user_guide_view", "render_user_guide", "decision_view",
    "render_decision",
)


# --- small synthesized fixtures (no cross-test imports, A53) -----------------

def _syn_summary(**over) -> dict:
    """A minimal, hand-computed summary the views can consume."""
    s = {
        "task": "parity-v1",
        "seed": 7,
        "finished_reason": "target",
        "baseline_score": 80.0,
        "final_best_score": 97.0,
        "improvement_factor": 1.2125,
        "experiments_run": 3,
        "wall_seconds": 12.5,
        "target": 95.0,
        "best_spec": {"model_family": "mlp", "architecture": [64, 32],
                      "activation": "tanh"},
    }
    s.update(over)
    return s


def _syn_entries() -> list:
    """Two scored candidates: one accepted (mutating ``hidden_dim``),
    one rejected (mutating ``lr``) — the win-rate rollup hand-computes to
    ``hidden_dim {1, 1, 1.0}`` and ``lr {1, 0, 0.0}``."""
    return [
        {"kind": "baseline", "accepted": True, "holdout_score": 80.0,
         "gen_gap": None, "mutation": []},
        {"kind": "experiment", "accepted": True, "holdout_score": 97.0,
         "gen_gap": 0.5, "mutation": ["hidden_dim"]},
        {"kind": "experiment", "accepted": False, "holdout_score": 85.0,
         "gen_gap": 1.0, "mutation": ["lr"]},
    ]


# --- 63.1 B2: formats (A53) ----------------------------------------------------

def test_report_formats_tuple():
    """A53 (SPEC.md 63.1.2): the three formats in documented order — the
    parser's ``choices`` and the renderer dispatcher share this tuple
    (one source)."""
    assert REPORT_FORMATS == ("md", "txt", "pdf")


def test_build_report_doc_headline_and_win_rate():
    """A53 (SPEC.md 63.1): the doc carries the headline in documented
    order, the *integer* per-field trials/wins/win_rate rollup
    (hand-computed: ``hidden_dim`` 1/1/1.0, ``lr`` 1/0/0.0), the
    per-entry log, and the best spec."""
    doc = build_report_doc(_syn_summary(), _syn_entries())
    assert [k for k, _v in doc["headline"]] == [
        "task", "seed", "finished_reason", "baseline_score",
        "final_best_score", "improvement_factor", "experiments_run",
        "wall_seconds"]
    assert doc["headline"][0] == ("task", "parity-v1")
    assert doc["headline"][4] == ("final_best_score", 97.0)
    wr = {r["field"]: r for r in doc["win_rate"]}
    assert wr["hidden_dim"] == {"field": "hidden_dim", "trials": 1,
                                "wins": 1, "win_rate": 1.0}
    assert wr["lr"] == {"field": "lr", "trials": 1, "wins": 0,
                        "win_rate": 0.0}
    assert all(isinstance(r["trials"], int) and isinstance(r["wins"], int)
               for r in doc["win_rate"])
    assert doc["log"] == [
        {"kind": "baseline", "accepted": True, "score": 80.0,
         "gen_gap": None, "mutation": []},
        {"kind": "experiment", "accepted": True, "score": 97.0,
         "gen_gap": 0.5, "mutation": ["hidden_dim"]},
        {"kind": "experiment", "accepted": False, "score": 85.0,
         "gen_gap": 1.0, "mutation": ["lr"]},
    ]
    assert doc["best_spec"]["architecture"] == [64, 32]
    assert doc["title"] == "AutoRefine report \u2014 parity-v1 (seed 7)"


def test_render_md_headings_tables_and_fenced_spec():
    """A53 (SPEC.md 63.1): the Markdown rendering — ``#`` headings,
    ``|`` tables, and a fenced ``json`` block for the best spec."""
    md = render_report_md(build_report_doc(_syn_summary(), _syn_entries()))
    assert md.startswith("# AutoRefine report \u2014 parity-v1 (seed 7)")
    assert "## Run" in md and "## Win rate (per-field)" in md
    assert "## Experiment log" in md and "## Best spec" in md
    assert "| task | parity-v1 |" in md
    assert "| hidden_dim | 1 | 1 | 100% |" in md
    assert "| lr | 1 | 0 | 0% |" in md
    assert "```json" in md and '"architecture": [' in md
    assert '"model_family": "mlp"' in md
    assert md.rstrip().endswith("```")


def test_render_txt_has_no_pipes_or_fences():
    """A53 (SPEC.md 63.1): the plain-text one-pager — aligned columns,
    no ``|`` pipes, no code fences (email / terminal friendly)."""
    txt = render_report_txt(build_report_doc(_syn_summary(), _syn_entries()))
    assert txt.startswith("AutoRefine report \u2014 parity-v1 (seed 7)")
    assert "|" not in txt and "```" not in txt
    assert re.search(r"^\s*task\s+parity-v1$", txt, re.MULTILINE)
    assert "100%" in txt and "hidden_dim" in txt
    assert '"activation": "tanh"' in txt  # the spec block, indented


def test_all_three_formats_are_byte_stable(tmp_path):
    """A53 (SPEC.md 63.1.3, G2): a re-render of the same doc is
    byte-identical — twice for md/txt, and two *independent* PDF builds
    (the /ID is doc-deterministic, the dates invariant)."""
    doc = build_report_doc(_syn_summary(), _syn_entries())
    assert render_report_md(doc) == render_report_md(doc)
    assert render_report_txt(doc) == render_report_txt(doc)
    p1, p2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    render_report_pdf(doc, p1)
    render_report_pdf(doc, p2)
    b1, b2 = p1.read_bytes(), p2.read_bytes()
    assert b1 == b2 and b1[:4] == b"%PDF"


def test_render_report_dispatch_and_errors():
    """A53 (SPEC.md 63.1): ``render_report`` serves the two text formats
    and rejects ``pdf`` (a file output — one path per output kind) and
    unknown formats with ``ValueError``."""
    doc = build_report_doc(_syn_summary(), _syn_entries())
    assert render_report("md", doc) == render_report_md(doc)
    assert render_report("txt", doc) == render_report_txt(doc)
    with pytest.raises(ValueError, match="file output"):
        render_report("pdf", doc)
    with pytest.raises(ValueError, match="unknown report format"):
        render_report("doc", doc)


# --- 63.2 B3: the end-user guide (A53) ----------------------------------------

class _SoftTask:
    """A fake 3-class softmax task with a per-row holdout split (28.2)."""
    head = "softmax"
    n_outputs = 3
    state_dim = 2
    class_values = ["cat", "dog", "bird"]

    def holdout_rows(self, n: int, model=None):
        x = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 0.0]]
        y = [0, 1, 2, 0, 1]
        return np.asarray(x, dtype=np.float64)[:n], list(y[:n])


class _MseTask:
    """A fake mse regression task with a per-row holdout split (28.2)."""
    head = "mse"
    n_outputs = 1
    state_dim = 2


class _MseTaskRows(_MseTask):
    def holdout_rows(self, n: int, model=None):
        x = [[0.5, 1.0], [1.0, 0.5]]
        y = [[1.0], [2.0]]
        return np.asarray(x, dtype=np.float64)[:n], [list(r) for r in
                                                      np.asarray(y)[:n]]


class _FakeLogits:
    """A model whose ``forward`` returns fixed logits (no training)."""

    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float64)

    def forward(self, x):
        return self._logits


def test_user_guide_what_it_predicts_softmax():
    """A53 (SPEC.md 63.2): the plain-English "what it predicts" block is
    derived from ``head`` / ``state_dim`` / ``class_values`` (3 classes,
    accuracy, higher-is-better)."""
    v = user_guide_view(_syn_summary(), [], task=_SoftTask(),
                        model=_FakeLogits([[3, 0, 0]] * 5))
    assert v["what_it_predicts"] == {
        "input": "2 numeric feature(s)",
        "output": "one of 3 classes",
        "metric": "accuracy",
        "direction": "higher is better",
        "n_classes": 3,
    }


def test_user_guide_examples_and_confidence_hand_computed():
    """A53 (SPEC.md 63.2): the 5 holdout examples with hand-computed
    softmax max-probabilities — rows 0–2 have one logit 3 above the
    other two, so the max prob is ``1/(1+2e^-3)`` ≈ 0.9094; rows 3–4 are
    uniform (1/3). ``mean_max_prob`` = ``(3·0.9094 + 2·0.3333)/5`` ≈
    0.679, ``frac_at_least_0.8`` = 3/5 = 0.6. The argmax tie (rows 3–4)
    resolves to index 0 (``cat``), so row 4 (true ``dog``) is wrong."""
    p3 = 1.0 / (1.0 + 2.0 * math.exp(-3.0))      # ≈ 0.9094
    p1 = 1.0 / 3.0                               # ≈ 0.3333
    v = user_guide_view(_syn_summary(), [], task=_SoftTask(),
                        model=_FakeLogits([[3, 0, 0], [0, 3, 0],
                                           [0, 0, 3], [1, 1, 1],
                                           [0, 0, 0]]),
                        n_examples=5)
    ex = v["examples"]
    assert [e["correct"] for e in ex] == [True, True, True, True, False]
    assert ex[0] == {"input": "0, 1", "true": "cat", "predicted": "cat",
                     "confidence": round(p3, 4), "correct": True}
    assert ex[2] == {"input": "0.5, 0.5", "true": "bird",
                     "predicted": "bird", "confidence": round(p3, 4),
                     "correct": True}
    assert ex[4]["true"] == "dog" and ex[4]["predicted"] == "cat"
    assert v["examples_note"] is None
    c = v["confidence"]
    assert c["n"] == 5
    assert c["mean_max_prob"] == round((3 * p3 + 2 * p1) / 5, 4)
    assert c["frac_at_least_0.8"] == 0.6
    # 63.2: the distrust + limitation lines follow from the sample
    assert "the model is <80% confident on 40% of the sampled rows" in \
        v["when_to_distrust"]
    assert "1 of the first 5 sampled row(s) are mispredicted" in \
        v["when_to_distrust"]
    assert "illustrated on a sample of 5 holdout row(s), not the full " \
           "holdout" in v["known_limitations"]
    assert "the search stopped for: target" in v["known_limitations"]


def test_user_guide_how_to_read_margin():
    """A53 (SPEC.md 63.2): the plain-words read of the number — 97 vs 95
    → margin +2 — plus the seed-variance caveat."""
    v = user_guide_view(_syn_summary(), [], task=_SoftTask(),
                        model=_FakeLogits([[3, 0, 0]] * 5))
    assert "final score 97 vs target 95 (margin +2)" in v["how_to_read"]
    assert "single-run point estimate" in v["how_to_read"]


def test_user_guide_degrades_without_split_task_or_model():
    """A53 (SPEC.md 63.2): a task with no ``holdout_rows`` (episode /
    synthetic), a missing task, or a missing model each degrade to an
    explanatory note — the guide never invents rows."""
    class _EpisodeTask:
        head = "softmax"
        n_outputs = 2
        state_dim = 4

    v = user_guide_view(_syn_summary(), [], task=_EpisodeTask(),
                        model=_FakeLogits([[1.0, 0.0]]))
    assert v["examples"] is None
    assert "no per-row holdout split" in v["examples_note"]
    # no `class_values` → the graceful generic output phrase
    assert v["what_it_predicts"]["output"] == "a class label"

    v2 = user_guide_view(_syn_summary(), [])
    assert v2["what_it_predicts"] is None and v2["examples"] is None
    assert "no task was supplied" in v2["examples_note"]

    v3 = user_guide_view(_syn_summary(), [], task=_SoftTask(), model=None)
    assert v3["examples"] is None
    assert "no best model to score with" in v3["examples_note"]


def test_user_guide_mse_head_examples_and_no_confidence():
    """A53 (SPEC.md 63.2): the mse path — examples judged by the 5%
    tolerance (row 0: |1.01−1.0| = 0.01 ≤ 0.05 → correct; row 1:
    |3.0−2.0| = 1.0 > 0.1 → wrong), integer-valued floats render without
    a trailing ``.0``, and there is no confidence read."""
    v = user_guide_view(_syn_summary(), [], task=_MseTaskRows(),
                        model=_FakeLogits([[1.01], [3.0]]), n_examples=2)
    ex = v["examples"]
    assert ex[0] == {"input": "0.5, 1", "true": 1, "predicted": 1.01,
                     "confidence": None, "correct": True}
    assert ex[1] == {"input": "1, 0.5", "true": 2, "predicted": 3,
                     "confidence": None, "correct": False}
    assert v["confidence"] is None
    assert v["what_it_predicts"] == {
        "input": "2 numeric feature(s)",
        "output": "1 numeric value(s)",
        "metric": "mean squared error (MSE)",
        "direction": "lower is better",
        "n_classes": None,
    }
    assert "1 of the first 2 sampled row(s) are mispredicted" in \
        v["when_to_distrust"]
    assert all("confident" not in d for d in v["when_to_distrust"])


def test_render_user_guide_is_byte_stable():
    """A53 (SPEC.md 63.2, G2): a re-render of the same view is
    byte-identical; the header identifies the view."""
    v = user_guide_view(_syn_summary(), _syn_entries(), task=_SoftTask(),
                        model=_FakeLogits([[3, 0, 0], [0, 3, 0],
                                           [0, 0, 3], [1, 1, 1],
                                           [0, 0, 0]]))
    t1 = render_user_guide(v)
    assert t1 == render_user_guide(v)
    assert t1.startswith("=== AutoRefine user guide (what this model "
                         "does) ===")
    assert "when_to_distrust" in t1 and "how_to_read" in t1


# --- 63.3 B4: the decision artifact (A53) -------------------------------------

def _decision_entries() -> list:
    """Baseline 80 + two rejected candidates above the baseline: the
    39.2.2 priority traces them to ``overfit`` (gen_gap 5.0 > 0.05·85 =
    4.25) and ``ci`` (gen_gap 1.0 ≤ 0.05·82 = 4.1) respectively."""
    return [
        {"kind": "baseline", "accepted": True, "holdout_score": 80.0,
         "gen_gap": None, "mutation": []},
        {"kind": "experiment", "accepted": False, "holdout_score": 85.0,
         "gen_gap": 5.0, "mutation": ["lr"]},
        {"kind": "experiment", "accepted": False, "holdout_score": 82.0,
         "gen_gap": 1.0, "mutation": ["hidden_dim"]},
    ]


def _syn_diag() -> dict:
    """A 28.2 diagnostics dict: 3 classes, weakest ``dog`` at 70%."""
    return {"class_labels": ["cat", "dog", "bird"],
            "per_class": [90.0, 70.0, 95.0],
            "confusion": [[9, 1], [3, 7], [1, 19]],
            "class_counts": [10, 10, 20]}


def test_decision_go_ships():
    """A53 (SPEC.md 63.3): target met (97 ≥ 95, the 62.2.2 rule, one
    home) → GO, and the next step is ``ship`` with the seed-variance
    confirmation hint (no other hints)."""
    v = decision_view(_syn_summary(), _syn_entries(),
                      proj={"verdict": "more", "more": 40, "vmax": 99.5})
    assert v["verdict"]["verdict"] == "GO"
    assert v["target_met"] == {"target": 95.0, "met": True, "margin": 2.0}
    assert v["next_step"]["action"] == "ship"
    assert "seed-variance sweep" in v["next_step"]["detail"]
    assert v["next_step"]["hints"] == []


def test_decision_below_target_more_budget_projection():
    """A53 (SPEC.md 63.3): 90 < 95 but above the 80 baseline → REVIEW;
    with a ``more`` projection the next step is ``more_budget``
    ("run ~40 more … asymptote 99.5")."""
    s = _syn_summary(final_best_score=90.0)
    v = decision_view(s, _syn_entries(),
                      proj={"verdict": "more", "more": 40, "vmax": 99.5})
    assert v["verdict"]["verdict"] == "REVIEW"
    assert v["target_met"] == {"target": 95.0, "met": False, "margin": -5.0}
    ns = v["next_step"]
    assert ns["action"] == "more_budget"
    assert ns["detail"] == ("run ~40 more experiment(s) to close the gap "
                            "(curve asymptote 99.5)")


def test_decision_ceiling_and_insufficient_and_absent():
    """A53 (SPEC.md 63.3): the three non-``more`` projections map to
    ``relax_or_expand`` (the asymptote bound), ``collect_history``
    (< 2 same-task runs), and ``investigate`` (no projection at all)."""
    s = _syn_summary(final_best_score=90.0)
    v = decision_view(s, [], proj={"verdict": "ceiling", "vmax": 92})
    assert v["next_step"]["action"] == "relax_or_expand"
    assert "relax the target to <= 92, or expand the model space" in \
        v["next_step"]["detail"]
    v2 = decision_view(s, [], proj={"verdict": "insufficient"})
    assert v2["next_step"]["action"] == "collect_history"
    assert ">= 2 same-task runs" in v2["next_step"]["detail"]
    v3 = decision_view(s, [])
    assert v3["next_step"]["action"] == "investigate"
    assert "no budget projection available" in v3["next_step"]["detail"]


def test_decision_seed_variance_block():
    """A53 (SPEC.md 63.3): the seed-variance read from a list of final
    scores (90/97/93 → spread 7); empty / non-numeric → ``None`` (the
    assessment says so instead)."""
    s = _syn_summary(final_best_score=90.0)
    v = decision_view(s, [], seed_spread=[90.0, 97.0, 93.0])
    assert v["confidence"]["seed_variance"] == {
        "n_runs": 3, "min": 90.0, "max": 97.0, "spread": 7.0}
    assert "across 3 seeds the final score spans 90-97 (spread 7)" in \
        v["confidence"]["assessment"]
    assert decision_view(s, [])["confidence"]["seed_variance"] is None
    assert "no seed-variance data" in decision_view(s, [])["confidence"][
        "assessment"]
    assert decision_view(s, [], seed_spread=["x"])["confidence"][
        "seed_variance"] is None


def test_decision_failure_modes_capped_and_hints():
    """A53 (SPEC.md 63.3): four candidate failure modes (weakest class
    70% < 100%, 1 overfit rejection, 1 CI rejection, stopped at budget
    below target) cap to the top 3 in documented order; the hints target
    the gen-gap gate and the weak class."""
    s = _syn_summary(final_best_score=90.0, finished_reason="budget")
    v = decision_view(s, _decision_entries(), seed_spread=[90.0, 97.0],
                      diag=_syn_diag())
    assert v["confidence"] == {
        "seed_variance": {"n_runs": 2, "min": 90.0, "max": 97.0,
                          "spread": 7.0},
        "ci_rejections": 1,
        "overfit_rejections": 1,
        "assessment": ("across 2 seeds the final score spans 90-97 "
                       "(spread 7); 1 candidate(s) rejected for "
                       "overfitting (gen-gap); 1 candidate(s) rejected "
                       "by the CI gate"),
    }
    assert [m["kind"] for m in v["failure_modes"]] == [
        "weakest_class", "overfit", "ci"]
    assert v["failure_modes"][0]["detail"] == \
        "class 'dog' at 70.0% holdout accuracy"
    hints = v["next_step"]["hints"]
    assert "relax the gen-gap (overfit) gate to admit stronger " \
           "candidates" in hints
    assert "add features / data targeting class 'dog'" in hints


def test_decision_no_target_is_exploratory():
    """A53 (SPEC.md 63.3): with no target the ``target_met`` block is
    all-None (exploratory), the verdict is REVIEW (the 62.2.2 rule),
    and the failure-mode set drops the budget row."""
    s = _syn_summary(target=None, final_best_score=90.0,
                     finished_reason="budget")
    v = decision_view(s, _decision_entries())
    assert v["target_met"] == {"target": None, "met": None, "margin": None}
    assert v["verdict"]["verdict"] == "REVIEW"
    assert "exploratory" in v["verdict"]["rationale"]
    assert all(m["kind"] != "budget" for m in v["failure_modes"])


def test_render_decision_is_byte_stable():
    """A53 (SPEC.md 63.3, G2): a re-render of the same decision view is
    byte-identical; the header identifies the artifact."""
    v = decision_view(_syn_summary(), _decision_entries(),
                      seed_spread=[90.0, 97.0], diag=_syn_diag())
    t1 = render_decision(v)
    assert t1 == render_decision(v)
    assert t1.startswith("=== AutoRefine decision (go/no-go + next "
                         "step) ===")
    assert "failure_modes" in t1 and "next_step" in t1


# --- exports + version (63.6.5 / 33.1, A53) ------------------------------------

def test_new_names_are_exported():
    """A53 (SPEC.md 63.6.5): the ten new top-level names are in
    ``__all__`` and resolvable (33.1)."""
    for name in NEW_EXPORTS:
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_version_round_v049():
    """A53 (33.1): the version stepped with the round — ``0.49.0`` in
    both sources (pyproject and the package)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.52.0"


# --- CLI (a real tiny parity run; A53) -----------------------------------------

@pytest.fixture
def real_run(tmp_path):
    """A tiny, self-consistent live run (parity-v1, 2 experiments, G2).
    Parity has no ``holdout_rows`` (episode task) — the ``--user``
    degradation path on a real run."""
    env = AutoRefineEnv(task="parity-v1", seed=7, budget=Budget(2, 300, 30),
                        runs_dir=tmp_path)
    policy = SearchPolicy(seed=7)
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    assert env.memory.load_summary()
    return Path(env.run_dir)


def test_cli_report_format_md(real_run, capsys):
    """A53 (SPEC.md 63.1): ``--format md`` rc 0 — the Markdown headings,
    the run table, and the fenced spec on stdout."""
    rc = cli_main(["report", "--run", str(real_run), "--format", "md"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("# AutoRefine report \u2014 parity-v1")
    assert "## Run" in out and "| task | parity-v1 |" in out
    assert "```json" in out


def test_cli_report_format_txt(real_run, capsys):
    """A53 (SPEC.md 63.1): ``--format txt`` rc 0 — no pipes, no fences
    (the terminal / email one-pager)."""
    rc = cli_main(["report", "--run", str(real_run), "--format", "txt"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("AutoRefine report \u2014 parity-v1")
    assert "|" not in out and "```" not in out
    assert "win rate (per-field)" in out


def test_cli_report_format_pdf_is_a_deterministic_file(real_run, capsys):
    """A53 (SPEC.md 63.1.3): ``--format pdf`` rc 0 writes
    ``report.pdf`` into the run dir (``%PDF``), and a re-run is
    byte-identical (the G2 CLI pin; the /ID is doc-deterministic)."""
    rc = cli_main(["report", "--run", str(real_run), "--format", "pdf"])
    out = capsys.readouterr().out
    assert rc == 0 and "report.pdf" in out
    pdf = real_run / "report.pdf"
    b1 = pdf.read_bytes()
    assert b1[:4] == b"%PDF"
    rc2 = cli_main(["report", "--run", str(real_run), "--format", "pdf"])
    assert rc2 == 0
    assert pdf.read_bytes() == b1


def test_cli_report_user(real_run, capsys):
    """A53 (SPEC.md 63.2): ``--user`` rc 0 — the guide's header, the
    "what it predicts" block, and (parity has no per-row holdout split)
    the explanatory no-split note rather than invented rows."""
    rc = cli_main(["report", "--run", str(real_run), "--user"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("=== AutoRefine user guide (what this model "
                          "does) ===")
    assert "what_it_predicts" in out and "how_to_read" in out
    assert "no per-row holdout split" in out


def test_cli_report_decision(real_run, capsys):
    """A53 (SPEC.md 63.3): ``--decision`` rc 0 — the artifact's header,
    the verdict (one home: the 62.2.2 rule), the confidence read, and
    the next step."""
    rc = cli_main(["report", "--run", str(real_run), "--decision"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("=== AutoRefine decision (go/no-go + next "
                          "step) ===")
    assert "verdict" in out and "target_met" in out
    assert "confidence" in out and "failure_modes" in out
    assert "next_step" in out and "action" in out


def test_cli_report_views_are_mutually_exclusive(real_run, capsys):
    """A53 (SPEC.md 63.4.1): each human view is mutually exclusive with
    the machine paths and the sibling human views (rc 1, the shared
    ``_EXCLUSIVE_FLAGS`` guard)."""
    cases = (
        (["--format", "md", "--json"], "--json"),
        (["--user", "--json"], "--json"),
        (["--decision", "--what-if", "score>=95"], "--what-if"),
        (["--user", "--format", "md"], "--format"),
        (["--decision", "--user"], "--user"),
        (["--decision", "--audience", "exec"], "--audience"),
        (["--audience", "exec", "--user"], "--user"),
        (["--audience", "exec", "--format", "md"], "--format"),
    )
    for flags, why in cases:
        rc = cli_main(["report", "--run", str(real_run), *flags])
        err = capsys.readouterr().err
        assert rc == 1, flags
        assert "mutually exclusive" in err, (flags, err)
        assert why.split("/")[0] in err or why in err


def test_cli_default_report_unchanged_and_technical_identical(
        real_run, capsys):
    """A53 (SPEC.md 63.5, the pin anchor): the default report is
    untouched (still the technical summary with ``final_best_score``)
    and byte-identical to ``--audience technical``."""
    rc1 = cli_main(["report", "--run", str(real_run)])
    out1 = capsys.readouterr().out
    assert rc1 == 0 and "final_best_score" in out1
    rc2 = cli_main(["report", "--run", str(real_run),
                    "--audience", "technical"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert out1 == out2
