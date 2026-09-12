"""v0.34 — beginner onboarding: guided setup, knob glossary, presets,
and the plain-English narration (SPEC.md 48, A38, M37).

Covers the pure core behind the app's five beginner features:

- 48.1 guided two-step setup — the BEGINNER_KNOBS / ADVANCED_KNOBS
  split over the ALL_KNOBS vocabulary (disjoint, exact union);
- 48.2 the knob glossary — one prose line per knob, keys exactly
  ALL_KNOBS (the 48.2.2 completeness invariant);
- 48.3 presets — GUI-knob-only flag bundles, copy semantics, and the
  apply-on-change merge (the preset touches only its knobs — including
  the Advanced ones; the input is not mutated; re-selecting the same
  preset keeps the user's manual edits);
- 48.4 the "what happened" narrative — narrate_run's four paragraphs
  (headline, tried, improved, recipe) + graceful degradation;
- 48.5 narrate the loop — narrate_step's accepted/rejected/duplicate
  forms + narrate_baseline; every rendered line cp1252-printable
  (Windows console-safe).

Plus the A38 round regression: the app renders the onboarding core
(source scan of dashboard_app.py) and the version stepped to `0.39.0`
in both sources (33.1).

House rules: no cross-test imports (all fixtures synthesized here);
stdlib + numpy only (narrate.py imports the dashboard core, which
needs numpy).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import autorefine
from autorefine import narrate, onboarding

REPO = Path(__file__).resolve().parent.parent
APP_SRC = REPO / "src" / "autorefine" / "dashboard_app.py"


# --- 48.1 guided two-step setup (A38) -----------------------------------------

def test_guided_setup_split():
    """A38 (SPEC.md 48.1): the beginner set is exactly
    (data, label, target); the advanced set the other six; the two
    sets are disjoint and their union is ALL_KNOBS, and knob_groups()
    covers ALL_KNOBS exactly (48.1.1–48.1.3)."""
    assert tuple(onboarding.BEGINNER_KNOBS) == ("data", "label", "target")
    assert tuple(onboarding.ADVANCED_KNOBS) == (
        "policy", "seed", "experiments", "max_train", "quality", "runs_dir")
    assert tuple(onboarding.ALL_KNOBS) == \
        onboarding.BEGINNER_KNOBS + onboarding.ADVANCED_KNOBS
    beginner, advanced = (set(onboarding.BEGINNER_KNOBS),
                          set(onboarding.ADVANCED_KNOBS))
    assert not (beginner & advanced)  # disjoint
    assert beginner | advanced == set(onboarding.ALL_KNOBS)  # exact union
    groups = onboarding.knob_groups()
    assert set(groups) == set(onboarding.ALL_KNOBS)
    assert set(groups.values()) <= {"beginner", "advanced"}
    assert {k for k, g in groups.items() if g == "beginner"} == beginner
    assert {k for k, g in groups.items() if g == "advanced"} == advanced


# --- 48.2 the knob glossary (A38) ----------------------------------------------

def test_glossary_complete_and_prose():
    """A38 (SPEC.md 48.2): KNOB_GLOSSARY's keys are exactly ALL_KNOBS
    (the 48.2.2 completeness invariant — a sidebar knob without a
    glossary line, or a glossary line for a nonexistent knob, fails
    here) and every entry is non-trivial, console-safe prose
    (48.2.1)."""
    assert set(onboarding.KNOB_GLOSSARY) == set(onboarding.ALL_KNOBS)
    for knob, text in onboarding.KNOB_GLOSSARY.items():
        assert isinstance(text, str), knob
        assert len(text) >= 10 and any(c.isalpha() for c in text), knob
        text.encode("cp1252")  # the app shows these as help= text


# --- 48.3 presets (A38) ---------------------------------------------------------

def test_presets_reference_gui_knobs_only():
    """A38 (SPEC.md 48.3.1): a preset may never reference a knob the
    sidebar does not expose (no CLI-only setting smuggled in) — every
    flag key is in ALL_KNOBS; each bundle has a label + description."""
    assert set(onboarding.PRESETS) == {"smoke", "classify", "thorough"}
    for name, p in onboarding.PRESETS.items():
        assert p["flags"], name
        assert set(p["flags"]) <= set(onboarding.ALL_KNOBS), name
        assert p["label"].strip() and p["description"].strip(), name


def test_preset_flags_copy_semantics():
    """A38 (SPEC.md 48.3.2): preset_flags returns a *copy* (mutating
    the result leaves PRESETS untouched), and an unknown name is a loud
    KeyError, not a silent no-op."""
    flags = onboarding.preset_flags("smoke")
    flags["experiments"] = 999
    assert onboarding.PRESETS["smoke"]["flags"]["experiments"] == 5
    with pytest.raises(KeyError):
        onboarding.preset_flags("no-such-preset")


def test_apply_preset_merge_semantics():
    """A38 (SPEC.md 48.3.3): apply_preset — the pure merge helper —
    overrides exactly the preset's knobs, keeps every other knob's
    current value, and does not mutate its input. (The app applies the
    bundle on selection change, 48.3.3; covered by the AppTest above.)"""
    current = {"label": "y", "target": 80.0, "seed": 42}
    before = dict(current)
    merged = onboarding.apply_preset("classify", current)
    assert current == before  # input not mutated
    # the preset's knobs are overridden...
    assert merged["target"] == 90.0
    assert merged["experiments"] == 40
    assert merged["max_train"] == 30.0
    assert merged["policy"] == "bandit"
    # ...and the rest keeps its current value
    assert merged["label"] == "y"
    assert merged["seed"] == 42
    merged["target"] = 1.0  # the result is its own dict
    assert current["target"] == 80.0
    with pytest.raises(KeyError):
        onboarding.apply_preset("no-such-preset", {})


def test_preset_choices_and_summary():
    """A38 (SPEC.md 48.3.2): preset_choices() is the stable
    (name, label) pair list in dict order; preset_summary is the
    one-line description the app shows under the selection."""
    choices = onboarding.preset_choices()
    assert [name for name, _lab in choices] == \
        list(onboarding.PRESETS)
    for name, label in choices:
        assert label == onboarding.PRESETS[name]["label"]
    assert onboarding.preset_summary("smoke") == \
        onboarding.PRESETS["smoke"]["description"]


# --- 48.4 the "what happened" narrative (A38) ----------------------------------

def _res_pass() -> dict:
    """A synthesized finish() result: a PASS run that improved
    60.00 -> 90.00 -> 96.40 over two accepted candidates."""
    return {
        "verdict": "PASS", "target": 95.0, "final_best_score": 96.4,
        "baseline_score": 60.0, "experiments_run": 12,
        "best_spec": {"family": "mlp", "hidden_dim": 256},
        "updates": [
            {"accepted": True, "candidate_score": 90.0,
             "mutation": ["hidden_dim"]},
            {"accepted": False, "candidate_score": 85.0,
             "mutation": ["lr"]},
            {"accepted": True, "candidate_score": 96.4,
             "mutation": ["hidden_dim", "lr"]},
        ],
    }


def test_narrate_run_four_paragraphs():
    """A38 (SPEC.md 48.4): narrate_run renders all four paragraphs —
    the headline (verdict, target, final, baseline, count), what it
    tried (the field_stats rollup), how it improved (the accepted
    chain), and the winning recipe (best_spec chips)."""
    text = narrate.narrate_run(_res_pass())
    # headline (48.4.1)
    assert "Verdict: PASS" in text
    assert "at least 95" in text
    assert "96.40" in text
    assert "60.00" in text
    assert "12 experiments" in text
    # what it tried (48.4.2) — hidden_dim: 2 trials/2 wins; lr: 2/1
    assert "It explored 2 model knob(s)" in text
    assert "most often hidden_dim" in text
    assert "improving 2" in text
    # how it improved (48.4.2) — the accepted chain
    assert "It made 2 improvement(s)" in text
    assert "60.00 -> 90.00 -> 96.40" in text
    # the winning recipe (48.4.2)
    assert "The winning recipe:" in text
    assert "family=mlp" in text
    assert "hidden_dim=256" in text


def test_narrate_run_degrades_gracefully():
    """A38 (SPEC.md 48.4.2): missing pieces degrade to honest "no …"
    lines — no updates logged, no accepted chain (fin < base), and no
    recorded recipe (28.5 style)."""
    res = {
        "verdict": "MISS", "target": 99.0, "final_best_score": 88.0,
        "baseline_score": 90.0, "experiments_run": 5,
        "best_spec": None, "updates": [],
    }
    text = narrate.narrate_run(res)
    assert "Verdict: MISS" in text
    assert "It did not log any model-knob changes to try." in text
    assert "It never beat its starting baseline" in text
    assert "No final recipe was recorded." in text
    # a run with scores but no accepted candidates and final >= baseline
    res2 = dict(res, final_best_score=90.0, baseline_score=85.0)
    text2 = narrate.narrate_run(res2)
    assert "It made no accepted improvements over the baseline." in text2


# --- 48.5 narrate the loop (A38) ------------------------------------------------

def test_narrate_baseline():
    """A38 (SPEC.md 48.5.2): narrate_baseline carries the start()
    info's baseline and target in the "Started the search …" line."""
    line = narrate.narrate_baseline(
        {"baseline_score": 60.0, "target": 95.0})
    assert line.startswith("Started the search")
    assert "60.00" in line and "95" in line
    # missing pieces degrade (no crash, no "None")
    assert "None" not in narrate.narrate_baseline({})


def test_narrate_step_three_forms():
    """A38 (SPEC.md 48.5.1): accepted, rejected, and duplicate update
    dicts render three distinct friendly forms (the app form of the
    41.2/41.3 terminal narration)."""
    acc = narrate.narrate_step({
        "index": 1, "mutation": ["hidden_dim"], "candidate_score": 90.0,
        "accepted": True, "best_score": 90.0})
    assert "Experiment 1" in acc and "tried hidden_dim" in acc
    assert "scored 90.00" in acc and "kept it" in acc and "new best" in acc
    rej = narrate.narrate_step({
        "index": 2, "mutation": [], "candidate_score": 85.0,
        "accepted": False})
    assert "Experiment 2" in rej and "85.00" in rej
    assert "not better than the best so far" in rej and "skipped it" in rej
    dup = narrate.narrate_step({
        "index": 3, "mutation": ["lr"], "candidate_score": None,
        "accepted": False})
    assert "Experiment 3" in dup
    assert "a duplicate, so it was skipped (no retrain)" in dup
    # the three forms are distinct
    assert len({acc, rej, dup}) == 3


def test_narration_console_safe():
    """A38 (SPEC.md 48.5.3): every rendered narration line is
    cp1252-printable (the Windows console never sees an unprintable
    glyph)."""
    lines = [
        narrate.narrate_baseline({"baseline_score": 60.0, "target": 95.0}),
        narrate.narrate_step({"index": 1, "mutation": ["hidden_dim"],
                              "candidate_score": 90.0, "accepted": True,
                              "best_score": 90.0}),
        narrate.narrate_step({"index": 2, "mutation": [],
                              "candidate_score": 85.0, "accepted": False}),
        narrate.narrate_run(_res_pass()),
    ]
    for line in lines:
        line.encode("cp1252")


# --- A38 round regression --------------------------------------------------------

def test_app_renders_onboarding_core():
    """A38 (SPEC.md 48.2/48.4/48.5): the app is a thin renderer over
    the onboarding + narrate core — the sidebar reads KNOB_GLOSSARY as
    widget help= text (48.2.1), shows the glossary expander + presets
    + narrate toggle (48.1/48.3/48.5), wires narrate into _run, and
    renders narrate_run in the result view (48.4.3)."""
    src = APP_SRC.read_text(encoding="utf-8")
    assert "help=KNOB_GLOSSARY[" in src          # 48.2.1 single source
    assert "for knob in ALL_KNOBS" in src        # 48.2 glossary expander
    assert "preset_pairs = preset_choices()" in src  # 48.3 selectbox
    assert 'key="preset"' in src
    assert 'key="narrate"' in src                # 48.5.3 toggle
    assert "narrate=narrate" in src              # _run call site
    assert "narrate_baseline(info)" in src       # 48.5.2
    assert "narrate_step(u)" in src              # 48.5.1
    assert "st.markdown(narrate_run(res))" in src  # 48.4.3
    # 48.3.3 apply-on-change: the bundle is written when the selection
    # changes (not the old setdefault pattern that silently no-oped
    # because the widgets always render before a preset is picked)
    assert "_preset_applied" in src
    assert 'st.session_state.get("_preset_applied") != preset_name' in src
    assert "if knob not in st.session_state" not in src


def test_app_preset_changes_advanced_widgets(tmp_path):
    """A38 (SPEC.md 48.3.3): picking a preset in the sidebar actually
    fills in the knobs it defines — including the Advanced ones (the
    regression for the preset dropdown silently doing nothing)."""
    pytest.importorskip("streamlit",
                        reason="dashboard app is optional (SPEC.md 23)")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_SRC), default_timeout=300)
    at.run()  # first render: widgets materialize at their defaults
    assert not at.exception
    assert at.session_state["experiments"] == 30  # default budget
    at.session_state["preset"] = "Thorough search"
    at.run()
    assert not at.exception
    # every flag the preset defines is now the preset's value (48.3.1)
    assert at.session_state["experiments"] == 120
    assert at.session_state["max_train"] == 60.0
    assert at.session_state["policy"] == "bandit"
    assert at.session_state["quality"] == "v04"
    assert at.session_state["target"] == 95.0
    # ...and a manual edit after the preset sticks (starting point, not
    # a lock): re-rendering with the same preset does not clobber it
    at.session_state["experiments"] = 77
    at.run()
    assert not at.exception
    assert at.session_state["experiments"] == 77


def test_version_round_v034():
    """A38 (SPEC.md 48.6, 33.1): the version stepped to `0.39.0` in
    both sources (v0.39 ⇒ `0.39.0`, M42, SPEC.md 53)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.46.0"
