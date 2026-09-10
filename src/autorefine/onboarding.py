"""SPEC.md 48 (v0.34): the app's beginner onboarding — pure, stdlib-only.

The Streamlit app (SPEC.md 23.2) is a thin renderer; the testable core
behind its three beginner features (48.1–48.3) lives here:

- **48.1 guided setup** — the sidebar knobs split into the *beginner* set
  (data + label + target) shown top-level and the *advanced* set (the rest)
  collapsed under an "Advanced" expander (48.1.1/48.1.2);
- **48.2 the knob glossary** — a plain-English description for *every* knob
  (48.2.1), with the completeness invariant that no knob is left un-glossed
  (48.2.2);
- **48.3 presets** — one-click sensible bundles that override only the knobs
  they define, leaving the rest untouched (48.3.2/48.3.3).

House rules: stdlib only (no numpy, no streamlit); no cross-test imports;
the app reads these constants so the split / glossary / presets are one
source of truth (not re-declared in the UI).
"""
from __future__ import annotations

# 48.1.1: the beginner knob set — everything a first-time user needs to
# start a run: the data, what to predict, and the bar to clear. Shown
# top-level in the sidebar (48.1.2).
BEGINNER_KNOBS: tuple[str, ...] = ("data", "label", "target")

# 48.1.2: the advanced knob set — the rest of the sidebar, collapsed under
# the "Advanced" expander by default. Same widgets as before 48.1; only
# their grouping changes (the defaults are unchanged, so a run that used to
# work does).
ADVANCED_KNOBS: tuple[str, ...] = (
    "policy", "seed", "experiments", "max_train", "quality", "runs_dir",
)

# 48.1.3: the full knob vocabulary, in sidebar display order. The two sets
# are disjoint and their union is exactly this (48.2.2 invariant).
ALL_KNOBS: tuple[str, ...] = BEGINNER_KNOBS + ADVANCED_KNOBS

# knob → group (48.1): every knob in exactly one of the two groups.
KNOB_GROUP: dict[str, str] = {
    **{k: "beginner" for k in BEGINNER_KNOBS},
    **{k: "advanced" for k in ADVANCED_KNOBS},
}

# 48.2.1: the plain-English glossary — one friendly line per knob. Written
# for someone who has never seen a bandit or a seed; technical names are
# only ever shown in parentheses.
KNOB_GLOSSARY: dict[str, str] = {
    "data": ("Your input. Upload a CSV (header + rows), or point at a folder "
             "of labelled images or audio."),
    "label": ("Which column is the thing you want to predict. Leave blank and "
              "we'll pick the label/target/y/class column (or the last)."),
    "target": ("The score you want to reach on data the model has never seen "
               "(0–100). Higher = a harder bar to clear."),
    "policy": ("How new candidates are picked. bandit (default) learns which "
               "knobs help as it goes; search tries the whole space "
               "deterministically."),
    "seed": ("The random seed. The same seed reproduces the same run exactly "
             "— change it to see how much the result moves."),
    "experiments": ("The budget: how many candidate models to try before "
                    "stopping. More = a better answer, but slower."),
    "max_train": ("The most seconds any single model may take to train. Keeps "
                  "one odd candidate from eating the whole budget."),
    "quality": ("The acceptance rule. v04 (default) is the current CLI rule "
                "(confidence + efficiency); legacy is the older strict-score "
                "rule."),
    "runs_dir": ("The folder where finished runs are saved, so you can re-open "
                 "and compare them later."),
}


def knob_groups() -> dict[str, str]:
    """48.1.1/48.1.2: the knob → group map (``"beginner"`` /
    ``"advanced"``). Pure; the app renders the beginner knobs top-level and
    the advanced ones under the expander. Every knob is in exactly one
    group (48.2.2 invariant: the map covers ALL_KNOBS exactly)."""
    groups = {k: KNOB_GROUP[k] for k in ALL_KNOBS}
    if set(groups) != set(ALL_KNOBS):
        raise AssertionError("KNOB_GROUP does not cover ALL_KNOBS exactly")
    return groups


# 48.3.1: the presets — one-click sensible bundles. Each ``flags`` dict
# overrides only the knobs it names (48.3.3); everything else keeps its
# current value. Written over the GUI's own knob vocabulary only — a preset
# never references a knob the sidebar does not expose (so a preset can't
# smuggle in a CLI-only setting the user can't see or change).
PRESETS: dict[str, dict] = {
    "smoke": {
        "label": "Quick smoke test",
        "description": ("A handful of experiments with short trains — fast, "
                        "just to see the loop work end to end."),
        "flags": {"experiments": 5, "max_train": 10.0},
    },
    "classify": {
        "label": "Classify my labels",
        "description": ("A solid default for a labelled table: a real budget "
                        "with the bandit policy and a 90 target."),
        "flags": {"target": 90.0, "experiments": 40, "max_train": 30.0,
                  "policy": "bandit"},
    },
    "thorough": {
        "label": "Thorough search",
        "description": ("A bigger budget and longer trains aimed at the "
                        "default 95 bar — the pick when accuracy matters."),
        "flags": {"target": 95.0, "experiments": 120, "max_train": 60.0,
                  "policy": "bandit", "quality": "v04"},
    },
}


def preset_choices() -> list[tuple[str, str]]:
    """48.3.2: the ``(name, display_label)`` pairs for the app's preset
    selectbox, in a stable order (dict insertion order). Pure."""
    return [(name, PRESETS[name]["label"]) for name in PRESETS]


def preset_flags(name: str) -> dict:
    """48.3.2: the flag overrides a preset applies — a *copy* (mutating the
    result never touches ``PRESETS``). ``KeyError`` names the preset, so a
    bad name is a loud failure, not a silent no-op (the app only offers
    valid names)."""
    return dict(PRESETS[name]["flags"])


def apply_preset(name: str, current: dict) -> dict:
    """48.3.3: merge a preset's flags over the current knob values — the
    preset touches only the knobs it defines; every other knob keeps its
    current value. Returns a new dict (``current`` is not mutated). Pure."""
    merged = dict(current)
    merged.update(preset_flags(name))
    return merged


def preset_summary(name: str) -> str:
    """48.3.2: a one-line human description of a preset (the app shows it
    under the selectbox). Pure."""
    return PRESETS[name]["description"]
