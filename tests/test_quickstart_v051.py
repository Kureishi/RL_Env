"""v0.51 — "Quickstart in the UI" (SPEC.md 65, A55, M54).

Covers the A55 items as implemented — the UI's onboarding, one source,
three surfaces, all additive (A1–A54 stay green; the app's data-loaded
path and every CLI byte-pin are untouched, 65.5):

- **65.1 the step model** — four steps in the stable order
  ``pick_data → run_loop → inspect_export → check_env``; every step dict
  has exactly the five keys ``{id, title, body, app, cli}`` (65.1.4);
  unique ids; ``quickstart_commands()`` equals the ``cli`` column in
  order; ``QUICKSTART_INTRO`` carries the "autonomous" loop sentence.
- **65.2 the renderers** — ``render_quickstart_md`` / ``render_quickstart``
  are byte-stable (G2), carry the intro + the four titles + all four CLI
  commands; the text rendering has no table pipes (65.2.2); the Markdown
  rendering is ``#``-headed and step-blocked (65.2.1).
- **65.4 the demo** — ``demo_recipe()`` is exactly the 41.3.1 recipe
  (parity-v1; 3 experiments / 60 s wall / 10 s per train; search policy;
  un-gated); ``run_demo`` returns the seven-key shape (six summary keys,
  a non-empty string ``trace``, a dict ``best_spec``, an existing
  ``run_dir`` with a registry entry — 41.3.2); two same-seed calls give
  the same summary (wall time excluded), the same trace, and the same
  best spec (G2).
- **65.3 the CLI** — ``autorefine quickstart`` rc 0 + the four titles +
  the intro on stdout; ``--json`` parses as ``{intro, steps}`` with the
  steps equal to ``quickstart_steps()``; the ``--json`` output is
  byte-stable.
- **exports (33.1)** — the eight new top-level names are in ``__all__``.
- **version + regression** — ``0.51.0`` in both sources (33.1); the A25
  index advances (51 rows; ``defined == set(range(1, 56))`` — asserted
  in ``test_coherency_v021.py``, A25).

House rules (A55): no cross-test imports (all fixtures synthesized
here); the pure core is tested by hand-computation; ``dashboard_app``
is never imported (23.1).
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import autorefine
from autorefine import (
    QUICKSTART_INTRO,
    QUICKSTART_STEPS,
    demo_recipe,
    quickstart_commands,
    quickstart_steps,
    render_quickstart,
    render_quickstart_md,
    run_demo,
)
from autorefine.cli import main as cli_main
from autorefine.registry import load_registry

REPO = Path(__file__).resolve().parents[1]

# A55 (65.1): the step model's stable shape.
STEP_IDS = ("pick_data", "run_loop", "inspect_export", "check_env")
STEP_KEYS = {"id", "title", "body", "app", "cli"}

# A55 (exports, 33.1): the eight new top-level names.
NEW_EXPORTS = (
    "QUICKSTART_INTRO", "QUICKSTART_STEPS", "quickstart_steps",
    "quickstart_commands", "render_quickstart", "render_quickstart_md",
    "demo_recipe", "run_demo",
)


# --- 65.1 the step model (A55) -------------------------------------------------

def test_steps_are_the_four_onboarding_steps_in_order():
    """A55 (SPEC.md 65.1.2): the four onboarding steps in the stable
    rendered order — ``pick_data → run_loop → inspect_export →
    check_env``."""
    steps = quickstart_steps()
    assert [s["id"] for s in steps] == list(STEP_IDS)
    assert len(steps) == 4


def test_step_dicts_carry_exactly_the_five_keys():
    """A55 (SPEC.md 65.1.4): every step dict has exactly the keys
    ``{id, title, body, app, cli}`` — a sixth key (or a missing one)
    is a suite failure, not a silent UI drift (the C4-registry
    pattern)."""
    for s in quickstart_steps():
        assert set(s) == STEP_KEYS, s["id"]


def test_step_ids_are_unique_and_commands_match_the_column():
    """A55 (SPEC.md 65.1.3/65.1.4): the ids are unique, and
    ``quickstart_commands()`` is exactly the ``cli`` column in step
    order (one home for the README Quickstart invocations)."""
    ids = [s["id"] for s in quickstart_steps()]
    assert len(set(ids)) == len(ids)
    assert quickstart_commands() == [s["cli"] for s in quickstart_steps()]


def test_intro_is_the_autonomous_loop_sentence():
    """A55 (SPEC.md 65.1.1): ``QUICKSTART_INTRO`` is the one-paragraph
    "what is this" — the README's own description (the autonomous
    improver loop, the hard budget, NumPy-only)."""
    assert "iteratively improve" in QUICKSTART_INTRO
    assert "autonomous" in QUICKSTART_INTRO
    assert "hard budget" in QUICKSTART_INTRO
    assert "NumPy-only" in QUICKSTART_INTRO


def test_steps_carry_non_trivial_app_and_cli_actions():
    """A55 (SPEC.md 65.1.2): every step has a non-empty plain-English
    body, an in-app action, and a copy-pasteable CLI command (the
    ``autorefine`` vocabulary)."""
    for s in quickstart_steps():
        assert s["body"].strip(), s["id"]
        assert s["app"].strip(), s["id"]
        assert s["cli"].startswith("autorefine "), s["id"]


# --- 65.2 the renderers (A55) --------------------------------------------------

def test_renderers_are_byte_stable():
    """A55 (SPEC.md 65.2.1/65.2.2): a re-render of the same step model
    is byte-identical (G2)."""
    assert render_quickstart_md() == render_quickstart_md()
    assert render_quickstart() == render_quickstart()


def test_renderers_carry_the_full_step_model():
    """A55 (SPEC.md 65.2): both renderings carry the intro, the four
    step titles, and all four CLI commands (nothing re-typed, nothing
    dropped)."""
    steps = quickstart_steps()
    for render in (render_quickstart_md(), render_quickstart()):
        assert QUICKSTART_INTRO in render
        for s in steps:
            assert s["title"] in render, s["id"]
            assert s["body"] in render, s["id"]
            assert s["cli"] in render, s["id"]


def test_text_rendering_is_terminal_friendly():
    """A55 (SPEC.md 65.2.2): the text rendering is the
    ``autorefine quickstart`` command's output — aligned columns,
    **no table pipes** (the 64.1.4 convention)."""
    out = render_quickstart()
    assert out.startswith("=== AutoRefine quickstart ===")
    assert "|" not in out


def test_markdown_rendering_is_headed_and_step_blocked():
    """A55 (SPEC.md 65.2.1): the Markdown rendering is the app's first
    screen — a ``#`` heading, the intro, one ``**title**`` block per
    step with the in-app action and the CLI command."""
    md = render_quickstart_md()
    assert md.startswith("# AutoRefine — Quickstart")
    assert "**in the app:**" in md
    assert "**on the CLI:**" in md
    for s in quickstart_steps():
        assert f"**{s['title']}**" in md


# --- 65.4 the demo action (A55) ------------------------------------------------

def test_demo_recipe_is_the_41_3_1_recipe():
    """A55 (SPEC.md 65.4.1): the demo recipe is exactly the 41.3.1
    tiny fixed budget — parity-v1, 3 experiments / 60 s wall /
    10 s per train, the search policy, un-gated, the v0.4 preset —
    one home, so the in-app action cannot drift from ``run --demo``."""
    r = demo_recipe()
    assert r["task"] == "parity-v1"
    assert r["budget"] == {"max_experiments": 3, "max_wall_seconds": 60.0,
                           "max_train_seconds": 10.0}
    assert r["policy"] == "search"
    assert r["target"] is None
    assert r["quality"] == "v04"
    # a copy: mutating the result never touches the recipe's identity
    r["budget"]["max_experiments"] = 99
    assert demo_recipe()["budget"]["max_experiments"] == 3


def test_run_demo_returns_the_narrated_shape(tmp_path):
    """A55 (SPEC.md 65.4.2): ``run_demo`` returns the seven-key shape —
    the six 41.3 summary keys, a non-empty string ``trace`` (41.2), a
    dict ``best_spec`` — and **a demo run is a run (41.3.2)**: the
    ``run_dir`` exists and the registry gained an entry."""
    res = run_demo(seed=7, runs_dir=str(tmp_path / "runs"))
    assert set(res) == {"task", "seed", "run_dir", "budget", "summary",
                        "trace", "best_spec"}
    assert res["task"] == "parity-v1" and res["seed"] == 7
    run_dir = Path(res["run_dir"])
    assert run_dir.is_dir()
    assert set(res["summary"]) == {"baseline_score", "final_best_score",
                                   "improvement_factor", "experiments_run",
                                   "wall_seconds", "finished_reason"}
    assert res["summary"]["experiments_run"] >= 1
    assert isinstance(res["trace"], list) and res["trace"]
    assert all(isinstance(line, str) and line for line in res["trace"])
    assert isinstance(res["best_spec"], dict) and res["best_spec"]
    reg = load_registry(str(tmp_path / "runs"))
    assert any(e.get("task") == "parity-v1" for e in reg), reg


def test_run_demo_is_deterministic_for_a_seed(tmp_path):
    """A55 (SPEC.md 65.4.2, G2): two same-seed calls give the same
    summary (the measured ``wall_seconds`` excluded — it is wall time,
    not computation), the same trace, and the same best spec."""
    a = run_demo(seed=7, runs_dir=str(tmp_path / "a"))
    b = run_demo(seed=7, runs_dir=str(tmp_path / "b"))
    for key in ("baseline_score", "final_best_score", "improvement_factor",
                "experiments_run", "finished_reason"):
        assert a["summary"][key] == b["summary"][key], key
    assert a["trace"] == b["trace"]
    assert a["best_spec"] == b["best_spec"]


# --- 65.3 the CLI (A55) ---------------------------------------------------------

def test_cli_quickstart_prints_the_steps(capsys):
    """A55 (SPEC.md 65.3.1): ``autorefine quickstart`` is rc 0 and
    prints the text rendering — the intro + the four step titles (zero
    training, zero writes, 65.3.2)."""
    rc = cli_main(["quickstart"])
    out = capsys.readouterr().out
    assert rc == 0
    assert QUICKSTART_INTRO in out
    for s in quickstart_steps():
        assert s["title"] in out, s["id"]


def test_cli_quickstart_json_is_the_step_model(capsys):
    """A55 (SPEC.md 65.3.1): ``--json`` prints the step model
    ``{intro, steps}`` — the steps equal ``quickstart_steps()``, and
    the JSON output is byte-stable across calls."""
    rc1 = cli_main(["quickstart", "--json"])
    j1 = capsys.readouterr().out
    rc2 = cli_main(["quickstart", "--json"])
    j2 = capsys.readouterr().out
    assert rc1 == rc2 == 0
    assert j1 == j2  # byte-stable (G2)
    data = json.loads(j1)
    assert set(data) == {"intro", "steps"}
    assert data["intro"] == QUICKSTART_INTRO
    assert data["steps"] == quickstart_steps()


# --- exports + version (A55) ----------------------------------------------------

def test_new_names_are_exported():
    """A55 (SPEC.md 65.6): the eight new top-level names are in
    ``__all__`` and resolvable (33.1)."""
    for name in NEW_EXPORTS:
        assert name in autorefine.__all__, name
        assert getattr(autorefine, name) is not None


def test_version_round_v051():
    """A55 (33.1): the version stepped with the round — ``0.51.0`` in
    both sources (pyproject and the package)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.57.0"
