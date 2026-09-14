"""v0.51 (SPEC.md 65, A55): the Quickstart — one source, three surfaces.

The README already carries a "Quickstart" section (CLI commands). This
round gives the *UI* (and the terminal) the same onboarding, from one
pure core module so it is testable without streamlit:

- 65.1 the step model — ``QUICKSTART_INTRO`` + ``QUICKSTART_STEPS``:
  the four onboarding steps (pick data → run the loop → inspect &
  export → check the environment), each with a plain-English ``body``,
  an in-app ``app`` action, and a copy-pasteable ``cli`` equivalent
  (the README Quickstart invocations, one home).
- 65.2 the renderers — ``render_quickstart_md`` (the app's first
  screen) and ``render_quickstart`` (the ``autorefine quickstart``
  command — no table pipes, terminal friendly). Both are pure and
  byte-stable (G2).
- 65.4 the demo action — ``demo_recipe`` (the 41.3.1 tiny parity-v1
  budget, one home) + ``run_demo``: runs that recipe and returns the
  narrated result (the 41.2 ``trace_lines`` + the six summary keys +
  the best spec + the run dir) — the app's "Run the demo now" button.
  A demo run is a run (41.3.2): a normal run dir with the full
  artifact set + a registry entry; deterministic for a given seed (G2).

House rules: pure and deterministic (G2); no ``st.*`` calls (the app
is the only renderer, 23.1); no ``print`` (the CLI is the only
printer); the core never imports ``cli`` (the 50.2 cycle-free rule).
"""
from __future__ import annotations

from .improver.meta_env import AutoRefineEnv, search_quality_v04
from .improver.policy import SearchPolicy
from .simulate import trace_lines  # 41.2: the decision trace (one renderer)

# --- 65.1 the step model (one home) -------------------------------------------

# 65.1.1: the one-paragraph intro — the README's own description, so the
# three surfaces (README / CLI / app) cannot drift apart.
QUICKSTART_INTRO = (
    "AutoRefine is an environment where ML models iteratively improve "
    "through autonomous means: the improver loop proposes model-"
    "configuration changes, trains, evaluates on splits it never saw, "
    "records the outcome, and proposes the next change — all within a "
    "hard budget, fully reproducible, NumPy-only."
)

# 65.1.2: the four onboarding steps, in stable order. Each step is a dict
# with exactly the keys {id, title, body, app, cli} (65.1.4):
#   id    — a stable machine name (the JSON shape's identity)
#   title — the rendered heading (already numbered)
#   body  — the plain-English "what/why" (app and CLI share it)
#   app   — what to do inside the dashboard
#   cli   — the copy-pasteable terminal equivalent (README Quickstart)
QUICKSTART_STEPS: tuple[dict, ...] = (
    {
        "id": "pick_data",
        "title": "1. Pick data (or skip it)",
        "body": ("Point AutoRefine at what you want to model: a CSV "
                 "(header + rows, one label column), or a directory of "
                 "labelled images / audio clips. No data yet? Skip this "
                 "step and press the demo button below."),
        "app": ("Sidebar: upload a CSV (or type its path / a media "
                "directory); the label column is auto-detected "
                "(label/target/y/class, else last)."),
        "cli": "autorefine fit --data examples/data/churn_sample.csv",
    },
    {
        "id": "run_loop",
        "title": "2. Run the loop",
        "body": ("The improver proposes a model change, trains it, "
                 "evaluates on splits it never saw, keeps what wins — "
                 "inside a hard budget. Every experiment is shown as it "
                 "happens, with the reason it was kept or skipped."),
        "app": ("Run tab: press 'Run the improvement loop' (or press "
                "R). Stop any time with 'Stop' (S) — an honest, "
                "partial run with full artifacts."),
        "cli": "autorefine fit --data sales.csv --label churn --target 95.0",
    },
    {
        "id": "inspect_export",
        "title": "3. Inspect & export",
        "body": ("The verdict (target met? by how much), the plots, the "
                 "per-candidate decisions, and downloads (model, spec, "
                 "config). Past runs stay in the registry — compare "
                 "them side by side."),
        "app": ("Results / Compare / Experiments tabs: verdict + plots, "
                "'Download run_config.json', 'Build share bundle', and "
                "the past-runs list."),
        "cli": "autorefine report --run runs/<run_id>",
    },
    {
        "id": "check_env",
        "title": "4. Check the environment",
        "body": ("Something off? One command checks the version, numpy, "
                 "the optional extras, a micro smoke train (< 1 s), and "
                 "whether the runs dir is writable."),
        "app": ("n/a — run it in a terminal next to the app; it is the "
                "onboarding check and support triage in one."),
        "cli": "autorefine doctor",
    },
)


def quickstart_steps() -> list[dict]:
    """65.1.3: the step model as a list (a copy — mutating the result
    never touches ``QUICKSTART_STEPS``), in the stable rendered order.
    Pure."""
    return [dict(s) for s in QUICKSTART_STEPS]


def quickstart_commands() -> list[str]:
    """65.1.3: the copy-pasteable CLI commands, in step order (the
    README Quickstart invocations read from their one home). Pure."""
    return [s["cli"] for s in QUICKSTART_STEPS]


# --- 65.2 the renderers (pure, byte-stable) -----------------------------------

def render_quickstart_md() -> str:
    """65.2.1: the Markdown rendering — the dashboard's first screen.
    ``#`` headings, one block per step (body, the in-app action, the CLI
    command). Pure and byte-stable (G2): a re-render is byte-identical.
    No table pipes (the app renders it with ``st.markdown``)."""
    out = ["# AutoRefine — Quickstart", "", QUICKSTART_INTRO, ""]
    for s in QUICKSTART_STEPS:
        out += [f"**{s['title']}**", "", s["body"], "",
                f"- **in the app:** {s['app']}",
                f"- **on the CLI:** `{s['cli']}`", ""]
    return "\n".join(out)


def render_quickstart() -> str:
    """65.2.2: the aligned-columns text rendering — the
    ``autorefine quickstart`` command. No table pipes (terminal
    friendly). Pure and byte-stable (G2)."""
    out = ["=== AutoRefine quickstart ===", "", QUICKSTART_INTRO, ""]
    for s in QUICKSTART_STEPS:
        out += [s["title"], f"   {s['body']}",
                f"   app : {s['app']}",
                f"   cli : {s['cli']}", ""]
    return "\n".join(out)


# --- 65.4 the demo action -------------------------------------------------------

def demo_recipe() -> dict:
    """65.4.1: the demo loop's recipe — the 41.3.1 tiny fixed budget
    (parity-v1, 3 experiments / 60 s wall / 10 s per train), the
    search policy, the v0.4 quality preset, un-gated — one home, so
    the app's "Run the demo now" button cannot drift from
    ``run --demo`` (41.3). A copy: the caller never mutates the
    recipe's identity. Pure."""
    return {
        "task": "parity-v1",
        "seed": 7,
        "budget": {"max_experiments": 3, "max_wall_seconds": 60.0,
                   "max_train_seconds": 10.0},
        "policy": "search",
        "target": None,
        "quality": "v04",
    }


def run_demo(seed: int = 7, runs_dir: str = "runs") -> dict:
    """65.4.2: run the 65.4.1 recipe and return its narrated result —
    the app's "Run the demo now" action (65.2). Pure return (no
    ``print``, no ``st.*``): the caller renders.

    A demo run is a run (41.3.2): a normal run dir with the full
    artifact set + a registry entry; deterministic for a given seed
    (G2 — two calls with the same seed and runs dir give the same
    summary numbers, the same trace, and the same best spec).
    """
    recipe = demo_recipe()
    from .config import Budget  # local: keep the top imports light

    env = AutoRefineEnv(
        task=recipe["task"], seed=int(seed),
        budget=Budget(**recipe["budget"]),  # 41.3.1: the tiny fixed budget
        runs_dir=runs_dir,
        policy=recipe["policy"], target=recipe["target"],  # un-gated (like `run`)
        **search_quality_v04())
    policy = SearchPolicy(seed=int(seed))
    state = env.reset()
    while not env.done:
        state, _r, _d, _i = env.step(policy.propose(state))
    summary = env.memory.load_summary() if env.memory else {}
    entries = env.memory.load_experiments()
    return {
        "task": recipe["task"],
        "seed": int(seed),
        "run_dir": str(env.run_dir),
        "budget": dict(recipe["budget"]),
        "summary": {k: summary.get(k) for k in (
            "baseline_score", "final_best_score", "improvement_factor",
            "experiments_run", "wall_seconds", "finished_reason")},
        "trace": trace_lines(entries),  # 41.2: the decision trace
        "best_spec": env.best_spec.to_dict(),
    }
