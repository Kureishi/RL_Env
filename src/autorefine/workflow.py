"""The procedural workflow's shared core (SPEC.md 69, v0.55, A59).

The app's core loop is a workflow — choose data → preview → run → read
the result — but the surfaces each hid it: the idle screen did not say
where the user was in the flow, and the verdict did not say what comes
next. This module is the one home for that state (69.4): a pure
state machine over (has_data, running, has_result), a hand-rolled SVG
strip that renders it, and the "next steps" guidance after the verdict.
Streamlit-free by construction (SPEC.md 23.1) — the app is a thin
renderer, the tests run without the optional dependency (G2: pure and
deterministic, stdlib only).
"""
from __future__ import annotations

# The four procedural steps, in order. User-facing labels — they render
# verbatim in the strip, the aria-label, and the tests' expectations.
# Deliberately not in ALL_KNOBS (SPEC.md 48.1): this is a view of the
# workflow, not a run setting (the 51.4.4 "view preference" invariant).
WORKFLOW_STEPS: tuple[str, ...] = ("Data", "Preview", "Run", "Results")

# 69.4.1: the per-step statuses. `done` — completed; `current` — the step
# the workflow is in right now (at most one); `todo` — not reached yet.
STEP_DONE = "done"
STEP_CURRENT = "current"
STEP_TODO = "todo"


def workflow_state(has_data: bool, running: bool,
                   has_result: bool) -> list[dict]:
    """SPEC.md 69.4.1 (v0.55, A59): the workflow's position — one
    ``{"step", "status"}`` row per entry of `WORKFLOW_STEPS`, in order.

    Pure over the three app facts (G2): a data path is set (the
    Setup tab's preview is then reachable), a run is in flight (the
    worker is alive and undrained), and a finished result is in session
    state. The rules:

      * Data — `done` once data is set, else the `current` step (nothing
        else can be reached before it).
      * Preview — `done` once data is set (the Setup tab renders it),
        else `todo`.
      * Run — `current` while a run is in flight; `done` once a result
        exists; else `todo`.
      * Results — `done` once a result exists and no run is in flight
        (the finished state is all four `done`), else `todo`.

    Deterministic: equal inputs give byte-equal rows; the default
    (no data, no run, no result) is exactly one `current` (Data), and
    the finished state (data, not running, result) is all four `done`
    (zero `current` — the `at most one` invariant still holds).
    """
    if not isinstance(has_data, bool) or not isinstance(running, bool) \
            or not isinstance(has_result, bool):
        raise ValueError("workflow_state: all three inputs must be bool")
    rows = []
    for i, name in enumerate(WORKFLOW_STEPS):
        if i == 0:  # Data
            status = STEP_DONE if has_data else STEP_CURRENT
        elif i == 1:  # Preview
            status = STEP_DONE if has_data else STEP_TODO
        elif i == 2:  # Run
            status = (STEP_CURRENT if running
                      else STEP_DONE if has_result else STEP_TODO)
        else:  # Results
            status = (STEP_DONE if (has_result and not running)
                      else STEP_TODO)
        rows.append({"step": name, "status": status})
    return rows


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def svg_workflow_strip(state: list[dict],
                       palette: str = "default", dark: bool = False) -> str:
    """SPEC.md 69.4.2 (v0.55, A59): the workflow strip — a compact
    horizontal stepper SVG over `workflow_state`'s rows.

    One node per step (circle + label), joined by connectors; the
    colors follow the design-token registry (SPEC.md 56.3): `done` =
    the accepted-green, `current` = the baseline-amber (the step the
    workflow is in), `todo` = the rejected-grey. ARIA per the 51.4.3
    convention (`role="img"` + `aria-label` naming the current step,
    plus a `<title>`); the `okabe` / `dark` overrides apply exactly like
    `resolve_tokens` (51.4.1/51.4.2). Pure and deterministic (G2): two
    calls over the same state are byte-equal, and the default call
    carries none of the Okabe-Ito hexes (the A41 palette pins apply).
    """
    from .plotting import resolve_tokens  # local: keeps import order stable
    if not isinstance(state, list) or not state:
        raise ValueError("svg_workflow_strip: `state` must be a non-empty "
                         "list of workflow_state rows")
    for row in state:
        if (not isinstance(row, dict)
                or set(row) != {"step", "status"}
                or not isinstance(row["step"], str)
                or row["status"] not in (STEP_DONE, STEP_CURRENT,
                                         STEP_TODO)):
            raise ValueError(f"svg_workflow_strip: bad state row {row!r}")
    tok = resolve_tokens(palette=palette, dark=dark)
    color = {
        STEP_DONE: tok["accepted"],
        STEP_CURRENT: tok["baseline"],
        STEP_TODO: tok["rejected"],
    }
    current = next((r["step"] for r in state
                    if r["status"] == STEP_CURRENT), None)

    n = len(state)
    node_r = 9.0
    gap = 92.0
    left = 18.0
    cy = 22.0
    width = left * 2 + (n - 1) * gap
    height = 52.0
    label = (f"Workflow — current step: {current}"
             if current else "Workflow — idle")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" role="img" '
        f'aria-label="{_esc(label)}">',
        f"<title>{_esc(label)}</title>",
        f'<rect width="100%" height="100%" fill="{tok["bg"]}"/>',
    ]
    # connectors first (under the nodes): each spans to the next node
    for i in range(n - 1):
        x1 = left + i * gap + node_r
        x2 = left + (i + 1) * gap - node_r
        # a connector into a completed/current step is colored; the last
        # segment into a todo step stays the axis grey
        lead = state[i + 1]["status"]
        seg = (color[lead] if lead in (STEP_DONE, STEP_CURRENT)
               else tok["axis"])
        out.append(
            f'<line x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}" '
            f'stroke="{seg}" stroke-width="2"/>')
    for i, row in enumerate(state):
        cx = left + i * gap
        c = color[row["status"]]
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{node_r:.1f}" '
            f'fill="{c}"><title>{_esc(row["step"])} — {row["status"]}</title>'
            f"</circle>")
        out.append(
            f'<text x="{cx:.1f}" y="{cy + 26:.1f}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" '
            f'font-size="11" fill="{tok["axis"]}">{_esc(row["step"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


# --- 69.4.3 the next-steps guidance (the verdict's "so what?") -------------

# One line per (verdict, situation) — plain user-facing English (the
# app-language contract, SPEC.md 35.2: no SPEC references, no version
# tags). Each line names a tab/panel the app already has, so the
# guidance is actionable, not aspirational.
_NEXT_STEPS_ALWAYS: tuple[str, ...] = (
    "Export & artifacts — download the model, the best spec, the "
    "copy-paste recipe, and the share bundle.",
    "Compare — the seed sweep answers 'is this improvement real?' and "
    "the past-runs list lets you diff this run against earlier ones.",
)
_NEXT_STEPS_PASS: tuple[str, ...] = (
    "Target met — confirm the margin holds across seeds (Compare tab) "
    "before shipping.",
    "Advanced analysis — re-score the logged history under a tighter "
    "bar, or A/B the other policy with one full budget.",
)
_NEXT_STEPS_MISS: tuple[str, ...] = (
    "Raise the budget (sidebar → advanced: experiments / max train "
    "seconds) or steer the search (Setup tab: pin / bias / constrain) "
    "and run again.",
    "Advanced analysis — re-gate the logged history against a different "
    "objective set; a passing candidate may already be in the run.",
)
_NEXT_STEPS_STOPPED: tuple[str, ...] = (
    "The run was stopped between experiments — the verdict and "
    "artifacts reflect the partial run; re-run with a larger budget "
    "for the full picture.",
)


def next_steps(verdict: str, finished_reason: str | None = None) -> list[str]:
    """SPEC.md 69.4.3 (v0.55, A59): the "Next steps" lines for a
    finished run — the verdict's "so what?", rendered as a bullet block
    right under the metrics.

    Pure over the result's two decision facts (G2): `verdict` ("PASS" /
    "MISS" — anything else is a `ValueError`, the 49.4.3 honesty rule:
    never guess a verdict) and the optional `finished_reason` (a
    user-stopped run carries "stopped"). The always-on lines (export,
    compare) come first, then the verdict-specific ones; a stopped run
    appends its partial-run line last. Deterministic: equal inputs give
    equal lists.
    """
    if verdict not in ("PASS", "MISS"):
        raise ValueError(f"next_steps: unknown verdict {verdict!r}")
    lines: list[str] = list(_NEXT_STEPS_ALWAYS)
    lines += list(_NEXT_STEPS_PASS if verdict == "PASS" else _NEXT_STEPS_MISS)
    if finished_reason == "stopped":
        lines += list(_NEXT_STEPS_STOPPED)
    return lines
