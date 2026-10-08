"""SPEC.md 87.1 (v0.73, A1) — plain-language goal entry.

"Get in with zero jargon": the user says what they want in their own
words — ``"reach 96% on this churn table, quick"`` — and the loop turns
that into the same recipe ``fit`` already understands. This leaf owns the
one pure resolver, ``resolve_goal(text, data_path=None)``; the CLI
(``autorefine ask``) and the app's "What do you want?" box are thin
surfaces over it (SPEC.md 87.1.4).

House rules (87.1.3): stdlib only; pure and deterministic (G2) — the same
goal text always resolves to the same recipe; fail-loud ``ValueError`` on
an empty goal or an out-of-range target (never a silent guess).
"""
from __future__ import annotations

import re

# 87.1.1: the defaults — an absent detail keeps the CLI's own default
# (``fit``'s target 95 / task auto / 30 experiments), so a bare goal is a
# valid plan, not an error.
DEFAULT_TARGET = 95.0
DEFAULT_TASK = "auto"
DEFAULT_EXPERIMENTS = 30

# 87.1.2: the "how long should it spend" words — quick/fast = a short
# budget, thorough/exhaustive = a long one (deterministic mapping).
_BUDGET_HINTS: tuple[tuple[str, int], ...] = (
    ("quick", 8),
    ("fast", 8),
    ("thorough", 60),
    ("exhaustive", 60),
)

# 87.1.2: the data-kind words, in the order they are checked (a goal that
# names one modality wins; first match is the decision, 87.1.3).
_TASK_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("csv", ("churn", "classify", "classification", "tabular",
             "spreadsheet", "customer", "rows")),
    ("image", ("image", "images", "picture", "pictures", "photo",
               "photos")),
    ("audio", ("audio", "sound", "sounds", "speech", "clip", "clips")),
    ("text", ("text", "words", "documents", "sentiment", "review",
              "reviews")),
)

# 87.1.2: the target read — "95%", "96 %", "at least 90 percent". The
# word boundary sits on ``percent`` only: ``%`` is a non-word char, so a
# trailing ``\b`` after ``%`` can never match (and "percentile" must not).
_TARGET_RE = re.compile(
    r"\b(?:at\s+least\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent\b)")

# 87.1.2: the known trigger words — a goal with *none* of them resolves to
# pure defaults and its text is reported back as unrecognized (87.1.4).
_TRIGGER_WORDS: tuple[str, ...] = (
    "at least",
    "%", "percent",
    *dict(_BUDGET_HINTS),
    *tuple(k for _t, kws in _TASK_HINTS for k in kws),
)


def _word_in(text: str, word: str) -> bool:
    """A case-insensitive word-boundary membership test (deterministic)."""
    return re.search(r"\b" + re.escape(word) + r"\b", text,
                     re.IGNORECASE) is not None


def resolve_goal(text: str, data_path: str | None = None) -> dict:
    """SPEC.md 87.1 (A1): a goal in the user's words -> a recipe dict.

    ``{data, task, target, experiments, plain_summary, unrecognized}``:

    - ``data`` — the given path, or ``None`` (the CLI asks for it);
    - ``task`` — the first modality the goal names, else ``"auto"``;
    - ``target`` — the first ``N%`` / ``"at least N percent"`` in the text;
      out of range (``<= 0`` or ``> 100``) is a fail-loud ``ValueError``
      (87.1.3); absent -> 95 (the ``fit`` default);
    - ``experiments`` — ``quick``/``fast`` -> 8, ``thorough``/``exhaustive``
      -> 60, absent -> 30 (the ``fit`` default);
    - ``plain_summary`` — the one ASCII sentence that reads the goal back
      to the user (no jargon — the 87.4 vocabulary);
    - ``unrecognized`` — ``[text]`` when no known trigger word appears
      (defaults only), else ``[]``.

    Pure (G2); an empty / whitespace goal is a fail-loud ``ValueError``
    (87.1.3) — a goal is what this command is for.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "tell me a goal in a sentence — e.g. "
            "\"reach 96% on this churn table, quick\"")
    lowered = text.lower()

    m = _TARGET_RE.search(text)
    if m is None:
        target = DEFAULT_TARGET
    else:
        target = float(m.group(1))
        if not (0.0 < target <= 100.0):
            raise ValueError(
                f"target {target:g}% is outside 0–100 — say a bar like "
                "\"95%\" or \"at least 90 percent\"")

    task = DEFAULT_TASK
    for name, words in _TASK_HINTS:
        if any(_word_in(lowered, w) for w in words):
            task = name
            break

    experiments = DEFAULT_EXPERIMENTS
    for word, budget in _BUDGET_HINTS:
        if _word_in(lowered, word):
            experiments = budget
            break

    matched = (m is not None
               or any(_word_in(lowered, w) for w in _TRIGGER_WORDS))
    unrecognized: list[str] = [] if matched else [text.strip()]

    # 88.2.2 (v0.74, A2): the sentence may carry the path itself — the
    # explicit ``data_path`` argument always wins; a path token in the
    # text (88.2.1) is the fallback, so ``ask "... on C:\data\churn.csv"``
    # embeds the real path instead of the <your data> placeholder.
    data = data_path if data_path else extract_data_path(text)
    data_part = (f"on {data}" if data else
                 "on your data (add a data path to run it)")
    task_part = task if task != DEFAULT_TASK else "your data's kind (auto)"
    plain_summary = (
        f"You want a model {data_part} that scores at least {target:g}% "
        f"(a {task_part} problem), spending {experiments} experiments."
    )
    return {
        "data": data,
        "task": task,
        "target": target,
        "experiments": experiments,
        "plain_summary": plain_summary,
        "unrecognized": unrecognized,
    }


def ask_command(resolved: dict) -> list[str]:
    """SPEC.md 87.1.4 (A1): the resolved goal as a copy-pasteable
    ``autorefine fit`` command (the 37.1.4 recipe pattern, goal-shaped).

    Defaults equal to the CLI's own are omitted (the 37.1.5 rule): the
    ``target`` is always the goal's point — it is printed; ``task`` is
    printed only when the goal named one (``auto`` is the CLI default);
    ``experiments`` only when it is not the default 30. With no data path
    the ``--data`` slot is the placeholder ``<your data>`` (the command is
    a plan, not a run — the ``ask`` surface says so, 87.1.4).
    """
    cmd = ["autorefine", "fit"]
    cmd += ["--data", resolved.get("data") or "<your data>"]
    task = resolved.get("task") or DEFAULT_TASK
    if task != DEFAULT_TASK:
        cmd += ["--task", task]
    cmd += ["--target", f"{float(resolved.get('target', DEFAULT_TARGET)):g}"]
    n = int(resolved.get("experiments", DEFAULT_EXPERIMENTS))
    if n != DEFAULT_EXPERIMENTS:
        cmd += ["--experiments", str(n)]
    return cmd


def go_budget(dataset_size: int | None) -> int:
    """SPEC.md 87.3.2 (A3): the ``go`` command's default budget from the
    data's size — small tables need few experiments, big tables a few
    more (a deterministic step function of the train-split size):

    - ``None`` (size unknown) -> 30 (the ``fit`` default);
    - ``< 100`` rows -> 15; ``< 1000`` -> 30; otherwise -> 40.
    """
    if dataset_size is None:
        return 30
    if dataset_size < 100:
        return 15
    if dataset_size < 1000:
        return 30
    return 40


# 88.2.1 (v0.74, A2): the data-extension vocabulary — a name ending in one
# of these is a data path (a drive path, below, is one too).
_DATA_EXTENSIONS = (".csv", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".txt")

# 88.2.1: a Windows drive path — ``C:\...`` or ``C:/...`` (either slash).
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

# 88.2.1: the sentence punctuation a path token may be wrapped in.
_PATH_TRIM = '(),.;:!"\''


def extract_data_path(text: str) -> str | None:
    r"""SPEC.md 88.2.1 (v0.74, A2): the first whitespace-delimited token of
    the goal sentence that *is* a data path — a Windows drive path
    (``C:\...`` / ``C:/...``) or a name ending in a known data extension
    (``.csv .png .jpg .jpeg .mp3 .wav .txt``) — with its surrounding
    sentence punctuation (``(),.;:!"'``) stripped; else ``None``.

    No other heuristics (88.2.1): a bare word — ``churn table``, ``this
    csv`` — is never a path. Pure and deterministic (G2)."""
    if not isinstance(text, str):
        return None
    for token in text.split():
        cand = token.strip(_PATH_TRIM)
        if not cand:
            continue
        if _DRIVE_PATH_RE.match(cand) or cand.lower().endswith(_DATA_EXTENSIONS):
            return cand
    return None


def auto_target(majority_frac: float) -> float:
    """SPEC.md 88.4.1 (v0.74, B4): the fair bar from the dataset's own
    class structure — ``min(99.5, max(95.0, majority*100 + 5))``: five
    points above the majority-class ceiling, floored at the 95 default,
    capped at 99.5 (0.5 -> 95, 0.9 -> 95, 0.95 -> 99.5). A fraction
    outside ``[0.5, 1.0]`` is a fail-loud ``ValueError`` (88.4.1) — below
    0.5 is not a majority, above 1.0 is impossible."""
    if not isinstance(majority_frac, (int, float)) or isinstance(majority_frac, bool):
        raise ValueError(
            f"auto_target: majority fraction must be a number, got {majority_frac!r}")
    if not (0.5 <= float(majority_frac) <= 1.0):
        raise ValueError(
            f"majority fraction {float(majority_frac):.3g} is outside "
            "[0.5, 1.0] - the larger class must be at least half of the rows")
    return min(99.5, max(95.0, float(majority_frac) * 100.0 + 5.0))


def fair_bar_line(majority_frac: float, target: float) -> str:
    """SPEC.md 88.4.1 (v0.74, B4): the one plain line that explains the
    auto-target (the 87.4 vocabulary, ASCII-safe): the majority class's
    share of the rows, and why the bar sits where it sits."""
    return (f"the majority class is {float(majority_frac) * 100.0:.1f}% of the "
            f"rows - a fair bar sits 5 points above it: {float(target):g}% "
            f"(held between the 95 default and the 99.5 cap)")


def measured_per_exp(entries, task: str) -> float | None:
    """SPEC.md 88.5.1 (v0.74, B5): the measured per-experiment wall time
    for one task — the mean ``wall_seconds / experiments_run`` over that
    task's registry entries with both values positive; ``None`` when no
    such entry exists (the caller falls back, 88.5.2). Deterministic
    (order-insensitive mean, G2)."""
    rates: list[float] = []
    for e in entries:
        if not isinstance(e, dict) or e.get("task") != task:
            continue
        wall = e.get("wall_seconds")
        exps = e.get("experiments_run")
        if (isinstance(wall, (int, float)) and not isinstance(wall, bool)
                and isinstance(exps, (int, float)) and not isinstance(exps, bool)
                and wall > 0 and exps > 0):
            rates.append(float(wall) / float(exps))
    if not rates:
        return None
    return sum(rates) / len(rates)


def experiments_for_time(seconds, per_exp) -> int:
    """SPEC.md 88.5.1 (v0.74, B5): the budget behind a wall-time ask —
    ``max(1, min(200, round(seconds / per_exp)))``; non-positive or
    missing inputs fall back to 1 (a run, not a crash)."""
    try:
        s = float(seconds)
        p = float(per_exp)
    except (TypeError, ValueError):
        return 1
    if not (s > 0.0 and p > 0.0):
        return 1
    return max(1, min(200, int(round(s / p))))
