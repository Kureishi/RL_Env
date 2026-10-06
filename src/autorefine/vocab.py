"""SPEC.md 87.4 (v0.73, B1) — the plain-language translation layer.

Jargon in, the reader's words out. One ordered table (``PLAIN``) maps the
loop's technical vocabulary to friendly words; ``plain(text)`` applies it
deterministically (longest key first, word boundaries, case-preserving)
so every surface — the app's "Plain language" toggle, the CLI's tour /
``go`` verdicts, the model card — says the same thing (87.4.3: one home).

House rules: stdlib only; pure (G2); the table and its replacements are
ASCII-safe (the 48.5 console rule) and carry no SPEC / version tokens
(the app-language scan applies to the rendered strings, 87.11).
"""
from __future__ import annotations

import re

# 87.4.1: the table — jargon -> plain words. Insertion order is the
# documented order; ``plain`` applies keys longest-first regardless, so a
# longer phrase always wins over its shorter sibling ("hyperparameters"
# before "hyperparameter", "mutations" before "mutation").
PLAIN: dict[str, str] = {
    "hyperparameters": "knobs",
    "hyperparameter": "knob",
    "convergence": "settling down",
    "converged": "settled down",
    "converge": "settle down",
    "overfitting": "memorizing the training data",
    "overfit": "memorizing the training data",
    "gen_gap": "train-vs-test gap",
    "holdout": "held-out",
    "candidates": "tries",
    "candidate": "try",
    "mutations": "small changes",
    "mutation": "small change",
    "baseline": "starter model",
    "bandit": "knob picker",
    "Pareto": "best trade-off",
    "gradient": "learning step",
    "epochs": "passes over the data",
    "epoch": "pass over the data",
    "features": "input columns",
    "feature": "input column",
    "frontier": "best trade-offs found",
    "UCB": "exploration score",
    "spec": "model recipe",
    "gate": "acceptance bar",
}

# 87.4.2: the idempotence guard — a replacement that contained a key would
# make ``plain(plain(x)) != plain(x)``. The house table is checked by the
# A77.4 invariant, not assumed.


def _replace(key: str, repl: str) -> callable:
    def sub(m: re.Match) -> str:
        out = repl
        # case-preserving: a capitalized key keeps a capitalized plain word
        if m.group(0)[0].isupper():
            out = out[0].upper() + out[1:]
        return out
    return sub


def plain(text: str) -> str:
    """SPEC.md 87.4 (B1): translate one string to the reader's words.

    Deterministic (G2): keys are applied longest-first with word
    boundaries, case-insensitively, and a capitalized match keeps a
    capitalized replacement. Idempotent — ``plain(plain(t)) == plain(t)``
    (the table's replacements contain no keys, 87.4.2). Non-strings and
    empty strings pass through unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for key in sorted(PLAIN, key=len, reverse=True):
        out = re.sub(r"\b" + re.escape(key) + r"\b",
                     _replace(key, PLAIN[key]), out, flags=re.IGNORECASE)
    return out


def plain_free(text: str) -> bool:
    """SPEC.md 87.4.4 (B1): ``True`` when ``text`` carries none of the
    table's jargon keys — the tour / ``go`` verdicts are asserted plain
    (A77.2/A77.3) and the app-language scan's spirit applies to the
    reader's words too."""
    if not isinstance(text, str):
        return False
    return not any(
        re.search(r"\b" + re.escape(key) + r"\b", text, re.IGNORECASE)
        for key in PLAIN)
