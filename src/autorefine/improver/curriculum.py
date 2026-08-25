"""Curriculum / adaptive difficulty (SPEC.md 20.1).

A curriculum watches the improver's best score and, when the current
difficulty saturates (best score reaches `trigger_fraction` of the level's
Bayes ceiling), steps the difficulty up so reward stays informative as
policies saturate.

ParityCurriculum: a two-axis ladder over (n_bits, p_flip). p_flip is swept
first (p_flip_start .. max_p_flip in steps of p_flip_step), then n_bits is
raised by one (re-sweeping p_flip), until max_n_bits. Level values come from
index arithmetic (no float accumulation), and every decision is a pure
function of (score history, seed) — deterministic (SPEC.md G2).

The curriculum only *decides*; AutoRefineEnv performs the task rebuild and
re-baselining (SPEC.md 20.1).
"""
from __future__ import annotations

from ..tasks.parity import ParityTask, parity_ceiling


class ParityCurriculum:
    """Difficulty ladder for the parity family (SPEC.md 20.1)."""

    def __init__(self,
                 seed: int,
                 p_flip_start: float = 0.08,
                 p_flip_step: float = 0.02,
                 max_p_flip: float = 0.16,
                 n_bits_start: int = 4,
                 max_n_bits: int = 5,
                 trigger_fraction: float = 0.9) -> None:
        if p_flip_step <= 0:
            raise ValueError(f"p_flip_step must be > 0, got {p_flip_step!r}")
        if not 0.0 <= p_flip_start <= max_p_flip < 0.5:
            raise ValueError(
                f"need 0 <= p_flip_start <= max_p_flip < 0.5, got "
                f"{p_flip_start!r} / {max_p_flip!r}")
        if not 1 <= n_bits_start <= max_n_bits <= 8:
            raise ValueError(
                f"need 1 <= n_bits_start <= max_n_bits <= 8, got "
                f"{n_bits_start!r} / {max_n_bits!r}")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError(
                f"trigger_fraction must be in (0, 1], got {trigger_fraction!r}")
        self.seed = int(seed)
        self.p_flip_start = float(p_flip_start)
        self.p_flip_step = float(p_flip_step)
        self.max_p_flip = float(max_p_flip)
        self.n_bits_start = int(n_bits_start)
        self.max_n_bits = int(max_n_bits)
        self.trigger_fraction = float(trigger_fraction)
        self._n_p = int(round((self.max_p_flip - self.p_flip_start)
                              / self.p_flip_step))  # max p_flip index
        self._p = 0  # current p_flip index
        self._b = 0  # current n_bits increment
        self.level = 0
        self.events: list[dict] = []

    # --- current level ---------------------------------------------------------
    @property
    def p_flip(self) -> float:
        return round(self.p_flip_start + self._p * self.p_flip_step, 10)

    @property
    def n_bits(self) -> int:
        return self.n_bits_start + self._b

    @property
    def ceiling(self) -> float:
        """Bayes ceiling (score points) of the current level."""
        return parity_ceiling(self.n_bits, self.p_flip)

    def level_description(self) -> str:
        return f"parity-{self.n_bits}-p{self.p_flip:.2f}"

    def levels_left(self) -> int:
        """Remaining step-ups (SPEC.md 20.1): the rest of the current p_flip
        sweep, plus one raise + a full p re-sweep for every n_bits level
        still to come (e.g. 9 at level 0 for the default ladder)."""
        raises = self.max_n_bits - self.n_bits_start - self._b
        return (self._n_p - self._p) + raises * (self._n_p + 1)

    def task(self) -> ParityTask:
        """A task instance at the current level (same seed; G2)."""
        return ParityTask(self.seed, n_bits=self.n_bits, p_flip=self.p_flip)

    # --- decision (pure function of the score, SPEC.md 20.1) -------------------
    def step_up_if_saturated(self, best_score: float) -> bool:
        """Step to the next difficulty level if the task saturated.

        Saturated: `best_score >= trigger_fraction * ceiling(current level)`
        and at least one level remains. Returns True iff a step-up happened."""
        if self.levels_left() == 0:
            return False
        if float(best_score) < self.trigger_fraction * self.ceiling:
            return False
        if self._p < self._n_p:
            self._p += 1
        else:
            self._b += 1
            self._p = 0
        self.level += 1
        self.events.append({
            "level": self.level,
            "n_bits": self.n_bits,
            "p_flip": self.p_flip,
            "ceiling": self.ceiling,
        })
        return True
