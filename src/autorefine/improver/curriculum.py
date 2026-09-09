"""Curriculum / adaptive difficulty (SPEC.md 20.1; v0.32 46.2).

A curriculum watches the improver's best score and, when the current
difficulty saturates (best score reaches `trigger_fraction` of the level's
Bayes ceiling), steps the difficulty up so reward stays informative as
policies saturate.

ParityCurriculum: a two-axis ladder over (n_bits, p_flip). p_flip is swept
first (p_flip_start .. max_p_flip in steps of p_flip_step), then n_bits is
raised by one (re-sweeping p_flip), until max_n_bits.

SineCurriculum (SPEC.md 46.2.3): a three-axis ladder over (amplitude,
freq_scale, noise) — noise swept first, then freq_scale, then amplitude
(3x3x3 levels by default).

CartPoleCurriculum (SPEC.md 46.2.4): a single-axis ladder over ic_scale
(1.0 -> 2.0 -> 3.0 by default; ceiling proxy = max_steps = 500).

Level values come from index arithmetic (no float accumulation), and
every decision is a pure function of (score history, seed) — deterministic
(SPEC.md G2). All three implement the same API (46.2.2), including
`level_params()` — the dict `_advance_curriculum` splats into the
curriculum event row (parity keeps its historical n_bits/p_flip keys).

The curriculum only *decides*; AutoRefineEnv performs the task rebuild and
re-baselining (SPEC.md 20.1).
"""
from __future__ import annotations

from ..tasks.cartpole import CartPoleV1
from ..tasks.parity import ParityTask, parity_ceiling
from ..tasks.sine import SineRegressionV1, sine_ceiling


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

    def level_params(self) -> dict:
        """SPEC.md 46.2.2: the level's task parameters — splatted into the
        curriculum event row (the historical n_bits/p_flip keys, so parity
        rows stay byte-identical to the A10 shape)."""
        return {"n_bits": self.n_bits, "p_flip": self.p_flip}

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
            **self.level_params(),
            "ceiling": self.ceiling,
        })
        return True


class SineCurriculum:
    """Difficulty ladder for the sine family (SPEC.md 46.2.3).

    Three axes, noise swept first (fastest), then freq_scale, then
    amplitude — the same index arithmetic as ParityCurriculum (G2):
    level (a, f, n) = (amp_start - a*amp_step, freq_start + f*freq_step,
    noise_start + n*noise_step). Defaults: a 3x3x3 grid (27 levels,
    26 step-ups). ceiling = sine_ceiling at the current level (46.2.3).
    """

    def __init__(self,
                 seed: int,
                 noise_start: float = 0.05,
                 noise_step: float = 0.05,
                 max_noise: float = 0.15,
                 freq_start: float = 1.0,
                 freq_step: float = 0.5,
                 max_freq: float = 2.0,
                 amp_start: float = 1.0,
                 amp_step: float = 0.25,
                 min_amp: float = 0.5,
                 trigger_fraction: float = 0.9) -> None:
        if noise_step <= 0 or not 0.0 <= noise_start <= max_noise:
            raise ValueError(
                f"need noise_step > 0 and 0 <= noise_start <= max_noise, got "
                f"{noise_start!r}/{noise_step!r}/{max_noise!r}")
        if freq_step <= 0 or not freq_start > 0 or not freq_start <= max_freq:
            raise ValueError(
                f"need freq_step > 0 and 0 < freq_start <= max_freq, got "
                f"{freq_start!r}/{freq_step!r}/{max_freq!r}")
        if amp_step <= 0 or not amp_start > 0 or not min_amp > 0 \
                or not min_amp <= amp_start:
            raise ValueError(
                f"need amp_step > 0, amp_start > 0, 0 < min_amp <= amp_start, "
                f"got {amp_start!r}/{amp_step!r}/{min_amp!r}")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError(
                f"trigger_fraction must be in (0, 1], got {trigger_fraction!r}")
        self.seed = int(seed)
        self.noise_start = float(noise_start)
        self.noise_step = float(noise_step)
        self.max_noise = float(max_noise)
        self.freq_start = float(freq_start)
        self.freq_step = float(freq_step)
        self.max_freq = float(max_freq)
        self.amp_start = float(amp_start)
        self.amp_step = float(amp_step)
        self.min_amp = float(min_amp)
        self.trigger_fraction = float(trigger_fraction)
        self._n_n = int(round((self.max_noise - self.noise_start)
                              / self.noise_step))  # max noise index
        self._n_f = int(round((self.max_freq - self.freq_start)
                              / self.freq_step))   # max freq index
        self._n_a = int(round((self.amp_start - self.min_amp)
                              / self.amp_step))    # max amplitude index
        self._a = 0  # current amplitude index
        self._f = 0  # current freq index
        self._n = 0  # current noise index (fastest axis)
        self.level = 0
        self.events: list[dict] = []

    # --- current level ---------------------------------------------------------
    @property
    def noise(self) -> float:
        return round(self.noise_start + self._n * self.noise_step, 10)

    @property
    def freq_scale(self) -> float:
        return round(self.freq_start + self._f * self.freq_step, 10)

    @property
    def amplitude(self) -> float:
        return round(self.amp_start - self._a * self.amp_step, 10)

    @property
    def ceiling(self) -> float:
        """Bayes ceiling (score points) of the current level (46.2.3)."""
        return sine_ceiling(self.noise, self.freq_scale, self.amplitude)

    def level_description(self) -> str:
        return (f"sine-n{self.noise:.2f}-f{self.freq_scale:.2f}"
                f"-a{self.amplitude:.2f}")

    def levels_left(self) -> int:
        """Remaining step-ups (SPEC.md 46.2.2): the rest of the 3-axis grid."""
        total = (self._n_a + 1) * (self._n_f + 1) * (self._n_n + 1)
        used = self._a * (self._n_f + 1) * (self._n_n + 1) \
            + self._f * (self._n_n + 1) + self._n + 1
        return total - used

    def level_params(self) -> dict:
        """SPEC.md 46.2.2: the level's task parameters — splatted into the
        curriculum event row (sine rows carry noise/freq_scale/amplitude)."""
        return {"noise": self.noise, "freq_scale": self.freq_scale,
                "amplitude": self.amplitude}

    def task(self) -> SineRegressionV1:
        """A task instance at the current level (same seed; G2)."""
        return SineRegressionV1(self.seed, noise=self.noise,
                                freq_scale=self.freq_scale,
                                amplitude=self.amplitude)

    # --- decision (pure function of the score, SPEC.md 20.1/46.2.3) ----------
    def step_up_if_saturated(self, best_score: float) -> bool:
        """Step to the next difficulty level if the task saturated (the
        same contract as ParityCurriculum)."""
        if self.levels_left() == 0:
            return False
        if float(best_score) < self.trigger_fraction * self.ceiling:
            return False
        if self._n < self._n_n:
            self._n += 1
        elif self._f < self._n_f:
            self._f += 1
            self._n = 0
        else:
            self._a += 1
            self._f = 0
            self._n = 0
        self.level += 1
        self.events.append({
            "level": self.level,
            **self.level_params(),
            "ceiling": self.ceiling,
        })
        return True


class CartPoleCurriculum:
    """Difficulty ladder for the cartpole family (SPEC.md 46.2.4).

    Single axis: ic_scale (the IC-box widening, 1.0 -> 2.0 -> 3.0 by
    default; capped at 3.0 — beyond it the th box leaves the AUG_BOX
    envelope). ceiling is the proxy `CartPoleV1.max_steps` (500.0 — the
    mean-steps metric's upper bound; cartpole has no closed-form Bayes
    ceiling, so the trigger is `best_score >= trigger_fraction * 500`).
    """

    def __init__(self,
                 seed: int,
                 ic_scale_start: float = 1.0,
                 ic_scale_step: float = 1.0,
                 max_ic_scale: float = 3.0,
                 trigger_fraction: float = 0.9) -> None:
        if ic_scale_step <= 0:
            raise ValueError(
                f"ic_scale_step must be > 0, got {ic_scale_step!r}")
        if not ic_scale_start >= 1.0 or not ic_scale_start <= max_ic_scale:
            raise ValueError(
                f"need 1 <= ic_scale_start <= max_ic_scale, got "
                f"{ic_scale_start!r} / {max_ic_scale!r}")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError(
                f"trigger_fraction must be in (0, 1], got {trigger_fraction!r}")
        self.seed = int(seed)
        self.ic_scale_start = float(ic_scale_start)
        self.ic_scale_step = float(ic_scale_step)
        self.max_ic_scale = float(max_ic_scale)
        self.trigger_fraction = float(trigger_fraction)
        self._n_ics = int(round((self.max_ic_scale - self.ic_scale_start)
                                / self.ic_scale_step))  # max level index
        self._i = 0  # current level index
        self.level = 0
        self.events: list[dict] = []

    # --- current level ---------------------------------------------------------
    @property
    def ic_scale(self) -> float:
        return round(self.ic_scale_start + self._i * self.ic_scale_step, 10)

    @property
    def ceiling(self) -> float:
        """Ceiling proxy (SPEC.md 46.2.4): max_steps — the mean-steps
        metric's upper bound (500.0)."""
        return float(CartPoleV1.max_steps)

    def level_description(self) -> str:
        return f"cartpole-ics{self.ic_scale:g}"

    def levels_left(self) -> int:
        """Remaining step-ups (SPEC.md 46.2.2)."""
        return self._n_ics - self._i

    def level_params(self) -> dict:
        """SPEC.md 46.2.2: the level's task parameters — splatted into the
        curriculum event row (cartpole rows carry ic_scale)."""
        return {"ic_scale": self.ic_scale}

    def task(self) -> CartPoleV1:
        """A task instance at the current level (same seed; G2)."""
        return CartPoleV1(self.seed, ic_scale=self.ic_scale)

    # --- decision (pure function of the score, SPEC.md 20.1/46.2.4) ----------
    def step_up_if_saturated(self, best_score: float) -> bool:
        """Step to the next difficulty level if the task saturated (the
        same contract as ParityCurriculum)."""
        if self.levels_left() == 0:
            return False
        if float(best_score) < self.trigger_fraction * self.ceiling:
            return False
        self._i += 1
        self.level += 1
        self.events.append({
            "level": self.level,
            **self.level_params(),
            "ceiling": self.ceiling,
        })
        return True
