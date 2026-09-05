"""v0.19 coherency — SPEC.md 33, A23:

- 33.1 (C1) version story — `pyproject [project].version` and
  `autorefine.__version__` agree (the single version source); the
  dashboard caption renders the same number.
- 33.2 (C2) knob registry — `autorefine.KNOBS` is the one table for every
  tunable `AutoRefineEnv` knob: every row's default equals the env kwarg
  default; both presets set only registry knobs with validator-passing
  values; the registry validator and the env agree on the A21/A22 value
  batteries; `cli.build_parser()` exposes each claimed `--flag` per
  subcommand and *not* elsewhere (32.3: no `--screen-frac` on
  `variance`), with parser defaults equal to registry defaults; registry
  `app_widget` claims appear in the dashboard source.
- 33.3 (C3) interaction matrix — screening × curriculum: a step-up
  re-pins the screen champion (the baseline spec re-screened on the new
  dataset, free), so after the step-up a candidate that beats the *old*
  champion but loses to the re-pinned one is screen-rejected while one
  that beats both is promoted (full-trained) even when gate-rejected; the
  curriculum event row carries `screen_champion`. screening × ensemble:
  a screen-rejected fingerprint never enters `pareto.points` or the
  ensemble top-k.

House rules: no cross-test imports (fixtures duplicated); tiny budgets
(scripted-score fake task, 4-experiment budget); determinism (G2) —
the matrix pin runs twice and compares.
"""
import argparse
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autorefine import AutoRefineEnv, BanditPolicy, Budget, KNOBS
from autorefine.improver.meta_env import candidate_screening, search_quality_v04
from autorefine.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# fakes (SPEC.md 33.3): a scripted-score task (screen vs full distinguished
# by dataset size — 16 rows, frac 0.25 → exactly 4 screen rows) and a
# two-level fake curriculum (easy ceiling 100 → hard ceiling 60, trigger
# best >= 50). `train` is monkeypatched to hand back models whose
# `.score_val` the task's `score()` returns; the model satisfies the
# `save_best` contract (`model.save(path)`).
# ---------------------------------------------------------------------------

FAKE_TASK = "fake-coh-v1"


class _FakeModel:
    """A stand-in model: the task scores it from `.score_val`."""

    def __init__(self, score_val: float) -> None:
        self.score_val = score_val

    def save(self, path: str) -> None:
        np.savez(path, a=np.zeros(1))  # the model.save contract


class _FakeTask:
    """Minimal AutoRefineEnv task (name/head/n_outputs/make_dataset/score)."""

    name = FAKE_TASK
    head = "softmax"
    n_outputs = 2
    default_dataset_size = 16

    def __init__(self, seed: int = 7, **_kw) -> None:
        pass

    def make_dataset(self, size: int):
        return (np.zeros((size, 2)), np.zeros(size, dtype=np.int64))

    def score(self, model, _split: str, _n: int) -> float:
        # _EnsembleModel (the 19.3 final-evaluation stand-in) has no
        # score_val — the fake scores it as 0.0; only direct _FakeModel
        # scores are asserted in the pins
        return float(getattr(model, "score_val", 0.0))


class _FakeCurriculum:
    """Two-level fake ladder (SPEC.md 33.3 pin): level 0 easy (ceiling
    100), level 1 hard (ceiling 60); one step-up, triggered at
    `best >= 50`. Implements the curriculum contract the env reads
    (task() + level metadata + step_up_if_saturated)."""

    def __init__(self) -> None:
        self.level = 0
        self.n_bits = 4
        self.p_flip = 0.08
        self.ceiling = 100.0

    def level_description(self) -> str:
        return f"fake-level-{self.level}"

    def levels_left(self) -> int:
        return 0 if self.level >= 1 else 1

    def task(self) -> _FakeTask:
        return _FakeTask(seed=7)

    def step_up_if_saturated(self, best_score: float) -> bool:
        if self.level >= 1 or float(best_score) < 50.0:
            return False
        self.level = 1
        self.ceiling = 60.0
        return True


def _install_screen_train(monkeypatch, screen_scores: list[float],
                          full_scores: list[float]) -> None:
    """`autorefine.improver.meta_env.train` → models scored from
    `screen_scores` (calls on the ≤4-row prefix subsample) or
    `full_scores` (calls on the full 16-row dataset); the last entry of
    each list repeats once exhausted (a constant tail)."""
    calls = {"s": 0, "f": 0}

    def fake_train(dataset, spec, seed, time_limit_seconds=0.0,
                   n_out=1, head="mse"):
        n = int(dataset[0].shape[0])
        if n <= 4:  # the prefix subsample (floor(16 * 0.25) = 4 rows)
            sv = screen_scores[min(calls["s"], len(screen_scores) - 1)]
            calls["s"] += 1
        else:
            sv = full_scores[min(calls["f"], len(full_scores) - 1)]
            calls["f"] += 1
        return SimpleNamespace(
            model=_FakeModel(sv), train_seconds=0.001,
            loss_history=[], time_capped=False)

    monkeypatch.setattr("autorefine.improver.meta_env.train", fake_train)


def _drive_to_done(env: AutoRefineEnv) -> list:
    """Drive the env with the bandit; returns the proposal list (so the
    matrix tests can fingerprint a rejected spec)."""
    policy = BanditPolicy(seed=7)
    state = env.reset()
    proposals = []
    while not env.done:
        proposals.append(policy.propose(state))
        state, _r, _d, _i = env.step(proposals[-1])
    return proposals


# --- 33.1 version story (C1, A23) --------------------------------------------

def test_version_single_source():
    """A23 (SPEC.md 33.1): `pyproject [project].version` and
    `autorefine.__version__` agree — one version, one meaning; the
    dashboard caption renders that same number."""
    import autorefine
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__
    # round label == package version (v0.21 ⇒ 0.21.0, M24 — the round
    # advanced per SPEC.md 33.1 with the v0.21 coherency-III batch, SPEC.md 35)
    assert autorefine.__version__ == "0.21.0"
    # the app caption is `v{__version__}` — the same number, no third copy
    app_src = (REPO / "src" / "autorefine" / "dashboard_app.py"
               ).read_text(encoding="utf-8")
    assert "v{__version__}" in app_src


# --- 33.2 knob registry (C2, A23) --------------------------------------------

def test_knob_registry_defaults_match_env_signature():
    """A23 (SPEC.md 33.2): every registry row's default equals the
    `AutoRefineEnv.__init__` kwarg default — the table and the env cannot
    drift."""
    sig = inspect.signature(AutoRefineEnv.__init__)
    assert set(KNOBS) == {
        "ci_blocks", "z_accept", "efficiency_weight", "gen_gap_penalty",
        "block_size", "ensemble_top_k", "stall_patience", "screen_frac"}
    for name, knob in KNOBS.items():
        assert name in sig.parameters, name
        assert sig.parameters[name].default == knob.default, name
        assert knob.spec_ref  # every row carries its spec ref (33.2)


def test_presets_set_only_registry_knobs_with_valid_values():
    """A23 (SPEC.md 33.2): presets may only set registry knobs with values
    that pass the registry validator; a typo'd knob name is a
    construction-time error."""
    for preset in (search_quality_v04(), candidate_screening(),
                   candidate_screening(0.5)):
        assert preset, "a preset must not be empty"
        for key, value in preset.items():
            assert key in KNOBS, key
            assert KNOBS[key].validate(value) == value, key
    # the 18.6/32.2 pins stay exact (values unchanged by the registry)
    assert search_quality_v04() == {
        "ci_blocks": 8, "z_accept": 1.0,
        "efficiency_weight": 0.5, "gen_gap_penalty": 0.5}
    assert candidate_screening() == {"screen_frac": 0.25}
    # an unknown knob is rejected at construction, not at runtime
    import autorefine.improver.meta_env as me
    with pytest.raises(KeyError, match="unknown knob"):
        me._preset("bogus", {"not_a_knob": 1})


def test_knob_validators_match_env_behavior(tmp_path):
    """A23 (SPEC.md 33.2): the registry validator and the env agree
    accept/reject on the A21/A22 batteries — one home for the rules,
    the same messages."""
    TASKS[FAKE_TASK] = _FakeTask
    try:
        for bad in (0, 1.5, "2", -1):  # the A21 battery (SPEC.md 31.1)
            with pytest.raises(ValueError, match="stall_patience"):
                AutoRefineEnv(task=FAKE_TASK, seed=7,
                              budget=Budget(3, 300, 30), runs_dir=tmp_path,
                              stall_patience=bad)
        for bad in (0.0, -0.5, 1.5, "abc", True, False, None,
                    float("nan"), float("inf")):  # A22 battery (32.2)
            with pytest.raises(ValueError, match="screen_frac"):
                AutoRefineEnv(task=FAKE_TASK, seed=7,
                              budget=Budget(3, 300, 30), runs_dir=tmp_path,
                              screen_frac=bad)
        # good values pass with the registry's normalization ("0.5" → 0.5)
        env = AutoRefineEnv(task=FAKE_TASK, seed=7,
                            budget=Budget(3, 300, 30), runs_dir=tmp_path,
                            stall_patience=2, screen_frac="0.5")
        assert env.stall_patience == 2 and env.screen_frac == 0.5
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_cli_flags_match_registry():
    """A23 (SPEC.md 33.2): the real parser (`cli.build_parser()`) exposes
    each claimed `--flag` per subcommand and *not* elsewhere (32.3:
    no `--screen-frac` on `variance`), with parser defaults equal to the
    registry defaults."""
    from autorefine.cli import build_parser
    parser = build_parser()
    sub = next(a for a in parser._actions
               if isinstance(a, argparse._SubParsersAction))
    subs = sub.choices
    for name, knob in KNOBS.items():
        flag = f"--{name.replace('_', '-')}"
        for sub_name, sp in subs.items():
            action = next((a for a in sp._actions
                           if flag in a.option_strings), None)
            assert (action is not None) == (sub_name in knob.cli), \
                (name, sub_name)
            # the exposed flags carry the registry default (read from the
            # action itself — `parse_args([])` would SystemExit on
            # subcommands with required args, e.g. `fit --data`)
            if action is not None:
                assert action.default == knob.default, (name, sub_name)


def test_registry_app_widget_claims():
    """A23 (SPEC.md 33.2): every registry `app_widget` claim appears in
    the dashboard source (none today — 32.3 keeps screening/stall
    out of the app; the check stays honest if one is added)."""
    app_src = (REPO / "src" / "autorefine" / "dashboard_app.py"
               ).read_text(encoding="utf-8")
    for name, knob in KNOBS.items():
        if knob.app_widget is not None:
            assert f'key="{knob.app_widget}"' in app_src, name


# --- 33.3 interaction matrix (C3, A23) ----------------------------------------

def test_screening_curriculum_repins_champion(tmp_path, monkeypatch):
    """A23 (SPEC.md 33.3): screening × curriculum — a step-up re-pins the
    champion (the baseline spec re-screened on the new dataset, free).
    Scripted scores, `screen_frac=0.25`, budget 4:
      reset: full baseline 50, screen champion 50
      exp1: screen 55 > 50 → promote; full 60 → accepted (best 60)
      step-up (best 60 ≥ 50): full re-baseline 45 (best 45),
             re-pin: screen 40 (champion 40)
      exp2: screen 45 > 40 → promote (it would be rejected against the
            stale champion 50); full 44 < 45 → gate-rejected
      exp3: screen 42 > 40 → promote; full 50 → accepted (best 50)
      exp4: screen 38 ≤ 40 → screen-rejected (budget spent)
    → kinds `baseline, experiment, curriculum, experiment, experiment,
    screen`; champion 40.0; the curriculum event carries
    `screen_champion=40.0`; bit-reproducible (G2)."""
    _install_screen_train(monkeypatch,
                          [50.0, 55.0, 40.0, 45.0, 42.0, 38.0],
                          [50.0, 60.0, 45.0, 44.0, 50.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(4, 300, 30), runs_dir=tmp_path,
            screen_frac=0.25, curriculum=_FakeCurriculum())
        _drive_to_done(env)
        assert env.done and env.done_reason == "budget_exhausted"
        assert env.curriculum.level == 1  # exactly one step-up

        entries = env.memory.load_experiments()
        assert [e["kind"] for e in entries] == [
            "baseline", "experiment", "curriculum",
            "experiment", "experiment", "screen"]
        # exp2 (index 3) was promoted past the re-pinned champion 40 even
        # though it loses to the stale champion 50 — and gate-rejected
        exp2 = entries[3]
        assert exp2["accepted"] is False and exp2["holdout_score"] == 44.0
        assert not exp2.get("screen_rejected")  # it full-trained
        # exp4 (index 5) is a plain screen rejection against the new bar
        exp4 = entries[5]
        assert exp4["accepted"] is False and exp4["screen_rejected"] is True
        assert exp4["holdout_score"] == 38.0

        # the champion re-pinned to the level-1 baseline screen score
        assert env._screen_champion == 40.0
        # the step-up event row carries the re-pinned champion (33.3)
        summary = env.memory.load_summary()
        levels = summary["curriculum"]["levels"]
        assert len(levels) == 1
        assert levels[0]["screen_champion"] == 40.0
        assert levels[0]["new_baseline_score"] == 45.0
        assert summary["screening"]["baseline_screen_score"] == 50.0
    finally:
        TASKS.pop(FAKE_TASK, None)

    # G2: a second drive is bit-identical (kinds, scores, champion)
    _install_screen_train(monkeypatch,
                          [50.0, 55.0, 40.0, 45.0, 42.0, 38.0],
                          [50.0, 60.0, 45.0, 44.0, 50.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env2 = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(4, 300, 30), runs_dir=tmp_path / "r2",
            screen_frac=0.25, curriculum=_FakeCurriculum())
        _drive_to_done(env2)
        e1 = env.memory.load_experiments()
        e2 = env2.memory.load_experiments()
        assert [
            (e["kind"], e.get("holdout_score"), e.get("accepted"))
            for e in e1
        ] == [
            (e["kind"], e.get("holdout_score"), e.get("accepted"))
            for e in e2
        ]
        assert env2._screen_champion == 40.0
        assert (env2.memory.load_summary()["curriculum"]["levels"][0]
                ["screen_champion"]) == 40.0
    finally:
        TASKS.pop(FAKE_TASK, None)


def test_screen_rejected_never_enters_frontier_or_ensemble(
        tmp_path, monkeypatch):
    """A23 (SPEC.md 33.3): screening × ensemble — a screen-rejected spec
    never full-trains, so its fingerprint is absent from `pareto.points`
    and the ensemble top-k (19.3). Scripted: champion 50; exp1 screen
    30 → rejected; exp2 screen 60 → promote, full 70 → accepted."""
    _install_screen_train(monkeypatch, [50.0, 30.0, 60.0], [50.0, 70.0])
    TASKS[FAKE_TASK] = _FakeTask
    try:
        env = AutoRefineEnv(
            task=FAKE_TASK, seed=7,
            budget=Budget(2, 300, 30), runs_dir=tmp_path,
            screen_frac=0.25, ensemble_top_k=2)
        proposals = _drive_to_done(env)
        assert env.done and env.done_reason == "budget_exhausted"

        entries = env.memory.load_experiments()
        assert [e["kind"] for e in entries] == ["baseline", "screen",
                                                "experiment"]
        rejected_fp = entries[1]["spec_hash"]
        assert entries[1]["screen_rejected"] is True
        # the rejected spec was proposed (so its fingerprint is meaningful)
        from autorefine.config import ModelSpec
        assert any(
            ModelSpec.from_dict(p).fingerprint() == rejected_fp
            for p in proposals)

        # 33.3: the frontier and the ensemble see only full-trained models
        frontier_hashes = {p["spec_hash"] for p in env.pareto.points}
        assert rejected_fp not in frontier_hashes
        top_scores = [score for score, _m in env._top]
        assert top_scores == [70.0, 50.0]  # exp2 + baseline, never the 30
    finally:
        TASKS.pop(FAKE_TASK, None)


# --- house invariants ----------------------------------------------------------

def test_import_autorefine_stays_clean():
    """A23 (SPEC.md 33.4): no new dependencies — a fresh interpreter
    importing `autorefine` never pulls in streamlit/PIL/soundfile."""
    code = (
        "import sys, autorefine; "
        "bad = [m for m in ('PIL', 'streamlit', 'soundfile') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(REPO))
