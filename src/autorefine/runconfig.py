"""The canonical run recipe — `RunConfig` (SPEC.md 37.1, v0.23 G3).

One frozen (immutable) dataclass — the complete recipe of a run.
Serialized **at reset** (the `run_config.json` artifact) and mirrored as
the additive `summary.json` key `run_config`; `fit_recipe` renders the
copy-pasteable CLI command from it, and `fit --from-run` re-runs a run
exactly from it (37.1.4).

Core module: stdlib only, no new dependencies (SPEC.md 3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any

RUN_CONFIG_SCHEMA = "autorefine.run_config/1"

# The canonical field set + order (SPEC.md 37.1.2 table order)
_FIELDS = (
    "task", "seed",
    "policy", "target", "rl_episodes",
    "max_experiments", "max_wall_seconds", "max_train_seconds",
    "dataset_size", "task_config",
    "ci_blocks", "z_accept", "efficiency_weight",
    "gen_gap_penalty", "block_size",
    "ensemble_top_k", "curriculum", "stall_patience", "screen_frac",
    "kfold",
    # SPEC.md 59.4 (v0.45): the steering rules (empty = pre-v0.45 runs;
    # optional in `from_dict` for back-compat — old run_config.json loads)
    "pins", "biases", "constraints",
    "autorefine_version",
)

# SPEC.md 59.4 (v0.45): the v0.45 fields are optional in `from_dict`
# (absent → the empty list) so pre-v0.45 run_config.json still loads
_OPTIONAL_FIELDS = ("pins", "biases", "constraints")

_INT = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
_NUM = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
_OPT = lambda fn: (lambda v: v is None or fn(v))  # noqa: E731

_VALIDATORS: dict[str, Any] = {
    "task": lambda v: isinstance(v, str) and v,
    "seed": _INT,
    "policy": _OPT(lambda v: isinstance(v, str) and v),
    "target": _OPT(_NUM),
    "rl_episodes": _OPT(_INT),
    "max_experiments": _INT,
    "max_wall_seconds": _NUM,
    "max_train_seconds": _NUM,
    "dataset_size": _OPT(_INT),
    "task_config": lambda v: isinstance(v, dict),
    "ci_blocks": _INT,
    "z_accept": _NUM,
    "efficiency_weight": _NUM,
    "gen_gap_penalty": _NUM,
    "block_size": _INT,
    "ensemble_top_k": _INT,
    "curriculum": lambda v: isinstance(v, bool),
    "stall_patience": _OPT(_INT),
    "screen_frac": _NUM,
    # SPEC.md 44.1.4 (v0.30): int >= 0 (0 = off; the legacy single split)
    "kfold": lambda v: _INT(v) and v >= 0,
    # SPEC.md 59.4 (v0.45): lists of [field, value] / [field, [values]] pairs
    "pins": lambda v: isinstance(v, list),
    "biases": lambda v: isinstance(v, list),
    "constraints": lambda v: isinstance(v, list),
    "autorefine_version": lambda v: isinstance(v, str) and v,
}


@dataclass(frozen=True)
class RunConfig:
    """The complete recipe of a run (SPEC.md 37.1.2). Immutable: a run's
    config is fixed at reset and never edited afterwards."""

    task: str
    seed: int
    # driver metadata — the env does not own these (None = env-level)
    policy: str | None = None
    target: float | None = None
    rl_episodes: int | None = None
    # budget (SPEC.md 5.2)
    max_experiments: int = 30
    max_wall_seconds: float = 900.0
    max_train_seconds: float = 30.0
    # data (SPEC.md 20.3 / 22.1)
    dataset_size: int | None = None
    task_config: dict = field(default_factory=dict)
    # search quality (SPEC.md 18)
    ci_blocks: int = 0
    z_accept: float = 0.0
    efficiency_weight: float = 0.0
    gen_gap_penalty: float = 0.0
    block_size: int = 512
    # options (SPEC.md 19.3 / 20.1 / 31.1 / 32.2)
    ensemble_top_k: int = 0
    curriculum: bool = False
    stall_patience: int | None = None
    screen_frac: float = 1.0
    # SPEC.md 44.1 (v0.30): k-fold holdout scoring (0 = off, legacy)
    kfold: int = 0
    # SPEC.md 59.4 (v0.45): the steering rules — lists of [field, value]
    # (pins/biases) and [field, [value, ...]] (constraints) pairs;
    # empty = pre-v0.45 behavior (the byte-identical default path)
    pins: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    # identity
    autorefine_version: str = "0.0.0"

    def to_dict(self) -> dict:
        """JSON-safe dict (SPEC.md 37.1.2): the full field set + schema."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["task_config"] = dict(self.task_config)  # detach the mutable value
        d["schema"] = RUN_CONFIG_SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Validate (schema + full field set + field types) and
        reconstruct (SPEC.md 37.1.2); a mutated/unknown dict is rejected."""
        if not isinstance(d, dict):
            raise ValueError(
                f"run_config must be a JSON object, got {type(d).__name__} "
                f"(SPEC.md 37.1.2)")
        if d.get("schema") != RUN_CONFIG_SCHEMA:
            raise ValueError(
                f"run_config schema mismatch: {d.get('schema')!r} != "
                f"{RUN_CONFIG_SCHEMA!r} (SPEC.md 37.1.2)")
        extra = set(d) - set(_FIELDS) - {"schema"}
        if extra:
            raise ValueError(
                f"run_config has unknown field(s): {sorted(extra)} "
                f"(SPEC.md 37.1.2)")
        kw: dict[str, Any] = {}
        for name in _FIELDS:
            if name not in d:
                if name in _OPTIONAL_FIELDS:
                    kw[name] = []  # SPEC.md 59.4 (v0.45): pre-v0.45 back-compat
                    continue
                raise ValueError(f"run_config is missing field {name!r} "
                                 f"(SPEC.md 37.1.2)")
            value = d[name]
            if not _VALIDATORS[name](value):
                raise ValueError(
                    f"run_config field {name!r} has an invalid value: "
                    f"{value!r} (SPEC.md 37.1.2)")
            kw[name] = value
        return cls(**kw)

    @classmethod
    def from_env(cls, env: Any, policy: str | None = None,
                 target: float | None = None,
                 rl_episodes: int | None = None) -> "RunConfig":
        """Build the recipe from a reset-ready `AutoRefineEnv` + the
        driver's own metadata (SPEC.md 37.1.3)."""
        import autorefine  # local: avoid the package-init cycle
        # SPEC.md 59.4 (v0.45): the steering rules (absent/empty = pre-v0.45)
        st = getattr(env, "steering", None)
        sd = st.to_dict() if st is not None else {}
        return cls(
            task=env.task_name,
            seed=int(env.seed),
            policy=policy,
            target=None if target is None else float(target),
            rl_episodes=None if rl_episodes is None else int(rl_episodes),
            max_experiments=int(env.budget.max_experiments),
            max_wall_seconds=float(env.budget.max_wall_seconds),
            max_train_seconds=float(env.budget.max_train_seconds),
            dataset_size=None if env.dataset_size is None else int(env.dataset_size),
            task_config=dict(env.task_config or {}),
            ci_blocks=int(env.ci_blocks),
            z_accept=float(env.z_accept),
            efficiency_weight=float(env.efficiency_weight),
            gen_gap_penalty=float(env.gen_gap_penalty),
            block_size=int(env.block_size),
            ensemble_top_k=int(env.ensemble_top_k),
            curriculum=bool(getattr(env, "curriculum", None) is not None),
            stall_patience=env.stall_patience,
            screen_frac=float(env.screen_frac),
            kfold=int(env.kfold),  # SPEC.md 44.1 (v0.30)
            pins=list(sd.get("pins") or []),  # SPEC.md 59.4 (v0.45)
            biases=list(sd.get("biases") or []),  # SPEC.md 59.4 (v0.45)
            constraints=list(sd.get("constraints") or []),  # 59.4 (v0.45)
            autorefine_version=autorefine.__version__,
        )


# --- the CLI recipe (SPEC.md 37.1.4) ----------------------------------------
# The CLI default values (SPEC.md 22.1 / 11) — flags equal to these are
# omitted from the rendered recipe.
_CLI_DEFAULTS = dict(
    seed=7, max_experiments=30, max_wall_seconds=900.0,
    max_train_seconds=30.0, rl_episodes=5, split_frac=0.2,
)
# The v0.4 search-quality preset (SPEC.md 18.6) = the CLI default; the
# legacy preset is the only non-default expressible via `--search-quality`.
_PRESET_V04 = dict(ci_blocks=8, z_accept=1.0, efficiency_weight=0.5,
                   gen_gap_penalty=0.5, block_size=512)
_PRESET_LEGACY = dict(ci_blocks=0, z_accept=0.0, efficiency_weight=0.0,
                      gen_gap_penalty=0.0, block_size=512)


def _steering_value(value: Any) -> str:
    """SPEC.md 59.4 (v0.45): one steering value as a CLI token — a
    list/tuple (an architecture tuple, e.g. ``[8, 16]``) is ``json.dumps``
    -ed; a scalar is ``str``-ed. A constraint's *set* of values is
    comma-joined by the caller (the CLI's comma split); architecture-tuple
    constraint values cannot be expressed via the comma split and are the
    app/Python API surface (59.4)."""
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return str(value)


def _quality(config: RunConfig) -> dict:
    d = config.to_dict()
    return {k: d[k] for k in ("ci_blocks", "z_accept", "efficiency_weight",
                              "gen_gap_penalty", "block_size")}


def fit_recipe(config: RunConfig) -> list[str]:
    """The copy-pasteable CLI recipe (SPEC.md 37.1.4): `autorefine fit …`
    for data tasks (`task_config` carries a path), `autorefine run …`
    for built-in tasks. A pure function of the `RunConfig` (37.1.5).

    Flags equal to the CLI default are omitted; the v04 preset is the CLI
    default (omitted); the legacy preset renders `--search-quality legacy`;
    non-preset knob values are noted (not expressible as flags).
    """
    d = config.to_dict()
    tc = d["task_config"]
    if tc.get("path"):  # data task → `fit` (`--data`, the CLI's flag)
        cmd = ["autorefine", "fit", "--data", str(tc["path"])]
        if tc.get("label") is not None:
            cmd += ["--label", str(tc["label"])]
        if tc.get("split_frac") not in (None, _CLI_DEFAULTS["split_frac"]):
            cmd += ["--split", str(tc["split_frac"])]
    else:  # built-in task → `run`
        cmd = ["autorefine", "run", "--task", d["task"]]
        if d["curriculum"]:
            cmd += ["--curriculum"]
    # budget (5.2) — non-default values only
    if d["max_experiments"] != _CLI_DEFAULTS["max_experiments"]:
        cmd += ["--experiments", str(d["max_experiments"])]
    if d["max_wall_seconds"] != _CLI_DEFAULTS["max_wall_seconds"]:
        cmd += ["--max-seconds", str(d["max_wall_seconds"])]
    if d["max_train_seconds"] != _CLI_DEFAULTS["max_train_seconds"]:
        cmd += ["--max-train-seconds", str(d["max_train_seconds"])]
    if d["seed"] != _CLI_DEFAULTS["seed"]:
        cmd += ["--seed", str(d["seed"])]
    # driver metadata (37.1.2)
    if d["policy"] in ("bandit", "search", "rl"):
        if d["policy"] != "bandit":  # the CLI default
            cmd += ["--policy", d["policy"]]
        if d["policy"] == "rl" and d["rl_episodes"] not in (None,
                                                            _CLI_DEFAULTS["rl_episodes"]):
            cmd += ["--rl-episodes", str(d["rl_episodes"])]
    if d["target"] is not None:
        cmd += ["--target", str(d["target"])]
    # search quality (18) — by preset, else noted
    quality = _quality(config)
    if quality == _PRESET_V04:
        pass  # the CLI default
    elif quality == _PRESET_LEGACY:
        cmd += ["--search-quality", "legacy"]
    else:
        cmd += ["# note: custom search-quality knobs (not a CLI preset):",
                str(quality)]
    # options (19.3 / 31.1 / 32.2) — non-default values only
    if d["ensemble_top_k"] > 0:
        cmd += ["--ensemble-final"]
    if d["stall_patience"] is not None:
        cmd += ["--stall-patience", str(d["stall_patience"])]
    if d["screen_frac"] < 1.0:
        cmd += ["--screen-frac", str(d["screen_frac"])]
    if d["kfold"] > 0:  # SPEC.md 44.1 (v0.30): non-default values only
        cmd += ["--kfold", str(d["kfold"])]
    # SPEC.md 59.4 (v0.45): the steering rules — repeatable flags, one per rule
    for f, v in d["pins"]:
        cmd += ["--pin", f"{f}={_steering_value(v)}"]
    for f, v in d["biases"]:
        cmd += ["--bias", f"{f}={_steering_value(v)}"]
    for f, vs in d["constraints"]:
        cmd += ["--constrain",
                f"{f}={','.join(_steering_value(x) for x in vs)}"]
    return cmd


def format_recipe(config: RunConfig) -> str:
    """The recipe as a single copy-pasteable line (37.1.4)."""
    return " ".join(fit_recipe(config))


def runconfig_to_flags(config: RunConfig) -> dict[str, Any]:
    """The config as plain CLI flag names (SPEC.md 47.1.4, v0.33):
    the canonical `RunConfig` translated into the CLI's kebab-case flag
    vocabulary, so `--config` can apply a canonical recipe and a plain
    flag file with one mechanism (47.1.2).

    A pure function of the `RunConfig`. Only non-CLI-default values are
    emitted (an absent key = the parser's default; the same omission
    rule as `fit_recipe`, 37.1.4). Raises `ValueError` when a field is
    not expressible as flags — a non-preset search-quality knob or an
    unknown `task_config` key: a wrong re-run is worse than a loud
    refusal (47.1.4), unlike `fit_recipe`'s note-and-continue.
    """
    d = config.to_dict()
    tc = d["task_config"]
    flags: dict[str, Any] = {}
    if tc.get("path"):  # data task (37.1.4) -> `fit`'s data flags
        flags["data"] = str(tc["path"])
        if tc.get("label") is not None:
            flags["label"] = str(tc["label"])
        if tc.get("split_frac") not in (None, _CLI_DEFAULTS["split_frac"]):
            flags["split"] = tc["split_frac"]
        if tc.get("split_mode") == "temporal":  # v0.31 (SPEC.md 45.1)
            flags["temporal"] = True
        if tc.get("metric") not in (None, "accuracy"):  # v0.32 (46.1)
            flags["metric"] = tc["metric"]
        unknown = set(tc) - {"path", "label", "split_frac",
                            "split_mode", "metric"}
        if unknown:
            raise ValueError(
                f"task_config key(s) not expressible as CLI flags: "
                f"{sorted(unknown)} (SPEC.md 47.1.4)")
    else:  # built-in task -> `run --task`
        flags["task"] = d["task"]
    # budget (5.2) — non-default values only (47.1.4)
    if d["max_experiments"] != _CLI_DEFAULTS["max_experiments"]:
        flags["experiments"] = d["max_experiments"]
    if d["max_wall_seconds"] != _CLI_DEFAULTS["max_wall_seconds"]:
        flags["max-seconds"] = d["max_wall_seconds"]
    if d["max_train_seconds"] != _CLI_DEFAULTS["max_train_seconds"]:
        flags["max-train-seconds"] = d["max_train_seconds"]
    if d["seed"] != _CLI_DEFAULTS["seed"]:
        flags["seed"] = d["seed"]
    # driver metadata (37.1.2) — emitted whenever recorded: the two
    # commands' defaults differ (run: search, fit: bandit), so the
    # parser default may not stand in for the recipe's policy
    if d["policy"] is not None:
        flags["policy"] = d["policy"]
    if (d["rl_episodes"] is not None
            and d["rl_episodes"] != _CLI_DEFAULTS["rl_episodes"]):
        flags["rl-episodes"] = d["rl_episodes"]
    if d["target"] is not None and d["target"] != 95.0:
        # 95.0 is `fit`'s parser default (omitted); `run` owns no
        # --target (ungated, 37.2.3), so it can never need one
        flags["target"] = d["target"]
    # search quality (18) — presets only (the v04 preset is the default)
    quality = _quality(config)
    if quality == _PRESET_LEGACY:
        flags["search-quality"] = "legacy"
    elif quality != _PRESET_V04:
        raise ValueError(
            f"custom search-quality knobs {quality} are not expressible "
            f"as CLI flags (SPEC.md 47.1.4)")
    # options (19.3 / 20.1 / 31.1 / 32.2 / 44.1) — non-default values only
    if d["ensemble_top_k"] > 0:
        flags["ensemble-final"] = True
    if d["curriculum"]:
        flags["curriculum"] = True
    if d["stall_patience"] is not None:
        flags["stall-patience"] = d["stall_patience"]
    if d["screen_frac"] < 1.0:
        flags["screen-frac"] = d["screen_frac"]
    if d["kfold"] > 0:  # v0.30 (SPEC.md 44.1)
        flags["kfold"] = d["kfold"]
    # SPEC.md 59.4 (v0.45): the steering rules — lists of "field=value" tokens
    # for the CLI's repeatable append flags (`_config_value` returns a list as-is)
    if d["pins"]:
        flags["pin"] = [f"{f}={_steering_value(v)}" for f, v in d["pins"]]
    if d["biases"]:
        flags["bias"] = [f"{f}={_steering_value(v)}" for f, v in d["biases"]]
    if d["constraints"]:
        flags["constrain"] = [
            f"{f}={','.join(_steering_value(x) for x in vs)}"
            for f, vs in d["constraints"]]
    return flags
