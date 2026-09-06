"""The canonical run recipe — `RunConfig` (SPEC.md 37.1, v0.23 G3).

One frozen (immutable) dataclass — the complete recipe of a run.
Serialized **at reset** (the `run_config.json` artifact) and mirrored as
the additive `summary.json` key `run_config`; `fit_recipe` renders the
copy-pasteable CLI command from it, and `fit --from-run` re-runs a run
exactly from it (37.1.4).

Core module: stdlib only, no new dependencies (SPEC.md 3).
"""
from __future__ import annotations

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
    "autorefine_version",
)

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
    return cmd


def format_recipe(config: RunConfig) -> str:
    """The recipe as a single copy-pasteable line (37.1.4)."""
    return " ".join(fit_recipe(config))
