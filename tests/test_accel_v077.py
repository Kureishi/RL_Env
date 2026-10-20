"""v0.77 dense parallel + the optional GPU backend — SPEC.md 91, A81.

SPEC.md 91 (v0.77): the single-threaded dense workloads become parallel
where the work is genuinely independent (the bagged trees, the per-seed
sweep units), and an optional GPU backend is added for the dense leaves.
Two invariants: the default path (``parallel=1``, ``backend="cpu"``) is
the exact pre-v0.77 code path (G2 — the bit-identical pins stay green),
and no new hard dependency is added (GPU libraries are discovered with
``find_spec`` and imported only when explicitly selected):

  * 91.1 — ``available_gpus`` / ``detect_backend`` / ``load_backend`` /
    ``backend_info`` (the backend discovery + report);
  * 91.2 — ``parallel_map`` (fixed-order; ``workers=1`` is the exact
    serial loop) + ``TreeEnsemble.fit(parallel=...)`` (the bags are
    drawn from the same seeded rng; the trees build on workers; the
    result is bit-identical at every ``parallel``) + the ``parallel``
    knob (cli: run/fit/variance) + ``trainer.train`` pass-through;
  * 91.3 — ``resolve_backend`` (graceful CPU fallback) + ``to_device`` /
    ``to_host`` / ``gpu_matmul`` (the GPU foundations) + the ``backend``
    knob (cli: run/fit; ``cpu`` default = the legacy path);
  * 91.4 — the ``RunConfig`` additive fields (back-compat: a pre-v0.77
    dict loads with the legacy defaults; ``fit_recipe`` /
    ``runconfig_to_flags`` emit the flags only when non-default);
  * 91.5 — the ``autorefine backend`` verb (+ ``--json``);
  * 91.7 — acceptance: the version steps to ``0.82.0`` in both sources,
    the A-index advances (A81, 77 acceptance rows, M80 resolving to
    this file), and the new exports land in ``autorefine.__all__``.

Host fact: this interpreter has NO GPU library (cupy / torch absent),
so the GPU-absent branches are exercised for real; every assertion is
written to stay correct on a GPU host as well (conditional on
``available_gpus()``).

House rules: stdlib + numpy only, no cross-test imports, deterministic,
tiny budgets, one module-scoped real-run fixture where a CLI end-to-end
pass is wanted.
"""
from __future__ import annotations

import contextlib
import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

import autorefine
from autorefine import (
    AutoRefineEnv,
    BackendUnavailable,
    available_gpus,
    backend_info,
    detect_backend,
    gpu_matmul,
    load_backend,
    parallel_map,
    resolve_backend,
    to_device,
    to_host,
)
from autorefine.cli import build_parser, main as cli_main
from autorefine.improver.meta_env import KNOBS
from autorefine.models.trees import TreeEnsemble
from autorefine.runconfig import RunConfig, fit_recipe, runconfig_to_flags
from autorefine.trainer import train as trainer_train
from autorefine.config import ModelSpec, SpecError

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "SPEC.md"

NEW_EXPORTS = (
    "BackendUnavailable",
    "available_gpus",
    "detect_backend",
    "load_backend",
    "backend_info",
    "resolve_backend",
    "to_device",
    "to_host",
    "gpu_matmul",
    "parallel_map",
)


# --- helpers (test-local; the house rules forbid cross-test imports) -----------

def _tree_data(n: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A seeded 2-class, 2-feature dataset — the tree-fit battery."""
    rnd = np.random.default_rng(seed)
    X = rnd.uniform(-1.0, 1.0, size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0.0).astype(np.int64)
    return X, y


def _tree_spec() -> ModelSpec:
    """A tiny tree-family spec (the ``trainer.train`` battery)."""
    return ModelSpec(
        model_family="tree",
        architecture=[3],     # tree depth (1..3, SPEC.md 15)
        train_steps=400,      # // 100 -> 4 trees
        batch_size=32,
        learning_rate=0.05,
        activation="relu",
        optimizer="adam",
        weight_decay=0.0,
        lr_schedule="constant",
        label_smoothing=0.0,
        input_noise=0.0,
    )


def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run one CLI command, capturing stdout + stderr (the house pattern)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(args)
    return rc, out.getvalue(), err.getvalue()


def _subs() -> dict:
    parser = build_parser()
    sub = next(a for a in parser._actions
               if type(a).__name__ == "_SubParsersAction")
    return sub.choices


# --- 91.2 parallel_map (A81.2) ---------------------------------------------------

class TestParallelMap:
    """SPEC.md 91.2 (A81.2): fixed-order, G2-safe ``parallel_map``."""

    def test_workers_one_is_the_exact_serial_result(self):
        res = parallel_map(8, lambda i: i * i, workers=1)
        assert res == [i * i for i in range(8)]

    def test_workers_gt_one_keeps_input_order(self):
        expected = [i * i for i in range(64)]
        for workers in (2, 4, 8):
            assert parallel_map(64, lambda i: i * i, workers=workers) == expected

    def test_zero_units_is_empty(self):
        assert parallel_map(0, lambda i: i, workers=4) == []

    @pytest.mark.parametrize("bad", [0, -1, True, "2", None, 1.0])
    def test_bad_workers_raise(self, bad):
        with pytest.raises(ValueError):
            parallel_map(3, lambda i: i, workers=bad)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            parallel_map(-1, lambda i: i, workers=1)


# --- 91.2 the bit-identical parallel tree fit (A81.2) ---------------------------

class TestParallelTreeFit:
    """SPEC.md 91.2 (A81.2): ``TreeEnsemble.fit(parallel=...)`` — the bags
    are drawn from the same seeded rng and the trees are assembled in the
    fixed legacy order, so every ``parallel`` value is bit-identical (G2)."""

    def test_trees_are_bit_identical_across_parallel_values(self):
        X, y = _tree_data()
        trees = {}
        for parallel in (1, 2, 4, 8):
            ens = TreeEnsemble(n_out=2, n_trees=6, max_depth=3,
                               head="softmax", seed=7)
            ens.fit(X, y, parallel=parallel)
            trees[parallel] = [(t.feat.tobytes(), t.thr.tobytes(),
                                t.left.tobytes(), t.right.tobytes(),
                                t.leafv.tobytes()) for t in ens.trees]
        assert trees[1] == trees[2] == trees[4] == trees[8]

    def test_forward_outputs_are_bit_identical_across_parallel_values(self):
        X, y = _tree_data()
        outs = {}
        for parallel in (1, 4):
            ens = TreeEnsemble(n_out=2, n_trees=6, max_depth=3,
                               head="softmax", seed=7)
            ens.fit(X, y, parallel=parallel)
            outs[parallel] = ens.forward(X)
        assert np.array_equal(outs[1], outs[4])  # bit-exact (==, not approx)

    def test_trainer_parallel_gives_identical_final_loss(self):
        X, y = _tree_data()
        losses = {}
        for parallel in (1, 4):
            res = trainer_train(
                (X, y), _tree_spec(), seed=7, n_out=2, head="softmax",
                parallel=parallel)
            losses[parallel] = res.final_loss
        assert losses[1] == losses[4]

    def test_trainer_rejects_bad_parallel(self):
        X, y = _tree_data()
        for bad in (0, -1, True, "4", None):
            with pytest.raises((SpecError, ValueError)):
                trainer_train((X, y), _tree_spec(), seed=7, parallel=bad)


# --- 91.2 / 91.3 the knobs (A81.2 / A81.3) ---------------------------------------

class TestKnobs:
    """SPEC.md 91.2/91.3 (A81.2/A81.3): the ``parallel`` / ``backend``
    registry knobs — validators, defaults, cli tuples."""

    def test_parallel_validator(self):
        v = KNOBS["parallel"].validate
        assert v(1) == 1 and v(8) == 8
        for bad in (0, -1, True, "4", None, 1.0):
            with pytest.raises(ValueError):
                v(bad)

    def test_backend_validator(self):
        v = KNOBS["backend"].validate
        for good in ("auto", "cpu", "gpu", "cupy", "torch"):
            assert v(good) == good
        for bad in ("cuda", "CPU", "", None, 1):
            with pytest.raises(ValueError):
                v(bad)

    def test_registry_defaults_and_cli_tuples(self):
        assert KNOBS["parallel"].default == 1
        assert KNOBS["parallel"].cli == ("run", "fit", "variance")
        assert KNOBS["backend"].default == "cpu"
        assert KNOBS["backend"].cli == ("run", "fit")
        assert KNOBS["parallel"].spec_ref == "91.2"
        assert KNOBS["backend"].spec_ref == "91.3"

    def test_env_defaults_and_attributes(self):
        env = AutoRefineEnv(task="parity-v1", runs_dir="runs-x")
        assert env.parallel == 1 and env.backend == "cpu"

    def test_env_rejects_bad_values(self):
        with pytest.raises(ValueError):
            AutoRefineEnv(task="parity-v1", runs_dir="runs-x", parallel=0)
        with pytest.raises(ValueError):
            AutoRefineEnv(task="parity-v1", runs_dir="runs-x",
                          backend="cuda")


# --- 91.3 / 91.5 the CLI (A81.3 / A81.5) -----------------------------------------

class TestCli:
    """SPEC.md 91.3/91.5 (A81.3/A81.5): the flags on exactly the owning
    subcommands + the ``backend`` report verb."""

    def test_parallel_flag_on_run_fit_variance_only(self):
        subs = _subs()
        where = [name for name, sp in subs.items()
                 if any("--parallel" in a.option_strings
                        for a in sp._actions)]
        assert where == ["run", "fit", "variance"]
        for name in where:
            action = next(a for a in subs[name]._actions
                          if "--parallel" in a.option_strings)
            assert action.default == KNOBS["parallel"].default == 1

    def test_backend_flag_on_run_fit_only(self):
        subs = _subs()
        where = [name for name, sp in subs.items()
                 if any("--backend" in a.option_strings
                        for a in sp._actions)]
        assert where == ["run", "fit"]
        for name in where:
            action = next(a for a in subs[name]._actions
                          if "--backend" in a.option_strings)
            assert action.default == KNOBS["backend"].default == "cpu"
            assert set(action.choices) == {"auto", "cpu", "gpu", "cupy",
                                           "torch"}

    def test_backend_subcommand_exists_and_exits_zero(self):
        assert "backend" in _subs()
        rc, out, err = _run_cmd(["backend"])
        assert rc == 0, out + err
        assert "auto picks" in out and "GPU" in out

    def test_backend_json_is_pure_json(self):
        rc, out, err = _run_cmd(["backend", "--json"])
        assert rc == 0, out + err
        info = json.loads(out)  # the whole stdout is one JSON object
        assert info == backend_info()


# --- 91.1 / 91.3 the backend cores (A81.1 / A81.3) -------------------------------

class TestBackendCores:
    """SPEC.md 91.1/91.3 (A81.1/A81.3): discovery, graceful fallback, and
    the GPU foundations (CPU-path assertions; GPU branches conditional)."""

    def test_available_gpus_is_a_pref_ordered_subset(self):
        gpus = available_gpus()
        assert isinstance(gpus, tuple)
        assert set(gpus) <= set(("cupy", "torch"))

    def test_detect_backend_cpu_and_auto(self):
        assert detect_backend("cpu") == "cpu"
        assert detect_backend("auto") in ("cpu", *available_gpus())

    def test_detect_backend_named_and_bogus(self):
        for name in ("cupy", "torch"):
            if name in available_gpus():
                assert detect_backend(name) == name
            else:
                with pytest.raises(BackendUnavailable):
                    detect_backend(name)
        with pytest.raises(ValueError):
            detect_backend("bogus")

    def test_load_backend_rules(self):
        with pytest.raises(ValueError):
            load_backend("cpu")       # the native path has no module
        with pytest.raises(ValueError):
            load_backend("bogus")
        for name in ("cupy", "torch"):
            if name in available_gpus():
                assert load_backend(name).__name__ == name
            else:
                with pytest.raises(BackendUnavailable):
                    load_backend(name)

    def test_backend_info_shape(self):
        info = backend_info()
        assert set(info) == {"auto", "libraries", "numpy", "cpu_count",
                             "gpu_available"}
        assert info["libraries"].keys() == {"cupy", "torch"}
        assert isinstance(info["gpu_available"], bool)
        assert info["gpu_available"] == (
            any(v is not None for v in info["libraries"].values()))

    def test_resolve_backend_cpu_is_a_noop(self):
        assert resolve_backend("cpu") == ("cpu", None)

    def test_resolve_backend_gpu_degrades_or_resolves(self):
        actual, warn = resolve_backend("gpu")
        if available_gpus():
            assert actual in available_gpus() and warn is None
        else:  # this host: no GPU library -> the friendly CPU fallback
            assert actual == "cpu" and warn

    def test_resolve_backend_named_degrades_or_resolves(self):
        for name in ("cupy", "torch", "auto"):
            actual, warn = resolve_backend(name)
            if name in available_gpus():
                assert actual == name and warn is None
            else:
                assert actual == "cpu" and warn
        with pytest.raises(ValueError):
            resolve_backend("bogus")
        with pytest.raises(ValueError):
            resolve_backend(3)

    def test_device_leaves_round_trip_on_cpu(self):
        x = np.arange(6, dtype=np.float64).reshape(2, 3)
        a = to_device(x, "cpu")
        assert a is x or np.array_equal(a, x)
        b = to_host(a, "cpu")
        assert np.array_equal(b, x) and b.dtype == np.float64

    def test_gpu_matmul_cpu_path_is_native_matmul(self):
        a = np.random.default_rng(0).normal(size=(5, 4))
        b = np.random.default_rng(1).normal(size=(4, 3))
        assert np.array_equal(gpu_matmul(a, b, "cpu"), a @ b)
        # "auto" on a GPU-less host degrades to the native path
        if not available_gpus():
            assert np.array_equal(gpu_matmul(a, b, "auto"), a @ b)
            assert np.array_equal(gpu_matmul(a, b, "gpu"), a @ b)


# --- 91.4 the RunConfig fields (A81.4) -------------------------------------------

class TestRunConfig:
    """SPEC.md 91.4 (A81.4): the additive ``parallel`` / ``backend``
    recipe fields — back-compat, validators, and the flag emitters."""

    def test_to_dict_carries_the_fields_with_defaults(self):
        cfg = RunConfig(task="parity-v1", seed=7)
        d = cfg.to_dict()
        assert d["parallel"] == 1 and d["backend"] == "cpu"

    def test_pre_v077_dict_loads_with_legacy_defaults(self):
        d = RunConfig(task="parity-v1", seed=7).to_dict()
        del d["parallel"], d["backend"]
        cfg = RunConfig.from_dict(d)
        assert cfg.parallel == 1 and cfg.backend == "cpu"

    def test_non_defaults_round_trip(self):
        cfg = RunConfig(task="parity-v1", seed=7, parallel=4,
                        backend="cupy")
        assert RunConfig.from_dict(cfg.to_dict()) == cfg

    def test_validators_reject_bad_values(self):
        base = RunConfig(task="parity-v1", seed=7).to_dict()
        for bad in (0, -1, True, "4", None):
            d = dict(base, parallel=bad)
            with pytest.raises(ValueError):
                RunConfig.from_dict(d)
        for bad in ("cuda", "CPU", "", None, 1):
            d = dict(base, backend=bad)
            with pytest.raises(ValueError):
                RunConfig.from_dict(d)

    def test_fit_recipe_omits_defaults_and_emits_non_defaults(self):
        base = fit_recipe(RunConfig(task="parity-v1", seed=7))
        assert "--parallel" not in base and "--backend" not in base
        flags = fit_recipe(RunConfig(task="parity-v1", seed=7, parallel=4,
                                     backend="cupy"))
        assert "--parallel" in flags and "4" in flags
        assert "--backend" in flags and "cupy" in flags

    def test_runconfig_to_flags_same_omission_rule(self):
        base = runconfig_to_flags(RunConfig(task="parity-v1", seed=7))
        assert "parallel" not in base and "backend" not in base
        flags = runconfig_to_flags(
            RunConfig(task="parity-v1", seed=7, parallel=4, backend="gpu"))
        assert flags["parallel"] == 4 and flags["backend"] == "gpu"


# --- 91.5 an end-to-end CLI pass (A81.5) ------------------------------------------

class TestEndToEnd:
    """SPEC.md 91.5 (A81.5): a tiny ``run --parallel 4`` — the flags flow
    through the env, the loop finishes, and the artifacts carry the
    config (91.4)."""

    def test_run_with_parallel_flag_finishes(self, tmp_path):
        runs = tmp_path / "runs"
        rc, out, err = _run_cmd([
            "run", "--task", "parity-v1", "--experiments", "2",
            "--parallel", "4", "--runs-dir", str(runs), "--quiet",
        ])
        assert rc == 0, out + err
        run_dirs = sorted(runs.glob("parity-v1-*"))
        assert run_dirs, f"no run dir in {runs}: {list(runs.iterdir())}"
        cfg = json.loads((run_dirs[0] / "run_config.json")
                         .read_text(encoding="utf-8"))
        assert cfg["parallel"] == 4 and cfg["backend"] == "cpu"

    def test_run_with_absent_gpu_backend_falls_back(self, tmp_path):
        if available_gpus():
            pytest.skip("this host has a GPU — no fallback to observe")
        runs = tmp_path / "runs"
        rc, out, err = _run_cmd([
            "run", "--task", "parity-v1", "--experiments", "2",
            "--backend", "gpu", "--runs-dir", str(runs), "--quiet",
        ])
        assert rc == 0, out + err
        assert "falling back to CPU" in err  # the 91.3 friendly note
        run_dirs = sorted(runs.glob("parity-v1-*"))
        cfg = json.loads((run_dirs[0] / "run_config.json")
                         .read_text(encoding="utf-8"))
        assert cfg["backend"] == "gpu"  # the request is recorded as-is


# --- 91.7 ceremony (A81.7) ---------------------------------------------------------

class TestCeremony:
    """SPEC.md 91.7 (A81.7): the version step + the index row + the
    exports."""

    def test_version_steps_to_077_in_both_sources(self):
        init = (REPO / "src" / "autorefine" / "__init__.py").read_text(
            encoding="utf-8")
        assert '__version__ = "0.82.0"' in init
        assert 'version = "0.82.0"' in (
            REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert autorefine.__version__ == "0.82.0"

    def test_spec_section_and_index_row(self):
        spec = SPEC.read_text(encoding="utf-8")
        assert "## 91." in spec
        assert "### 91.7 Acceptance (A81)" in spec
        row = next(l for l in spec.splitlines() if l.startswith("| M80 |"))
        assert "91" in row and "A81" in row
        assert "tests/test_accel_v077.py" in row

    def test_new_exports_in_all(self):
        for name in NEW_EXPORTS:
            assert name in autorefine.__all__
            assert hasattr(autorefine, name)

    def test_pyproject_gpu_extra_is_optional_metadata(self):
        py = tomllib.loads((REPO / "pyproject.toml")
                           .read_text(encoding="utf-8"))
        extras = py["project"]["optional-dependencies"]
        assert "cupy-cuda12x>=13.0" in extras["gpu"]
        # the hard dependency set is untouched (no new hard dep)
        assert py["project"]["dependencies"] == ["numpy>=1.26"]
