"""Parallel dense workloads + the optional GPU backend (SPEC.md 91, v0.77).

Core module: stdlib + numpy only. GPU libraries (cupy / torch) are NEVER
imported at module load — they are discovered with
`importlib.util.find_spec` and imported only when a GPU backend is
explicitly selected (91.3), degrading to CPU with a friendly note when
absent. The default path is CPU and bit-identical to the pre-v0.77 code
(G2): `parallel_map(n, fn, workers=1)` is the exact serial loop, and the
bag draw order in the tree family is untouched (91.2).

  * 91.1 — backend discovery + report (`available_gpus`,
    `detect_backend`, `backend_info`, `load_backend`)
  * 91.2 — `parallel_map` (fixed-order, G2-safe parallelism over
    independent units)
  * 91.3 — `resolve_backend` (graceful CPU fallback), `to_device`,
    `to_host`, `gpu_matmul` (the GPU foundations; per-family GPU
    training is the documented follow-up, 91.6)
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "BackendUnavailable",
    "GPU_BACKENDS",
    "available_gpus",
    "detect_backend",
    "load_backend",
    "backend_info",
    "resolve_backend",
    "to_device",
    "to_host",
    "gpu_matmul",
    "parallel_map",
]


class BackendUnavailable(RuntimeError):
    """A requested GPU backend is not importable (SPEC.md 91.3)."""


# Discovery order = preference order (cupy first: it is the lighter,
# array-native backend; torch is the broader fallback).
GPU_BACKENDS = ("cupy", "torch")


def available_gpus() -> tuple:
    """91.1: the importable GPU backends (preference order). `find_spec`
    only — no import, no side effects, no GPU work at discovery time."""
    return tuple(b for b in GPU_BACKENDS
                 if importlib.util.find_spec(b) is not None)


def detect_backend(prefer: str = "auto") -> str:
    """91.1: resolve a preference to a concrete backend name.

    "cpu"  -> "cpu" (always); "auto" -> the first importable GPU backend,
    else "cpu"; "gpu" -> the first importable GPU backend or
    `BackendUnavailable`; "cupy"/"torch" -> that backend when importable,
    else `BackendUnavailable`. Anything else is a `ValueError`."""
    if prefer == "cpu":
        return "cpu"
    if prefer == "auto":
        gpus = available_gpus()
        return gpus[0] if gpus else "cpu"
    if prefer == "gpu":
        gpus = available_gpus()
        if not gpus:
            raise BackendUnavailable(
                "no GPU backend found (looked for cupy, torch) — "
                "install one (e.g. `pip install cupy` or `pip install "
                "torch`) or use --backend cpu (SPEC.md 91.3)")
        return gpus[0]
    if prefer in GPU_BACKENDS:
        if prefer not in available_gpus():
            raise BackendUnavailable(
                f"GPU backend {prefer!r} is not importable on this "
                f"interpreter — install it (e.g. `pip install {prefer}`) "
                f"or use --backend cpu (SPEC.md 91.3)")
        return prefer
    raise ValueError(
        f"backend must be one of auto/cpu/gpu/cupy/torch, got {prefer!r} "
        f"(SPEC.md 91.3)")


def load_backend(name: str):
    """91.1: import a GPU backend by name (never "cpu" — it has no module).

    Raises `BackendUnavailable` when it is not importable."""
    if name == "cpu":
        raise ValueError(
            "the cpu backend is the native numpy path — there is no module "
            "to load (SPEC.md 91.3)")
    if name not in GPU_BACKENDS:
        raise ValueError(
            f"unknown backend {name!r} (SPEC.md 91.3)")
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise BackendUnavailable(
            f"GPU backend {name!r} is not importable: {exc} — install it "
            f"(e.g. `pip install {name}`) or use --backend cpu "
            f"(SPEC.md 91.3)") from exc


def _library_version(name: str):
    """91.1: the installed version string of a GPU library, or None. The
    import is guarded — a broken install reads as absent, never a crash."""
    if importlib.util.find_spec(name) is None:
        return None
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def backend_info() -> dict:
    """91.1: the backend report — JSON-safe, deterministic for a given
    interpreter (the `autorefine backend` verb renders it)."""
    libraries = {name: _library_version(name) for name in GPU_BACKENDS}
    return {
        "auto": detect_backend("auto"),
        "libraries": libraries,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "gpu_available": any(v is not None for v in libraries.values()),
    }


_FALLBACK_NOTE = ("no GPU library (cupy/torch) found; falling back to CPU "
                  "(SPEC.md 91.3)")


def resolve_backend(requested: str = "cpu") -> tuple:
    """91.3: `(actual, warning)` — the graceful-degradation contract.

    "cpu" is a no-op. A GPU request with no importable backend degrades to
    "cpu" with a friendly note (the CLI/app prints it) instead of failing
    the run. A GPU request with a backend present returns it and no
    warning. Bad names are a `ValueError` (the knob validator catches them
    first in the env/CLI paths)."""
    if not isinstance(requested, str):
        raise ValueError(
            f"backend must be a string, got {type(requested).__name__} "
            f"(SPEC.md 91.3)")
    if requested == "cpu":
        return "cpu", None
    if requested not in ("auto", "gpu", "cupy", "torch"):
        raise ValueError(
            f"backend must be one of auto/cpu/gpu/cupy/torch, "
            f"got {requested!r} (SPEC.md 91.3)")
    try:
        actual = detect_backend(requested if requested in GPU_BACKENDS
                                else "auto")
    except BackendUnavailable:
        return "cpu", _FALLBACK_NOTE
    if actual == "cpu":
        return "cpu", _FALLBACK_NOTE
    return actual, None


# --- 91.3: the GPU foundations (device arrays + the dense op) -------------
# cupy and torch both accept numpy in / emit numpy out via these leaves,
# so a caller never sees a device type. `backend == "cpu"` is the native
# path (no conversion, no copy of already-float64 input semantics beyond
# `np.asarray` — bit-identical to the legacy `a @ b`).

def to_device(x: np.ndarray, backend: str = "cpu"):
    """91.3: a numpy array as a device array (cpu = `np.asarray`)."""
    a = np.asarray(x, dtype=np.float64)
    if backend == "cpu":
        return a
    module = load_backend(backend)
    if backend == "cupy":
        return module.asarray(a)
    return module.as_tensor(a)  # torch


def to_host(x, backend: str = "cpu") -> np.ndarray:
    """91.3: a device array back to numpy (cpu = `np.asarray`)."""
    if backend == "cpu":
        return np.asarray(x, dtype=np.float64)
    if backend == "cupy":
        return np.asarray(x, dtype=np.float64)  # DLPack: cupy -> numpy
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)  # torch


def gpu_matmul(a: np.ndarray, b: np.ndarray, backend: str = "auto") -> np.ndarray:
    """91.3: the dense (n,k) @ (k,m) op on the selected backend, returned
    as host numpy. "auto" = the first importable GPU backend, else CPU.

    Documented caveat (91.3): a GPU run is bit-DIFFERENT from CPU — a
    different device and a different float reduction order are both
    allowed; the G2 bit-exact pins are CPU-path pins by contract."""
    # "cpu" is always the native path; a named GPU backend is strict
    # (missing -> BackendUnavailable); "auto"/"gpu" degrade gracefully.
    if backend == "cpu":
        resolved = "cpu"
    elif backend in GPU_BACKENDS:
        resolved = detect_backend(backend)
    else:
        resolved = detect_backend("auto")
    if resolved == "cpu":
        return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    A = to_device(a, resolved)
    B = to_device(b, resolved)
    return to_host(A @ B, resolved)


# --- 91.2: fixed-order parallel map ----------------------------------------

def parallel_map(n: int, fn, workers: int = 1) -> list:
    """91.2: run `fn(i)` for `i in 0..n-1` and return the results in INPUT
    ORDER.

    `workers == 1` (the default) is the exact serial loop — the
    pre-v0.77 path, bit-identical (G2). `workers > 1` runs the independent
    units on a `ThreadPoolExecutor` (numpy releases the GIL inside its
    array/BLAS work; the units must be side-effect-free and share no
    mutable state — the bagged-tree fit is: each unit is one tree on its
    own bag, 91.2). `pool.map` preserves order, so the aggregation is the
    fixed legacy order regardless of scheduling."""
    if workers is None or isinstance(workers, bool) \
            or not isinstance(workers, int) or workers < 1:
        raise ValueError(
            f"parallel must be an int >= 1 (1 = serial, bit-identical), "
            f"got {workers!r} (SPEC.md 91.2)")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n} (SPEC.md 91.2)")
    if workers == 1:
        return [fn(i) for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda i: fn(i), range(n)))
