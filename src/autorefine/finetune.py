"""Fine-tuning support — checkpoint loading + warm start (SPEC.md 80, v0.66).

The user's entry point for "start from a trained model": a saved model
(``*.npz``, the plain-array ``save()`` format of any family, loadable with
``allow_pickle=False``) is loaded here, its spec reconstructed, and its
compatibility with a task checked. The improver then uses the model as the
baseline seed (scored, never retrained) and offers its weights as a warm
start for compatible candidates (``copy_weights``).

Pure module: stdlib + numpy + the existing model classes only (SPEC.md 3).
No import cycle — ``finetune`` depends on ``config`` + ``models``; the
trainer and the meta env may import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_SPEC, ModelSpec
from .models.conv1d import Conv1D
from .models.convnet import ConvNet
from .models.gam import GAM
from .models.gp import GP
from .models.knn import KNN
from .models.mlp import MLP
from .models.rnn import RNN
from .models.trees import BoostingEnsemble, TreeEnsemble

# Families that carry transferable weights (a warm start copies their layer
# weights). The non-parametric families (tree/boost/knn/gp/gam) memorize
# the training rows — there is nothing to copy (80.3.3).
# SPEC.md 83 (v0.69): the native temporal families carry weights too.
PARAMETRIC = ("mlp", "convnet", "conv1d", "rnn")


@dataclass(frozen=True)
class Checkpoint:
    """One loaded model + its reconstructed spec (SPEC.md 80.2)."""

    path: str
    family: str
    model: Any
    spec: ModelSpec
    head: str
    n_out: int


def _detect_family(files: set[str]) -> str:
    """Family by npz key set (80.2.1). Check order: the discriminating
    keys never collide between the families' ``save()`` formats.

    SPEC.md 83 (v0.69): the temporal families share ``C``/``n_layers``/``layer*_w``
    with convnet, so their discriminating keys go first — conv1d stores
    ``F`` + ``K`` (convnet has neither), convnet stores ``W`` (conv1d/rnn
    have neither), rnn stores ``T`` + ``H`` (conv1d has no ``H``; convnet
    has no ``T``). The mlp check (``n_layers`` + ``layer0_w``) stays last
    of the neural families — every family above has those keys too."""
    if "F" in files and "K" in files and "n_layers" in files:
        return "conv1d"
    if "C" in files and "W" in files and "n_layers" in files:
        return "convnet"
    if "T" in files and "H" in files and "n_layers" in files:
        return "rnn"
    if "X" in files and "k" in files:
        return "knn"
    if "omega" in files and "w" in files:
        return "gp"
    if "betas" in files and "p" in files:
        return "gam"
    if "n_trees" in files and "base" in files:
        return "boost"
    if "n_trees" in files and "max_depth" in files:
        return "tree"
    if "n_layers" in files and "layer0_w" in files:
        return "mlp"
    raise ValueError(
        f"unrecognized model checkpoint — keys {sorted(files)} do not "
        "match any AutoRefine model family (SPEC.md 80.2.1)")


def _spec_for(family: str, model: Any) -> ModelSpec:
    """Reconstruct a valid ``ModelSpec`` from a trained model (80.2.2):
    ``DEFAULT_SPEC`` with the family's fields overridden from the model's
    stored parameters, validated through ``ModelSpec.from_dict`` (an
    out-of-space value is a loud ``SpecError``, never a silent clip)."""
    d = DEFAULT_SPEC.to_dict()
    if family == "mlp":
        # layers[0] is input->first hidden; layers[:-1] are the hidden ones
        d["architecture"] = [int(w.shape[1]) for w, _b in model.layers[:-1]]
        d["activation"] = model.activation
        d["fourier_features"] = int(model.fourier_K)
    elif family == "convnet":
        d["model_family"] = "convnet"
        d["architecture"] = [int(model.c1), int(model.c2)]
        d["activation"] = model.activation
        d["init_scale"] = float(model.init_scale)
    elif family == "conv1d":  # SPEC.md 83.2: the temporal knobs are spec fields
        d["model_family"] = "conv1d"
        d["conv1d_filters"] = int(model.F)
        d["conv1d_kernel"] = int(model.K)
        d["activation"] = model.activation
        d["init_scale"] = float(model.init_scale)
    elif family == "rnn":  # SPEC.md 83.3: the Elman hidden width is a spec field
        d["model_family"] = "rnn"
        d["rnn_hidden"] = int(model.H)
        d["activation"] = model.activation
        d["init_scale"] = float(model.init_scale)
    elif family == "knn":
        d["model_family"] = "knn"
        d["knn_k"] = int(model.k)
        d["fourier_features"] = int(model.fourier_K)
    elif family == "gp":
        d["model_family"] = "gp"
        d["gp_length_scale"] = float(model.length_scale)
        d["fourier_features"] = int(model.fourier_K)
    elif family == "gam":
        d["model_family"] = "gam"
        d["weight_decay"] = float(model.weight_decay)
        d["gam_interactions"] = len(model.interactions)
    elif family in ("tree", "boost"):
        d["model_family"] = family
        d["architecture"] = [int(model.max_depth)]
        d["train_steps"] = max(200, len(model.trees) * 100)
    else:  # pragma: no cover — _detect_family only yields known families
        raise ValueError(f"unknown family {family!r} (SPEC.md 80.2.2)")
    return ModelSpec.from_dict(d)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Load a saved model + reconstruct its spec (SPEC.md 80.2).

    ``ValueError`` on a missing file, an unrecognized key set, or a
    spec that no longer validates (fail-loud — the env's ``ValueError``
    contract the CLI/app surface as a friendly error, 80.2.4)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such model file: {p} (SPEC.md 80.2.4)")
    with np.load(p, allow_pickle=False) as z:
        family = _detect_family(set(z.files))
    loaders = {
        "mlp": MLP.load, "convnet": ConvNet.load, "conv1d": Conv1D.load,
        "rnn": RNN.load, "knn": KNN.load,
        "gp": GP.load, "gam": GAM.load,
        "tree": TreeEnsemble.load, "boost": BoostingEnsemble.load,
    }
    model = loaders[family](str(p))
    spec = _spec_for(family, model)
    return Checkpoint(
        path=str(p), family=family, model=model, spec=spec,
        head=model.head, n_out=int(model.n_out))


def validate_for_task(cp: Checkpoint, task: Any) -> tuple[bool, str]:
    """Compatibility of a checkpoint with a task (SPEC.md 80.2.3).

    ``head`` + ``n_out`` must match for every family; parametric families
    additionally check the input contract: mlp/gp against
    ``state_dim * (2K+1)`` (the model owns its spectral map, 78.2.1), gam
    against ``state_dim``, convnet against the task's grid. Returns
    ``(ok, reason)`` — the reason is the user-facing failure text."""
    if cp.head != task.head:
        return False, (f"head mismatch: the model's head is {cp.head!r} "
                       f"but the task's is {task.head!r}")
    if cp.n_out != task.n_outputs:
        return False, (f"output count mismatch: the model has {cp.n_out} "
                       f"output(s) but the task has {task.n_outputs}")
    if cp.family == "mlp":
        K = int(cp.model.fourier_K)
        want = int(task.state_dim) * (2 * K + 1)
        if cp.model.in_dim != want:
            return False, (f"input dimension mismatch: the model expects "
                           f"{cp.model.in_dim} feature(s) but the task has "
                           f"{want} (with its spectral expansion)")
    elif cp.family == "convnet":
        if not getattr(task, "grid_capable", False):
            return False, "the task is not grid-capable (no image/audio grid)"
        grid = tuple(getattr(task, "feature_grid", ()) or ())
        want = (int(cp.model.C), int(cp.model.H), int(cp.model.W))
        if grid != want:
            return False, (f"grid mismatch: the model expects {want} "
                           f"but the task's grid is {grid}")
    elif cp.family in ("conv1d", "rnn"):  # SPEC.md 83.1: the sequence layout
        if "sequence" not in getattr(task, "capabilities", frozenset()):
            return False, ("the task has no 'sequence' capability (no "
                           "(timesteps, channels) layout)")
        T = getattr(task, "T", None)
        C = getattr(task, "C", None)
        if T is not None and C is not None:
            want = (int(T), int(C))
            have = (int(cp.model.T), int(cp.model.C))
            if want != have:
                return False, (f"sequence layout mismatch: the model expects "
                               f"T={have[0]} timesteps and {have[1]} channel(s) "
                               f"but the task has T={want[0]} and {want[1]}")
    elif cp.family == "gp":
        K = int(cp.model.fourier_K)
        want = int(task.state_dim) * (2 * K + 1)
        if cp.model.in_dim != want:
            return False, (f"input dimension mismatch: the model expects "
                           f"{cp.model.in_dim} feature(s) but the task has "
                           f"{want} (with its spectral expansion)")
    elif cp.family == "gam":
        if cp.model.in_dim != int(task.state_dim):
            return False, (f"input dimension mismatch: the model expects "
                           f"{cp.model.in_dim} feature(s) but the task has "
                           f"{task.state_dim}")
    return True, ""


def can_warm_start(cp: Checkpoint, spec: ModelSpec, n_out: int, head: str,
                   in_dim: int | None = None) -> bool:
    """Whether `cp`'s weights can seed a model of `spec` (SPEC.md 80.3).

    Parametric families only (80.3.3): the family, head, and output count
    must match; mlp additionally requires the same activation, the same
    hidden architecture, the same spectral order, and (when known) the
    same input width. Non-parametric families always return False — the
    candidate trains fresh, exactly as pre-v0.66."""
    if cp.family != spec.model_family:
        return False
    if cp.head != head or cp.n_out != n_out:
        return False
    if cp.family not in PARAMETRIC:
        return False
    m = cp.model
    if m.activation != spec.activation:
        return False
    if cp.family == "mlp":
        hidden = tuple(int(w.shape[1]) for w, _b in m.layers[:-1])
        if hidden != tuple(spec.architecture):
            return False
        if int(m.fourier_K) != int(spec.fourier_features):
            return False
        if in_dim is not None and m.in_dim != int(in_dim):
            return False
    if cp.family == "conv1d":  # SPEC.md 83.2: the temporal knobs must match
        if int(m.F) != int(spec.conv1d_filters) or int(m.K) != int(spec.conv1d_kernel):
            return False
    if cp.family == "rnn":  # SPEC.md 83.3: the Elman hidden width must match
        if int(m.H) != int(spec.rnn_hidden):
            return False
    return True


def copy_weights(dst: Any, src: Any) -> bool:
    """Copy `src`'s layer weights into `dst` (SPEC.md 80.3.2).

    Every layer shape is checked *before* any mutation: on a mismatch
    nothing is copied and `dst` is left exactly as built (a failed warm
    start must never leave a half-seeded model). ``forward()`` rebuilds
    its internal cache on entry, so the stale-cache state is irrelevant."""
    if (not hasattr(dst, "layers") or not hasattr(src, "layers")
            or len(dst.layers) != len(src.layers)):
        return False
    for (dw, db), (sw, sb) in zip(dst.layers, src.layers):
        if dw.shape != sw.shape or db.shape != sb.shape:
            return False
    dst.layers = [(w.copy(), b.copy()) for w, b in src.layers]
    return True
