"""Cross-modality warm start (SPEC.md 99, v0.85 C2).

Shape-mapped weight transfer between two `.layers` models of different
modalities — e.g. a trained `conv1d` (temporal) champion seeding a
`convnet` (spatial) candidate, or `rnn` -> `mlp`. Where the 80.3
`copy_weights` contract is strict (all-or-nothing, identical shapes),
this transfer is **partial by design**: cross-modality rarely shares
full layer shapes, so each slot is transferred when it can and honestly
reported when it cannot.

Per-slot rules (slot i of the target):
  1. the source's slot i has an **identical** shape        -> copy;
  2. both weight and bias have an **equal element count** -> reshape
     (row-major) and copy;
  3. otherwise                                              -> the
     target's slot stays at its (seeded) init.

A pair is never half-transferred (weight reshaped but bias not — the
two must move together). Deterministic: pure array copies, no RNG (G2).
"""
from __future__ import annotations

from typing import Any

__all__ = ["cross_modal_warm_start"]


def cross_modal_warm_start(dst: Any, src: Any) -> dict:
    """Transfer `src`'s weights into `dst` slot by slot (see the module
    docstring). Returns a JSON-safe report:

    {"copied": n, "reshaped": n, "kept_init": n,
     "slots": [{"index": i, "mode": "copy"|"reshape"|"init",
                "w_shape": [...]}...]}

    `dst` and `src` must expose `.layers` (lists of (w, b)); anything
    else is a `ValueError` (fail loud, like 80.3)."""
    if not hasattr(dst, "layers") or not hasattr(src, "layers"):
        raise ValueError(
            "cross_modal_warm_start needs both models to expose .layers "
            "(SPEC.md 99)")
    src_layers = list(src.layers)
    slots: list[dict] = []
    n_copied = n_reshaped = n_init = 0
    for i, (dw, db) in enumerate(list(dst.layers)):
        if i >= len(src_layers):
            slots.append({"index": i, "mode": "init",
                          "w_shape": list(dw.shape)})
            n_init += 1
            continue
        sw, sb = src_layers[i]
        if dw.shape == sw.shape and db.shape == sb.shape:
            dst.layers[i] = (sw.copy(), sb.copy())
            slots.append({"index": i, "mode": "copy",
                          "w_shape": list(dw.shape)})
            n_copied += 1
        elif dw.size == sw.size and db.size == sb.size:
            dst.layers[i] = (sw.reshape(dw.shape).copy(),
                             sb.reshape(db.shape).copy())
            slots.append({"index": i, "mode": "reshape",
                          "w_shape": list(dw.shape)})
            n_reshaped += 1
        else:
            slots.append({"index": i, "mode": "init",
                          "w_shape": list(dw.shape)})
            n_init += 1
    return {"copied": n_copied, "reshaped": n_reshaped, "kept_init": n_init,
            "slots": slots}
