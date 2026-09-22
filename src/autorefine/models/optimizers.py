"""NumPy optimizers: SGD, momentum, Adam. All deterministic given the input."""
from __future__ import annotations

from typing import Any

import numpy as np


class SGD:
    name = "sgd"

    def step(self, param: np.ndarray, grad: np.ndarray, state: Any, t: int) -> None:
        param -= self.lr * grad

    def __init__(self, lr: float) -> None:
        self.lr = lr


class Momentum:
    name = "momentum"
    mu = 0.9

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, param: np.ndarray, grad: np.ndarray, state: dict, t: int) -> None:
        # SPEC.md 70: lazy default — `dict.get(key, default)` evaluates the
        # default on EVERY call; create the zero vector only on the first step
        # (identical semantics, no per-step allocation).
        v = state.get("v")
        if v is None:
            v = np.zeros_like(param)
        # SPEC.md 74: in-place moment update — element-wise bit-identical to
        # `v = self.mu * v + grad` (same two ops, same order, per element),
        # with one fewer temporary allocation.
        v *= self.mu
        v += grad
        state["v"] = v
        param -= self.lr * v


class Adam:
    name = "adam"
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, param: np.ndarray, grad: np.ndarray, state: dict, t: int) -> None:
        # SPEC.md 70: lazy defaults (see Momentum.step) — identical semantics.
        m = state.get("m")
        if m is None:
            m = np.zeros_like(param)
        v = state.get("v")
        if v is None:
            v = np.zeros_like(param)
        # SPEC.md 74: in-place moment updates — element-wise bit-identical to
        # `m = b1*m + (1-b1)*grad` and `v = b2*v + (1-b2)*grad*grad` (same
        # ops, same per-element order), with two fewer temporary
        # allocations per call.
        m *= self.b1
        m += (1 - self.b1) * grad
        v *= self.b2
        vg = (1 - self.b2) * grad
        vg *= grad
        v += vg
        mhat = m / (1 - self.b1 ** t)
        vhat = v / (1 - self.b2 ** t)
        state["m"] = m
        state["v"] = v
        param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def make_optimizer(name: str, lr: float):
    from ..config import SpecError

    try:
        return {"sgd": SGD, "momentum": Momentum, "adam": Adam}[name](lr)
    except KeyError:
        raise SpecError(f"unknown optimizer {name!r}") from None
