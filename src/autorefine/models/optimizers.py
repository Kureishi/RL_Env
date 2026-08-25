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
        v = state.get("v", np.zeros_like(param))
        v = self.mu * v + grad
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
        m = state.get("m", np.zeros_like(param))
        v = state.get("v", np.zeros_like(param))
        m = self.b1 * m + (1 - self.b1) * grad
        v = self.b2 * v + (1 - self.b2) * grad * grad
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
