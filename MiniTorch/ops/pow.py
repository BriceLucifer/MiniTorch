from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Pow(Function):
    def __init__(self, c: float | int) -> None:
        super().__init__()
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = x**self.c
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        (x,) = self.inputs  # type: ignore[misc]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: Variable, c: float | int) -> Variable:
    return Pow(c)(x)  # type: ignore[return-value]
