from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Exp(Function):
    """
    implement a Exp Function
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.exp(x)

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        x = self.inputs[0].data  # type: ignore[index, union-attr]
        return np.exp(x) * gy  # type: ignore[return-value, arg-type]


def exp(x: Variable) -> Variable:
    """
    exp():
        params:
            x: Variable
        return:
            Variable
    """
    return Exp()(x)  # type: ignore[return-value]
