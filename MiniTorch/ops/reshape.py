from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_variable

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        return reshape(gy, self.x_shape)


def reshape(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)  # type: ignore[return-value]
