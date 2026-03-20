from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_variable

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x_shape = x.shape
        from MiniTorch.utils.sumto import sumto

        y = sumto(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        from MiniTorch.ops.broadcast import broadcast_to

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)  # type: ignore[return-value]
