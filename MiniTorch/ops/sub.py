from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:  # type: ignore[override]
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            from MiniTorch.ops.sum_to import sum_to

            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)

        return gx0, gx1


def sub(x0: Variable, x1: Variable | float | int) -> Variable:
    x1_arr = as_array(x1)
    return Sub()(x0, x1_arr)  # type: ignore[return-value]


def rsub(x0: Variable, x1: Variable | float | int) -> Variable:
    x1_arr = as_array(x1)
    return Sub()(x1_arr, x0)  # type: ignore[return-value]
