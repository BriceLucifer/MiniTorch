from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 * x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:  # type: ignore[override]
        x0, x1 = self.inputs  # type: ignore[misc]
        gx0, gx1 = gy * x1, gy * x0
        if x0.shape != x1.shape:  # for broadcast
            from MiniTorch.ops.sum_to import sum_to

            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0: Variable, x1: Variable | float | int) -> Variable:
    x1_arr = as_array(x1)
    return Mul()(x0, x1_arr)  # type: ignore[return-value]
