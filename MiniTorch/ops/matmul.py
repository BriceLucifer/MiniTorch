from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class MatMul(Function):
    def forward(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = x.dot(w)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:  # type: ignore[override]
        x, W = self.inputs  # type: ignore[misc]
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

    def backward_array(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.input_data(0)
        weight = self.input_data(1)
        return gy.dot(weight.T), x.T.dot(gy)


def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)  # type: ignore[return-value]
