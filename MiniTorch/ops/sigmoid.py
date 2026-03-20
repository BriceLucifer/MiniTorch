from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        # Numerically stable: avoid overflow for large negative x
        y = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        y = self.outputs[0]()  # type: ignore[index]  # cached output
        return gy * y * (1 - y)  # type: ignore[return-value]


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)  # type: ignore[return-value]
