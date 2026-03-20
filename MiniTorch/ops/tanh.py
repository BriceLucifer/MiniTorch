from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        y = self.outputs[0]()  # type: ignore[index, misc]
        gx = gy * (1 - y * y)  # type: ignore[operator]
        return gx


def tanh(x: Variable) -> Variable:
    return Tanh()(x)  # type: ignore[return-value]
