from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Transpose(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.transpose(x)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        gx = transpose(gy)
        return gx


def transpose(x: Variable) -> Variable:
    return Transpose()(x)  # type: ignore[return-value]
