from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        (x,) = self.inputs  # type: ignore[misc]
        from MiniTorch.ops.sin import sin

        gx = gy * -sin(x)  # type: ignore[operator]
        return gx


def cos(x: Variable) -> Variable:
    return Cos()(x)  # type: ignore[return-value]
