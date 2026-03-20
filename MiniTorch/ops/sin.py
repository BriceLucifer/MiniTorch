from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        (x,) = self.inputs  # type: ignore[misc]
        from MiniTorch.ops.cos import cos

        gx = gy * cos(x)  # type: ignore[operator]
        return gx


def sin(x: Variable) -> Variable:
    return Sin()(x)  # type: ignore[return-value]
