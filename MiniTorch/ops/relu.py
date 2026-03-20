from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class ReLU(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.maximum(0, x)

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        x, = self.inputs  # type: ignore[misc]
        mask = (x.data > 0).astype(x.data.dtype)  # type: ignore[union-attr, operator]
        return gy * mask  # type: ignore[return-value]


def relu(x: Variable) -> Variable:
    return ReLU()(x)  # type: ignore[return-value]
