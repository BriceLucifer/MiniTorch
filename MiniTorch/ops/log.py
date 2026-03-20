from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.log(x)

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        x, = self.inputs  # type: ignore[misc]
        return gy / x  # type: ignore[return-value]


def log(x: Variable) -> Variable:
    return Log()(x)  # type: ignore[return-value]
