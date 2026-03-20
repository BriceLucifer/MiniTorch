from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return -x

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        return -gy


def neg(x: Variable) -> Variable:
    return Neg()(x)  # type: ignore[return-value]
