from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.reshape_sum_backward import reshape_sum_backward

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Sum(Function):
    def __init__(self, axis: int | tuple[int, ...] | None, keepdims: bool) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        from MiniTorch.ops.broadcast import broadcast_to

        gx = broadcast_to(gy, self.x_shape)
        return gx  # type: ignore[return-value]


def sum(
    x: Variable,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Variable:
    return Sum(axis=axis, keepdims=keepdims)(x)  # type: ignore[return-value]
