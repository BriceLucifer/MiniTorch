from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from MiniTorch.core import Function
from MiniTorch.utils.type_check import as_array

if TYPE_CHECKING:
    from MiniTorch.core.variable import Variable


class Square(Function):
    """
    implement a Square Function based on Funtion object
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return x**2

    def backward(self, gy: Variable) -> Variable:  # type: ignore[override]
        x = self.inputs[0].data  # type: ignore[index, union-attr]
        gx = 2 * x * gy  # type: ignore[operator]
        return gx  # type: ignore[return-value]


# more easy to use
def square(x: Variable) -> Variable:
    """
    square():
        params:
            x: Variable
        return:
            Variable
    """
    return Square()(x)  # type: ignore[return-value]
