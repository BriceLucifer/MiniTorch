from typing import override

import numpy.typing as npt

from MiniTorch.core import Function, Variable
from MiniTorch.utils.type_check import as_array


class Square(Function):
    """
    implement a Square Function based on Funtion object
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):  # type: ignore[override]
        return x**2

    def backward(self, gy):  # type: ignore[override]
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


# more easy to use
def square(x):
    """
    square():
        params:
            x: Variable
        return:
            Variable
    """
    return Square()(x)
