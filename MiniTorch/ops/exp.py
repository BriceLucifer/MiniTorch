import numpy as np

from MiniTorch.core import Function


class Exp(Function):
    """
    implement a Exp Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):  # type: ignore[override]
        return np.exp(x)

    def backward(self, gy):  # type: ignore[override]
        x = self.inputs[0].data  # type: ignore
        return np.exp(x) * gy


def exp(x):
    """
    exp():
        params:
            x: Variable
        return:
            Variable
    """
    return Exp()([x])
