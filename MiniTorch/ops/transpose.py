import numpy as np

from MiniTorch.core.function import Function


class Transpose(Function):
    def forward(self, x):  # type: ignore
        y = np.transpose(x)
        return y

    def backward(self, gy):  # type: ignore
        gx = transpose(gy)
        return gx


def transpose(x):
    return Transpose()(x)
