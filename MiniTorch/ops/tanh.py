import numpy as np

from MiniTorch.core.function import Function


class Tanh(Function):
    def forward(self, x):  # type: ignore
        y = np.tanh(x)
        return y

    def backward(self, gy):  # type: ignore
        y = self.outputs[0]()  # type: ignore
        gx = gy * (1 - y * y)  # type: ignore
        return gx


def tanh(x):
    return Tanh()(x)
