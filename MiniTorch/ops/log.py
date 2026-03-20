import numpy as np

from MiniTorch.core.function import Function


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x, = self.inputs
        return gy / x


def log(x):
    return Log()(x)
