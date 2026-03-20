import numpy as np

from MiniTorch.core.function import Function


class Sigmoid(Function):
    def forward(self, x):
        # Numerically stable: avoid overflow for large negative x
        y = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # cached output
        return gy * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)
