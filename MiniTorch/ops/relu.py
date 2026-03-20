import numpy as np

from MiniTorch.core.function import Function


class ReLU(Function):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(x.data.dtype)
        return gy * mask


def relu(x):
    return ReLU()(x)
