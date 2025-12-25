import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_variable


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):  # type: ignore
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):  # type: ignore
        from MiniTorch.ops.sum_to import sum_to

        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
