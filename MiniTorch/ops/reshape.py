from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_variable


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):  # type: ignore
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):  # type: ignore
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
