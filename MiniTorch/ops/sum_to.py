from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_variable


class SumTo(Function):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):  # type: ignore
        self.x_shape = x.shape
        from MiniTorch.utils.sumto import sumto

        y = sumto(x, self.shape)
        return y

    def backward(self, gy):  # type: ignore
        from MiniTorch.ops.broadcast import broadcast_to

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
