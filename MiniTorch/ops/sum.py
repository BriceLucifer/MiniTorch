from MiniTorch.core.function import Function
from MiniTorch.utils.reshape_sum_backward import reshape_sum_backward


class Sum(Function):
    def __init__(self, axis, keepdims: bool):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):  # type: ignore
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):  # type: ignore
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        from MiniTorch.ops.broadcast import broadcast_to

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)
