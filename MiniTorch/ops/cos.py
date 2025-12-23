import numpy as np

from MiniTorch.core.function import Function


class Cos(Function):
    def forward(self, x):  # type: ignore
        y = np.cos(x)
        return y

    def backward(self, gy):  # type: ignore
        (x,) = self.inputs  # type: ignore
        from MiniTorch.ops.sin import sin

        gx = gy * -sin(x)  # type: ignore
        return gx


def cos(x):
    return Cos()(x)
