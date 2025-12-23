import numpy as np

from MiniTorch.core.function import Function


class Sin(Function):
    def forward(self, x):  # type: ignore
        y = np.sin(x)
        return y

    def backward(self, gy):  # type: ignore
        (x,) = self.inputs  # type: ignore
        from MiniTorch.ops.cos import cos

        gx = gy * cos(x)  # type: ignore
        return gx


def sin(x):
    return Sin()(x)
