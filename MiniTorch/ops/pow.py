from MiniTorch.core.function import Function


class Pow(Function):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):  # type: ignore
        y = x**self.c
        return y

    def backward(self, gy):  # type:ignore
        x = self.inputs[0].data  # type: ignore
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)
