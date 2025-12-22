from MiniTorch.core.function import Function


class Neg(Function):
    def forward(self, x):  # type: ignore
        return -x

    def backward(self, gy):  # type: ignore
        return -gy


def neg(x):
    return Neg()(x)
