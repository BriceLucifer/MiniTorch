from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Sub(Function):
    def forward(self, x0, x1):  # type: ignore
        y = x0 - x1
        return y

    def backward(self, gy):  # type: ignore
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
