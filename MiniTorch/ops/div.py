from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Div(Function):
    def forward(self, x0, x1):  # type: ignore
        y = x0 / x1
        return y

    def backward(self, gy):  # type: ignore
        x0, x1 = self.inputs[0].data, self.inputs[1].data  # type: ignore
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)
