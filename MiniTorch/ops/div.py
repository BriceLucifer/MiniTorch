from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Div(Function):
    def forward(self, x0, x1):  # type: ignore
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):  # type: ignore
        x0, x1 = self.inputs  # type: ignore
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)  # type: ignore
        if x0.shape != x1.shape:
            from MiniTorch.ops.sum_to import sum_to

            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)

        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)
