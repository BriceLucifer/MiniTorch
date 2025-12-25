from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Sub(Function):
    def forward(self, x0, x1):  # type: ignore
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):  # type: ignore
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            from MiniTorch.ops.sum_to import sum_to

            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)

        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
