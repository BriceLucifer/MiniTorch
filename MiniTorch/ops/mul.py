from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Mul(Function):
    def forward(self, x0, x1):  # type:ignore
        return x0 * x1

    def backward(self, gy):  # type: ignore
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
