from MiniTorch.core.function import Function


class MeanSquaredError(Function):
    def forward(self, x0, x1):  # type: ignore
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):  # type: ignore
        x0, x1 = self.inputs  # type: ignore
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
