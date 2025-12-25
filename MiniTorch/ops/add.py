from MiniTorch.core.function import Function
from MiniTorch.utils.type_check import as_array


class Add(Function):
    def forward(self, x0, x1):  # type: ignore[override]
        """
        params:
            xs: a list of inputs data
        return:
            tuple with y (y, )
        """
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):  # type: ignore[override]
        """
        params:
            gy: a grad based on additon rule
        return:
            two value with the same gy
        """
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            from MiniTorch.ops.sum_to import sum_to

            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)

        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)
