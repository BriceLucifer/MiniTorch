from MiniTorch.core.function import Function


class Add(Function):
    def forward(self, x0, x1):  # type: ignore[override]
        """
        params:
            xs: a list of inputs data
        return:
            tuple with y (y, )
        """
        y = x0 + x1
        return y

    def backward(self, gy):  # type: ignore[override]
        """
        params:
            gy: a grad based on additon rule
        return:
            two value with the same gy
        """
        return gy, gy
