from MiniTorch.core.function import Function


class MatMul(Function):
    def forward(self, x, w):  # type: ignore
        y = x.dot(w)
        return y

    def backward(self, gy):  # type: ignore
        x, W = self.inputs  # type: ignore
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)
