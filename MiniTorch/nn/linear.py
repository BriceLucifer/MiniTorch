import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn.module import Module
from MiniTorch.ops.matmul import matmul


class Linear(Module):
    """
    Fully-connected linear layer: y = x @ W + b

    Parameters
    ----------
    in_features  : input dimension
    out_features : output dimension
    bias         : whether to include a bias term (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # He (Kaiming) initialisation — good default for ReLU networks
        scale = np.sqrt(2.0 / in_features)
        self.W = Variable(
            (np.random.randn(in_features, out_features) * scale).astype(np.float64),
            name="W",
        )
        self.b = None
        if bias:
            self.b = Variable(
                np.zeros(out_features, dtype=np.float64),
                name="b",
            )

    def forward(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b
        return y

    def __repr__(self):
        in_f, out_f = self.W.shape
        return f"Linear(in={in_f}, out={out_f}, bias={self.b is not None})"
