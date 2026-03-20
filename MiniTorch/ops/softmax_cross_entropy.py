from __future__ import annotations

import numpy as np

from MiniTorch.core.function import Function
from MiniTorch.core.variable import Variable


class SoftmaxCrossEntropy(Function):
    """
    Fused numerically-stable Softmax + Cross-Entropy loss.

    forward(x, t):
        x : (N, C) logits
        t : (N,)   integer class labels
    returns scalar loss.
    """

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:  # type: ignore[override]
        N = x.shape[0]
        # Log-softmax (subtract max for numerical stability)
        x_s = x - x.max(axis=1, keepdims=True)
        log_p = x_s - np.log(np.exp(x_s).sum(axis=1, keepdims=True))
        t_int = t.ravel().astype(int)
        loss = -log_p[np.arange(N), t_int].sum() / np.float64(N)
        return loss

    def backward(self, gy: Variable) -> tuple[Variable, None]:  # type: ignore[override]
        x, t = self.inputs  # type: ignore[misc]
        N, C = x.data.shape  # type: ignore[union-attr]
        t_int = t.data.ravel().astype(int)  # type: ignore[union-attr]

        # Recompute softmax from stored input
        x_data: np.ndarray = x.data  # type: ignore[union-attr, assignment]
        x_s = x_data - x_data.max(axis=1, keepdims=True)
        softmax = np.exp(x_s) / np.exp(x_s).sum(axis=1, keepdims=True)

        # Gradient: (softmax - one_hot) / N * upstream_grad
        one_hot = np.zeros_like(softmax)
        one_hot[np.arange(N), t_int] = 1
        dx = (softmax - one_hot) * gy.data / N

        # Labels have no meaningful gradient
        return Variable(dx), None


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy()(x, t)  # type: ignore[return-value]
