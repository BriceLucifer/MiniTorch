from __future__ import annotations

import numpy as np

from MiniTorch.core.variable import Variable


class SGD:
    """
    Stochastic Gradient Descent (with optional momentum).

    Parameters
    ----------
    parameters   : list of Variable
    lr           : learning rate
    momentum     : momentum factor (0 = plain SGD)
    weight_decay : L2 regularisation coefficient
    """

    def __init__(
        self,
        parameters: list[Variable],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for v, p in zip(self._velocity, self.parameters):
            if p.grad is None:
                continue
            g: np.ndarray = p.grad.data  # type: ignore[assignment]
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data  # type: ignore[operator]
            v[:] = self.momentum * v + g
            p.data -= self.lr * v  # type: ignore[operator]

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.clear_grad()
