from __future__ import annotations

import numpy as np

from MiniTorch.core.variable import Variable


class Adam:
    """
    Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    parameters   : list of Variable
    lr           : step size (α)
    beta1        : first-moment decay (default 0.9)
    beta2        : second-moment decay (default 0.999)
    eps          : numerical stability term (default 1e-8)
    weight_decay : L2 regularisation coefficient
    """

    def __init__(
        self,
        parameters: list[Variable],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t: int = 0
        self._m: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]
        self._v: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        self.t += 1
        lr_t = (
            self.lr
            * np.sqrt(1 - self.beta2 ** self.t)
            / (1 - self.beta1 ** self.t)
        )
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g: np.ndarray = p.grad.data  # type: ignore[assignment]
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data  # type: ignore[operator]
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g ** 2
            p.data -= lr_t * self._m[i] / (np.sqrt(self._v[i]) + self.eps)  # type: ignore[operator]

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.clear_grad()
