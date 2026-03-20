import numpy as np


class Adam:
    """
    Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    parameters  : list of Variable
    lr          : step size (α)
    beta1       : first-moment decay (default 0.9)
    beta2       : second-moment decay (default 0.999)
    eps         : numerical stability term (default 1e-8)
    weight_decay: L2 regularisation coefficient
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in parameters]
        self._v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        # Bias-corrected learning rate
        lr_t = (
            self.lr
            * np.sqrt(1 - self.beta2 ** self.t)
            / (1 - self.beta1 ** self.t)
        )
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g ** 2
            p.data -= lr_t * self._m[i] / (np.sqrt(self._v[i]) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.clear_grad()
