class SGD:
    """
    Stochastic Gradient Descent (with optional momentum).

    Parameters
    ----------
    parameters : list of Variable
    lr         : learning rate
    momentum   : momentum factor (0 = plain SGD)
    weight_decay : L2 regularisation coefficient
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        import numpy as np

        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for v, p in zip(self._velocity, self.parameters):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            v[:] = self.momentum * v + g
            p.data -= self.lr * v

    def zero_grad(self):
        for p in self.parameters:
            p.clear_grad()
