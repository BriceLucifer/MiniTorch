import numpy as np


def as_array(x):
    """
    check if x is a numpy array
    """
    if np.isscalar(x):
        return np.array(x)
    else:
        return x


def as_variable(x):
    from MiniTorch.core.variable import Variable

    if isinstance(x, Variable):
        return x
    else:
        return Variable(x)
