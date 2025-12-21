import numpy as np


def as_array(x):
    """
    check if x is a numpy array
    """
    if np.isscalar(x):
        return np.array(x)
    else:
        return x
