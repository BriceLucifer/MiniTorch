import numpy as np
import numpy.typing as npt


def as_array(x) -> npt.NDArray:
    if np.isscalar(x):
        return np.array(x)
    else:
        return x
