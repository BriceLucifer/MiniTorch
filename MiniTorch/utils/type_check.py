from __future__ import annotations

from typing import Any

import numpy as np


def as_array(x: Any) -> np.ndarray:
    """Wrap scalars in a 0-d NumPy array; pass arrays through unchanged."""
    if np.isscalar(x):
        return np.array(x)
    return x  # type: ignore[return-value]


def as_variable(x: Any) -> Any:
    """Return x unchanged if it is already a Variable, otherwise wrap it."""
    from MiniTorch.core.variable import Variable

    if isinstance(x, Variable):
        return x
    return Variable(x)
