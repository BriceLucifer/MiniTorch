from __future__ import annotations

import numpy as np


class DataLoader:
    """
    Simple mini-batch iterator over (x, y) numpy arrays.

    Parameters
    ----------
    x          : feature array, shape (N, ...)
    y          : label array, shape (N, ...)
    batch_size : number of samples per batch
    shuffle    : whether to shuffle before each epoch
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        assert len(x) == len(y), "x and y must have the same length"
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(x)

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n, self.batch_size):
            idx = indices[start : start + self.batch_size]
            yield self.x[idx], self.y[idx]

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.n + self.batch_size - 1) // self.batch_size
