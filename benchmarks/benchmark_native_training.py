"""Compare eager MiniTorch training with the compiled Sequential trainer."""
from __future__ import annotations

import time

import numpy as np

from MiniTorch import Variable
from MiniTorch.native import train as native_train
from MiniTorch.nn import Linear, ReLU, Sequential
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim import Adam


def make_model() -> Sequential:
    return Sequential(
        Linear(64, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )


def eager_train(
    model: Sequential,
    x: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
) -> None:
    optimizer = Adam(model.parameters())
    for _ in range(epochs):
        for start in range(0, len(x), batch_size):
            logits = model(Variable(x[start : start + batch_size]))
            loss = softmax_cross_entropy(
                logits, Variable(labels[start : start + batch_size])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8192, 64)).astype(np.float32)
    labels = rng.integers(0, 10, size=8192, dtype=np.int64)
    epochs = 3
    batch_size = 128

    native_model = make_model()
    start = time.perf_counter()
    native_train(
        native_model,
        x,
        labels,
        epochs=epochs,
        batch_size=batch_size,
    )
    native_seconds = time.perf_counter() - start

    eager_model = make_model()
    start = time.perf_counter()
    eager_train(eager_model, x, labels, epochs, batch_size)
    eager_seconds = time.perf_counter() - start

    print(f"Native compiled loop : {native_seconds:.4f}s")
    print(f"Eager autograd loop  : {eager_seconds:.4f}s")
    print(f"Speedup              : {eager_seconds / native_seconds:.2f}x")


if __name__ == "__main__":
    main()
