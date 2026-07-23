"""Train a dense classifier without a Python mini-batch loop.

Run with:
    uv run python examples/native_training.py
"""

import numpy as np

from MiniTorch.native import train
from MiniTorch.nn import Linear, ReLU, Sequential


rng = np.random.default_rng(7)
x = rng.normal(size=(4_096, 32)).astype(np.float32)
teacher = rng.normal(size=(32, 4)).astype(np.float32)
labels = np.argmax(x @ teacher, axis=1).astype(np.int64)

model = Sequential(
    Linear(32, 64),
    ReLU(),
    Linear(64, 4),
)

history = train(
    model,
    x,
    labels,
    epochs=10,
    batch_size=128,
    lr=3e-3,
)

print(f"steps: {history.steps}")
print(f"first loss: {history.losses[0]:.4f}")
print(f"last loss:  {history.losses[-1]:.4f}")
