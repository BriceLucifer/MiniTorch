"""Small reverse-mode autodiff example.

Run with:
    uv run python examples/basic_autograd.py
"""

import numpy as np

from MiniTorch import Variable


x = Variable(np.array([[2.0]], dtype=np.float32), name="x")
w = Variable(np.array([[3.0]], dtype=np.float32), name="w")
b = Variable(np.array([[1.0]], dtype=np.float32), name="b")

y = x * w + b
loss = (y - 10.0) ** 2
loss.backward()

print(f"loss:   {loss.data.item():.1f}")
print(f"x.grad: {x.grad.data.item():.1f}")
print(f"w.grad: {w.grad.data.item():.1f}")
print(f"b.grad: {b.grad.data.item():.1f}")
