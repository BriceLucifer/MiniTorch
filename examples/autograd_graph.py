"""Generate an interactive graph for one autograd expression.

Run with:
    uv run python examples/autograd_graph.py
"""

import numpy as np

from MiniTorch import Variable, visualize_graph


x = Variable(np.array([[2.0]], dtype=np.float32), name="x")
w = Variable(np.array([[3.0]], dtype=np.float32), name="w")
b = Variable(np.array([[1.0]], dtype=np.float32), name="b")

y = x * w + b
y.name = "prediction"
loss = (y - 10.0) ** 2
loss.name = "loss"
loss.backward(retain_grad=True)

visualize_graph(loss, filename="autograd_graph.html")
print("Wrote autograd_graph.html")
