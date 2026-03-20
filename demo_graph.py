"""
demo_graph.py — Computation graph visualisation demo.

Run:  uv run python demo_graph.py

Produces:
    graph_simple.html  — basic expression: loss = (x*w + b - target)^2
    graph_mlp.html     — 2-layer MLP forward + backward pass
"""
import numpy as np

import MiniTorch as mt
from MiniTorch import visualize_graph
from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy


# ── Demo 1: simple expression ────────────────────────────────────────────────
print("=" * 55)
print("  Demo 1: simple expression")
print("  loss = (x*w + b - target)^2")
print("=" * 55)

x      = Variable(np.array([[2.0]]), name="x")
w      = Variable(np.array([[3.0]]), name="w")
b      = Variable(np.array([[1.0]]), name="b")
target = Variable(np.array([[10.0]]), name="target")

y       = x * w + b;   y.name    = "y"
loss    = (y - target) ** 2;  loss.name = "loss"

loss.backward(retain_grad=True)

print(f"  y      = {y.data.item():.1f}   (expected 7)")
print(f"  loss   = {loss.data.item():.1f}  (expected 9)")
print(f"  x.grad = {x.grad.data.item():.1f}  (expected -18)")
print(f"  w.grad = {w.grad.data.item():.1f}  (expected -12)")
print(f"  b.grad = {b.grad.data.item():.1f}   (expected  -6)")

visualize_graph(loss, filename="graph_simple.html")
print("  Saved → graph_simple.html\n")


# ── Demo 2: 2-layer MLP ──────────────────────────────────────────────────────
print("=" * 55)
print("  Demo 2: 2-layer MLP  (Linear→ReLU→Linear→SoftmaxCE)")
print("=" * 55)

np.random.seed(0)
l1 = Linear(4, 8)
l2 = Linear(8, 3)

x2     = Variable(np.random.randn(2, 4).astype(np.float64), name="x")
labels = Variable(np.array([0, 2]))

h      = relu(l1(x2));   h.name      = "hidden"
logits = l2(h);          logits.name = "logits"
loss2  = softmax_cross_entropy(logits, labels);  loss2.name = "loss"

loss2.backward(retain_grad=True)

print(f"  loss             = {loss2.data.item():.4f}")
print(f"  l1.W grad norm   = {np.linalg.norm(l1.W.grad.data):.4f}")
print(f"  l2.W grad norm   = {np.linalg.norm(l2.W.grad.data):.4f}")

visualize_graph(loss2, filename="graph_mlp.html")
print("  Saved → graph_mlp.html\n")

print("Open both HTML files in your browser to explore the graph.")
print("Hover over any node for full details. Red edges = gradient computed.")
