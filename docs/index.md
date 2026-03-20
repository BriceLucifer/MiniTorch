---
layout: home

hero:
  name: MiniTorchBR
  text: Autograd from scratch
  tagline: A lightweight PyTorch-inspired deep learning framework built on NumPy — learn how backpropagation really works.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/BriceLucifer/MiniTorch

features:
  - icon: ⚡
    title: Reverse-mode Autograd
    details: Full automatic differentiation engine with dynamic computation graphs. Every operation tracks its gradient function for efficient backprop.
  - icon: 🧠
    title: Neural Network Modules
    details: Familiar nn.Module API with Linear layers, Sequential containers, activation functions, and loss functions — just like PyTorch.
  - icon: 🔧
    title: Optimizers
    details: SGD with momentum and Adam with bias correction, ready to train any model you build.
  - icon: 📊
    title: Rich Visualizations
    details: Interactive HTML computation-graph rendering, training-curve plots, weight histograms, and confusion matrices out of the box.
  - icon: 📦
    title: Install with pip
    details: Available on PyPI. Zero heavy dependencies — just NumPy, Matplotlib, and PyVis.
  - icon: 🎓
    title: Educational Design
    details: Every module is readable source code. Ideal for learning or teaching the internals of modern deep learning frameworks.
---

## Quick Install

```bash
pip install minitorchbr
```

## 30-Second Example

```python
import numpy as np
from MiniTorch.core.variable import Variable

# Create tensors with gradient tracking
x = Variable(np.array([[1.0, 2.0, 3.0]]))
w = Variable(np.random.randn(3, 1))

# Forward pass — graph is built automatically
y = x @ w          # matmul
loss = (y ** 2).sum()

# Backward pass
loss.backward()

print(w.grad)      # dL/dw computed via reverse-mode AD
```
