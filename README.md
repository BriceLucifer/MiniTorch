# MiniTorch

A lightweight autograd framework for AI training, built from scratch with NumPy.

[![CI/CD Pipeline](https://github.com/BriceLucifer/MiniTorch/actions/workflows/workflow.yml/badge.svg)](https://github.com/BriceLucifer/MiniTorch/actions/workflows/workflow.yml)
[![PyPI Version](https://img.shields.io/pypi/v/minitorchbr.svg)](https://pypi.org/project/minitorchbr/)
[![Python Versions](https://img.shields.io/pypi/pyversions/minitorchbr.svg)](https://pypi.org/project/minitorchbr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is MiniTorch?

MiniTorch is a PyTorch-inspired deep learning framework built entirely on NumPy. It implements reverse-mode automatic differentiation, neural network layers, optimizers, and rich visualizations — all from scratch, so you can read every line and understand exactly how it works.

**Use it to:**
- Learn how autograd and backpropagation actually work
- Trace computation graphs interactively
- Train real models (MLP on MNIST reaches ~97% accuracy)
- Understand optimizers, loss functions, and gradient flow

---

## Installation

```bash
pip install minitorchbr
```

Or from source:

```bash
git clone https://github.com/BriceLucifer/MiniTorch.git
cd MiniTorch
uv pip install -e .
```

---

## Quick Start

### Autograd in 10 lines

```python
import numpy as np
from MiniTorch import Variable

x = Variable(np.array([[2.0]]), name="x")
w = Variable(np.array([[3.0]]), name="w")
b = Variable(np.array([[1.0]]), name="b")

y    = x * w + b          # forward pass builds the graph
loss = (y - 10.0) ** 2

loss.backward()           # reverse-mode autodiff

print(x.grad)   # -18.0
print(w.grad)   # -12.0
print(b.grad)   #  -6.0
```

### Neural network layers

```python
import numpy as np
from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear, Sequential
from MiniTorch.ops.relu import relu
from MiniTorch.optim import Adam

model = Sequential(
    Linear(784, 256),
    # relu is applied manually between layers
    Linear(256, 10),
)

optimizer = Adam(model.parameters(), lr=1e-3)
```

### MNIST training

```bash
uv run python mnist_train.py
```

Trains a 3-layer MLP (784→256→128→10) with Adam and Softmax Cross-Entropy.
Reaches **~97% test accuracy** in 15 epochs.

Produces three plots automatically:

| File | Contents |
|------|----------|
| `mnist_training_history.png` | Loss & accuracy curves |
| `mnist_predictions.png` | 32 sample predictions (green = correct) |
| `mnist_confusion_matrix.png` | 10×10 digit confusion matrix |

---

## Computation Graph Visualization

```python
import numpy as np
from MiniTorch import Variable, visualize_graph

x = Variable(np.array([[2.0]]), name="x")
w = Variable(np.array([[3.0]]), name="w")
b = Variable(np.array([[1.0]]), name="b")

y    = x * w + b;  y.name = "y"
loss = (y - 10.0) ** 2;  loss.name = "loss"
loss.backward(retain_grad=True)

visualize_graph(loss, filename="graph.html")
# Open graph.html in your browser
```

The graph is **strictly structured**:

```
col 0 (gen 0)     col 1      col 2 (gen 1)     col 3      col 4 (gen 2)
┌──────────┐                 ┌──────────┐                  ┌──────────┐
│  x       │──►  (Mul)  ──►  │  y       │──►  (Pow)  ──►  │  loss    │
│ INPUT    │                 │ TENSOR   │                  │ OUTPUT   │
└──────────┘  ╱             └──────────┘                  └──────────┘
┌──────────┐╱
│  w       │
│ INPUT    │
└──────────┘
```

- **Blue box** — input / parameter (leaf Variable)
- **Violet box** — intermediate tensor
- **Green box** — output / loss
- **Orange ellipse** — operation (Function)
- **Red edge** — gradient has been computed on this path

Run the demo:

```bash
uv run python demo_graph.py
# → graph_simple.html  (basic expression)
# → graph_mlp.html     (2-layer MLP)
```

---

## Training Visualization

```bash
uv run python demo_training_viz.py
```

Trains on a synthetic spiral dataset and saves:

- **`training_history.png`** — loss & accuracy per epoch, annotates best epoch
- **`weight_histograms.png`** — weight and gradient distributions per layer
- **`confusion_matrix.png`** — class confusion matrix with per-cell counts

Or call directly in your own training loop:

```python
from MiniTorch.utils.training_viz import (
    plot_training_history,
    plot_predictions,
    plot_weight_histograms,
    plot_confusion_matrix,
)

plot_training_history(train_losses, test_losses, train_accs, test_accs)
plot_predictions(model, x_test, y_test)
plot_confusion_matrix(model, x_test, y_test)
plot_weight_histograms(model)
```

---

## API Reference

### Core

| Symbol | Description |
|--------|-------------|
| `Variable(data, name=)` | Tensor with autograd. Wraps a NumPy array. |
| `variable.backward()` | Run reverse-mode autodiff from this node. |
| `no_grad()` | Context manager — disables graph construction. |
| `visualize_graph(var)` | Render interactive HTML computation graph. |

### Operations (`MiniTorch.ops`)

| Category | Ops |
|----------|-----|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow` |
| Math | `exp`, `log`, `sin`, `cos`, `tanh`, `square` |
| Activations | `relu`, `sigmoid` |
| Matrix / shape | `matmul`, `reshape`, `transpose`, `broadcast_to` |
| Reduction | `sum`, `sum_to` |
| Loss | `mean_squared_error`, `softmax_cross_entropy` |

### Neural Network (`MiniTorch.nn`)

| Class | Description |
|-------|-------------|
| `Module` | Base class. Auto-discovers `Variable` parameters recursively. |
| `Linear(in, out, bias=True)` | Fully-connected layer with He initialisation. |
| `Sequential(*layers)` | Chains modules in order. |

### Optimizers (`MiniTorch.optim`)

| Class | Description |
|-------|-------------|
| `SGD(params, lr, momentum, weight_decay)` | Stochastic gradient descent with momentum. |
| `Adam(params, lr, beta1, beta2, eps)` | Adam with bias correction and weight decay. |

### Data (`MiniTorch.data`)

| Symbol | Description |
|--------|-------------|
| `load_mnist()` | Downloads and caches MNIST. Returns `(x_train, y_train), (x_test, y_test)`. |
| `DataLoader(x, y, batch_size, shuffle)` | Mini-batch iterator. |

---

## Project Structure

```
MiniTorch/
├── core/
│   ├── variable.py          # Variable — tensor + autograd
│   ├── function.py          # Function — base op class
│   └── config.py            # no_grad / with_grad context managers
├── ops/                     # All differentiable operations
│   ├── add.py  sub.py  mul.py  div.py  neg.py  pow.py
│   ├── exp.py  log.py  sin.py  cos.py  tanh.py  square.py
│   ├── relu.py  sigmoid.py
│   ├── matmul.py  reshape.py  transpose.py
│   ├── sum.py  sum_to.py  broadcast.py
│   ├── mean_squared_error.py  softmax_cross_entropy.py
├── nn/
│   ├── module.py            # Base Module class
│   ├── linear.py            # Linear layer
│   └── sequential.py        # Sequential container
├── optim/
│   ├── sgd.py               # SGD + momentum
│   └── adam.py              # Adam
├── data/
│   ├── mnist.py             # MNIST downloader + parser
│   └── dataloader.py        # Mini-batch DataLoader
└── utils/
    ├── visualize.py         # Computation graph → HTML
    ├── training_viz.py      # Loss curves, predictions, histograms
    └── numer_diff.py        # Numerical gradient checker

mnist_train.py               # Full MNIST training script
demo_graph.py                # Graph visualisation demo
demo_training_viz.py         # Training visualisation demo
tests/test_new_ops.py        # 30 unit tests
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

```
30 passed in 0.57s
```

Tests cover: ReLU, Sigmoid, Log, Softmax-CE (forward + gradient check), Linear, Sequential, SGD, Adam, and an end-to-end MLP convergence test.

---

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest tests/ -v
uv build
```

---

## License

MIT

---

## Acknowledgments

MiniTorch is inspired by PyTorch and [DezeroBook](https://github.com/oreilly-japan/deep-learning-from-scratch-3).
Built as a teaching tool to understand automatic differentiation and deep learning frameworks from first principles — every line is readable NumPy.
