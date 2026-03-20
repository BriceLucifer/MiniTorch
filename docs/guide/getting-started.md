# Getting Started

## Installation

MiniTorchBR is available on PyPI and requires Python 3.10+.

```bash
pip install minitorchbr
```

Or install from source:

```bash
git clone https://github.com/BriceLucifer/MiniTorch.git
cd MiniTorch
pip install -e .
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy ≥ 1.24 | Tensor computation |
| matplotlib ≥ 3.7 | Training plots |
| pyvis ≥ 0.3 | Interactive graph rendering |

## Project Layout

```
MiniTorch/
├── core/       # Variable (tensor) + Function (op base)
├── ops/        # 20+ differentiable operations
├── nn/         # Module, Linear, Sequential
├── optim/      # SGD, Adam
├── data/       # MNIST loader, DataLoader
└── utils/      # Graph viz, training viz, numerical diff
```

## Your First Computation

```python
import numpy as np
from MiniTorch.core.variable import Variable

# Scalars
a = Variable(np.array(2.0))
b = Variable(np.array(3.0))

c = a * b + a   # c = a*b + a  →  dc/da = b+1 = 4,  dc/db = a = 2
c.backward()

print(a.grad)   # 4.0
print(b.grad)   # 2.0
```

## Disabling Gradient Tracking

Use `no_grad` for inference to save memory and speed up computation:

```python
from MiniTorch.core.config import no_grad

with no_grad():
    out = model(x)   # no graph is built
```

## Next Steps

- [Autograd System](./autograd) — understand how the computation graph works
- [Neural Networks](./neural-networks) — build and train models
- [Examples](../examples/basic-autograd) — runnable code samples
