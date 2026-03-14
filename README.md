# MiniTorch

A lightweight autograd framework for AI training, built from scratch with NumPy.

[![CI/CD Pipeline](https://github.com/BriceLucifer/MiniTorch/actions/workflows/workflow.yml/badge.svg)](https://github.com/BriceLucifer/MiniTorch/actions/workflows/workflow.yml)
[![PyPI Version](https://img.shields.io/pypi/v/minitorchbr.svg)](https://pypi.org/project/minitorchbr/)
[![Python Versions](https://img.shields.io/pypi/pyversions/minitorchbr.svg)](https://pypi.org/project/minitorchbr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Automatic Differentiation**: Reverse-mode automatic differentiation with computational graph
- **NumPy Backend**: All tensor operations powered by NumPy
- **Computational Graph Visualization**: Interactive graph visualization using pyvis
- **Gradient Checking**: Numerical gradient verification for debugging
- **Clean API**: Simple and intuitive interface inspired by PyTorch

## Installation

### From PyPI (coming soon)

```bash
pip install minitorch
```

### From Source

```bash
git clone <repository-url>
cd framework
uv pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from MiniTorch import Variable, square, add, mul

# Create variables
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

# Perform operations
z = add(square(x), square(y))  # z = x² + y²

# Compute gradients
z.backward()

print(f"z = {z.data}")        # z = 13.0
print(f"dz/dx = {x.grad}")    # dz/dx = 4.0
print(f"dz/dy = {y.grad}")    # dz/dy = 6.0
```

### Linear Regression Example

```python
import numpy as np
from MiniTorch import Variable, square, mul, sub, sum

# Generate synthetic data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2.0 + 3.0 * x + 0.1 * np.random.randn(100, 1)

# Initialize parameters
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    return add(mul(x, W), b)

def loss(y_pred, y_true):
    diff = sub(y_pred, y_true)
    return mean_squared_error(diff)

# Training loop
lr = 0.1
for i in range(100):
    y_pred = predict(x)
    l = loss(y_pred, y)
    
    # Reset gradients
    W.grad = None
    b.grad = None
    
    # Backward pass
    l.backward()
    
    # Update parameters
    W.data -= lr * W.grad
    b.data -= lr * b.grad
    
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {l.data}")

print(f"W = {W.data}, b = {b.data}")
```

### Computational Graph Visualization

```python
from MiniTorch import Variable, square, add, visualize_graph

x = Variable(np.array(2.0), name='x')
y = Variable(np.array(3.0), name='y')
z = add(square(x), square(y), name='z')

# Generate interactive visualization
visualize_graph(z, filename='graph.html')
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Variable` | Main tensor class with automatic differentiation |
| `Function` | Base class for all operations |

### Operations

#### Basic Math
- `add`, `sub`, `mul`, `div`, `neg`, `pow`, `square`

#### Math Functions
- `exp`, `sin`, `cos`, `tanh`

#### Matrix Operations
- `matmul`, `reshape`, `transpose`

#### Reduction Operations
- `sum`, `sum_to`, `broadcast_to`

#### Loss Functions
- `mean_squared_error`

### Utilities

| Function | Description |
|----------|-------------|
| `numerical_diff` | Compute numerical gradient for verification |
| `as_array` | Convert input to NumPy array |
| `visualize_graph` | Generate interactive computational graph |
| `no_grad` | Context manager to disable gradient computation |
| `with_grad` | Context manager to enable gradient computation |

## Project Structure

```
MiniTorch/
├── core/              # Core autograd engine
│   ├── variable.py    # Variable class definition
│   ├── function.py    # Function base class
│   └── config.py      # Configuration and context managers
├── ops/               # Operation implementations
│   ├── add.py, mul.py, exp.py, etc.
└── utils/             # Utility functions
    ├── numer_diff.py  # Numerical differentiation
    └── visualize.py   # Graph visualization
```

## Development

### Setup Development Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .[dev]
```

### Running Tests

```bash
python -m unittest tests/test.py
```

### Building the Package

```bash
uv build
```

## License

This project is provided for educational purposes.

## Acknowledgments

MiniTorch is inspired by PyTorch and designed as a teaching tool to understand automatic differentiation and deep learning frameworks from first principles.
