# Autograd System

MiniTorchBR implements **reverse-mode automatic differentiation** (backpropagation). This page explains the design so you can understand what happens under the hood.

## Core Concepts

### Variable

`Variable` is the central object — it wraps a NumPy array and optionally participates in a computation graph.

```python
from MiniTorch.core.variable import Variable
import numpy as np

x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
```

Key attributes:

| Attribute | Description |
|-----------|-------------|
| `data` | The underlying NumPy array |
| `grad` | Accumulated gradient (same shape as `data`) |
| `creator` | The `Function` that produced this variable |
| `requires_grad` | Whether to track gradients |

### Function

Every differentiable operation is a subclass of `Function` with two methods:

- **`forward(*inputs)`** — computes the output value
- **`backward(grad_output)`** — returns gradients w.r.t. each input

```python
from MiniTorch.core.function import Function

class MySquare(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x ** 2

    def backward(self, grad_output):
        (x,) = self.saved_tensors
        return 2 * x * grad_output
```

## Computation Graph

Each call to a `Function` builds a node in the graph:

```
x ──┐
    ├─► Mul ──► z ──► Sum ──► loss
y ──┘
```

When you call `loss.backward()`, MiniTorchBR:

1. Starts with gradient `1.0` at the loss node
2. Walks the graph in **reverse topological order**
3. Calls each `Function.backward()` to propagate gradients

## Example: Manual Inspection

```python
import numpy as np
from MiniTorch.core.variable import Variable

x = Variable(np.array([2.0, 3.0]))
y = Variable(np.array([4.0, 5.0]))

z = x * y        # element-wise multiply
loss = z.sum()   # scalar

loss.backward()

print(x.grad)    # [4. 5.]  (dL/dx = y)
print(y.grad)    # [2. 3.]  (dL/dy = x)
```

## Gradient Accumulation

Gradients **accumulate** across multiple `.backward()` calls (like PyTorch). Zero them before each training step:

```python
optimizer.zero_grad()   # or manually: param.grad = None
loss.backward()
optimizer.step()
```

## Numerical Gradient Check

Use the built-in checker to verify custom ops:

```python
from MiniTorch.utils.numer_diff import numerical_gradient_check

numerical_gradient_check(my_function, inputs)
```

This compares analytical gradients against finite differences and raises if they diverge beyond tolerance.

## no_grad Context

```python
from MiniTorch.core.config import no_grad

with no_grad():
    prediction = model(x_test)  # no graph allocated
```
