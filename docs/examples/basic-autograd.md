# Basic Autograd

This example covers the core automatic differentiation API.

## Scalar Gradients

```python
import numpy as np
from MiniTorch.core.variable import Variable

a = Variable(np.array(2.0))
b = Variable(np.array(3.0))

# f(a, b) = a^2 + a*b
# df/da = 2a + b = 7
# df/db = a = 2
f = a ** 2 + a * b
f.backward()

print(a.grad)  # 7.0
print(b.grad)  # 2.0
```

## Vector Gradients

```python
x = Variable(np.array([1.0, 2.0, 3.0]))
y = (x ** 2).sum()   # y = sum(x_i^2),  dy/dx_i = 2*x_i
y.backward()

print(x.grad)  # [2. 4. 6.]
```

## Matrix Operations

```python
X = Variable(np.random.randn(4, 3))
W = Variable(np.random.randn(3, 2))

Z = X @ W          # (4, 2)
loss = (Z ** 2).sum()
loss.backward()

print(X.grad.shape)  # (4, 3)
print(W.grad.shape)  # (3, 2)
```

## Chain Rule Through Multiple Ops

```python
from MiniTorch.ops.relu import relu
from MiniTorch.ops.sigmoid import sigmoid

x = Variable(np.linspace(-2, 2, 5))

h = relu(x)
y = sigmoid(h)
loss = y.sum()
loss.backward()

print(x.grad)
```

## Verifying Gradients Numerically

```python
from MiniTorch.utils.numer_diff import numerical_gradient_check
from MiniTorch.ops.relu import relu

x = Variable(np.random.randn(3, 3))
numerical_gradient_check(relu, [x])   # raises if analytical ≠ finite-diff
```

## Inference Mode (no_grad)

```python
from MiniTorch.core.config import no_grad

with no_grad():
    # No graph is built — faster and lower memory
    out = (x ** 2).sum()

print(out.creator)  # None
```
