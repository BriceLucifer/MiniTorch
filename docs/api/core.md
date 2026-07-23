# Core API

The core package contains the tensor object and operation base class used by
every differentiable computation.

## Variable

`Variable` wraps a NumPy array, stores an optional accumulated gradient, and
points to the `Function` that produced it.

```python
import numpy as np
from MiniTorch import Variable

x = Variable(np.array([1.0, 2.0], dtype=np.float32), name="x")
loss = (x ** 2).sum()
loss.backward()
```

!!! note

    MiniTorch gradients are themselves `Variable` objects. Read the raw array
    through `x.grad.data`.

::: MiniTorch.core.variable.Variable
    options:
      members:
        - backward
        - clear_grad
        - reshape
        - T
      show_root_heading: true

## Function

Subclass `Function` to implement a differentiable operation. The ordinary
first-order path can provide `backward_array` to propagate raw NumPy arrays
without building another autograd graph.

::: MiniTorch.core.function.Function
    options:
      members:
        - forward
        - backward
        - input_data
        - output_data
      show_root_heading: true

## Gradient configuration

::: MiniTorch.core.config.no_grad
    options:
      show_root_heading: true

::: MiniTorch.core.config.with_grad
    options:
      show_root_heading: true
