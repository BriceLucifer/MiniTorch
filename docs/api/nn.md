# nn API

## Module

```python
from MiniTorch.nn.module import Module
```

Base class for all neural network components.

### `parameters() → list[Variable]`

Returns a flat list of all learnable parameters in the module (and any sub-modules).

```python
model = Sequential(Linear(4, 8), Linear(8, 2))
params = model.parameters()   # [w1, b1, w2, b2]
```

### `__call__(x)`

Calls `forward(x)`. Build your computation graph here.

### Custom Module Example

```python
import numpy as np
from MiniTorch.core.variable import Variable
from MiniTorch.nn.module import Module
from MiniTorch.ops.relu import relu

class TwoLayerNet(Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden)
        self.fc2 = Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(relu(self.fc1(x)))
```

---

## Linear

```python
from MiniTorch.nn.linear import Linear
```

Fully-connected layer: `y = x @ W + b`

### Constructor

```python
Linear(in_features: int, out_features: int, bias: bool = True)
```

Parameters are initialized with Kaiming uniform.

### Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `weight` | `(in_features, out_features)` | Weight matrix |
| `bias` | `(out_features,)` | Bias vector |

### Example

```python
layer = Linear(128, 64)
out = layer(x)   # x: (batch, 128) → out: (batch, 64)
```

---

## Sequential

```python
from MiniTorch.nn.sequential import Sequential
```

Chains modules in order. Input is passed through each layer sequentially.

### Constructor

```python
Sequential(*layers)
```

### Example

```python
model = Sequential(
    Linear(784, 256),
    Linear(256, 128),
    Linear(128, 10),
)

logits = model(x_batch)
```
