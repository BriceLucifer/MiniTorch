# Core API

## Variable

```python
from MiniTorch.core.variable import Variable
```

The fundamental tensor class with automatic differentiation support.

### Constructor

```python
Variable(data, requires_grad=True)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | The tensor data |
| `requires_grad` | `bool` | Enable gradient tracking (default `True`) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | Raw NumPy array |
| `grad` | `np.ndarray \| None` | Accumulated gradient |
| `shape` | `tuple` | Shape of the underlying array |
| `creator` | `Function \| None` | Op that created this variable |

### Methods

#### `.backward(grad=None)`

Triggers reverse-mode autodiff from this variable.

```python
loss.backward()          # grad defaults to ones
loss.backward(np.ones((1,)))
```

#### `.zero_grad()`

Resets `.grad` to `None`.

#### Operator Overloads

`Variable` supports standard Python operators which dispatch to the corresponding `Function`:

| Operator | Function |
|----------|----------|
| `a + b` | `Add` |
| `a - b` | `Sub` |
| `a * b` | `Mul` |
| `a / b` | `Div` |
| `a ** n` | `Pow` |
| `a @ b` | `MatMul` |
| `-a` | `Neg` |

---

## Function

```python
from MiniTorch.core.function import Function
```

Base class for all differentiable operations.

### Implementing a Custom Op

```python
class MyOp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.save_for_backward(x)
        return np.tanh(x)

    def backward(self, grad_output: np.ndarray):
        (x,) = self.saved_tensors
        return (1 - np.tanh(x) ** 2) * grad_output
```

### `save_for_backward(*tensors)`

Stash NumPy arrays needed during the backward pass.

### `saved_tensors`

Retrieve stashed arrays in `.backward()`.

---

## no_grad

```python
from MiniTorch.core.config import no_grad

with no_grad():
    y = model(x)   # no graph is built
```

Context manager that disables gradient tracking globally.
