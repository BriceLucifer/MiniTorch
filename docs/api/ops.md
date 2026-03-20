# Operations API

All ops live in `MiniTorch.ops.*`. Each is a `Function` subclass supporting forward and backward passes.

## Arithmetic

| Op | Import | Description |
|----|--------|-------------|
| Add | `from MiniTorch.ops.add import Add` | Element-wise addition |
| Sub | `from MiniTorch.ops.sub import Sub` | Element-wise subtraction |
| Mul | `from MiniTorch.ops.mul import Mul` | Element-wise multiplication |
| Div | `from MiniTorch.ops.div import Div` | Element-wise division |
| Pow | `from MiniTorch.ops.pow import Pow` | Element-wise power |
| Neg | `from MiniTorch.ops.neg import Neg` | Negation |
| MatMul | `from MiniTorch.ops.matmul import MatMul` | Matrix multiplication |

These are also available via operator overloading on `Variable` (e.g. `a + b`, `a @ b`).

## Reductions

| Op | Import | Description |
|----|--------|-------------|
| Sum | `from MiniTorch.ops.sum import Sum` | Sum all elements (scalar output) |
| Mean | `from MiniTorch.ops.mean import Mean` | Mean of all elements |

```python
x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
s = x.sum()    # Variable(10.0)
s.backward()
print(x.grad)  # [[1. 1.] [1. 1.]]
```

## Activations

### ReLU

```python
from MiniTorch.ops.relu import relu

y = relu(x)   # max(0, x) element-wise
```

Gradient: `1` where `x > 0`, else `0`.

### Sigmoid

```python
from MiniTorch.ops.sigmoid import sigmoid

y = sigmoid(x)   # 1 / (1 + exp(-x))
```

Gradient: `sigmoid(x) * (1 - sigmoid(x))`.

## Loss Functions

### MSE Loss

```python
from MiniTorch.ops.mse import mse_loss

loss = mse_loss(predictions, targets)
# equivalent to ((predictions - targets) ** 2).mean()
```

### Softmax Cross-Entropy

```python
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy

loss = softmax_cross_entropy(logits, labels)
```

- `logits`: `Variable` of shape `(batch, num_classes)`
- `labels`: integer class indices, shape `(batch,)`

The op fuses softmax and cross-entropy for numerical stability.
