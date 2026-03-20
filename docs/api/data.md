# data API

## MNIST Loader

```python
from MiniTorch.data.mnist import load_mnist
```

Downloads and caches the MNIST dataset, returning NumPy arrays.

### Usage

```python
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalize` | `True` | Scale pixel values to [0, 1] |
| `flatten` | `True` | Flatten `28×28` images to `784`-dim vectors |

Returned shapes:

| Array | Shape |
|-------|-------|
| `x_train` | `(60000, 784)` |
| `y_train` | `(60000,)` — integer labels `0–9` |
| `x_test` | `(10000, 784)` |
| `y_test` | `(10000,)` |

---

## DataLoader

```python
from MiniTorch.data.dataloader import DataLoader
```

Mini-batch iterator with optional shuffling.

### Constructor

```python
DataLoader(x, y, batch_size=32, shuffle=True)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | — | Input array `(N, ...)` |
| `y` | — | Label array `(N,)` |
| `batch_size` | `32` | Number of samples per batch |
| `shuffle` | `True` | Shuffle before each epoch |

### Iteration

```python
loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)

for x_batch, y_batch in loader:
    # x_batch: np.ndarray (64, 784)
    # y_batch: np.ndarray (64,)
    ...
```

Each `x_batch` / `y_batch` is a plain NumPy array — wrap in `Variable` as needed:

```python
for x_np, y_np in loader:
    x = Variable(x_np)
    logits = model(x)
    loss = softmax_cross_entropy(logits, y_np)
```
