# optim API

## SGD

```python
from MiniTorch.optim.sgd import SGD
```

Stochastic Gradient Descent with optional momentum.

### Constructor

```python
SGD(parameters, lr=0.01, momentum=0.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parameters` | — | List of `Variable` objects |
| `lr` | `0.01` | Learning rate |
| `momentum` | `0.0` | Momentum factor (`0` = plain SGD) |

### Methods

#### `.step()`

Updates each parameter: `p ← p - lr * p.grad` (with momentum if set).

#### `.zero_grad()`

Resets all parameter gradients to `None`.

### Example

```python
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Adam

```python
from MiniTorch.optim.adam import Adam
```

Adam optimizer with bias correction (Kingma & Ba, 2015).

### Constructor

```python
Adam(parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parameters` | — | List of `Variable` objects |
| `lr` | `1e-3` | Learning rate |
| `beta1` | `0.9` | First-moment decay |
| `beta2` | `0.999` | Second-moment decay |
| `eps` | `1e-8` | Numerical stability |

### Methods

Same as SGD: `.step()` and `.zero_grad()`.

### Example

```python
optimizer = Adam(model.parameters(), lr=1e-3)

for x_batch, y_batch in dataloader:
    logits = model(x_batch)
    loss = softmax_cross_entropy(logits, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
