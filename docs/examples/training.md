# Training a Network

MiniTorch provides an eager autograd loop for arbitrary models and a compiled
loop for static dense classifiers.

Run the complete synthetic compiled-training example:

```bash
uv run python examples/native_training.py
```

## Eager autograd

```python
from MiniTorch import Variable
from MiniTorch.nn import Linear, ReLU, Sequential
from MiniTorch.ops import softmax_cross_entropy
from MiniTorch.optim import Adam

model = Sequential(
    Linear(64, 128),
    ReLU(),
    Linear(128, 10),
)
optimizer = Adam(model.parameters(), lr=1e-3)

for xb, yb in loader:
    logits = model(Variable(xb))
    loss = softmax_cross_entropy(logits, Variable(yb))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The eager loop supports arbitrary MiniTorch operations, custom modules, dynamic
Python model code, and higher-order differentiation.

## Compiled NumPy training

Static dense classifiers can move their epoch, batch, forward, backward, and
Adam control loops into compiled C:

```python
from MiniTorch.native import train
from MiniTorch.nn import Linear, ReLU, Sequential

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
)

history = train(
    model,
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    lr=1e-3,
)

print(history.losses)
```

NumPy still executes matrix multiplication through its optimized native
libraries. The C extension removes repeated Python model, operation,
autograd-object, and optimizer dispatch from every mini-batch.

The compiled path currently supports:

- `nn.Sequential`;
- alternating `Linear` and `ReLU` modules;
- a final `Linear` classification layer;
- float32 parameters and inputs;
- integer class labels;
- softmax cross-entropy and Adam.

The model parameters are updated in place, and the last mini-batch parameter
gradients remain available programmatically. The model explorer shows
activation `Value` and `Grad`; run one eager probe with
`backward(retain_grad=True)` before opening it. Use eager training for other
architectures.
