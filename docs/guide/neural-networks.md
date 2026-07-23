# Neural Networks

MiniTorchBR's `nn` module provides a familiar PyTorch-like API for building and training neural networks.

## Module Base Class

All layers inherit from `nn.Module`:

```python
from MiniTorch.nn.module import Module

class MyLayer(Module):
    def __init__(self):
        super().__init__()
        self.w = Variable(np.random.randn(4, 2))

    def forward(self, x):
        return x @ self.w
```

`Module` automatically collects parameters from all attributes that are `Variable` instances.

## Built-in Layers

### Linear

Fully-connected layer: `y = xW + b`

```python
from MiniTorch.nn.linear import Linear

layer = Linear(in_features=784, out_features=128)
out = layer(x)  # x shape: (batch, 784) → out shape: (batch, 128)
```

### Sequential

Chain layers in order:

```python
from MiniTorch.nn import Linear, ReLU, Sequential

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

logits = model(x)
```

## Activations

Apply activation ops directly as functions:

```python
from MiniTorch.ops.relu import relu
from MiniTorch.ops.sigmoid import sigmoid

h = relu(layer(x))
```

## Loss Functions

### Mean Squared Error

```python
from MiniTorch.ops import mean_squared_error

loss = mean_squared_error(predictions, targets)
```

### Softmax Cross-Entropy

```python
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy

loss = softmax_cross_entropy(logits, labels)  # labels: integer class indices
```

## Training Loop

```python
from MiniTorch import Variable
from MiniTorch.nn import Linear, ReLU, Sequential
from MiniTorch.ops import softmax_cross_entropy
from MiniTorch.optim import Adam

# Build model
model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        # Forward
        logits = model(Variable(x_batch))
        loss = softmax_cross_entropy(logits, Variable(y_batch))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}  loss={loss.data.item():.4f}")
```

## Accessing Parameters

```python
params = model.parameters()   # list of Variable objects
for p in params:
    print(p.data.shape)
```
