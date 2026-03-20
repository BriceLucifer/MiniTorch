# Training a Network

A complete example training a two-layer MLP on synthetic data.

## Setup

```python
import numpy as np
from MiniTorch.core.variable import Variable
from MiniTorch.nn.sequential import Sequential
from MiniTorch.nn.linear import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.mse import mse_loss
from MiniTorch.optim.adam import Adam
```

## Synthetic Dataset

```python
np.random.seed(42)
N = 200   # samples

# Regression: y = sin(x1) + cos(x2)
X_np = np.random.uniform(-np.pi, np.pi, (N, 2))
y_np = np.sin(X_np[:, 0]) + np.cos(X_np[:, 1])
y_np = y_np.reshape(-1, 1)
```

## Model & Optimizer

```python
class MLP(Sequential):
    def forward(self, x):
        h = relu(self.layers[0](x))
        return self.layers[1](h)

model = MLP(Linear(2, 64), Linear(64, 1))
optimizer = Adam(model.parameters(), lr=1e-3)
```

## Training Loop

```python
EPOCHS = 200
BATCH  = 32

indices = np.arange(N)

for epoch in range(EPOCHS):
    np.random.shuffle(indices)
    epoch_loss = 0.0

    for start in range(0, N, BATCH):
        idx = indices[start:start + BATCH]
        x_batch = Variable(X_np[idx])
        y_batch = Variable(y_np[idx])

        pred = model(x_batch)
        loss = mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.data)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  loss={epoch_loss:.4f}")
```

## Evaluation

```python
from MiniTorch.core.config import no_grad

with no_grad():
    x_val = Variable(X_np[:20])
    preds = model(x_val)

for pred, true in zip(preds.data.flatten(), y_np[:20].flatten()):
    print(f"pred={pred:.3f}  true={true:.3f}")
```

## Training Visualization

```python
from MiniTorch.utils.training_viz import plot_training_history

history = {"train_loss": [...], "val_loss": [...]}
plot_training_history(history)
```
