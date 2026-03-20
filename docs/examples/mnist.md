# MNIST Classifier

Train a fully-connected network on MNIST to ~97% test accuracy in under a minute.

## Full Script

```python
import numpy as np
from MiniTorch.core.variable import Variable
from MiniTorch.data.mnist import load_mnist
from MiniTorch.data.dataloader import DataLoader
from MiniTorch.nn.sequential import Sequential
from MiniTorch.nn.linear import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim.adam import Adam
from MiniTorch.core.config import no_grad

# ── Data ─────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True)

train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
test_loader  = DataLoader(x_test,  y_test,  batch_size=256, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class MNISTNet(Sequential):
    def forward(self, x):
        h1 = relu(self.layers[0](x))
        h2 = relu(self.layers[1](h1))
        return self.layers[2](h2)

model = MNISTNet(
    Linear(784, 256),
    Linear(256, 128),
    Linear(128, 10),
)

optimizer = Adam(model.parameters(), lr=1e-3)

# ── Training ──────────────────────────────────────────────────────────────────
for epoch in range(10):
    total_loss = 0.0
    for x_np, y_np in train_loader:
        x_var = Variable(x_np)
        logits = model(x_var)
        loss = softmax_cross_entropy(logits, y_np)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data)

    # ── Evaluation ────────────────────────────────────────────────────────────
    correct = 0
    total = 0
    with no_grad():
        for x_np, y_np in test_loader:
            logits = model(Variable(x_np))
            preds = np.argmax(logits.data, axis=1)
            correct += int((preds == y_np).sum())
            total += len(y_np)

    acc = correct / total * 100
    print(f"Epoch {epoch+1:2d}  loss={total_loss:.2f}  test_acc={acc:.2f}%")
```

## Expected Output

```
Epoch  1  loss=312.47  test_acc=93.21%
Epoch  2  loss=189.33  test_acc=95.18%
...
Epoch 10  loss=58.12   test_acc=97.34%
```

## Visualizing Results

```python
from MiniTorch.utils.training_viz import (
    plot_training_history,
    plot_confusion_matrix,
    plot_weight_histograms,
)

# Confusion matrix
with no_grad():
    all_preds = np.concatenate([
        np.argmax(model(Variable(xb)).data, axis=1)
        for xb, _ in test_loader
    ])

plot_confusion_matrix(y_test, all_preds, class_names=[str(i) for i in range(10)])
plot_weight_histograms(model)
```
