"""
MNIST training example using MiniTorch.

Architecture
------------
  Linear(784 → 256) → ReLU
  Linear(256 → 128) → ReLU
  Linear(128 →  10)
  Loss: Softmax Cross-Entropy

Run
---
  python mnist_train.py
"""
from __future__ import annotations

import time

import numpy as np

import MiniTorch as mt
from MiniTorch.core.config import no_grad
from MiniTorch.core.variable import Variable
from MiniTorch.data import DataLoader, load_mnist
from MiniTorch.nn import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim import Adam
from MiniTorch.utils.training_viz import (
    plot_confusion_matrix,
    plot_predictions,
    plot_training_history,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP:
    """
    Three-layer multi-layer perceptron for 10-class classification.
    """

    def __init__(self, in_dim: int = 784, hidden: tuple = (256, 128), out_dim: int = 10):
        dims = [in_dim] + list(hidden) + [out_dim]
        self._layers = [Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def __call__(self, x):
        h = x
        for i, layer in enumerate(self._layers[:-1]):
            h = relu(layer(h))
        return self._layers[-1](h)   # logits (no softmax — fused into loss)

    def parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int = 512):
    """Return (avg_loss, accuracy_pct) over the full array."""
    loader = DataLoader(x, y, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0

    with no_grad():
        for xb, yb in loader:
            x_var = Variable(xb.astype(np.float64))
            t_var = Variable(yb)
            logits = model(x_var)
            loss   = softmax_cross_entropy(logits, t_var)
            total_loss    += float(loss.data)
            total_correct += int((np.argmax(logits.data, axis=1) == yb).sum())

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / len(x)
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs:     int   = 15,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    seed:       int   = 42,
):
    np.random.seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # ── Model & optimiser ─────────────────────────────────────────────────
    model     = MLP()
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"\n{'─'*60}")
    print(f"  MiniTorch MNIST  |  epochs={epochs}  bs={batch_size}  lr={lr}")
    print(f"  train={len(x_train):,}  test={len(x_test):,}  params={sum(p.data.size for p in model.parameters()):,}")
    print(f"{'─'*60}\n")

    # ── History ───────────────────────────────────────────────────────────
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
        epoch_loss = 0.0

        for xb, yb in loader:
            x_var = Variable(xb.astype(np.float64))
            t_var = Variable(yb)

            logits = model(x_var)
            loss   = softmax_cross_entropy(logits, t_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)

        train_loss              = epoch_loss / len(loader)
        test_loss, test_acc     = evaluate(model, x_test, y_test)
        _,          train_acc   = evaluate(model, x_train[:6000], y_train[:6000])
        elapsed                 = time.time() - t0

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        bar_len  = 20
        bar_fill = int(bar_len * test_acc / 100)
        bar      = "█" * bar_fill + "░" * (bar_len - bar_fill)

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"loss {train_loss:.4f} → {test_loss:.4f} | "
            f"acc {train_acc:5.1f}% / {test_acc:5.1f}% [{bar}] | "
            f"{elapsed:.1f}s"
        )

    print(f"\n{'─'*60}")
    print(f"  Final test accuracy : {test_accs[-1]:.2f}%")
    print(f"  Best  test accuracy : {max(test_accs):.2f}%  (epoch {test_accs.index(max(test_accs)) + 1})")
    print(f"{'─'*60}\n")

    # ── Visualise results ─────────────────────────────────────────────────
    plot_training_history(
        train_losses, test_losses,
        train_accs,   test_accs,
        save_path="mnist_training_history.png",
        title="MiniTorch — MNIST Training",
    )

    plot_predictions(
        model, x_test, y_test,
        n_samples=32,
        save_path="mnist_predictions.png",
    )

    plot_confusion_matrix(
        model, x_test, y_test,
        class_names=[str(i) for i in range(10)],
        save_path="mnist_confusion_matrix.png",
    )


if __name__ == "__main__":
    train()
