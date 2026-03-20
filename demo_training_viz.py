"""
demo_training_viz.py — Training visualisation demo (no MNIST download needed).

Trains a small MLP on a synthetic 2-class spiral dataset and produces:
    training_history.png   — loss + accuracy curves
    predictions.png        — sample predictions on test set
    weight_histograms.png  — parameter distributions
    confusion_matrix.png   — 2-class confusion matrix

Run:  uv run python demo_training_viz.py
"""
import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim import Adam
from MiniTorch.data import DataLoader
from MiniTorch.utils.training_viz import (
    plot_training_history,
    plot_weight_histograms,
    plot_confusion_matrix,
)


# ── Synthetic spiral dataset ─────────────────────────────────────────────────
def make_spiral(n_per_class: int = 200, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for cls in range(2):
        t   = np.linspace(0, 1, n_per_class)
        ang = t * 3 * np.pi + cls * np.pi
        r   = t
        x   = r * np.cos(ang) + rng.normal(0, noise, n_per_class)
        y   = r * np.sin(ang) + rng.normal(0, noise, n_per_class)
        X.append(np.stack([x, y], axis=1))
        Y.append(np.full(n_per_class, cls, dtype=np.int32))
    X = np.concatenate(X).astype(np.float64)
    Y = np.concatenate(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


# ── Model ────────────────────────────────────────────────────────────────────
class SmallMLP:
    def __init__(self):
        self.l1 = Linear(2, 32)
        self.l2 = Linear(32, 16)
        self.l3 = Linear(16, 2)

    def __call__(self, x):
        return self.l3(relu(self.l2(relu(self.l1(x)))))

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters() + self.l3.parameters()


# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(model, x, y):
    from MiniTorch.core.config import no_grad
    with no_grad():
        logits = model(Variable(x))
        preds  = np.argmax(logits.data, axis=1)
    loss_var = softmax_cross_entropy(model(Variable(x)), Variable(y))
    acc = 100.0 * (preds == y).mean()
    return float(loss_var.data), acc


# ── Train ────────────────────────────────────────────────────────────────────
print("Generating spiral dataset …")
X, Y = make_spiral(n_per_class=300)
split = int(0.8 * len(X))
x_train, y_train = X[:split], Y[:split]
x_test,  y_test  = X[split:], Y[split:]
print(f"  train={len(x_train)}  test={len(x_test)}")

model = SmallMLP()
opt   = Adam(model.parameters(), lr=5e-3)

EPOCHS     = 40
BATCH_SIZE = 64
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

print(f"\nTraining for {EPOCHS} epochs …")
for epoch in range(1, EPOCHS + 1):
    loader = DataLoader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    total  = 0.0
    for xb, yb in loader:
        logits = model(Variable(xb))
        loss   = softmax_cross_entropy(logits, Variable(yb))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.data)

    tr_loss, tr_acc = evaluate(model, x_train, y_train)
    te_loss, te_acc = evaluate(model, x_test, y_test)
    train_losses.append(tr_loss);  test_losses.append(te_loss)
    train_accs.append(tr_acc);     test_accs.append(te_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS} | loss {tr_loss:.4f}/{te_loss:.4f} "
              f"| acc {tr_acc:.1f}%/{te_acc:.1f}%")

print(f"\nFinal test accuracy: {test_accs[-1]:.1f}%")

# ── Visualise ─────────────────────────────────────────────────────────────────
print("\nGenerating plots …")

plot_training_history(
    train_losses, test_losses,
    train_accs,   test_accs,
    save_path="training_history.png",
    title="MiniTorch — Spiral Dataset Training",
)

plot_weight_histograms(model, save_path="weight_histograms.png")

plot_confusion_matrix(
    model, x_test, y_test,
    class_names=["class 0", "class 1"],
    save_path="confusion_matrix.png",
)

print("\nDone! Files saved:")
print("  training_history.png")
print("  weight_histograms.png")
print("  confusion_matrix.png")
