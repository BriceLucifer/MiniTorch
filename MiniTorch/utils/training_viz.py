"""
Training visualisation utilities for MiniTorch.

Functions
---------
plot_training_history  – static loss / accuracy curves
plot_predictions       – grid of model predictions on test images
plot_weight_histograms – weight & gradient distributions
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────
# Colour palette (consistent across all plots)
# ──────────────────────────────────────────────────────────────
_TRAIN_COLOR = "#3b82f6"   # blue-500
_TEST_COLOR  = "#ef4444"   # red-500
_GREEN       = "#22c55e"   # green-500


def plot_training_history(
    train_losses: list[float],
    test_losses: list[float] | None = None,
    train_accs:  list[float] | None = None,
    test_accs:   list[float] | None = None,
    save_path: str = "training_history.png",
    title: str = "Training History",
) -> None:
    """
    Plot loss and accuracy curves and save to *save_path*.

    Parameters
    ----------
    train_losses : training loss per epoch
    test_losses  : test/validation loss per epoch (optional)
    train_accs   : training accuracy % per epoch (optional)
    test_accs    : test/validation accuracy % per epoch (optional)
    save_path    : output PNG file
    title        : figure title
    """
    n_panels = 1 + (1 if (train_accs or test_accs) else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # ── Loss panel ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_losses, color=_TRAIN_COLOR, linewidth=2,
            marker="o", markersize=4, label="Train")
    if test_losses:
        ax.plot(epochs, test_losses, color=_TEST_COLOR, linewidth=2,
                marker="s", markersize=4, label="Test")
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    ax.annotate(f"{train_losses[-1]:.4f}",
                xy=(len(train_losses), train_losses[-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=_TRAIN_COLOR)
    if test_losses:
        ax.annotate(f"{test_losses[-1]:.4f}",
                    xy=(len(test_losses), test_losses[-1]),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=8, color=_TEST_COLOR)

    # ── Accuracy panel ──────────────────────────────────────────
    if n_panels > 1:
        ax = axes[1]
        if train_accs:
            ax.plot(epochs, train_accs, color=_TRAIN_COLOR, linewidth=2,
                    marker="o", markersize=4, label="Train")
        if test_accs:
            ax.plot(epochs, test_accs, color=_TEST_COLOR, linewidth=2,
                    marker="s", markersize=4, label="Test")
        ax.set_title("Accuracy (%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # Annotate best test accuracy
        if test_accs:
            best_epoch = int(np.argmax(test_accs)) + 1
            best_acc   = max(test_accs)
            ax.axhline(best_acc, linestyle="--", color=_TEST_COLOR, alpha=0.4)
            ax.annotate(f"best {best_acc:.2f}% @ ep {best_epoch}",
                        xy=(best_epoch, best_acc),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=8, color=_TEST_COLOR)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved training history → {save_path}")
    plt.show()
    plt.close(fig)


def plot_predictions(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 32,
    save_path: str = "predictions.png",
) -> None:
    """
    Display a grid of sample predictions.

    Correct predictions have a green title; wrong ones have red.
    """
    from MiniTorch.core.variable import Variable
    from MiniTorch.core.config import no_grad

    # Pick random samples
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    x_sample = x_test[indices]
    y_true   = y_test[indices]

    with no_grad():
        logits = model(Variable(x_sample.astype(np.float64)))
        y_pred = np.argmax(logits.data, axis=1)

    cols = min(8, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0))
    axes_flat = np.array(axes).flatten()

    for i in range(n_samples):
        ax = axes_flat[i]
        ax.imshow(x_sample[i].reshape(28, 28), cmap="gray", interpolation="nearest")
        correct = y_pred[i] == y_true[i]
        color = _GREEN if correct else _TEST_COLOR
        ax.set_title(f"P:{y_pred[i]}  T:{y_true[i]}", color=color,
                     fontsize=8, fontweight="bold")
        ax.axis("off")

    for i in range(n_samples, len(axes_flat)):
        axes_flat[i].axis("off")

    acc = 100 * (y_pred == y_true).mean()
    fig.suptitle(
        f"Model Predictions  (sample acc: {acc:.1f}%)\n"
        f"Green = correct  |  Red = wrong",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved predictions → {save_path}")
    plt.show()
    plt.close(fig)


def plot_weight_histograms(
    model,
    save_path: str = "weight_histograms.png",
) -> None:
    """
    Plot weight and gradient histograms for all parameters in the model.
    """
    params = model.parameters()
    n = len(params)
    if n == 0:
        print("[viz] No parameters found.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i, p in enumerate(params):
        name = p.name or f"param_{i}"
        # Weights
        axes[i][0].hist(p.data.flatten(), bins=50, color=_TRAIN_COLOR, alpha=0.8)
        axes[i][0].set_title(f"{name}  [weights]", fontsize=9)
        axes[i][0].set_ylabel("count")
        axes[i][0].grid(True, alpha=0.3)

        # Gradients
        ax_g = axes[i][1]
        if p.grad is not None:
            ax_g.hist(p.grad.data.flatten(), bins=50, color=_TEST_COLOR, alpha=0.8)
            ax_g.set_title(f"{name}  [gradients]", fontsize=9)
        else:
            ax_g.text(0.5, 0.5, "no gradient", ha="center", va="center",
                      transform=ax_g.transAxes, fontsize=10, color="gray")
            ax_g.set_title(f"{name}  [gradients]", fontsize=9)
        ax_g.grid(True, alpha=0.3)

    fig.suptitle("Parameter Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved weight histograms → {save_path}")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str = "confusion_matrix.png",
    batch_size: int = 512,
) -> None:
    """
    Compute and display a confusion matrix over the full test set.
    """
    from MiniTorch.core.variable import Variable
    from MiniTorch.core.config import no_grad
    from MiniTorch.data.dataloader import DataLoader

    n_classes = int(y_test.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)
    with no_grad():
        for xb, yb in loader:
            logits = model(Variable(xb.astype(np.float64)))
            preds = np.argmax(logits.data, axis=1)
            for t, p in zip(yb, preds):
                cm[t, p] += 1

    acc = 100.0 * cm.diagonal().sum() / cm.sum()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = class_names or [str(i) for i in range(n_classes)]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix  (acc: {acc:.2f}%)",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved confusion matrix → {save_path}")
    plt.show()
    plt.close(fig)
