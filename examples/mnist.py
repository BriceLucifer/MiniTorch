"""Train a dense MNIST classifier with the compiled MiniTorch loop.

The first run downloads MNIST into the local cache.

Run with:
    uv run python examples/mnist.py
"""

import numpy as np

from MiniTorch import Variable, no_grad, sum as tensor_sum, visualize
from MiniTorch.data import DataLoader, load_mnist
from MiniTorch.native import train
from MiniTorch.nn import Linear, ReLU, Sequential


def accuracy(model: Sequential, x: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    loader = DataLoader(x, labels, batch_size=512, shuffle=False)
    with no_grad():
        for features, targets in loader:
            prediction = model(Variable(features.astype(np.float32, copy=False)))
            correct += int((np.argmax(prediction.data, axis=1) == targets).sum())
    return correct / len(labels)


(x_train, y_train), (x_test, y_test) = load_mnist()
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

print(f"final loss: {history.losses[-1]:.4f}")
print(f"test accuracy: {accuracy(model, x_test, y_test):.2%}")

# Capture one concrete neuron's value and gradient for the interactive viewer.
probe = Variable(x_test[:1].astype(np.float32, copy=False))
probe_output = model(probe)
probe_loss = tensor_sum(probe_output)
probe_loss.backward(retain_grad=True)
visualize(model, filename="mnist_model.html", input_shape=(None, 784))
