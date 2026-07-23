# MNIST Classifier

The repository includes a complete compiled-training example at
`examples/mnist.py`.

```bash
uv run python examples/mnist.py
```

On its first run, the loader downloads and caches MNIST. The example:

1. loads flattened, normalized images;
2. builds a `784 → 256 → 128 → 10` dense classifier;
3. trains with the compiled C control loop and NumPy matrix kernels;
4. evaluates test accuracy in eager inference mode;
5. writes and opens `mnist_model.html`.

The core setup is:

```python
from MiniTorch.data import load_mnist
from MiniTorch.native import train
from MiniTorch.nn import Linear, ReLU, Sequential

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
```

Training results vary with initialization, NumPy, BLAS, and hardware. The
example prints the measured loss and accuracy rather than promising a fixed
runtime or score.

To generate the explorer without opening a browser:

```python
from MiniTorch import Variable, sum as tensor_sum, visualize

probe = Variable(x_test[:1].astype("float32"))
probe_output = model(probe)
probe_loss = tensor_sum(probe_output)
probe_loss.backward(retain_grad=True)

visualize(
    model,
    filename="mnist_model.html",
    input_shape=(None, 784),
    open_browser=False,
)
```
