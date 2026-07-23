import numpy as np
import pytest

from MiniTorch import Variable
from MiniTorch.native import train
from MiniTorch.nn import Linear, ReLU, Sequential


def test_native_training_reduces_loss_and_updates_model():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 8)).astype(np.float32)
    labels = (x[:, 0] - x[:, 1] > 0).astype(np.int64)
    model = Sequential(Linear(8, 16), ReLU(), Linear(16, 2))

    try:
        history = train(
            model,
            x,
            labels,
            epochs=6,
            batch_size=32,
            lr=1e-2,
            seed=7,
        )
    except RuntimeError as exc:
        if "not built" in str(exc):
            pytest.skip("native extension is not built in this source checkout")
        raise

    assert history.losses[-1] < history.losses[0]
    assert history.steps == 48
    predictions = model(Variable(x)).data.argmax(axis=1)
    assert (predictions == labels).mean() > 0.85


def test_native_training_rejects_unsupported_module_order():
    model = Sequential(Linear(4, 8), Linear(8, 2))
    with pytest.raises(TypeError, match="must be ReLU"):
        train(
            model,
            np.zeros((8, 4), dtype=np.float32),
            np.zeros(8, dtype=np.int64),
        )
