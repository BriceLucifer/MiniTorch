from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn.activations import ReLU
from MiniTorch.nn.linear import Linear
from MiniTorch.nn.module import Module
from MiniTorch.nn.sequential import Sequential

train_mlp: Any
_IMPORT_ERROR: ImportError | None

try:
    from MiniTorch.native._native_trainer import (  # type: ignore[import-not-found]
        train_mlp as _compiled_train_mlp,
    )
except ImportError as exc:  # pragma: no cover - exercised by source-only installs
    train_mlp = None
    _IMPORT_ERROR = exc
else:
    train_mlp = _compiled_train_mlp
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class NativeTrainingHistory:
    losses: list[float]
    steps: int
    samples: int


def _linear_layers(model: Module) -> list[Linear]:
    if not isinstance(model, Sequential):
        raise TypeError("native training currently requires nn.Sequential")

    linear_layers: list[Linear] = []
    expect_linear = True
    for index, layer in enumerate(model.layers):
        if expect_linear:
            if not isinstance(layer, Linear):
                raise TypeError(
                    f"layer {index} must be Linear; got {type(layer).__name__}"
                )
            if layer.b is None:
                raise ValueError("native training currently requires Linear bias")
            linear_layers.append(layer)
        elif not isinstance(layer, ReLU):
            raise TypeError(
                f"layer {index} must be ReLU; got {type(layer).__name__}"
            )
        expect_linear = not expect_linear

    if not linear_layers or expect_linear:
        raise ValueError(
            "expected Sequential(Linear, ReLU, ..., Linear) with no final activation"
        )
    return linear_layers


def train(
    model: Module,
    x: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    shuffle: bool = True,
    seed: int = 42,
) -> NativeTrainingHistory:
    """Train a static Linear/ReLU classifier in a compiled C control loop.

    Matrix operations remain NumPy operations, preserving its optimized BLAS.
    The compiled path owns its Adam state and updates model parameters in place.
    """
    if train_mlp is None:
        raise RuntimeError(
            "MiniTorch native training is not built. Reinstall the package from "
            "source so its Cython extension can be compiled."
        ) from _IMPORT_ERROR
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    layers = _linear_layers(model)
    features = np.ascontiguousarray(x, dtype=np.float32)
    targets = np.ascontiguousarray(labels, dtype=np.int64)
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("x must be 2-D and labels must be 1-D")
    if len(features) != len(targets):
        raise ValueError("x and labels must contain the same number of samples")
    if features.shape[1] != layers[0].W.shape[0]:
        raise ValueError("input feature count does not match the first Linear layer")

    weights = []
    biases = []
    for layer in layers:
        if layer.W.data is None or layer.b is None or layer.b.data is None:
            raise RuntimeError("model parameter data is unavailable")
        if layer.W.data.dtype != np.float32 or layer.b.data.dtype != np.float32:
            raise TypeError("native training requires float32 model parameters")
        if not layer.W.data.flags.c_contiguous or not layer.b.data.flags.c_contiguous:
            raise ValueError("native training requires contiguous parameters")
        weights.append(layer.W.data)
        biases.append(layer.b.data)

    result = train_mlp(
        weights,
        biases,
        features,
        targets,
        epochs,
        batch_size,
        lr,
        shuffle,
        seed,
    )
    for layer, weight_grad, bias_grad in zip(
        layers,
        result["weight_gradients"],
        result["bias_gradients"],
    ):
        layer.W.grad = Variable(np.asarray(weight_grad))
        if layer.b is not None:
            layer.b.grad = Variable(np.asarray(bias_grad))
    return NativeTrainingHistory(
        losses=[float(loss) for loss in result["losses"]],
        steps=int(result["steps"]),
        samples=int(result["samples"]),
    )
