from __future__ import annotations

from MiniTorch.core.variable import Variable
from MiniTorch.nn.module import Module
from MiniTorch.ops.relu import relu
from MiniTorch.ops.sigmoid import sigmoid
from MiniTorch.ops.tanh import tanh


class ReLU(Module):
    def forward(self, x: Variable) -> Variable:  # type: ignore[override]
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Variable) -> Variable:  # type: ignore[override]
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x: Variable) -> Variable:  # type: ignore[override]
        return tanh(x)
