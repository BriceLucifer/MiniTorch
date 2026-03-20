from __future__ import annotations

from MiniTorch.core.variable import Variable
from MiniTorch.nn.module import Module


class Sequential(Module):
    """
    A container that chains modules in order.

    Example
    -------
    model = Sequential(
        Linear(784, 256),
        Linear(256, 10),
    )
    out = model(x)
    """

    def __init__(self, *layers: Module) -> None:
        self.layers: list[Module] = list(layers)

    def forward(self, x: Variable) -> Variable:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Variable]:
        params: list[Variable] = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)
