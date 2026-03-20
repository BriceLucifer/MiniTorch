from MiniTorch.nn.module import Module


class Sequential(Module):
    """
    A container that chains modules in order.

    Example
    -------
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
    )
    out = model(x)
    """

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)
