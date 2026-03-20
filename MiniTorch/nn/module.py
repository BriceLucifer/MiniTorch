from MiniTorch.core.variable import Variable


class Module:
    """
    Base class for all neural network modules.

    Subclass this and implement forward(). Parameters are discovered
    automatically by scanning instance attributes for Variable objects.
    """

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError

    def parameters(self):
        """Return all leaf Variable parameters (recursive)."""
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Variable):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Variable):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.clear_grad()

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for name, value in self.__dict__.items():
            if isinstance(value, (Module, Variable)):
                lines.append(f"  ({name}): {repr(value)}")
        lines.append(")")
        return "\n".join(lines)
