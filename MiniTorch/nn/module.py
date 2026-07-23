from __future__ import annotations

from MiniTorch.core.variable import Variable


class Module:
    """
    Base class for all neural network modules.

    Subclass this and implement forward(). Parameters are discovered
    automatically by scanning instance attributes for Variable objects.
    """

    def __call__(self, *inputs: Variable) -> Variable:
        return self.forward(*inputs)

    def forward(self, *inputs: Variable) -> Variable:
        raise NotImplementedError

    def parameters(self) -> list[Variable]:
        """Return all leaf Variable parameters (recursive)."""
        return [parameter for _, parameter in self.named_parameters()]

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Variable]]:
        """Return stable, deduplicated parameter names and values."""
        result: list[tuple[str, Variable]] = []
        seen: set[int] = set()

        def visit(value, path: str) -> None:
            if isinstance(value, Variable):
                if id(value) not in seen:
                    seen.add(id(value))
                    result.append((path, value))
            elif isinstance(value, Module):
                for name, child in value.__dict__.items():
                    visit(child, f"{path}.{name}" if path else name)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{path}.{index}" if path else str(index))

        for name, value in self.__dict__.items():
            visit(value, f"{prefix}.{name}" if prefix else name)
        return result

    def named_modules(self, prefix: str = "") -> list[tuple[str, Module]]:
        """Return this module and all registered child modules."""
        result: list[tuple[str, Module]] = [(prefix, self)]
        seen = {id(self)}

        def visit(value, path: str) -> None:
            if isinstance(value, Module):
                if id(value) in seen:
                    return
                seen.add(id(value))
                result.append((path, value))
                for name, child in value.__dict__.items():
                    visit(child, f"{path}.{name}" if path else name)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{path}.{index}" if path else str(index))

        for name, value in self.__dict__.items():
            visit(value, f"{prefix}.{name}" if prefix else name)
        return result

    def summary(self) -> str:
        """Return a compact architecture and parameter summary."""
        lines = ["Name                  Type             Parameters"]
        lines.append("-" * 55)
        total = 0
        for name, module in self.named_modules():
            if not name:
                continue
            count = sum(parameter.size for parameter in module.parameters())
            total += count if not any(
                other_name.startswith(name + ".")
                for other_name, _ in self.named_modules()
                if other_name != name
            ) else 0
            lines.append(
                f"{name:<21} {module.__class__.__name__:<16} {count:>10,}"
            )
        lines.append("-" * 55)
        lines.append(f"Total parameters: {len(self.parameters()):,} tensors / "
                     f"{sum(p.size for p in self.parameters()):,} values")
        return "\n".join(lines)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.clear_grad()

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for name, value in self.__dict__.items():
            if isinstance(value, (Module, Variable)):
                lines.append(f"  ({name}): {repr(value)}")
        lines.append(")")
        return "\n".join(lines)
