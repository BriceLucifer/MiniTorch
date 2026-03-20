from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, Optional

from MiniTorch.core.config import Config, using_config

if TYPE_CHECKING:
    from MiniTorch.core.function import Function


class Variable:
    __array_priority__ = 200

    def __init__(self, data: Optional[np.ndarray], name: Optional[str] = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data: Optional[np.ndarray] = data
        self.name: Optional[str] = name
        self.grad: Optional[Variable] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape  # type: ignore[union-attr]

    @property
    def ndim(self) -> int:
        return self.data.ndim  # type: ignore[union-attr]

    @property
    def size(self) -> int:
        return self.data.size  # type: ignore[union-attr]

    @property
    def dtype(self) -> np.dtype:  # type: ignore[type-arg]
        return self.data.dtype  # type: ignore[union-attr]

    def __len__(self) -> int:
        return len(self.data)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def __mul__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.mul import mul
        return mul(self, other)

    def __rmul__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.mul import mul
        return mul(self, other)

    def __add__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.add import add
        return add(self, other)

    def __radd__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.add import add
        return add(self, other)

    def __neg__(self) -> Variable:
        from MiniTorch.ops.neg import neg
        return neg(self)

    def __sub__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.sub import sub
        return sub(self, other)

    def __rsub__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.sub import rsub
        return rsub(self, other)

    def __truediv__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.div import div
        return div(self, other)

    def __rtruediv__(self, other: Variable | float | int) -> Variable:
        from MiniTorch.ops.div import rdiv
        return rdiv(self, other)

    def __pow__(self, other: float | int) -> Variable:
        from MiniTorch.ops.pow import pow
        return pow(self, other)

    def reshape(self, *shape: Any) -> Variable:
        resolved: Any = shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
        from MiniTorch.ops.reshape import reshape
        return reshape(self, resolved)

    @property
    def T(self) -> Variable:
        from MiniTorch.ops.transpose import transpose
        return transpose(self)

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        if not Config.enable_backprob:
            raise RuntimeError("backward() is not allowed in no_grad mode")

        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)  # type: ignore[arg-type]

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # type: ignore[union-attr, misc]

            with using_config("enable_backprob", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):  # type: ignore[arg-type]
                    if gx is None:
                        continue
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if (not retain_grad) and (not create_graph):
                for y in f.outputs:  # type: ignore[union-attr]
                    y().grad = None  # type: ignore[misc]
