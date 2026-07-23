from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, Optional

from MiniTorch.core.config import Config, using_config

if TYPE_CHECKING:
    from MiniTorch.core.function import Function


class Variable:
    __array_priority__ = 200
    __slots__ = ("data", "name", "grad", "creator", "generation", "__weakref__")

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

        if self.creator is None:
            return

        # Build a compact topological tape in O(functions + edges). Forward
        # execution already defines a DAG, so backward can simply replay this
        # tape in reverse instead of repeatedly sorting a ready list.
        tape: list[Function] = []
        discovered: set[Function] = set()
        stack: list[tuple[Function, bool]] = [(self.creator, False)]

        while stack:
            func, expanded = stack.pop()
            if expanded:
                tape.append(func)
                continue
            if func in discovered:
                continue
            discovered.add(func)
            stack.append((func, True))
            if func.inputs is not None:
                for input_var in func.inputs:
                    if input_var.creator is not None and input_var.creator not in discovered:
                        stack.append((input_var.creator, False))

        for func in reversed(tape):
            if func.outputs is None or func.inputs is None:
                continue

            outputs = [output_ref() for output_ref in func.outputs]
            if create_graph:
                output_grads = [
                    output.grad if output is not None else None
                    for output in outputs
                ]
                with using_config("enable_backprob", True):
                    input_grads = func.backward(*output_grads)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for input_var, grad in zip(func.inputs, input_grads):
                    if grad is None:
                        continue
                    input_var.grad = grad if input_var.grad is None else input_var.grad + grad
            else:
                output_grad_arrays = [
                    output.grad.data
                    if output is not None and output.grad is not None
                    else None
                    for output in outputs
                ]
                array_backward = getattr(func, "backward_array", None)
                try:
                    if array_backward is None:
                        raise NotImplementedError
                    input_grad_arrays = array_backward(*output_grad_arrays)
                except NotImplementedError:
                    # Compatibility path for third-party operations that have
                    # not implemented the raw-array training API yet.
                    output_grads = [
                        output.grad if output is not None else None
                        for output in outputs
                    ]
                    with using_config("enable_backprob", False):
                        legacy_grads = func.backward(*output_grads)
                    if not isinstance(legacy_grads, tuple):
                        legacy_grads = (legacy_grads,)
                    input_grad_arrays = tuple(
                        grad.data if isinstance(grad, Variable) else grad
                        for grad in legacy_grads
                    )

                if not isinstance(input_grad_arrays, tuple):
                    input_grad_arrays = (input_grad_arrays,)

                for input_var, grad_array in zip(func.inputs, input_grad_arrays):
                    if grad_array is None:
                        continue
                    grad_array = np.asarray(grad_array)
                    if input_var.grad is None:
                        input_var.grad = Variable(grad_array)
                    else:
                        # Allocate the accumulation result instead of mutating
                        # in place: some derivative rules intentionally return
                        # aliased upstream gradients.
                        input_var.grad.data = np.asarray(
                            input_var.grad.data + grad_array
                        )

            if not retain_grad and not create_graph:
                for output in outputs:
                    if output is not None and output is not self:
                        output.grad = None
