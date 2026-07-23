from __future__ import annotations

import weakref
from typing import Any, Optional

import numpy as np

from MiniTorch.core.config import Config
from MiniTorch.core.variable import Variable
from MiniTorch.utils.type_check import as_array, as_variable


class Function:
    """Base class for all differentiable operations."""

    def __init__(self) -> None:
        self.inputs: Optional[list[Variable]] = None
        self.outputs: Optional[list[Any]] = None
        self.generation: int = 0

    def __call__(self, *inputs: Any) -> Any:
        vars_in: list[Variable] = [as_variable(x) for x in inputs]
        xs = [x.data for x in vars_in]
        ys = self.forward(*xs)  # type: ignore[arg-type]
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprob:
            self.generation = max(x.generation for x in vars_in)
            for output in outputs:
                output.set_creator(self)
            self.inputs = vars_in
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError

    def backward(self, *gys: Optional[Variable]) -> Any:
        raise NotImplementedError

    def input_data(self, index: int) -> np.ndarray:
        """Return a validated saved input for a raw NumPy backward kernel."""
        if self.inputs is None:
            raise RuntimeError("operation input data is no longer available")
        data = self.inputs[index].data
        if data is None:
            raise RuntimeError("operation input data is no longer available")
        return data

    def output_data(self, index: int) -> np.ndarray:
        """Return a validated output for a raw NumPy backward kernel."""
        if self.outputs is None:
            raise RuntimeError("operation output data is no longer available")
        output = self.outputs[index]()
        if output is None or output.data is None:
            raise RuntimeError("operation output data is no longer available")
        return output.data
