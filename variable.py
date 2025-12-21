from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from function import Function


class Variable:
    def __init__(self, data: npt.NDArray) -> None:
        self.data: npt.NDArray = data
        self.grad: Optional[npt.NDArray] = None
        self.creator: Optional["Function"] = None

    def set_creator(self, func: "Function"):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list["Function"] = [self.creator]  # type: ignore
        # bfs
        while funcs:
            f: "Function" = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)  # type: ignore

            if x.creator is not None:  # type: ignore
                funcs.append(x.creator)  # type: ignore
