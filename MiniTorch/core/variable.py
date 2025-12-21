from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from MiniTorch.core.function import Function


class Variable:
    def __init__(self, data) -> None:
        # I use type check to avoid input problem
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: "Function"):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list["Function"] = [self.creator]  # type: ignore
        # bfs
        while funcs:
            f: "Function" = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)
