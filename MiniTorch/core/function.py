from typing import List, Optional, Tuple

import numpy.typing as npt

from MiniTorch.core.variable import Variable
from MiniTorch.utils.type_check import as_array


class Function:
    def __init__(self):
        """
        we save input and output of the function
        """
        self.inputs: Tuple = ()
        self.output: Tuple = ()

    """
    using it as a base virtual table
    """

    def __call__(self: "Function", *inputs):
        # we save the input variable
        xs = [x.data for x in inputs]
        # forward()
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        # let the outputs save the creator function
        for output in outputs:
            output.set_creator(self)
        # save the input and output
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError
