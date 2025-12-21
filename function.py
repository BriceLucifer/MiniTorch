from typing import Optional

import numpy as np
import numpy.typing as npt

from variable import Variable


class Function:
    def __init__(self):
        """
        we save input and output of the function
        """
        self.input: Optional[Variable] = None
        self.output: Optional[Variable] = None

    """
    using it as a base virtual table
    """

    def __call__(self, input: Variable) -> Variable:
        # we save the input variable
        x = input.data
        # forward()
        y = self.forward(x)
        output = Variable(y)
        # let the output save the creator function
        output.set_creator(self)
        # save the input and output
        self.input = input
        self.output = output
        return output

    def forward(self, x: npt.NDArray):
        raise NotImplementedError

    def backward(self, gy: npt.NDArray):
        raise NotImplementedError


class Square(Function):
    """
    implement a Square Function based on Funtion object
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: npt.NDArray):
        return x**2

    def backward(self, gy: npt.NDArray):
        if self.input is not None:
            x = self.input.data
            gx = 2 * x * gy
            return gx


# more easy to use
def square(x: Variable) -> Variable:
    """
    square():
        params:
            x: Variable
        return:
            Variable
    """
    f: Square = Square()
    return f(x)


class Exp(Function):
    """
    implement a Exp Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: npt.NDArray):
        return np.exp(x)

    def backward(self, gy: npt.NDArray):
        if self.input is not None:
            x = self.input.data
            return np.exp(x) * gy


def exp(x: Variable) -> Variable:
    """
    exp():
        params:
            x: Variable
        return:
            Variable
    """
    f: Exp = Exp()
    return f(x)
