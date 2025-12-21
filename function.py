import numpy as np

from variable import Variable


class Function:
    """
    using it as a base virtual table
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    """
    implement a Square Function based on Funtion object
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x**2


class Exp(Function):
    """
    implement a Exp Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.exp(x)
