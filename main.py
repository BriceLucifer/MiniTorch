import numpy as np

from function import Exp, Function, Square
from numer_diff import numerical_diff
from variable import Variable


def f(x: Variable):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


def main():
    data = np.array(2.0)
    x = Variable(data)
    print(x.data)

    f = Square()
    y = f(x)
    dy = numerical_diff(f, x)
    print(y.data)
    print(dy)
    print(type(y))


if __name__ == "__main__":
    main()
