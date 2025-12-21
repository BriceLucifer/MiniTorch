import numpy as np

from function import Exp, Square, exp, square
from variable import Variable


def f(x: Variable):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


def main():
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
