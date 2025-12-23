import math

import matplotlib.pyplot as plt
import numpy as np

from MiniTorch import Variable, cos, sin, tanh, visualize_graph


def sphere(x, y):
    z = x**2 + y**2
    return z


def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


def f(x):
    y = x**4 - 2 * x**2
    return y


def main():
    x = Variable(np.array(1.0))
    y = tanh(x)
    x.name = "x"
    y.name = "y"  # type: ignore
    y.backward(create_graph=True)  # type: ignore
    iters = 0
    for i in range(iters):
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)
    gx = x.grad
    gx.name = "gx" + str(iters + 1)
    visualize_graph(gx, filename="graph.html")


if __name__ == "__main__":
    main()
