import numpy as np

from MiniTorch import Variable, visualize_graph


def sphere(x, y):
    z = x**2 + y**2
    return z


def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def main():
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))

    z = sphere(x, y)
    z.backward(retain_grad=True)
    visualize_graph(z, "graph.html")


if __name__ == "__main__":
    main()
