import numpy as np

from MiniTorch import Variable, visualize_graph


def test_simple_backward():
    x = Variable(np.array([[2.0]]), name="x")
    w = Variable(np.array([[3.0]]), name="w")
    b = Variable(np.array([[1.0]]), name="b")

    y = x * w + b
    y.name = "y"

    target = np.array([[10.0]])
    loss = (y - target) ** 2
    loss.name = "loss"

    loss.backward(retain_grad=True)  # open retain grad to see the inner node's grad

    visualize_graph(loss)

    print(f"Loss: {loss.data}")
    print(f"x.grad: {x.grad.data}")  # 2*(7-10) * 3 = -18
    print(f"w.grad: {w.grad.data}")  # 2*(7-10) * 2 = -12
    print(f"b.grad: {b.grad.data}")  # 2*(7-10) * 1 = -6


if __name__ == "__main__":
    test_simple_backward()
