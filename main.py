import numpy as np

from MiniTorch import Function, Variable, matmul, sum


def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = matmul(x, W) + b
        return y

    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return sum(diff**2) / len(diff)  # type: ignore

    lr = 0.1
    iters = 100
    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.clear_grad()
        b.clear_grad()
        loss.backward()
        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)


if __name__ == "__main__":
    main()
