import unittest

import numpy as np

from MiniTorch import Variable, numerical_diff, square


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(expected, y.data)  # type: ignore

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()  # type: ignore
        expected = np.array(6.0)
        self.assertEqual(x.grad.data, expected)  # type: ignore

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)  # type: ignore
        y.backward()  # type: ignore
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad.data, num_grad)  # type: ignore
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main()
