import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear, ReLU, Sequential


def test_backward_on_leaf_seeds_gradient():
    x = Variable(np.array([1.0, 2.0]))
    x.backward()
    np.testing.assert_array_equal(x.grad.data, np.ones(2))


def test_wide_graph_accumulates_all_branches():
    x = Variable(np.array(2.0))
    branches = [x * float(index) for index in range(1, 101)]
    result = branches[0]
    for branch in branches[1:]:
        result = result + branch
    result.backward()
    np.testing.assert_allclose(x.grad.data, sum(range(1, 101)))


def test_create_graph_keeps_symbolic_second_derivative():
    x = Variable(np.array(3.0))
    y = x**3
    y.backward(create_graph=True)
    first_derivative = x.grad
    x.clear_grad()
    first_derivative.backward()
    np.testing.assert_allclose(x.grad.data, 18.0)


def test_linear_defaults_to_float32():
    layer = Linear(4, 3)
    assert layer.W.data.dtype == np.float32
    assert layer.b.data.dtype == np.float32


def test_named_parameters_are_stable_and_deduplicated():
    shared = Linear(3, 3)
    model = Sequential(shared, ReLU(), shared)
    assert [name for name, _ in model.named_parameters()] == [
        "layers.0.W",
        "layers.0.b",
    ]
