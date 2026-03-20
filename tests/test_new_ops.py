"""
Tests for new ops: relu, sigmoid, log, softmax_cross_entropy
and nn / optim components.
"""
import numpy as np
import pytest

import MiniTorch as mt
from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear, Sequential
from MiniTorch.ops.log import log
from MiniTorch.ops.relu import relu
from MiniTorch.ops.sigmoid import sigmoid
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim import Adam, SGD
from MiniTorch.utils.numer_diff import numerical_diff


# ─────────────────────────────────────────────────────────────────────────────
# ReLU
# ─────────────────────────────────────────────────────────────────────────────

class TestReLU:
    def test_forward_positive(self):
        x = Variable(np.array([1.0, 2.0, 3.0]))
        y = relu(x)
        np.testing.assert_allclose(y.data, [1.0, 2.0, 3.0])

    def test_forward_negative(self):
        x = Variable(np.array([-1.0, -2.0, 0.0]))
        y = relu(x)
        np.testing.assert_allclose(y.data, [0.0, 0.0, 0.0])

    def test_forward_mixed(self):
        x = Variable(np.array([-1.0, 0.0, 1.0, 2.0]))
        y = relu(x)
        np.testing.assert_allclose(y.data, [0.0, 0.0, 1.0, 2.0])

    def test_backward(self):
        x = Variable(np.array([-1.0, 0.5, 2.0]))
        y = relu(x)
        y.backward(retain_grad=True)
        np.testing.assert_allclose(x.grad.data, [0.0, 1.0, 1.0])

    def test_gradient_check(self):
        x = Variable(np.array([1.0]))          # positive region only
        grad_auto = relu(x).data               # we just want forward here
        x2 = Variable(np.array([1.0]))
        y2 = relu(x2)
        y2.backward()
        nd = numerical_diff(relu, Variable(np.array([1.0])))
        np.testing.assert_allclose(x2.grad.data, nd, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Sigmoid
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_forward_zero(self):
        x = Variable(np.array([0.0]))
        y = sigmoid(x)
        np.testing.assert_allclose(y.data, [0.5], atol=1e-7)

    def test_forward_large(self):
        x = Variable(np.array([100.0]))
        y = sigmoid(x)
        np.testing.assert_allclose(y.data, [1.0], atol=1e-6)

    def test_forward_large_neg(self):
        x = Variable(np.array([-100.0]))
        y = sigmoid(x)
        np.testing.assert_allclose(y.data, [0.0], atol=1e-6)

    def test_backward(self):
        x = Variable(np.array([0.0]))
        y = sigmoid(x)
        y.backward()
        # σ'(0) = 0.25
        np.testing.assert_allclose(x.grad.data, [0.25], atol=1e-7)

    def test_gradient_check(self):
        x = Variable(np.array([0.5]))
        nd = numerical_diff(sigmoid, Variable(np.array([0.5])))
        y = sigmoid(Variable(np.array([0.5])))
        y.backward()
        np.testing.assert_allclose(
            Variable(np.array([0.5])).grad if False else nd,
            nd, rtol=1e-3
        )
        # Direct check
        x2 = Variable(np.array([0.5]))
        sigmoid(x2).backward()
        np.testing.assert_allclose(x2.grad.data, nd, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Log
# ─────────────────────────────────────────────────────────────────────────────

class TestLog:
    def test_forward(self):
        x = Variable(np.array([1.0, np.e]))
        y = log(x)
        np.testing.assert_allclose(y.data, [0.0, 1.0], atol=1e-7)

    def test_backward(self):
        x = Variable(np.array([2.0]))
        log(x).backward()
        np.testing.assert_allclose(x.grad.data, [0.5], atol=1e-7)

    def test_gradient_check(self):
        x = Variable(np.array([2.0]))
        nd = numerical_diff(log, Variable(np.array([2.0])))
        x2 = Variable(np.array([2.0]))
        log(x2).backward()
        np.testing.assert_allclose(x2.grad.data, nd, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Softmax Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftmaxCrossEntropy:
    def test_forward_perfect_prediction(self):
        # If the correct class has a very large logit, loss ≈ 0
        logits = Variable(np.array([[100.0, 0.0, 0.0]]))
        labels = Variable(np.array([0]))
        loss = softmax_cross_entropy(logits, labels)
        assert float(loss.data) < 1e-3

    def test_forward_uniform(self):
        # Uniform logits → loss = log(n_classes)
        n = 5
        logits = Variable(np.zeros((1, n)))
        labels = Variable(np.array([0]))
        loss = softmax_cross_entropy(logits, labels)
        np.testing.assert_allclose(float(loss.data), np.log(n), atol=1e-5)

    def test_backward_shape(self):
        N, C = 4, 10
        logits = Variable(np.random.randn(N, C))
        labels = Variable(np.arange(N))
        loss = softmax_cross_entropy(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.data.shape == (N, C)

    def test_backward_gradient_check(self):
        """Finite-difference check on the loss w.r.t. a single logit."""
        np.random.seed(0)
        N, C = 2, 4
        x_data = np.random.randn(N, C)
        t_data = np.array([1, 2])
        eps = 1e-5

        # Analytic gradient
        x_var = Variable(x_data.copy())
        t_var = Variable(t_data.copy())
        loss = softmax_cross_entropy(x_var, t_var)
        loss.backward()
        analytic_grad = x_var.grad.data.copy()

        # Numerical gradient for one element (0, 0)
        def f_scalar(delta):
            xd = x_data.copy()
            xd[0, 0] += delta
            l = softmax_cross_entropy(Variable(xd), Variable(t_data.copy()))
            return float(l.data)

        num_grad_00 = (f_scalar(eps) - f_scalar(-eps)) / (2 * eps)
        np.testing.assert_allclose(analytic_grad[0, 0], num_grad_00, rtol=1e-4)

    def test_batch(self):
        N, C = 32, 10
        logits = Variable(np.random.randn(N, C))
        labels = Variable(np.random.randint(0, C, size=N))
        loss = softmax_cross_entropy(logits, labels)
        assert loss.data.shape == ()   # scalar


# ─────────────────────────────────────────────────────────────────────────────
# Linear layer
# ─────────────────────────────────────────────────────────────────────────────

class TestLinear:
    def test_output_shape(self):
        layer = Linear(4, 8)
        x = Variable(np.random.randn(3, 4))
        y = layer(x)
        assert y.shape == (3, 8)

    def test_no_bias(self):
        layer = Linear(4, 8, bias=False)
        assert layer.b is None
        x = Variable(np.random.randn(2, 4))
        y = layer(x)
        assert y.shape == (2, 8)

    def test_parameters_count(self):
        layer = Linear(4, 8)
        params = layer.parameters()
        assert len(params) == 2           # W and b

    def test_parameters_no_bias(self):
        layer = Linear(4, 8, bias=False)
        params = layer.parameters()
        assert len(params) == 1           # W only

    def test_backward(self):
        layer = Linear(3, 2)
        x = Variable(np.ones((1, 3)))
        y = layer(x)
        loss = mt.sum(y)
        loss.backward()
        assert layer.W.grad is not None
        assert layer.b.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Sequential
# ─────────────────────────────────────────────────────────────────────────────

class TestSequential:
    def _make_model(self):
        from MiniTorch.nn import Sequential
        return Sequential(Linear(4, 8), Linear(8, 2))

    def test_forward_shape(self):
        model = self._make_model()
        x = Variable(np.random.randn(5, 4))
        y = model(x)
        assert y.shape == (5, 2)

    def test_parameters(self):
        model = self._make_model()
        params = model.parameters()
        # Two Linear layers × 2 params each = 4
        assert len(params) == 4


# ─────────────────────────────────────────────────────────────────────────────
# SGD optimiser
# ─────────────────────────────────────────────────────────────────────────────

class TestSGD:
    def test_parameter_update(self):
        layer = Linear(2, 1)
        opt   = SGD(layer.parameters(), lr=0.1)
        x     = Variable(np.ones((1, 2)))
        t     = Variable(np.array([[1.0]]))
        loss  = mt.mean_squared_error(layer(x), t)
        loss.backward()
        w_before = layer.W.data.copy()
        opt.step()
        assert not np.allclose(layer.W.data, w_before)

    def test_zero_grad(self):
        layer = Linear(2, 1)
        opt   = SGD(layer.parameters(), lr=0.1)
        x     = Variable(np.ones((1, 2)))
        loss  = mt.sum(layer(x))
        loss.backward()
        opt.zero_grad()
        for p in layer.parameters():
            assert p.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser
# ─────────────────────────────────────────────────────────────────────────────

class TestAdam:
    def test_loss_decreases(self):
        """Adam should reduce a simple linear regression loss over 50 steps."""
        np.random.seed(0)
        layer = Linear(4, 1)
        opt   = Adam(layer.parameters(), lr=1e-2)
        x     = Variable(np.random.randn(32, 4))
        t     = Variable(np.random.randn(32, 1))

        losses = []
        for _ in range(50):
            loss = mt.mean_squared_error(layer(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0], "Adam did not reduce the loss"

    def test_step_counter(self):
        layer = Linear(2, 1)
        opt   = Adam(layer.parameters(), lr=1e-3)
        x     = Variable(np.ones((1, 2)))
        for _ in range(3):
            mt.sum(layer(x)).backward()
            opt.step()
            opt.zero_grad()
        assert opt.t == 3


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: tiny XOR-like training
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_mlp_learns_linear(self):
        """A 2-layer MLP should fit a linear function in a few hundred steps."""
        np.random.seed(1)
        N = 64
        x_np = np.random.randn(N, 2)
        y_np = (x_np @ np.array([1.0, -1.0]) > 0).astype(np.int32)

        l1 = Linear(2, 16)
        l2 = Linear(16, 2)
        params = l1.parameters() + l2.parameters()
        opt = Adam(params, lr=5e-3)

        for _ in range(200):
            x_var = Variable(x_np)
            t_var = Variable(y_np)
            logits = l2(relu(l1(x_var)))
            loss   = softmax_cross_entropy(logits, t_var)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Final accuracy should be reasonable
        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y_np).mean()
        assert acc > 0.80, f"Expected >80% accuracy, got {acc*100:.1f}%"
