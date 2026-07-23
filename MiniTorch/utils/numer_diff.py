from MiniTorch.core.variable import Variable


def numerical_diff(f, x: Variable, eps=1e-4):
    """
    numeriacl_differential function:
        params:
            f: function which is callable
            x: Variable
            eps: 1e-4 default
    """
    if x.data is None:
        raise ValueError("numerical_diff requires variable data")
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    if y0.data is None or y1.data is None:
        raise ValueError("numerical_diff function returned an empty variable")
    return (y1.data - y0.data) / (2 * eps)
