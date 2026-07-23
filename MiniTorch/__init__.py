from MiniTorch.core import Config, Function, Variable, no_grad, with_grad
from MiniTorch.ops import (
    add,
    broadcast_to,
    cos,
    div,
    exp,
    log,
    matmul,
    mean_squared_error,
    mul,
    neg,
    pow,
    relu,
    reshape,
    sigmoid,
    sin,
    softmax_cross_entropy,
    square,
    sub,
    sum,
    sum_to,
    tanh,
    transpose,
)
from MiniTorch.utils import as_array, numerical_diff

__all__ = [
    # Core
    "Variable",
    "Function",
    "Config",
    "with_grad",
    "no_grad",
    # Arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    # Math
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "square",
    # Activations
    "relu",
    "sigmoid",
    # Matrix / shape
    "matmul",
    "reshape",
    "transpose",
    "broadcast_to",
    "sum",
    "sum_to",
    # Loss
    "mean_squared_error",
    "softmax_cross_entropy",
    # Utilities
    "numerical_diff",
    "as_array",
    "visualize_graph",
    "plot_training_history",
    "plot_predictions",
    "plot_weight_histograms",
    "plot_confusion_matrix",
    "visualize",
]


def __getattr__(name: str):
    """Keep the core import lightweight by lazily loading visual tools."""
    if name == "visualize_graph":
        from MiniTorch.utils.visualize import visualize_graph

        return visualize_graph
    if name == "visualize":
        from MiniTorch.visualization import visualize

        return visualize
    if name in {
        "plot_training_history",
        "plot_predictions",
        "plot_weight_histograms",
        "plot_confusion_matrix",
    }:
        from MiniTorch.utils import training_viz

        return getattr(training_viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
