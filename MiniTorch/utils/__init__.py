# MiniTorch/utils/__init__.py
from MiniTorch.utils.numer_diff import numerical_diff
from MiniTorch.utils.type_check import as_array, as_variable
from MiniTorch.utils.visualize import visualize_graph

__all__ = [
    "as_array",
    "numerical_diff",
    "as_variable",
    "visualize_graph",
]
