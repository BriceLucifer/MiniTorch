from __future__ import annotations

from typing import Any

from MiniTorch.utils.numer_diff import numerical_diff
from MiniTorch.utils.reshape_sum_backward import reshape_sum_backward
from MiniTorch.utils.type_check import as_array, as_variable

__all__ = [
    "as_array",
    "numerical_diff",
    "as_variable",
    "visualize_graph",
    "reshape_sum_backward",
    "plot_training_history",
    "plot_predictions",
    "plot_weight_histograms",
    "plot_confusion_matrix",
]


def __getattr__(name: str) -> Any:
    """Load optional visualization dependencies only when requested."""
    if name == "visualize_graph":
        from MiniTorch.utils.visualize import visualize_graph

        return visualize_graph
    if name in {
        "plot_training_history",
        "plot_predictions",
        "plot_weight_histograms",
        "plot_confusion_matrix",
    }:
        from MiniTorch.utils import training_viz

        return getattr(training_viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
