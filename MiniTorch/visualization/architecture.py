from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn.linear import Linear
from MiniTorch.nn.module import Module


def _parameter_metadata(parameter: Variable) -> dict[str, Any]:
    data = parameter.data
    if data is None or data.size == 0:
        return {}
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "count": int(data.size),
    }


def _neuron_scalar(values: np.ndarray | None, index: int) -> float | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] <= index:
        return None
    return float(np.asarray(array[..., index], dtype=np.float64).mean())


def _captured_neuron_state(
    layer: Linear,
    role: str,
    index: int,
) -> tuple[float | None, float | None]:
    reference = (
        layer._last_input_ref if role == "input" else layer._last_output_ref
    )
    variable = reference() if reference is not None else None
    if variable is None:
        return None, None
    value = _neuron_scalar(variable.data, index)
    gradient = _neuron_scalar(
        variable.grad.data
        if variable.grad is not None and variable.grad.data is not None
        else None,
        index,
    )
    return value, gradient


def _neuron_stats(layer: Linear) -> dict[str, list[dict[str, Any]]]:
    weights = layer.W.data
    if weights is None:
        return {"input": [], "output": []}

    input_neurons = []
    for index in range(weights.shape[0]):
        value, gradient = _captured_neuron_state(layer, "input", index)
        input_neurons.append(
            {
                "index": index,
                "value": value,
                "gradient": gradient,
            }
        )

    output_neurons = []
    for index in range(weights.shape[1]):
        value, gradient = _captured_neuron_state(layer, "output", index)
        output_neurons.append(
            {
                "index": index,
                "value": value,
                "gradient": gradient,
            }
        )
    return {"input": input_neurons, "output": output_neurons}


def _leaf_modules(model: Module) -> list[tuple[str, Module]]:
    named = model.named_modules()
    parent_names = {
        name
        for name, _ in named
        if any(
            other_name.startswith(name + ".")
            for other_name, _ in named
            if name and other_name != name
        )
    }
    return [
        (name or "model", module)
        for name, module in named
        if name and name not in parent_names
    ]


def export_architecture(
    model: Module,
    input_shape: Sequence[int | None] | None = None,
) -> dict[str, Any]:
    """Export a stable, JSON-serializable neural-network description."""
    if not isinstance(model, Module):
        raise TypeError("visualize(model) expects a MiniTorch.nn.Module")

    modules = _leaf_modules(model)
    first_linear = next(
        (module for _, module in modules if isinstance(module, Linear)),
        None,
    )
    if input_shape is None:
        if first_linear is None:
            input_shape = (None,)
        else:
            input_shape = (None, first_linear.W.shape[0])

    current_shape = list(input_shape)
    input_neurons = (
        int(current_shape[-1])
        if current_shape and current_shape[-1] is not None
        else None
    )
    nodes: list[dict[str, Any]] = [
        {
            "id": "input",
            "name": "input",
            "type": "Input",
            "shape": current_shape,
            "neurons": input_neurons,
            "parameters": 0,
            "parameterTensors": [],
        }
    ]
    edges: list[dict[str, str]] = []
    previous_id = "input"

    for index, (name, module) in enumerate(modules):
        node_id = f"layer-{index}"
        parameters = module.named_parameters()
        parameter_tensors = [
            {
                "name": parameter_name.rsplit(".", 1)[-1],
                **_parameter_metadata(parameter),
            }
            for parameter_name, parameter in parameters
        ]

        node: dict[str, Any] = {
            "id": node_id,
            "name": name,
            "type": module.__class__.__name__,
            "shape": current_shape,
            "neurons": current_shape[-1] if current_shape else None,
            "parameters": int(sum(parameter.size for _, parameter in parameters)),
            "parameterTensors": parameter_tensors,
        }

        if isinstance(module, Linear):
            in_features, out_features = module.W.shape
            current_shape = [*current_shape[:-1], out_features]
            node.update(
                {
                    "inputFeatures": in_features,
                    "outputFeatures": out_features,
                    "shape": current_shape,
                    "neurons": out_features,
                    "bias": module.b is not None,
                    "dtype": str(module.W.dtype),
                    "neuronStats": _neuron_stats(module),
                }
            )

        nodes.append(node)
        edges.append(
            {
                "id": f"{previous_id}-{node_id}",
                "source": previous_id,
                "target": node_id,
            }
        )
        previous_id = node_id

    return {
        "schemaVersion": 1,
        "model": {
            "name": model.__class__.__name__,
            "totalParameters": int(sum(parameter.size for parameter in model.parameters())),
            "parameterTensors": len(model.parameters()),
            "inputShape": list(input_shape),
            "outputShape": current_shape,
        },
        "nodes": nodes,
        "edges": edges,
    }
