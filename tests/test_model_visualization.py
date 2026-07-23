from pathlib import Path

import numpy as np

from MiniTorch import Variable, sum as tensor_sum
from MiniTorch.nn import Linear, ReLU, Sequential
from MiniTorch.visualization import export_architecture, visualize


def make_model():
    return Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )


def test_export_architecture_describes_layers_and_parameters():
    architecture = export_architecture(make_model())
    assert architecture["schemaVersion"] == 1
    assert architecture["model"]["inputShape"] == [None, 4]
    assert architecture["model"]["outputShape"] == [None, 2]
    assert [node["type"] for node in architecture["nodes"]] == [
        "Input",
        "Linear",
        "ReLU",
        "Linear",
    ]
    assert architecture["model"]["totalParameters"] == 58
    first_linear = architecture["nodes"][1]
    assert len(first_linear["neuronStats"]["input"]) == 4
    assert len(first_linear["neuronStats"]["output"]) == 8


def test_export_architecture_includes_captured_neuron_value_and_grad():
    model = make_model()
    output = model(Variable(np.ones((2, 4), dtype=np.float32)))
    tensor_sum(output).backward(retain_grad=True)
    architecture = export_architecture(model)

    first_input = architecture["nodes"][1]["neuronStats"]["input"][0]
    final_output = architecture["nodes"][3]["neuronStats"]["output"][0]
    assert first_input["value"] == 1.0
    assert first_input["gradient"] is not None
    assert final_output["value"] is not None
    assert final_output["gradient"] is not None
    assert set(final_output) == {"index", "value", "gradient"}


def test_visualize_writes_a_self_contained_html(tmp_path: Path):
    destination = tmp_path / "nested" / "architecture.html"
    result = visualize(make_model(), destination, open_browser=False)
    html = destination.read_text(encoding="utf-8")
    assert result.path == destination
    assert "minitorch-model-data" in html
    assert "react-flow" in html
    assert "Collapse architecture" in html
    assert "Collapse neuron inspector" in html
    assert "representative neurons" not in html
    assert '"totalParameters":58' in html


def test_visualize_adapts_to_a_small_10_to_1_model(tmp_path: Path):
    model = Sequential(Linear(10, 1))
    output = model(Variable(np.ones((1, 10), dtype=np.float32)))
    tensor_sum(output).backward(retain_grad=True)

    result = visualize(
        model,
        tmp_path / "ten-to-one.html",
        open_browser=False,
    )
    architecture = result.architecture
    linear = architecture["nodes"][1]

    assert architecture["model"]["inputShape"] == [None, 10]
    assert architecture["model"]["outputShape"] == [None, 1]
    assert architecture["model"]["totalParameters"] == 11
    assert linear["inputFeatures"] == 10
    assert linear["outputFeatures"] == 1
    assert len(linear["neuronStats"]["input"]) == 10
    assert len(linear["neuronStats"]["output"]) == 1
    assert linear["neuronStats"]["output"][0]["value"] is not None
    assert linear["neuronStats"]["output"][0]["gradient"] == 1.0

    html = result.path.read_text(encoding="utf-8")
    assert '"inputFeatures":10' in html
    assert '"outputFeatures":1' in html
