from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from MiniTorch.nn.module import Module
from MiniTorch.visualization.architecture import export_architecture


_STATIC_DIR = Path(__file__).with_name("static")


@dataclass(frozen=True)
class ModelVisualization:
    """A generated model visualization that can be saved or opened."""

    path: Path
    architecture: dict

    def show(self) -> ModelVisualization:
        webbrowser.open(self.path.resolve().as_uri())
        return self

    def save(self, filename: str | Path) -> ModelVisualization:
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        return ModelVisualization(destination, self.architecture)


def _render_html(architecture: dict) -> str:
    script_path = _STATIC_DIR / "model-viewer.js"
    style_path = _STATIC_DIR / "model-viewer.css"
    if not script_path.exists() or not style_path.exists():
        raise RuntimeError(
            "The model viewer assets are missing. Build them with "
            "`cd lib/graph-viewer && npm install && npm run build`."
        )

    payload = json.dumps(
        architecture, separators=(",", ":"), ensure_ascii=False
    ).replace("<", "\\u003c")
    css = style_path.read_text(encoding="utf-8")
    javascript = script_path.read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{architecture["model"]["name"]} · MiniTorch</title>
  <style>{css}</style>
</head>
<body>
  <div id="minitorch-model-viewer"></div>
  <script id="minitorch-model-data" type="application/json">{payload}</script>
  <script>{javascript}</script>
</body>
</html>
"""


def visualize(
    model: Module,
    filename: str | Path = "model_architecture.html",
    *,
    input_shape: Sequence[int | None] | None = None,
    open_browser: bool = True,
) -> ModelVisualization:
    """Create a standalone architecture overview and full-neuron map."""
    architecture = export_architecture(model, input_shape=input_shape)
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_html(architecture), encoding="utf-8")
    result = ModelVisualization(destination, architecture)
    if open_browser:
        result.show()
    return result
