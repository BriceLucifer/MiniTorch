"""Create the interactive scientific model explorer.

Run with:
    uv run python examples/model_visualization.py
"""

import numpy as np

from MiniTorch import Variable, sum as tensor_sum, visualize
from MiniTorch.nn import Linear, ReLU, Sequential


model = Sequential(
    Linear(64, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
)

# Values depend on an input, and gradients depend on a backward pass. Keep
# intermediate gradients for the inspector before exporting the viewer.
probe = Variable(np.random.default_rng(7).normal(size=(1, 64)).astype(np.float32))
probe_output = model(probe)
probe_loss = tensor_sum(probe_output)
probe_loss.backward(retain_grad=True)

result = visualize(
    model,
    filename="model_architecture.html",
    input_shape=(None, 64),
)
print(f"Wrote {result.path}")
