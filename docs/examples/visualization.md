# Graph Visualization

MiniTorchBR can render the computation graph as an interactive HTML file using PyVis.

## Basic Graph Rendering

```python
import numpy as np
from MiniTorch.core.variable import Variable
from MiniTorch.utils.visualize import visualize_graph

x = Variable(np.array([1.0, 2.0]))
w = Variable(np.array([0.5, -0.5]))

y = (x * w).sum()
y.backward()

# Renders an interactive HTML file
visualize_graph(y, output_path="graph.html")
```

Open `graph.html` in any browser. Nodes represent variables and edges represent operations. You can drag, zoom, and hover for details.

## Larger Network Graph

```python
from MiniTorch.nn.linear import Linear
from MiniTorch.ops.relu import relu
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy

x_np = np.random.randn(4, 8)
x = Variable(x_np)

fc1 = Linear(8, 4)
fc2 = Linear(4, 2)

h = relu(fc1(x))
logits = fc2(h)
loss = softmax_cross_entropy(logits, np.array([0, 1, 0, 1]))
loss.backward()

visualize_graph(loss, output_path="network_graph.html")
```

## Training Curves

```python
from MiniTorch.utils.training_viz import plot_training_history

history = {
    "train_loss": [0.95, 0.72, 0.55, 0.42, 0.34],
    "val_loss":   [0.98, 0.75, 0.58, 0.46, 0.39],
    "train_acc":  [0.65, 0.76, 0.83, 0.87, 0.90],
    "val_acc":    [0.63, 0.74, 0.81, 0.85, 0.88],
}

plot_training_history(history)   # saves training_history.png
```

## Weight Histograms

```python
from MiniTorch.utils.training_viz import plot_weight_histograms

plot_weight_histograms(model)    # saves weight_histograms.png
```

Histograms show the distribution of weights in each layer — useful for diagnosing vanishing/exploding gradients.

## Confusion Matrix

```python
from MiniTorch.utils.training_viz import plot_confusion_matrix
import numpy as np

y_true = np.array([0, 1, 2, 1, 0, 2])
y_pred = np.array([0, 1, 2, 0, 0, 1])

plot_confusion_matrix(y_true, y_pred, class_names=["cat", "dog", "bird"])
# saves confusion_matrix.png
```
