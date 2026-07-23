# Operations API

All public operations accept or return `Variable` objects and provide both
symbolic and raw-array gradient paths.

| Category | Functions |
|---|---|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow` |
| Mathematical | `exp`, `log`, `sin`, `cos`, `tanh`, `square` |
| Activation | `relu`, `sigmoid` |
| Matrix and shape | `matmul`, `reshape`, `transpose`, `broadcast_to` |
| Reduction | `sum`, `sum_to` |
| Loss | `mean_squared_error`, `softmax_cross_entropy` |

## Matrix multiplication

::: MiniTorch.ops.matmul.matmul
    options:
      show_root_heading: true

## Reductions

::: MiniTorch.ops.sum.sum
    options:
      show_root_heading: true

::: MiniTorch.ops.sum_to.sum_to
    options:
      show_root_heading: true

## Activations

::: MiniTorch.ops.relu.relu
    options:
      show_root_heading: true

::: MiniTorch.ops.sigmoid.sigmoid
    options:
      show_root_heading: true

::: MiniTorch.ops.tanh.tanh
    options:
      show_root_heading: true

## Losses

::: MiniTorch.ops.meansquarederror.mean_squared_error
    options:
      show_root_heading: true

::: MiniTorch.ops.softmax_cross_entropy.softmax_cross_entropy
    options:
      show_root_heading: true
