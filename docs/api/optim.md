# Optimizer API

Both optimizers update parameter arrays in place and expose `zero_grad()` for
clearing accumulated gradients.

## SGD

::: MiniTorch.optim.sgd.SGD
    options:
      members:
        - step
        - zero_grad
      show_root_heading: true

## Adam

::: MiniTorch.optim.adam.Adam
    options:
      members:
        - step
        - zero_grad
      show_root_heading: true
