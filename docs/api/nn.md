# Neural-network API

## Module

`Module` discovers nested parameters, provides stable names for visualization,
and supports compact model summaries.

::: MiniTorch.nn.module.Module
    options:
      members:
        - forward
        - parameters
        - named_parameters
        - named_modules
        - summary
        - zero_grad
      show_root_heading: true

## Linear

::: MiniTorch.nn.linear.Linear
    options:
      members:
        - forward
      show_root_heading: true

Weights use He initialization and `float32` by default.

## Sequential

::: MiniTorch.nn.sequential.Sequential
    options:
      members:
        - forward
        - parameters
      show_root_heading: true

## Activation modules

`ReLU`, `Sigmoid`, and `Tanh` wrap their matching differentiable operations so
they can be placed directly inside `Sequential`.

::: MiniTorch.nn.activations.ReLU
    options:
      show_root_heading: true

::: MiniTorch.nn.activations.Sigmoid
    options:
      show_root_heading: true

::: MiniTorch.nn.activations.Tanh
    options:
      show_root_heading: true
