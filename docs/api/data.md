# Data API

## MNIST

The loader downloads the four standard MNIST archives on first use, caches
them, normalizes images to `float32`, and returns flattened 784-value samples.

::: MiniTorch.data.mnist.load_mnist
    options:
      show_root_heading: true

## DataLoader

`DataLoader` yields NumPy feature and label arrays. Wrap features in a
`Variable` when using eager autograd.

::: MiniTorch.data.dataloader.DataLoader
    options:
      members:
        - __iter__
        - __len__
      show_root_heading: true
