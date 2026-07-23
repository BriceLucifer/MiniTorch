# Native training API

The native path supports a static
`Sequential(Linear, ReLU, ..., Linear)` classifier with contiguous `float32`
features and parameters, integer labels, softmax cross-entropy, and Adam.

::: MiniTorch.native.trainer.train
    options:
      show_root_heading: true

::: MiniTorch.native.trainer.NativeTrainingHistory
    options:
      show_root_heading: true
