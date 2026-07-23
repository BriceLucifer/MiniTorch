from __future__ import annotations

import numpy as np
cimport numpy as cnp

cnp.import_array()


def train_mlp(
    list weights,
    list biases,
    cnp.ndarray x,
    cnp.ndarray labels,
    int epochs,
    int batch_size,
    double lr,
    bint shuffle,
    int seed,
):
    """Compiled control loop for Linear/ReLU classifiers with Adam."""
    cdef Py_ssize_t sample_count = x.shape[0]
    cdef Py_ssize_t layer_count = len(weights)
    cdef Py_ssize_t epoch
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t layer_index
    cdef Py_ssize_t batch_count
    cdef long step = 0
    cdef double beta1 = 0.9
    cdef double beta2 = 0.999
    cdef double epsilon = 1e-8
    cdef double learning_rate
    cdef double epoch_loss

    cdef object rng = np.random.default_rng(seed)
    cdef cnp.ndarray indices = np.arange(sample_count, dtype=np.int64)
    cdef cnp.ndarray batch_indices
    cdef cnp.ndarray xb
    cdef cnp.ndarray yb
    cdef cnp.ndarray z
    cdef cnp.ndarray probabilities
    cdef cnp.ndarray delta
    cdef cnp.ndarray previous_delta
    cdef cnp.ndarray grad_weight
    cdef cnp.ndarray grad_bias
    cdef cnp.ndarray previous_activation
    cdef list activations
    cdef list preactivations
    cdef list losses = []
    cdef list first_weight_moments = [np.zeros_like(weight) for weight in weights]
    cdef list second_weight_moments = [np.zeros_like(weight) for weight in weights]
    cdef list first_bias_moments = [np.zeros_like(bias) for bias in biases]
    cdef list second_bias_moments = [np.zeros_like(bias) for bias in biases]
    cdef list last_weight_gradients = [np.zeros_like(weight) for weight in weights]
    cdef list last_bias_gradients = [np.zeros_like(bias) for bias in biases]

    for epoch in range(epochs):
        if shuffle:
            rng.shuffle(indices)
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_indices = indices[start:end]
            xb = x[batch_indices]
            yb = labels[batch_indices]
            activations = [xb]
            preactivations = []

            for layer_index in range(layer_count):
                z = np.matmul(activations[layer_index], weights[layer_index])
                z += biases[layer_index]
                preactivations.append(z)
                if layer_index + 1 < layer_count:
                    activations.append(np.maximum(z, 0.0))
                else:
                    activations.append(z)

            z = activations[layer_count]
            z = z - z.max(axis=1, keepdims=True)
            probabilities = np.exp(z)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            epoch_loss += float(
                -np.log(
                    probabilities[np.arange(end - start), yb] + 1e-12
                ).mean()
            )
            batch_count += 1
            step += 1

            delta = probabilities.copy()
            delta[np.arange(end - start), yb] -= 1.0
            delta /= end - start

            learning_rate = (
                lr
                * ((1.0 - beta2 ** step) ** 0.5)
                / (1.0 - beta1 ** step)
            )

            for layer_index in range(layer_count - 1, -1, -1):
                previous_activation = activations[layer_index]
                grad_weight = np.matmul(previous_activation.T, delta)
                grad_bias = delta.sum(axis=0)
                last_weight_gradients[layer_index] = grad_weight.copy()
                last_bias_gradients[layer_index] = grad_bias.copy()

                if layer_index > 0:
                    previous_delta = np.matmul(delta, weights[layer_index].T)
                    previous_delta *= preactivations[layer_index - 1] > 0.0

                first_weight_moments[layer_index] *= beta1
                first_weight_moments[layer_index] += (1.0 - beta1) * grad_weight
                second_weight_moments[layer_index] *= beta2
                second_weight_moments[layer_index] += (
                    (1.0 - beta2) * grad_weight * grad_weight
                )
                first_bias_moments[layer_index] *= beta1
                first_bias_moments[layer_index] += (1.0 - beta1) * grad_bias
                second_bias_moments[layer_index] *= beta2
                second_bias_moments[layer_index] += (
                    (1.0 - beta2) * grad_bias * grad_bias
                )

                weights[layer_index] -= (
                    learning_rate
                    * first_weight_moments[layer_index]
                    / (np.sqrt(second_weight_moments[layer_index]) + epsilon)
                )
                biases[layer_index] -= (
                    learning_rate
                    * first_bias_moments[layer_index]
                    / (np.sqrt(second_bias_moments[layer_index]) + epsilon)
                )

                if layer_index > 0:
                    delta = previous_delta

        losses.append(epoch_loss / batch_count)

    return {
        "losses": losses,
        "steps": step,
        "samples": sample_count * epochs,
        "weight_gradients": last_weight_gradients,
        "bias_gradients": last_bias_gradients,
    }
