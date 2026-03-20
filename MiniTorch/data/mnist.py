"""
MNIST data loader — downloads and caches the dataset locally.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".minitorch_data", "mnist")

# Primary mirrors (tried in order)
_SOURCES: dict[str, list[str]] = {
    "train_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}


def _download(key: str, cache_dir: str) -> str:
    filename = _SOURCES[key][0].split("/")[-1]
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path

    os.makedirs(cache_dir, exist_ok=True)
    for url in _SOURCES[key]:
        try:
            print(f"  Downloading {filename} …", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"  Saved → {path}", flush=True)
            return path
        except Exception as exc:
            print(f"  Failed ({url}): {exc}")

    raise RuntimeError(f"Could not download {filename} from any mirror.")


def _parse_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _parse_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">2I", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def load_mnist(cache_dir: str | None = None):
    """
    Load MNIST, downloading if not already cached.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x : float32 ndarray of shape (N, 784), values in [0, 1]
        y : int32   ndarray of shape (N,),     values in 0–9
    """
    if cache_dir is None:
        cache_dir = _CACHE_DIR

    print("Loading MNIST …")
    x_train = _parse_images(_download("train_images", cache_dir))
    y_train = _parse_labels(_download("train_labels", cache_dir))
    x_test  = _parse_images(_download("test_images",  cache_dir))
    y_test  = _parse_labels(_download("test_labels",  cache_dir))
    print(f"  train: {x_train.shape}  test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)
