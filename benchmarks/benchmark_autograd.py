"""Repeatable NumPy/autograd microbenchmarks for MiniTorch.

Run:
    python benchmarks/benchmark_autograd.py
    python benchmarks/benchmark_autograd.py --quick --json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from MiniTorch.core.variable import Variable
from MiniTorch.nn import Linear, ReLU, Sequential
from MiniTorch.ops.softmax_cross_entropy import softmax_cross_entropy
from MiniTorch.optim import Adam


def measure(
    operation: Callable[[], None],
    *,
    warmups: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "minimum_ms": min(samples),
    }


def chain_backward(nodes: int) -> None:
    x = Variable(np.array(1.0))
    y = x
    for _ in range(nodes // 2):
        y = y * 1.000001 + 0.000001
    y.backward()


def fanout_backward(branches: int) -> None:
    x = Variable(np.array(1.0))
    outputs = [x * (index + 1.0) for index in range(branches)]
    result = outputs[0]
    for output in outputs[1:]:
        result = result + output
    result.backward()


def training_step_factory(batch_size: int = 256) -> Callable[[], None]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch_size, 784)).astype(np.float32)
    labels = rng.integers(0, 10, size=batch_size, dtype=np.int32)
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    )
    optimizer = Adam(model.parameters())

    def step() -> None:
        logits = model(Variable(x))
        loss = softmax_cross_entropy(logits, Variable(labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return step


def run(quick: bool = False) -> dict[str, Any]:
    repeats = 5 if quick else 30
    warmups = 1 if quick else 5
    training_step = training_step_factory()
    scenarios: dict[str, Callable[[], None]] = {
        "chain_2000_nodes": lambda: chain_backward(2_000),
        "fanout_1000_branches": lambda: fanout_backward(1_000),
        "mlp_train_batch_256": training_step,
    }
    results = {
        name: measure(operation, warmups=warmups, repeats=repeats)
        for name, operation in scenarios.items()
    }
    results["mlp_train_batch_256"]["samples_per_second"] = (
        256_000 / results["mlp_train_batch_256"]["median_ms"]
    )
    return {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "dtype": "float32",
        },
        "settings": {"warmups": warmups, "repeats": repeats},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    report = run(quick=arguments.quick)
    if arguments.json:
        print(json.dumps(report, indent=2))
        return

    print("MiniTorch NumPy performance")
    print(f"Python {report['environment']['python']} · NumPy {report['environment']['numpy']}")
    for name, values in report["results"].items():
        suffix = (
            f" · {values['samples_per_second']:,.0f} samples/s"
            if "samples_per_second" in values
            else ""
        )
        print(
            f"{name:<25} {values['median_ms']:>8.3f} ms median"
            f" · {values['p95_ms']:>8.3f} ms p95{suffix}"
        )


if __name__ == "__main__":
    main()
