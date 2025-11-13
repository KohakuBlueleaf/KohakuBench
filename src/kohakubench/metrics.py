"""Utility helpers for counting MACs and FLOPs."""

from typing import Iterable

__all__ = [
    "matmul_macs",
    "linear_macs",
    "conv2d_macs",
    "elementwise_flops",
    "activation_flops",
    "total_ops",
]


def matmul_macs(batch: int, m: int, n: int, k: int) -> float:
    return float(batch) * m * n * k


def linear_macs(batch_tokens: int, in_features: int, out_features: int) -> float:
    return float(batch_tokens) * in_features * out_features


def conv2d_macs(
    batch: int,
    in_channels: int,
    out_channels: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    groups: int = 1,
) -> float:
    per_filter = kernel_h * kernel_w * (in_channels / groups)
    return float(batch) * out_channels * out_h * out_w * per_filter


def elementwise_flops(num_elements: int, ops_per_element: float = 1.0) -> float:
    return float(num_elements) * ops_per_element


def activation_flops(num_elements: int, approx_ops: float = 4.0) -> float:
    """Approximate nonlinear activation cost."""
    return float(num_elements) * approx_ops


def total_ops(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total
