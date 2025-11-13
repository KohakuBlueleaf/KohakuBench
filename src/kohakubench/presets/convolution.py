"""Convolution benchmark presets."""

from __future__ import annotations

import torch

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import conv2d_macs, elementwise_flops

__all__ = ["conv2d_case", "depthwise_conv2d_case"]


def conv2d_case(
    *,
    batch_size: int = 4,
    in_channels: int = 128,
    out_channels: int = 256,
    image_size: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    padding = kernel_size // 2 if padding is None else padding
    conv = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    ).to(device=device, dtype=dtype)
    conv.eval()
    sample = torch.randn(
        batch_size, in_channels, image_size, image_size, device=device, dtype=dtype
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return conv(x)

    out_dim = _conv_out_dim(image_size, kernel_size, padding, stride)

    metadata = {
        "batch_size": batch_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "out_dim": out_dim,
        "kernel_size": kernel_size,
        "groups": 1,
    }

    def mac_counter(meta: dict) -> float:
        return conv2d_macs(
            meta["batch_size"],
            meta["in_channels"],
            meta["out_channels"],
            meta["out_dim"],
            meta["out_dim"],
            meta["kernel_size"],
            meta["kernel_size"],
            meta["groups"],
        )

    return BenchmarkCase(
        name="conv2d",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=None,
    )


def depthwise_conv2d_case(
    *,
    batch_size: int = 4,
    channels: int = 256,
    image_size: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    pointwise: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    padding = kernel_size // 2 if padding is None else padding
    depthwise = torch.nn.Conv2d(
        channels,
        channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=channels,
        bias=False,
    )
    layers = [depthwise]
    if pointwise:
        layers.append(torch.nn.Conv2d(channels, channels, kernel_size=1, bias=False))
    block = torch.nn.Sequential(*layers).to(device=device, dtype=dtype)
    block.eval()
    sample = torch.randn(
        batch_size, channels, image_size, image_size, device=device, dtype=dtype
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return block(x)

    out_dim = _conv_out_dim(image_size, kernel_size, padding, stride)

    metadata = {
        "batch_size": batch_size,
        "channels": channels,
        "out_dim": out_dim,
        "kernel_size": kernel_size,
        "pointwise": pointwise,
    }

    def mac_counter(meta: dict) -> float:
        dw = conv2d_macs(
            meta["batch_size"],
            meta["channels"],
            meta["channels"],
            meta["out_dim"],
            meta["out_dim"],
            meta["kernel_size"],
            meta["kernel_size"],
            groups=meta["channels"],
        )
        if not meta["pointwise"]:
            return dw
        pw = conv2d_macs(
            meta["batch_size"],
            meta["channels"],
            meta["channels"],
            meta["out_dim"],
            meta["out_dim"],
            1,
            1,
            groups=1,
        )
        return dw + pw

    def flop_counter(meta: dict) -> float:
        elements = (
            meta["batch_size"] * meta["channels"] * meta["out_dim"] * meta["out_dim"]
        )
        return elementwise_flops(elements, ops_per_element=1.0)

    return BenchmarkCase(
        name="depthwise_conv2d",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )


def _conv_out_dim(size: int, kernel: int, padding: int, stride: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1
