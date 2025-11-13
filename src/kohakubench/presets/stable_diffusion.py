"""Stable Diffusion style residual block preset."""

from __future__ import annotations

import torch

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import conv2d_macs, elementwise_flops, total_ops

__all__ = ["stable_diffusion_resblock_case"]


def stable_diffusion_resblock_case(
    *,
    batch_size: int = 2,
    channels: int = 320,
    height: int = 64,
    width: int = 64,
    groups: int = 32,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    block = _SDResBlock(channels, groups=groups).to(device=device, dtype=dtype)
    block.eval()
    sample = torch.randn(
        batch_size, channels, height, width, device=device, dtype=dtype
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return block(x)

    metadata = {
        "batch_size": batch_size,
        "channels": channels,
        "height": height,
        "width": width,
        "groups": groups,
    }

    def mac_counter(meta: dict) -> float:
        conv1 = conv2d_macs(
            meta["batch_size"],
            meta["channels"],
            meta["channels"],
            meta["height"],
            meta["width"],
            3,
            3,
        )
        conv2 = conv2d_macs(
            meta["batch_size"],
            meta["channels"],
            meta["channels"],
            meta["height"],
            meta["width"],
            3,
            3,
        )
        return conv1 + conv2

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["channels"] * meta["height"] * meta["width"]
        gn = elementwise_flops(elems, ops_per_element=6.0) * 2
        silu = elementwise_flops(elems, ops_per_element=5.0) * 2
        add = elementwise_flops(elems, ops_per_element=1.0)
        return total_ops([gn, silu, add])

    return BenchmarkCase(
        name="sd_resblock",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )


class _SDResBlock(torch.nn.Module):
    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(groups, channels)
        self.norm2 = torch.nn.GroupNorm(groups, channels)
        self.act = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h
