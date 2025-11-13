"""Normalization benchmark presets."""

import torch

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import elementwise_flops

__all__ = ["layer_norm_case", "group_norm_case", "rms_norm_case"]


def layer_norm_case(
    *,
    batch_size: int = 4,
    seq_len: int = 512,
    embed_dim: int = 2048,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    norm = torch.nn.LayerNorm(embed_dim).to(device=device, dtype=dtype)
    sample = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return norm(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
    }

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["seq_len"] * meta["embed_dim"]
        return elementwise_flops(elems, ops_per_element=6.0)

    return BenchmarkCase(
        name="layer_norm",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        flop_counter=flop_counter,
    )


def group_norm_case(
    *,
    batch_size: int = 4,
    channels: int = 256,
    height: int = 64,
    width: int = 64,
    groups: int = 32,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    norm = torch.nn.GroupNorm(groups, channels).to(device=device, dtype=dtype)
    sample = torch.randn(
        batch_size, channels, height, width, device=device, dtype=dtype
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return norm(x)

    metadata = {
        "batch_size": batch_size,
        "channels": channels,
        "height": height,
        "width": width,
        "groups": groups,
    }

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["channels"] * meta["height"] * meta["width"]
        return elementwise_flops(elems, ops_per_element=6.0)

    return BenchmarkCase(
        name="group_norm",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        flop_counter=flop_counter,
    )


def rms_norm_case(
    *,
    batch_size: int = 4,
    seq_len: int = 512,
    embed_dim: int = 2048,
    eps: float = 1e-6,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    norm = _RMSNorm(embed_dim, eps=eps).to(device=device, dtype=dtype)
    sample = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return norm(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
    }

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["seq_len"] * meta["embed_dim"]
        return elementwise_flops(elems, ops_per_element=4.0)

    return BenchmarkCase(
        name="rms_norm",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        flop_counter=flop_counter,
    )


class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight
