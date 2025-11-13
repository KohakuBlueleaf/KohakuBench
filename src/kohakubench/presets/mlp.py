"""MLP benchmark presets."""

import torch
import torch.nn.functional as F

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import activation_flops, elementwise_flops, linear_macs, total_ops

__all__ = ["standard_mlp_case", "swiglu_mlp_case"]


def standard_mlp_case(
    *,
    batch_size: int = 4,
    seq_len: int = 512,
    embed_dim: int = 2048,
    hidden_dim: int = 8192,
    activation: str = "gelu",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    act_fn = _get_activation(activation)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, hidden_dim, bias=False),
        act_fn,
        torch.nn.Linear(hidden_dim, embed_dim, bias=False),
    ).to(device=device, dtype=dtype)
    mlp.eval()
    tokens = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return mlp(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "activation": activation,
    }

    def mac_counter(meta: dict) -> float:
        tokens_ct = meta["batch_size"] * meta["seq_len"]
        up = linear_macs(tokens_ct, meta["embed_dim"], meta["hidden_dim"])
        down = linear_macs(tokens_ct, meta["hidden_dim"], meta["embed_dim"])
        return total_ops([up, down])

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["seq_len"] * meta["hidden_dim"]
        if meta["activation"].lower() == "relu":
            return elementwise_flops(elems, ops_per_element=1.0)
        return activation_flops(elems, approx_ops=6.0)

    return BenchmarkCase(
        name=f"mlp_{activation.lower()}",
        fn=fn,
        args=(tokens,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )


def swiglu_mlp_case(
    *,
    batch_size: int = 4,
    seq_len: int = 512,
    embed_dim: int = 2048,
    hidden_dim: int = 8192,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    block = _SwiGLU(embed_dim, hidden_dim).to(device=device, dtype=dtype)
    block.eval()
    tokens = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return block(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
    }

    def mac_counter(meta: dict) -> float:
        tokens_ct = meta["batch_size"] * meta["seq_len"]
        up = linear_macs(tokens_ct, meta["embed_dim"], meta["hidden_dim"] * 2)
        down = linear_macs(tokens_ct, meta["hidden_dim"], meta["embed_dim"])
        return total_ops([up, down])

    def flop_counter(meta: dict) -> float:
        elems = meta["batch_size"] * meta["seq_len"] * meta["hidden_dim"]
        silu = activation_flops(elems, approx_ops=5.0)
        gate = elementwise_flops(elems, ops_per_element=1.0)
        return total_ops([silu, gate])

    return BenchmarkCase(
        name="mlp_swiglu",
        fn=fn,
        args=(tokens,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )


def _get_activation(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'")


class _SwiGLU(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up = torch.nn.Linear(embed_dim, hidden_dim * 2, bias=False)
        self.down = torch.nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * x_proj)
