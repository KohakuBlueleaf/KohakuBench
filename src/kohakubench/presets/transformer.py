"""Transformer block preset."""

import torch
import torch.nn.functional as F

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import (
    activation_flops,
    elementwise_flops,
    linear_macs,
    matmul_macs,
    total_ops,
)
from ..modules import ScaledDotProductAttention

__all__ = ["transformer_block_case"]


def transformer_block_case(
    *,
    batch_size: int = 2,
    seq_len: int = 512,
    embed_dim: int = 1024,
    num_heads: int = 16,
    mlp_ratio: int = 4,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    block = _TransformerBlock(embed_dim, num_heads, mlp_ratio).to(
        device=device, dtype=dtype
    )
    block.eval()
    sample = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return block(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
    }

    def mac_counter(meta: dict) -> float:
        batch = meta["batch_size"]
        seq = meta["seq_len"]
        dim = meta["embed_dim"]
        heads = meta["num_heads"]
        head_dim = dim // heads
        tokens = batch * seq
        hidden = dim * meta["mlp_ratio"]
        qkv = 3 * linear_macs(tokens, dim, dim)
        attn_scores = matmul_macs(batch * heads, seq, seq, head_dim)
        attn_value = matmul_macs(batch * heads, seq, head_dim, seq)
        proj = linear_macs(tokens, dim, dim)
        up = linear_macs(tokens, dim, hidden)
        down = linear_macs(tokens, hidden, dim)
        return total_ops([qkv, attn_scores, attn_value, proj, up, down])

    def flop_counter(meta: dict) -> float:
        batch = meta["batch_size"]
        seq = meta["seq_len"]
        dim = meta["embed_dim"]
        heads = meta["num_heads"]
        tokens = batch * seq
        hidden = dim * meta["mlp_ratio"]
        softmax = elementwise_flops(batch * heads * seq * seq, ops_per_element=4.0)
        scale = elementwise_flops(batch * heads * seq * seq, ops_per_element=2.0)
        act = activation_flops(tokens * hidden, approx_ops=6.0)
        return total_ops([softmax, scale, act])

    return BenchmarkCase(
        name="transformer_block",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )


class _TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = ScaledDotProductAttention(dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(dim)
        hidden_dim = dim * mlp_ratio
        self.up = torch.nn.Linear(dim, hidden_dim)
        self.down = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out = self.attn(attn_input)
        x = x + attn_out
        mlp_in = self.norm2(x)
        mlp_out = self.down(F.gelu(self.up(mlp_in)))
        return x + mlp_out
