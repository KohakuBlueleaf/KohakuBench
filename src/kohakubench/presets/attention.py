"""Attention benchmark presets."""

import torch

from ..core import BenchmarkCase, resolve_device, resolve_dtype
from ..metrics import elementwise_flops, linear_macs, matmul_macs, total_ops
from ..modules import ScaledDotProductAttention

__all__ = ["multi_head_attention_case"]


def multi_head_attention_case(
    *,
    batch_size: int = 4,
    seq_len: int = 1024,
    embed_dim: int = 1024,
    num_heads: int = 16,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> BenchmarkCase:
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    attn = ScaledDotProductAttention(embed_dim, num_heads).to(
        device=device, dtype=dtype
    )
    attn.eval()
    sample = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return attn(x)

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
    }

    def mac_counter(meta: dict) -> float:
        batch = meta["batch_size"]
        seq = meta["seq_len"]
        dim = meta["embed_dim"]
        heads = meta["num_heads"]
        head_dim = dim // heads
        tokens = batch * seq
        qkv = 3 * linear_macs(tokens, dim, dim)
        attn_scores = matmul_macs(batch * heads, seq, seq, head_dim)
        attn_value = matmul_macs(batch * heads, seq, head_dim, seq)
        proj_out = linear_macs(tokens, dim, dim)
        return total_ops([qkv, attn_scores, attn_value, proj_out])

    def flop_counter(meta: dict) -> float:
        batch = meta["batch_size"]
        seq = meta["seq_len"]
        heads = meta["num_heads"]
        scale_flops = elementwise_flops(batch * heads * seq * seq, ops_per_element=2.0)
        softmax_flops = elementwise_flops(
            batch * heads * seq * seq, ops_per_element=4.0
        )
        return total_ops([scale_flops, softmax_flops])

    return BenchmarkCase(
        name="multi_head_attention",
        fn=fn,
        args=(sample,),
        metadata=metadata,
        mac_counter=mac_counter,
        flop_counter=flop_counter,
    )
