"""Reusable building blocks for benchmark presets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["ScaledDotProductAttention"]


class ScaledDotProductAttention(torch.nn.Module):
    """Minimal multi-head attention using torch.nn.functional.scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, *, bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape

        def _reshape(proj: torch.nn.Linear) -> torch.Tensor:
            y = proj(x)
            return y.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = _reshape(self.q_proj)
        k = _reshape(self.k_proj)
        v = _reshape(self.v_proj)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(bsz, seq, self.embed_dim)
        return self.out_proj(attn)
