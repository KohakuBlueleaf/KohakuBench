"""Example showing how to benchmark a custom architecture with KoBench."""

from __future__ import annotations

import torch

from kohakubench.core import BenchmarkCase, KoBench
from kohakubench.modules import ScaledDotProductAttention
from kohakubench.report import print_table


class HybridConvAttentionBlock(torch.nn.Module):
    """Simple block mixing Conv2d, GroupNorm, SiLU, and scaled-dot attention."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(32, channels)
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()
        self.attn = ScaledDotProductAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.act(self.norm(x)))
        b, c, height, width = h.shape
        tokens = h.view(b, c, height * width).transpose(1, 2)
        tokens = self.attn(tokens)
        tokens = tokens.transpose(1, 2).view(b, c, height, width)
        return x + tokens


def _build_case(bench: KoBench, cfg: dict) -> BenchmarkCase:
    block = HybridConvAttentionBlock(cfg["channels"], cfg["heads"]).to(
        device=bench.device,
        dtype=bench.dtype,
    )
    block.eval()
    sample = torch.randn(
        cfg["batch_size"],
        cfg["channels"],
        cfg["height"],
        cfg["width"],
        device=bench.device,
        dtype=bench.dtype,
    )
    name = f"hybrid_block_b{cfg['batch_size']}_c{cfg['channels']}_h{cfg['height']}"
    return BenchmarkCase(
        name=name,
        fn=block,
        args=(sample,),
        metadata=cfg,
    )


def main() -> None:
    bench = KoBench()
    configs = [
        {"batch_size": 2, "channels": 256, "height": 32, "width": 32, "heads": 8},
        {"batch_size": 1, "channels": 320, "height": 64, "width": 64, "heads": 10},
    ]
    cases = [_build_case(bench, cfg) for cfg in configs]
    results = bench.benchmark_many(cases)
    print_table(results)


if __name__ == "__main__":
    main()
