import torch

from kohakubench.core import KoBench
from kohakubench.presets import (
    attention,
    convolution,
    mlp,
    norms,
    stable_diffusion,
    transformer,
)


def _cpu_bench() -> KoBench:
    return KoBench(device="cpu", dtype=torch.float32, warmup=1, iters=1, sync=False)


def test_presets_execute_on_cpu():
    bench = _cpu_bench()
    cases = [
        attention.multi_head_attention_case(
            batch_size=1,
            seq_len=8,
            embed_dim=32,
            num_heads=4,
            device=bench.device,
            dtype=bench.dtype,
        ),
        mlp.standard_mlp_case(
            batch_size=1,
            seq_len=16,
            embed_dim=64,
            hidden_dim=128,
            device=bench.device,
            dtype=bench.dtype,
        ),
        mlp.swiglu_mlp_case(
            batch_size=1,
            seq_len=16,
            embed_dim=64,
            hidden_dim=96,
            device=bench.device,
            dtype=bench.dtype,
        ),
        convolution.conv2d_case(
            batch_size=1,
            in_channels=8,
            out_channels=8,
            image_size=16,
            device=bench.device,
            dtype=bench.dtype,
        ),
        convolution.depthwise_conv2d_case(
            batch_size=1,
            channels=8,
            image_size=16,
            device=bench.device,
            dtype=bench.dtype,
        ),
        norms.layer_norm_case(
            batch_size=1,
            seq_len=16,
            embed_dim=64,
            device=bench.device,
            dtype=bench.dtype,
        ),
        norms.group_norm_case(
            batch_size=1,
            channels=8,
            height=8,
            width=8,
            groups=4,
            device=bench.device,
            dtype=bench.dtype,
        ),
        norms.rms_norm_case(
            batch_size=1,
            seq_len=16,
            embed_dim=64,
            device=bench.device,
            dtype=bench.dtype,
        ),
        transformer.transformer_block_case(
            batch_size=1,
            seq_len=8,
            embed_dim=64,
            num_heads=4,
            mlp_ratio=2,
            device=bench.device,
            dtype=bench.dtype,
        ),
        stable_diffusion.stable_diffusion_resblock_case(
            batch_size=1,
            channels=8,
            height=8,
            width=8,
            groups=4,
            device=bench.device,
            dtype=bench.dtype,
        ),
    ]
    for case in cases:
        result = bench.benchmark(case)
        assert result.latency_ms >= 0
