"""Example of a user-defined preset file for `kobench --preset`.

Expose one of the following symbols:
- build_cases(opts) -> Iterable[BenchmarkCase]
- make_cases(opts) -> Iterable[BenchmarkCase]
- get_cases(opts) -> Iterable[BenchmarkCase]
- class Preset with build(opts) -> Iterable[BenchmarkCase]
- CASES: Iterable[BenchmarkCase]

`opts` contains {'device': torch.device, 'dtype': torch.dtype}.
"""

from __future__ import annotations

import torch
from typing import Iterable, List

from kohakubench.core import BenchmarkCase
from kohakubench.presets import attention, mlp


def build_cases(opts) -> Iterable[BenchmarkCase]:
    device, dtype = opts["device"], opts["dtype"]
    cases: List[BenchmarkCase] = []
    # Multiple shapes for attention
    attn_cfgs = [
        dict(batch_size=2, seq_len=1024, embed_dim=1024, num_heads=16),
        dict(batch_size=1, seq_len=2048, embed_dim=2048, num_heads=32),
    ]
    for cfg in attn_cfgs:
        case = attention.multi_head_attention_case(**cfg, device=device, dtype=dtype)
        case.name += f":b{cfg['batch_size']}_s{cfg['seq_len']}_d{cfg['embed_dim']}_h{cfg['num_heads']}"
        cases.append(case)

    # Multiple shapes for MLPs
    mlp_cfgs = [
        dict(batch_size=2, seq_len=512, embed_dim=2048, hidden_dim=8192),
        dict(batch_size=4, seq_len=256, embed_dim=1536, hidden_dim=6144),
    ]
    for cfg in mlp_cfgs:
        cases.append(
            mlp.standard_mlp_case(**cfg, activation="gelu", device=device, dtype=dtype)
        )
        cases.append(mlp.swiglu_mlp_case(**cfg, device=device, dtype=dtype))
    return cases
