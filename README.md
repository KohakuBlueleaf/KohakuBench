# KohakuBench

KohakuBench is a lightweight PyTorch benchmarking toolkit that focuses on realistic CUDA performance checks. It ships with a configurable benchmarking harness (`KoBench`), VRAM-aware timing utilities, MACs/MACS/FLOPs/FLOPS estimators, and a library of ready-to-run presets covering attention, MLPs, convolutions, normalization, transformer blocks, and Stable Diffusion-style residual blocks.

## Features

* CUDA-aware benchmarking loop with warmup control (8) and measured iterations (32), event-based timing, and inference-mode toggles.
* VRAM tracking that differentiates idle usage versus runtime allocations and workspace reservations.
* MACs/MACS and FLOPs/FLOPS helper functions that capture both matmul-heavy and element-wise workloads.
* Presets, reporting utilities, and example scripts that double as reference implementations.

## Installation

```bash
git clone https://github.com/KohakuBlueleaf/KohakuBench
cd KohakuBench
pip install -e .
```

## CLI Quick Start

```bash
kobench --preset attn --device cuda:0 --dtype float16
```

Use `kobench --list` to show available presets. Override the default warmup/measured iterations via `--warmup` and `--iters` when needed.

## Examples

All scripts live in `examples/` and sweep multiple preset shapes:

* `attention_bench.py` - multi-head attention workloads powered by scaled-dot product attention.
* `mlp_bench.py` - GELU/RELU/SwiGLU feed-forward networks across several widths.
* `linear_bench.py` - dense feed-forward layers used as simple linear workloads.
* `conv_bench.py` - standard and depthwise-plus-pointwise convolutions.
* `norm_bench.py` - LayerNorm/GroupNorm/RMSNorm from sequence to image scales.
* `custom_bench.py` - template showing how to benchmark your own modules.

Run any script directly, e.g.

```bash
python examples/attention_bench.py
```

## Custom Benchmarks

Use `KoBench` directly when you need custom workloads:

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device="cuda:0", dtype=torch.float16)
model = torch.nn.Sequential(...).to(bench.device, bench.dtype).eval()
sample = torch.randn(..., device=bench.device, dtype=bench.dtype)
case = BenchmarkCase(name="my_block", fn=model, args=(sample,), metadata={"notes": "custom"})
result = bench.benchmark(case)
print(result.to_dict())
```

See `examples/custom_bench.py` for a full script that mixes convolutions, attention, and nonlinearities.

## Testing

```bash
python -m pytest
```

## License

KohakuBench is distributed under the Apache 2.0 License. See `LICENSE` for details.
