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

## Quick Examples

### 1. CLI benchmark (built-in preset)

```bash
kobench --preset mlp --device cuda:0 --dtype bf16 --iters 48
```

### 2. Example script (multiple shapes)

```bash
python examples/attention_bench.py
```

### 3. Python API (inline KoBench usage)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device="cuda:0", dtype="float16")
model = torch.nn.Linear(2048, 4096, bias=False).to(bench.device, bench.dtype).eval()
sample = torch.randn(8, 2048, device=bench.device, dtype=bench.dtype)
case = BenchmarkCase(name="linear_demo", fn=model, args=(sample,))
print(bench.benchmark(case).to_dict())
```

### 4. CLI with a user preset file

```bash
kobench --preset examples/user_preset_example.py
```

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


### 4. Programmatic presets (multiple cases)

```python
from kohakubench.core import KoBench
from kohakubench.presets import attention, mlp
from kohakubench.report import print_table

bench = KoBench(device='cuda:0', dtype='float16')
cases = [
    attention.multi_head_attention_case(batch_size=2, seq_len=1024, embed_dim=1024, num_heads=16,
                                        device=bench.device, dtype=bench.dtype),
    mlp.swiglu_mlp_case(batch_size=2, seq_len=512, embed_dim=2048, hidden_dim=8192,
                        device=bench.device, dtype=bench.dtype),
]
results = bench.benchmark_many(cases)
print_table(results)
```

### 5. Custom counters (MACs/FLOPs)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device='cuda:0', dtype='float16')
conv = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False).to(bench.device, bench.dtype).eval()
x = torch.randn(4, 128, 64, 64, device=bench.device, dtype=bench.dtype)

meta = {'batch': 4, 'in_ch': 128, 'out_ch': 256, 'h': 64, 'w': 64, 'k': 3}

def mac_counter(m):
    per = m['k'] * m['k'] * m['in_ch']
    return m['batch'] * m['out_ch'] * m['h'] * m['w'] * per

def flop_counter(m):
    return m['batch'] * m['out_ch'] * m['h'] * m['w']  # simple elementwise add approx

case = BenchmarkCase(name='conv2d_custom', fn=conv, args=(x,), metadata=meta,
                     mac_counter=mac_counter, flop_counter=flop_counter)
print(bench.benchmark(case).to_dict())
```

### 6. Backward-pass benchmarking (training step)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

torch.manual_seed(0)
bench = KoBench(device='cuda:0', dtype='float16', inference_mode=False)
model = torch.nn.Linear(1024, 1024).to(bench.device, bench.dtype)
model.train()

x = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
y = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.enable_grad()
def train_step(x, y):
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    opt.step()
    return loss

print(bench.benchmark(BenchmarkCase(name='linear_train_step', fn=train_step, args=(x, y))).to_dict())
```

## Documentation

See the full docs in the `docs/` folder:

* docs/README.md — index
* docs/manual.md — quickstart and core concepts
* docs/kobench-api.md — API reference for KoBench, cases, results
* docs/presets.md — all preset builders
* docs/metrics.md — counters and definitions
* docs/reporting.md — output and tables
* docs/cli.md — CLI usage
* docs/recipes.md — practical patterns
* docs/modules.md — extra modules
