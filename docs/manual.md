# Manual

This manual introduces KoBench and shows how to run benchmarks using the CLI and Python API.

## Install

```bash
pip install -e .
```

## Concepts

- Warmup iterations: default 8 unmeasured passes to stabilize kernels.
- Measured iterations: default 32 timed passes.
- CUDA timing: CUDA events on GPU; CPU timing: `time.perf_counter`.
- Memory tracing: records idle VRAM, peak allocations, and workspace usage.
- MACs/MACS and FLOPs/FLOPS: operation counts and per-second rates when counters are available.

## CLI

```bash
kobench --preset attn --device cuda:0 --dtype float16
kobench --preset mlp  --warmup 8 --iters 32
kobench --list
```

## Python API

### Single case

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device='cuda:0', dtype='float16')
model = torch.nn.Linear(2048, 4096, bias=False).to(bench.device, bench.dtype).eval()
x = torch.randn(8, 2048, device=bench.device, dtype=bench.dtype)
case = BenchmarkCase('linear_demo', model, args=(x,))
print(bench.benchmark(case).to_dict())
```

### Multiple cases (presets)

```python
from kohakubench.core import KoBench
from kohakubench.presets import attention, mlp
from kohakubench.report import print_table

bench = KoBench()
cases = [
    attention.multi_head_attention_case(batch_size=2, seq_len=1024, embed_dim=1024, num_heads=16,
                                        device=bench.device, dtype=bench.dtype),
    mlp.swiglu_mlp_case(batch_size=2, seq_len=512, embed_dim=2048, hidden_dim=8192,
                        device=bench.device, dtype=bench.dtype),
]
print_table(bench.benchmark_many(cases))
```

### Custom counters

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench()
proj = torch.nn.Linear(768, 768, bias=False).to(bench.device, bench.dtype).eval()
q = torch.randn(2, 1024, 768, device=bench.device, dtype=bench.dtype)

meta = {'batch_tokens': 2*1024, 'in': 768, 'out': 768}
mac_counter = lambda m: m['batch_tokens'] * m['in'] * m['out']
flop_counter = lambda m: m['batch_tokens'] * m['out']

case = BenchmarkCase('proj', proj, args=(q,), metadata=meta,
                     mac_counter=mac_counter, flop_counter=flop_counter)
print(bench.benchmark(case).to_dict())
```

### Training step (backward)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

torch.manual_seed(0)
bench = KoBench(inference_mode=False)
model = torch.nn.Linear(1024, 1024).to(bench.device, bench.dtype)
model.train()

x = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
y = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.enable_grad()
def step(x, y):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss

print(bench.benchmark(BenchmarkCase('train_step', step, args=(x, y))).to_dict())
```
