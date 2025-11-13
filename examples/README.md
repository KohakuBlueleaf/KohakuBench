# Examples

This directory contains runnable scripts and snippets that show how to drive KoBench presets, tweak benchmarking parameters, and integrate custom modules. Treat it as a mini cookbook for building your own benchmarking workflows.

## Prerequisites

* Install KohakuBench in editable mode (`pip install -e .` from the project root).
* Have a working PyTorch environment with CUDA if you plan to profile GPUs (scripts fall back to CPU automatically).
* Optional: set `CUDA_VISIBLE_DEVICES` or `PYTORCH_CUDA_ALLOC_CONF` before running scripts to control device placement or allocator behavior.

## Using KoBench Manually

1. Instantiate `KoBench` to pick device, dtype, warmup, and measurement loops.
2. Move your module and sample inputs to the same device/dtype.
3. Wrap the callable in a `BenchmarkCase`.
4. Call `bench.benchmark(case)` (single workload) or `bench.benchmark_many([...])` (batch of workloads).

```python
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device="cuda:0", dtype="float16", warmup=4, iters=16)
model = MyModule().to(bench.device, bench.dtype).eval()
sample = build_input().to(bench.device, bench.dtype)
case = BenchmarkCase(name="custom", fn=model, args=(sample,), metadata={"desc": "demo"})
result = bench.benchmark(case)
print(result.to_dict())
```

## Running Preset Scripts

Every script inherits KoBench defaults (8 warmup + 32 timed iterations) and prints a MACs/MACS/FLOPs/FLOPS table. Run any script from this folder:

```bash
python attention_bench.py        # multi-head attention sweep
python mlp_bench.py              # various MLP/SwiGLU settings
python custom_bench.py           # bespoke hybrid block example
```

Override the device/dtype or loop counts by editing the script or wrapping commands with environment variables (e.g., `CUDA_VISIBLE_DEVICES=1 python attention_bench.py`). Each script tags cases with their shapes so report rows stay distinguishable.

### Sample Output

```
name                         | mean_ms | std_ms | idle_vram | peak_vram | workspace | MACs | MACS | FLOPs | FLOPS
multi_head_attention:b2_s1024 | 2.315  | 0.041  | 420.12 MB | 512.34 MB | 40.00 MB  | 5.12 G | 2.21 T/s | 1.02 G | 442.6 G/s
...
```

## Script Reference

| Script | Workloads | Default shapes/configurations |
| ------ | --------- | ----------------------------- |
| `attention_bench.py` | Scaled-dot product attention | Batches {1,2,4}, sequence lengths {512,1024,2048}, head counts {12,16,32} |
| `mlp_bench.py` | Standard + SwiGLU MLPs | GELU/RELU/SwiGLU with embed dims {1536-4096} and hidden ratios up to 8x |
| `linear_bench.py` | Lightweight dense layers | Smaller token counts for quick smoke tests |
| `conv_bench.py` | Conv2d + depthwise separable blocks | Image sizes 32-128 with varying stride/padding |
| `norm_bench.py` | LayerNorm/GroupNorm/RMSNorm | Sequence-style LN/RMS and image-style GN |
| `custom_bench.py` | Hybrid conv + attention module | Demonstrates custom `BenchmarkCase` wiring |

Feel free to duplicate a script and adjust the config arrays to cover your own shapes—KoBench will reuse the same configuration per run.

## Building Custom Benchmarks

The `custom_bench.py` script mixes convolution, SiLU, GroupNorm, and scaled-dot attention to illustrate how to benchmark arbitrary PyTorch modules. To adapt it:

1. Replace `HybridConvAttentionBlock` with your module (e.g., diffusion U-Net block, ConvNeXt layer).
2. Update the `configs` list to enumerate relevant shapes (batch, channels, spatial size, heads, etc.).
3. Adjust `BenchmarkCase` metadata to surface the parameters you care about in reports.
4. Run `python custom_bench.py` and inspect the printed table or convert `BenchmarkResult` objects to dictionaries/JSON.

For even finer control (e.g., logging to CSV, integrating into CI), import the helper functions from these scripts and call `print_table(results)` or `format_table(results)` directly.


## Programmatic Presets (Code)

```python
from kohakubench.core import KoBench
from kohakubench.presets import attention, mlp, convolution
from kohakubench.report import print_table

bench = KoBench(device='cuda:0', dtype='float16')
cases = [
    attention.multi_head_attention_case(batch_size=2, seq_len=1024, embed_dim=1024, num_heads=16,
                                        device=bench.device, dtype=bench.dtype),
    mlp.standard_mlp_case(batch_size=2, seq_len=512, embed_dim=2048, hidden_dim=8192,
                          activation='gelu', device=bench.device, dtype=bench.dtype),
    convolution.depthwise_conv2d_case(batch_size=4, channels=256, image_size=64,
                                     pointwise=True, device=bench.device, dtype=bench.dtype),
]
print_table(bench.benchmark_many(cases))
```

## Custom Counters (Code)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device='cuda:0', dtype='float16')
proj = torch.nn.Linear(768, 768, bias=False).to(bench.device, bench.dtype).eval()
q = torch.randn(2, 1024, 768, device=bench.device, dtype=bench.dtype)

meta = {'batch_tokens': 2*1024, 'in': 768, 'out': 768}
mac_counter = lambda m: m['batch_tokens'] * m['in'] * m['out']
flop_counter = lambda m: m['batch_tokens'] * m['out']  # activation-like

case = BenchmarkCase('proj', proj, args=(q,), metadata=meta,
                     mac_counter=mac_counter, flop_counter=flop_counter)
print(bench.benchmark(case).to_dict())
```

## Backward Steps (Code)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device='cuda:0', dtype='float16', inference_mode=False)
model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024)).to(bench.device, bench.dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)
y = torch.randn(32, 1024, device=bench.device, dtype=bench.dtype)

@torch.enable_grad()
def step(x, y):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss

print(bench.benchmark(BenchmarkCase('mlp_train_step', step, args=(x, y))).to_dict())
```

## Scaled-Dot Product Attention (Code)

```python
import torch
from kohakubench.core import BenchmarkCase, KoBench
from kohakubench.modules import ScaledDotProductAttention

bench = KoBench()
attn = ScaledDotProductAttention(embed_dim=1024, num_heads=16).to(bench.device, bench.dtype).eval()
x = torch.randn(2, 1024, 1024, device=bench.device, dtype=bench.dtype)
print(bench.benchmark(BenchmarkCase('sdp_attn', attn, args=(x,))).to_dict())
```

## Further Reading

For deeper explanations and APIs, see the docs in `../docs/`:

* ../docs/manual.md
* ../docs/kobench-api.md
* ../docs/presets.md
* ../docs/metrics.md
* ../docs/reporting.md
* ../docs/cli.md
* ../docs/architecture.md
* ../docs/recipes.md
