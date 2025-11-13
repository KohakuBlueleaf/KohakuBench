# Examples

This directory contains runnable scripts that demonstrate how to drive KohakuBench presets and custom workloads.

## Using KoBench

1. Instantiate `KoBench` to select device, dtype, and iteration counts.
2. Move your module/tensors to the same device + dtype.
3. Wrap the callable in a `BenchmarkCase`.
4. Run `bench.benchmark(case)` or `bench.benchmark_many([...])`.

```python
from kohakubench.core import BenchmarkCase, KoBench

bench = KoBench(device="cuda:0", dtype="float16")
model = MyModule().to(bench.device, bench.dtype).eval()
sample = build_input().to(bench.device, bench.dtype)
case = BenchmarkCase(name="custom", fn=model, args=(sample,), metadata={"desc": "demo"})
print(bench.benchmark(case).to_dict())
```

## Scripts

* `attention_bench.py` - Scaled-dot attention workloads at multiple sequence lengths.
* `mlp_bench.py` - GELU/RELU/SwiGLU feed-forward blocks across various hidden sizes.
* `linear_bench.py` - Dense linear workloads for quick sanity checks.
* `conv_bench.py` - Dense and depthwise-plus-pointwise convolution blocks.
* `norm_bench.py` - LayerNorm, GroupNorm, and RMSNorm for both sequence and image shapes.
* `custom_bench.py` - End-to-end template that benchmarks a bespoke architecture.

Run any script with Python:

```bash
python attention_bench.py     # optionally set CUDA_VISIBLE_DEVICES
```

Each script inherits KoBench defaults (8 warmup + 32 timed iterations); override by editing the script or constructing `KoBench` with new parameters.
