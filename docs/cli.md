# CLI

KohakuBench provides the `kobench` command.

## Usage

```bash
kobench --preset <attn|mlp|conv|norm|transformer|sdres|path/to/preset.py|module.path> \
        --device <cuda:0|cpu> \
        --dtype <float16|bf16|float32> \
        --warmup <int> \
        --iters <int>
```

## Examples

```bash
kobench --preset attn --device cuda:0 --dtype float16
kobench --preset mlp  --warmup 8 --iters 32
kobench --list

# User-defined preset from file
kobench --preset examples/user_preset_example.py

# User-defined preset from module
kobench --preset mypkg.my_preset
```

The CLI expands a preset into one or more `BenchmarkCase` instances, runs them sequentially with KoBench, and prints a table including latency, VRAM, MACs/MACS, and FLOPs/FLOPS.

## Writing a user preset

Create a Python file that exposes one of the following (searched in order):

- function `build_cases(opts) -> Iterable[BenchmarkCase]`
- function `make_cases(opts) -> Iterable[BenchmarkCase]`
- function `get_cases(opts) -> Iterable[BenchmarkCase]`
- class `Preset` with method `build(opts) -> Iterable[BenchmarkCase]`
- variable `CASES: Iterable[BenchmarkCase]`

`opts` is a dictionary with:

- `device`: `torch.device`
- `dtype`: `torch.dtype`

Return a list/iterable of `BenchmarkCase` objects (multiple shapes are encouraged). See `examples/user_preset_example.py` for a reference implementation.
