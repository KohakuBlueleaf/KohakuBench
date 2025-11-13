# KoBench API

This page documents the core classes and helpers exposed by KohakuBench.

## resolve_device(device=None) -> torch.device
Normalizes a device input and defaults to CUDA when available.

## resolve_dtype(dtype=None, device=None) -> torch.dtype
Maps strings ("float16", "bf16", "float32", etc.) to a torch dtype, defaulting to a GPU-friendly dtype when applicable.

## class BenchmarkCase
Container describing the callable to benchmark.

- name: str — label for reports
- fn: Callable — function or module to call
- args: tuple — positional args passed to fn
- kwargs: dict — keyword args passed to fn
- metadata: dict — shape/config info surfaced in results
- mac_counter: Optional[Callable[[dict], float]] — function returning MACs for the case
- flop_counter: Optional[Callable[[dict], float]] — function returning FLOPs for the case

## class BenchmarkResult
Structured output from KoBench.

- name, iters, warmup
- latency_ms, latency_std_ms
- device, dtype
- idle_vram_mb, peak_vram_mb, workspace_vram_mb
- macs, macs_rate, flops, flops_rate
- metadata (dict)

## class KoBench
Core benchmarking utility.

Constructor:
```python
KoBench(
  device: Optional[Union[str, torch.device]] = None,
  dtype: Optional[Union[str, torch.dtype]] = None,
  warmup: int = 8,
  iters: int = 32,
  sync: bool = True,
  inference_mode: bool = True,
)
```

Methods:
- benchmark(target, name=None, warmup=None, iters=None, metadata=None, flop_counter=None, mac_counter=None) -> BenchmarkResult
- benchmark_many(cases: Iterable[BenchmarkCase]) -> List[BenchmarkResult]

Notes:
- CUDA timing uses events with stream sync; CPU uses perf_counter.
- Memory tracing separates idle from runtime and workspace reservations.
- When inference_mode=True, wraps calls in torch.inference_mode for speed and determinism.
