import torch

from kohakubench.core import BenchmarkCase, KoBench


def test_kobench_basic_cpu_run():
    bench = KoBench(device="cpu", dtype=torch.float32, warmup=1, iters=2, sync=False)
    model = torch.nn.Linear(32, 32).to(bench.device, bench.dtype)
    sample = torch.randn(4, 32, device=bench.device, dtype=bench.dtype)
    case = BenchmarkCase(name="linear_cpu", fn=model, args=(sample,))

    result = bench.benchmark(case)

    assert result.latency_ms > 0
    assert result.iters == 2
    assert result.warmup == 1
    assert result.idle_vram_mb is None
    assert result.macs is None
    assert result.macs_rate is None
    assert result.flops is None
    assert result.flops_rate is None


def test_benchmark_many_returns_results():
    bench = KoBench(device="cpu", dtype=torch.float32, warmup=1, iters=1, sync=False)
    linear = torch.nn.Linear(16, 16).to(bench.device, bench.dtype)
    sample = torch.randn(2, 16, device=bench.device, dtype=bench.dtype)
    case = BenchmarkCase(name="linear", fn=linear, args=(sample,))
    results = bench.benchmark_many([case, case])
    assert len(results) == 2
