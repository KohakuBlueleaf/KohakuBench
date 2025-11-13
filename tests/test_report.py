from kohakubench.core import BenchmarkResult
from kohakubench.report import format_table


def test_format_table_renders_text():
    result = BenchmarkResult(
        name="sample",
        iters=2,
        warmup=1,
        latency_ms=1.23,
        latency_std_ms=0.1,
        device="cpu",
        dtype="float32",
        idle_vram_mb=None,
        peak_vram_mb=None,
        workspace_vram_mb=None,
        macs=1e9,
        macs_rate=1e9 / (1.23 / 1000),
        flops=5e8,
        flops_rate=5e8 / (1.23 / 1000),
        metadata={"batch_size": 1},
    )
    table = format_table([result])
    assert "sample" in table
    assert "mean_ms" in table.splitlines()[0]
