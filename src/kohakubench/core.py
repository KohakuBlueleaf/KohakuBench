import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

Number = int | float

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "KoBench",
    "resolve_device",
    "resolve_dtype",
]

_MB = 1024**2


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Normalize device input and default to CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(
    dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.dtype:
    """Resolve dtype names/aliases to an actual torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        table = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if normalized not in table:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return table[normalized]
    resolved_device = resolve_device(device)
    return torch.float16 if resolved_device.type == "cuda" else torch.float32


@dataclass
class BenchmarkCase:
    """Container describing a callable to benchmark."""

    name: str
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    flop_counter: Callable[[dict[str, Any]], Number] | None = None
    mac_counter: Callable[[dict[str, Any]], Number] | None = None


@dataclass
class BenchmarkResult:
    """Structured benchmark output."""

    name: str
    iters: int
    warmup: int
    latency_ms: float
    latency_std_ms: float
    device: str
    dtype: str
    idle_vram_mb: float | None
    peak_vram_mb: float | None
    workspace_vram_mb: float | None
    macs: float | None
    macs_rate: float | None
    flops: float | None
    flops_rate: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iters": self.iters,
            "warmup": self.warmup,
            "latency_ms": self.latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "device": self.device,
            "dtype": self.dtype,
            "idle_vram_mb": self.idle_vram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "workspace_vram_mb": self.workspace_vram_mb,
            "macs": self.macs,
            "macs_rate": self.macs_rate,
            "flops": self.flops,
            "flops_rate": self.flops_rate,
            "metadata": dict(self.metadata),
        }


class KoBench:
    """Core benchmarking utility with CUDA-aware timing and memory tracing."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        warmup: int = 8,
        iters: int = 32,
        sync: bool = True,
        inference_mode: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.warmup = warmup
        self.iters = iters
        self.sync = sync
        self.inference_mode = inference_mode

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def benchmark(
        self,
        target: BenchmarkCase | Callable[..., Any],
        *,
        name: str | None = None,
        warmup: int | None = None,
        iters: int | None = None,
        metadata: dict[str, Any] | None = None,
        flop_counter: Callable[[dict[str, Any]], Number] | None = None,
        mac_counter: Callable[[dict[str, Any]], Number] | None = None,
    ) -> BenchmarkResult:
        """Benchmark a single callable or BenchmarkCase."""
        case = self._normalize_case(
            target,
            name=name,
            metadata=metadata,
            flop_counter=flop_counter,
            mac_counter=mac_counter,
        )
        warmup_runs = self._validate_positive("warmup", warmup or self.warmup)
        iter_runs = self._validate_positive("iters", iters or self.iters)

        exec_ctx = (
            torch.inference_mode if self.inference_mode else contextlib.nullcontext
        )

        with exec_ctx():
            self._run_warmup(case, warmup_runs)
            idle_stats = self._start_memory_trace()
            durations = self._run_iterations(case, iter_runs)
            mem_stats = self._finish_memory_trace(idle_stats)

        latency_ms = statistics.fmean(durations)
        std_ms = statistics.pstdev(durations) if len(durations) > 1 else 0.0
        metrics_context = {**case.metadata, "device": self.device, "dtype": self.dtype}
        macs = case.mac_counter(metrics_context) if case.mac_counter else None
        flops = case.flop_counter(metrics_context) if case.flop_counter else None
        macs_rate = _ops_per_second(macs, latency_ms)
        flops_rate = _ops_per_second(flops, latency_ms)

        return BenchmarkResult(
            name=case.name,
            iters=iter_runs,
            warmup=warmup_runs,
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            device=str(self.device),
            dtype=str(self.dtype).split(".")[-1],
            idle_vram_mb=_bytes_to_mb(mem_stats.get("idle_allocated")),
            peak_vram_mb=_bytes_to_mb(mem_stats.get("peak_allocated")),
            workspace_vram_mb=_bytes_to_mb(mem_stats.get("workspace_reserved")),
            macs=macs,
            macs_rate=macs_rate,
            flops=flops,
            flops_rate=flops_rate,
            metadata=dict(case.metadata),
        )

    def benchmark_many(
        self,
        cases: Iterable[BenchmarkCase | Callable[..., Any]],
        **kwargs: Any,
    ) -> list[BenchmarkResult]:
        """Benchmark a collection of cases sequentially."""
        return [self.benchmark(case, **kwargs) for case in cases]

    # Internal helpers -----------------------------------------------------

    def _normalize_case(
        self,
        target: BenchmarkCase | Callable[..., Any],
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        flop_counter: Callable[[dict[str, Any]], Number] | None = None,
        mac_counter: Callable[[dict[str, Any]], Number] | None = None,
    ) -> BenchmarkCase:
        if isinstance(target, BenchmarkCase):
            return BenchmarkCase(
                name=target.name,
                fn=target.fn,
                args=target.args,
                kwargs=dict(target.kwargs),
                metadata={**target.metadata, **(metadata or {})},
                flop_counter=target.flop_counter or flop_counter,
                mac_counter=target.mac_counter or mac_counter,
            )
        if callable(target):
            resolved_name = name or getattr(target, "__name__", "anonymous_case")
            return BenchmarkCase(
                name=resolved_name,
                fn=target,
                metadata=metadata or {},
                flop_counter=flop_counter,
                mac_counter=mac_counter,
            )
        raise TypeError("target must be a callable or BenchmarkCase instance")

    def _validate_positive(self, label: str, value: int) -> int:
        if value <= 0:
            raise ValueError(f"{label} must be > 0 (got {value})")
        return value

    def _run_warmup(self, case: BenchmarkCase, repeats: int) -> None:
        if repeats <= 0:
            return
        for _ in range(repeats):
            self._sync_cuda()
            case.fn(*case.args, **case.kwargs)
            self._sync_cuda()

    def _run_iterations(self, case: BenchmarkCase, repeats: int) -> list[float]:
        durations: list[float] = []
        for _ in range(repeats):
            self._sync_cuda()
            if self.is_cuda:
                duration = self._time_with_cuda_events(case)
            else:
                duration = self._time_with_perf_counter(case)
            durations.append(duration)
            self._sync_cuda()
        return durations

    def _time_with_cuda_events(self, case: BenchmarkCase) -> float:
        stream = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream=stream)
        case.fn(*case.args, **case.kwargs)
        end.record(stream=stream)
        torch.cuda.synchronize(self.device)
        return float(start.elapsed_time(end))  # milliseconds

    def _time_with_perf_counter(self, case: BenchmarkCase) -> float:
        start = time.perf_counter()
        case.fn(*case.args, **case.kwargs)
        return (time.perf_counter() - start) * 1000.0

    def _sync_cuda(self) -> None:
        if self.sync and self.is_cuda:
            torch.cuda.synchronize(self.device)

    def _start_memory_trace(self) -> dict[str, int] | None:
        if not self.is_cuda:
            return None
        self._sync_cuda()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        return {
            "idle_allocated": torch.cuda.memory_allocated(self.device),
            "idle_reserved": torch.cuda.memory_reserved(self.device),
        }

    def _finish_memory_trace(
        self, idle_stats: dict[str, int] | None
    ) -> dict[str, int | None]:
        if not self.is_cuda or idle_stats is None:
            return {
                "idle_allocated": None,
                "peak_allocated": None,
                "workspace_reserved": None,
            }
        self._sync_cuda()
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        torch.cuda.empty_cache()
        runtime_alloc = max(0, peak_allocated - idle_stats["idle_allocated"])
        runtime_reserved = max(0, peak_reserved - idle_stats["idle_reserved"])
        return {
            "idle_allocated": idle_stats["idle_allocated"],
            "peak_allocated": idle_stats["idle_allocated"] + runtime_alloc,
            "workspace_reserved": runtime_reserved,
        }


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / _MB, 4)


def _ops_per_second(ops: float | None, latency_ms: float) -> float | None:
    if ops is None or latency_ms <= 0:
        return None
    seconds = latency_ms / 1000.0
    if seconds == 0:
        return None
    return ops / seconds
