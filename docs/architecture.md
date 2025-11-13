# Architecture

This document explains how KoBench measures runtime and memory.

## Timing
- CUDA: Uses `torch.cuda.Event(enable_timing=True)` on the current stream to measure kernel time. Synchronizes the device to ensure correctness.
- CPU: Uses `time.perf_counter()` around the callable.

## Memory
- Clears caches and resets peak memory before measurement.
- Records `idle_allocated` and `idle_reserved`.
- After iterations, reads `max_memory_allocated` and `max_memory_reserved`.
- Reports:
  - idle_vram_mb: baseline allocations (MB)
  - peak_vram_mb: idle + runtime allocation peak (MB)
  - workspace_vram_mb: increase in reserved memory (MB)

## Modes
- `inference_mode=True` wraps calls in `torch.inference_mode()` for speed and to disable autograd.
- `sync=True` forces a device synchronize between steps when enabled.

## Extensibility
- Counters are pluggable via `BenchmarkCase` fields; add realistic MACs/FLOPs functions for your domain.
- Results are structured (`BenchmarkResult`) and convertible to dict for export.
