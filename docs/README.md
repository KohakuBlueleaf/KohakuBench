# KohakuBench Documentation

This is the central hub for KohakuBench docs. It explains how to benchmark PyTorch workloads on CPU/CUDA with KoBench, how presets work, how metrics are computed, and how to extend the system.

## Contents

- [Manual](manual.md) — step-by-step quickstart and core concepts
- [KoBench API](kobench-api.md) — classes, parameters, return types
- [Modules](modules.md) — building blocks like ScaledDotProductAttention
- [Presets](presets.md) — attention, MLP, convolution, norms, transformer, SD resblock
- [Metrics](metrics.md) — MACs/MACS and FLOPs/FLOPS definitions and counters
- [Reporting](reporting.md) — printing and formatting results
- [CLI](cli.md) — `kobench` usage and options
- [Architecture](architecture.md) — timing, memory tracing, design notes
- [Recipes](recipes.md) — programmatic usage, training steps, custom counters, CSV logging

Start with the Manual, then jump to Recipes for practical patterns. Refer to the KoBench API when coding.
