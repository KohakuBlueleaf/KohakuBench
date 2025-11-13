"""Text formatting helpers for KoBench results."""

from typing import Iterable, Sequence

from .core import BenchmarkResult

__all__ = ["format_table", "print_table"]


def format_table(results: Sequence[BenchmarkResult], precision: int = 3) -> str:
    headers = [
        "name",
        "mean_ms",
        "std_ms",
        "idle_vram",
        "peak_vram",
        "workspace",
        "MACs",
        "MACS",
        "FLOPs",
        "FLOPS",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.name,
                _format_ms(result.latency_ms, precision),
                _format_ms(result.latency_std_ms, precision),
                _format_memory(result.idle_vram_mb),
                _format_memory(result.peak_vram_mb),
                _format_memory(result.workspace_vram_mb),
                _format_ops(result.macs),
                _format_rate(result.macs_rate),
                _format_ops(result.flops),
                _format_rate(result.flops_rate),
            ]
        )
    widths = [len(head) for head in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = " | "
    header_line = sep.join(head.ljust(widths[idx]) for idx, head in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    data_lines = [
        sep.join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, divider, *data_lines])


def print_table(results: Sequence[BenchmarkResult], precision: int = 3) -> None:
    print(format_table(results, precision=precision))


def _format_ms(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _format_memory(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024:
        return f"{value/1024:.2f} GB"
    return f"{value:.2f} MB"


def _format_ops(value: float | None) -> str:
    if value is None or value == 0:
        return "n/a"
    units = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for scale, suffix in units:
        if value >= scale:
            return f"{value/scale:.2f} {suffix}"
    return f"{value:.2f}"


def _format_rate(value: float | None) -> str:
    if value is None or value == 0:
        return "n/a"
    units = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for scale, suffix in units:
        if value >= scale:
            return f"{value/scale:.2f} {suffix}/s"
    return f"{value:.2f}/s"
