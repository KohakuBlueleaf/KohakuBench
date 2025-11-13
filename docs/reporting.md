# Reporting

`kohakubench.report.format_table(results)`
`kohakubench.report.print_table(results)`

Columns:
- name, mean_ms, std_ms
- idle_vram, peak_vram, workspace
- MACs, MACS, FLOPs, FLOPS

Memory stats:
- idle_vram: allocated bytes before warmup (MB)
- peak_vram: peak allocated bytes (idle + runtime) during iterations (MB)
- workspace: reserved bytes increase (MB)

Use `format_table` to render a string or `print_table` to print directly. For structured processing, consume `BenchmarkResult.to_dict()`.
