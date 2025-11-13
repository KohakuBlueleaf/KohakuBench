"""Example script for multi-head attention workloads."""

from kohakubench.core import KoBench
from kohakubench.presets import attention
from kohakubench.report import print_table


def _tagged(case, cfg):
    tag = f"b{cfg['batch_size']}_s{cfg['seq_len']}_d{cfg['embed_dim']}_h{cfg['num_heads']}"
    case.metadata["shape_tag"] = tag
    case.name = f"{case.name}:{tag}"
    return case


def main() -> None:
    bench = KoBench()
    configs = [
        {"batch_size": 2, "seq_len": 1024, "embed_dim": 1024, "num_heads": 16},
        {"batch_size": 1, "seq_len": 2048, "embed_dim": 2048, "num_heads": 32},
        {"batch_size": 4, "seq_len": 512, "embed_dim": 768, "num_heads": 12},
    ]
    cases = []
    for cfg in configs:
        case = attention.multi_head_attention_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        cases.append(_tagged(case, cfg))
    results = bench.benchmark_many(cases)
    print_table(results)


if __name__ == "__main__":
    main()
