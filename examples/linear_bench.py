"""Example script for linear/MLP style workloads."""

from kohakubench.core import KoBench
from kohakubench.presets import mlp
from kohakubench.report import print_table


def _tagged(case, cfg, suffix):
    tag = f"b{cfg['batch_size']}_s{cfg['seq_len']}_din{cfg['embed_dim']}_dhid{cfg['hidden_dim']}_{suffix}"
    case.metadata["shape_tag"] = tag
    case.name = f"{case.name}:{tag}"
    return case


def main() -> None:
    bench = KoBench()
    configs = [
        {"batch_size": 2, "seq_len": 256, "embed_dim": 1024, "hidden_dim": 4096},
        {"batch_size": 4, "seq_len": 128, "embed_dim": 1536, "hidden_dim": 6144},
        {"batch_size": 1, "seq_len": 1536, "embed_dim": 3072, "hidden_dim": 12288},
    ]
    cases = []
    for cfg in configs:
        standard = mlp.standard_mlp_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        cases.append(_tagged(standard, cfg, "standard"))
        swiglu = mlp.swiglu_mlp_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        cases.append(_tagged(swiglu, cfg, "swiglu"))
    results = bench.benchmark_many(cases)
    print_table(results)


if __name__ == "__main__":
    main()
