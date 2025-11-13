"""Example script for normalization workloads."""

from kohakubench.core import KoBench
from kohakubench.presets import norms
from kohakubench.report import print_table


def _tagged(case, tag):
    case.metadata["shape_tag"] = tag
    case.name = f"{case.name}:{tag}"
    return case


def main() -> None:
    bench = KoBench()
    cases = []

    layer_norm_cfgs = [
        {"batch_size": 2, "seq_len": 256, "embed_dim": 2048},
        {"batch_size": 4, "seq_len": 128, "embed_dim": 1024},
    ]
    for cfg in layer_norm_cfgs:
        case = norms.layer_norm_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        tag = f"ln_b{cfg['batch_size']}_s{cfg['seq_len']}_d{cfg['embed_dim']}"
        cases.append(_tagged(case, tag))

    group_norm_cfgs = [
        {"batch_size": 2, "channels": 256, "height": 32, "width": 32, "groups": 32},
        {"batch_size": 1, "channels": 512, "height": 64, "width": 64, "groups": 64},
    ]
    for cfg in group_norm_cfgs:
        case = norms.group_norm_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        tag = f"gn_b{cfg['batch_size']}_c{cfg['channels']}_h{cfg['height']}"
        cases.append(_tagged(case, tag))

    rms_norm_cfgs = [
        {"batch_size": 2, "seq_len": 256, "embed_dim": 2048},
        {"batch_size": 1, "seq_len": 2048, "embed_dim": 4096},
    ]
    for cfg in rms_norm_cfgs:
        case = norms.rms_norm_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        tag = f"rms_b{cfg['batch_size']}_s{cfg['seq_len']}_d{cfg['embed_dim']}"
        cases.append(_tagged(case, tag))

    results = bench.benchmark_many(cases)
    print_table(results)


if __name__ == "__main__":
    main()
