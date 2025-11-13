"""Example script for convolution workloads."""

from kohakubench.core import KoBench
from kohakubench.presets import convolution
from kohakubench.report import print_table


def _tagged(case, tag):
    case.metadata["shape_tag"] = tag
    case.name = f"{case.name}:{tag}"
    return case


def main() -> None:
    bench = KoBench()
    cases = []
    conv_configs = [
        {"batch_size": 4, "in_channels": 128, "out_channels": 256, "image_size": 64, "kernel_size": 3},
        {"batch_size": 8, "in_channels": 64, "out_channels": 192, "image_size": 128, "kernel_size": 5, "stride": 2},
    ]
    for cfg in conv_configs:
        case = convolution.conv2d_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        tag = f"b{cfg['batch_size']}_c{cfg['in_channels']}to{cfg['out_channels']}_n{cfg['image_size']}"
        cases.append(_tagged(case, tag))

    depthwise_configs = [
        {"batch_size": 4, "channels": 256, "image_size": 64, "kernel_size": 3, "pointwise": True},
        {"batch_size": 2, "channels": 512, "image_size": 32, "kernel_size": 5, "pointwise": False},
    ]
    for cfg in depthwise_configs:
        case = convolution.depthwise_conv2d_case(
            **cfg,
            device=bench.device,
            dtype=bench.dtype,
        )
        tag = (
            f"dw_b{cfg['batch_size']}_c{cfg['channels']}_n{cfg['image_size']}"
            f"_{'pw' if cfg.get('pointwise', True) else 'nopw'}"
        )
        cases.append(_tagged(case, tag))
    results = bench.benchmark_many(cases)
    print_table(results)


if __name__ == "__main__":
    main()
