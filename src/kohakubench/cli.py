"""Command line interface for KohakuBench."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, List

from .core import BenchmarkCase, KoBench, resolve_dtype
from .presets import attention, convolution, mlp, norms, stable_diffusion, transformer
from .report import print_table


def _build_preset_registry() -> (
    Dict[str, Callable[[Dict[str, object]], Iterable[BenchmarkCase]]]
):
    return {
        "attn": lambda opts: [attention.multi_head_attention_case(**opts)],
        "mlp": lambda opts: [
            mlp.standard_mlp_case(**opts),
            mlp.swiglu_mlp_case(**opts),
        ],
        "conv": lambda opts: [
            convolution.conv2d_case(**opts),
            convolution.depthwise_conv2d_case(**opts),
        ],
        "norm": lambda opts: [
            norms.layer_norm_case(**opts),
            norms.group_norm_case(**opts),
            norms.rms_norm_case(**opts),
        ],
        "transformer": lambda opts: [transformer.transformer_block_case(**opts)],
        "sdres": lambda opts: [stable_diffusion.stable_diffusion_resblock_case(**opts)],
    }


PRESET_BUILDERS = _build_preset_registry()
PRESET_COUNTS = {
    "attn": 1,
    "mlp": 2,
    "conv": 2,
    "norm": 3,
    "transformer": 1,
    "sdres": 1,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KohakuBench presets.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_BUILDERS.keys()),
        default="attn",
        help="Preset or workload to benchmark.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device string, eg cuda:0 or cpu."
    )
    parser.add_argument(
        "--dtype", default=None, help="Torch dtype (float16, float32, bf16)."
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="Number of measured iterations."
    )
    parser.add_argument("--list", action="store_true", help="List presets and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for key in sorted(PRESET_BUILDERS.keys()):
            count = PRESET_COUNTS.get(key, "?")
            print(f"{key:12s} ({count} cases)")
        return 0

    bench_kwargs = {}
    if args.device:
        bench_kwargs["device"] = args.device
    if args.dtype:
        bench_kwargs["dtype"] = resolve_dtype(args.dtype, device=args.device)
    if args.warmup is not None:
        bench_kwargs["warmup"] = args.warmup
    if args.iters is not None:
        bench_kwargs["iters"] = args.iters

    bench = KoBench(**bench_kwargs)
    preset_opts = {"device": bench.device, "dtype": bench.dtype}
    cases = list(PRESET_BUILDERS[args.preset](preset_opts))
    results = bench.benchmark_many(cases)
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
