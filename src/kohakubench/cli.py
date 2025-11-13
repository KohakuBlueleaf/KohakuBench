"""Command line interface for KohakuBench.

Supports built-in presets by name and user-provided presets via a Python file path
or importable module path passed to --preset.
"""

import argparse
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .core import BenchmarkCase, KoBench, resolve_dtype
from .presets import attention, convolution, mlp, norms, stable_diffusion, transformer
from .report import print_table


def _build_preset_registry() -> (
    dict[str, Callable[[dict[str, object]], Iterable[BenchmarkCase]]]
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
PRESET_COUNTS: dict[str, int] = {
    "attn": 1,
    "mlp": 2,
    "conv": 2,
    "norm": 3,
    "transformer": 1,
    "sdres": 1,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KohakuBench presets.")
    parser.add_argument(
        "--preset",
        default="attn",
        help=(
            "Preset name (built-in) or a Python file/module providing a preset, "
            "e.g. 'attn' or 'examples/user_preset_example.py' or 'mypkg.my_preset'."
        ),
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
        print("\nTo use a user preset: kobench --preset path/to/preset.py")
        return 0

    bench_kwargs: dict[str, object] = {}
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

    # Resolve preset: built-in key, python file, or module path
    cases: Sequence[BenchmarkCase]
    preset_key = args.preset
    if preset_key in PRESET_BUILDERS:
        cases = list(PRESET_BUILDERS[preset_key](preset_opts))
    else:
        builder = _load_user_preset(preset_key)
        result = builder(preset_opts)
        # Normalize return value
        if isinstance(result, BenchmarkCase):
            cases = [result]
        else:
            cases = list(result)
    results = bench.benchmark_many(cases)
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _load_user_preset(
    path_or_module: str,
) -> Callable[[dict[str, object]], Iterable[BenchmarkCase]]:
    """Load a user preset from a file path or module path.

    The loaded module should expose one of the following (searched in order):
    - function build_cases(opts) -> Iterable[BenchmarkCase]
    - function make_cases(opts) -> Iterable[BenchmarkCase]
    - function get_cases(opts) -> Iterable[BenchmarkCase]
    - class Preset with method build(opts) -> Iterable[BenchmarkCase]
    - variable CASES: Iterable[BenchmarkCase]

    The opts dict will contain keys {"device": torch.device, "dtype": torch.dtype}.
    """
    module = _import_user_module(path_or_module)
    # Functions
    for name in ("build_cases", "make_cases", "get_cases"):
        fn = getattr(module, name, None)
        if callable(fn):
            return lambda opts: fn(opts)
    # Class Preset
    Preset = getattr(module, "Preset", None)
    if inspect.isclass(Preset):
        inst = Preset()
        if hasattr(inst, "build") and callable(getattr(inst, "build")):
            return lambda opts: inst.build(opts)
        raise RuntimeError(
            "Preset class found but missing a callable 'build(opts)' method."
        )
    # Variable CASES
    cases = getattr(module, "CASES", None)
    if cases is not None:

        def _from_cases(_: dict[str, object]) -> Iterable[BenchmarkCase]:
            return cases

        return _from_cases
    raise RuntimeError(
        "No valid preset entry found. Define one of: build_cases(opts), make_cases(opts), "
        "get_cases(opts), class Preset with build(opts), or a CASES list."
    )


def _import_user_module(path_or_module: str):
    p = Path(path_or_module)
    if p.suffix == ".py" or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Preset file not found: {path_or_module}")
        spec = importlib.util.spec_from_file_location("kobench_user_preset", p)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load preset module from {path_or_module}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    # Try importing as module path
    try:
        return importlib.import_module(path_or_module)
    except Exception as e:
        raise ImportError(
            f"Could not import user preset '{path_or_module}'. Provide a built-in name, "
            f"a .py file path, or an importable module path. Error: {e}"
        ) from e
