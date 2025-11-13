"""Benchmark presets covering common deep learning workloads."""

from . import attention, convolution, mlp, norms, stable_diffusion, transformer

__all__ = [
    "attention",
    "convolution",
    "mlp",
    "norms",
    "stable_diffusion",
    "transformer",
]
