"""Release framework memory caches between pipeline stages.

MLX and torch both keep allocator pools that survive a forward pass. With four
models resident on a 16 GB machine those pools are large enough that generation
starts paging, and the planner slows from under 2 s to over 35 s -- a 20x
penalty that is pure allocator pressure, not compute. Dropping the caches at
stage boundaries costs a few milliseconds and recovers all of it.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def release_caches() -> None:
    """Drop MLX and torch allocator caches. Safe to call when neither is loaded."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # mlx absent or no Metal device
        pass

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def stats() -> dict[str, float]:
    """Current MLX allocator figures in GB, for the dashboard and benchmarks."""
    try:
        import mlx.core as mx

        return {
            "mlx_active_gb": mx.get_active_memory() / 1e9,
            "mlx_peak_gb": mx.get_peak_memory() / 1e9,
            "mlx_cache_gb": mx.get_cache_memory() / 1e9,
        }
    except Exception:
        return {}
