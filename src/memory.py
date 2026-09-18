"""Drop the framework's memory caches between stages.

PyTorch keeps an allocator pool alive after a forward pass. With three models
loaded on a 16 GB machine those pools are big enough to push the machine into
swap, and a single plan goes from under two seconds to minutes. Clearing them
between stages costs a few milliseconds.

Clearing after *every* call is worse, not better -- the next operation has to
reallocate from scratch. Only the handoff between models is worth clearing.
"""
from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)


def release_caches() -> None:
    """Free cached blocks. Safe to call whether or not a model is loaded."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
