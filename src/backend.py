"""Runtime backend selection.

The local development machine is Apple Silicon, where MLX is much the fastest
way to run quantized models. Hugging Face Spaces is x86 Linux with an NVIDIA
slice, where MLX does not exist at all. Rather than fork the project, every
model-loading site asks this module which backend is live and which model id to
use, so one codebase runs in both places.

Set VOICEARM_BACKEND=torch to force the portable path on an Apple machine --
that is how the Spaces code path gets tested without deploying.
"""
from __future__ import annotations

import logging
import os
import platform

log = logging.getLogger(__name__)

_FORCED = os.environ.get("VOICEARM_BACKEND", "").strip().lower()


def _mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


if _FORCED in ("mlx", "torch"):
    BACKEND = _FORCED
    if BACKEND == "mlx" and not _mlx_available():
        raise RuntimeError("VOICEARM_BACKEND=mlx but MLX is not importable here")
else:
    BACKEND = "mlx" if _mlx_available() else "torch"

IS_MLX = BACKEND == "mlx"

# Quantized MLX repos on Apple Silicon; the upstream fp16 repos elsewhere. The
# Tamil ASR model and the detector are torch-only in both cases.
LLM_MODEL = "mlx-community/Qwen3-4B-4bit" if IS_MLX else "Qwen/Qwen3-4B-Instruct-2507"
ASR_MULTILINGUAL = ("mlx-community/whisper-large-v3-mlx" if IS_MLX
                    else "openai/whisper-large-v3-turbo")


def torch_device() -> str:
    """Best available torch device: CUDA on Spaces, MPS locally, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe() -> str:
    return f"backend={BACKEND} device={torch_device()} llm={LLM_MODEL}"
