"""Frame-sequence helpers shared by the milestone scripts and the dashboard."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> bool:
    """Encode frames to mp4. Returns False when no encoder is available."""
    if not frames:
        return False
    try:
        import cv2
    except ImportError:
        return False

    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():                      # H.264 unavailable, fall back
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            return False
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 1024


def frame_grid(frames: list[np.ndarray], path: Path, cols: int = 4, rows: int = 2) -> None:
    """Even sample of `frames` laid out as a contact sheet."""
    idx = np.linspace(0, len(frames) - 1, cols * rows).astype(int)
    picked = [frames[i] for i in idx]
    grid = np.concatenate(
        [np.concatenate(picked[r * cols:(r + 1) * cols], axis=1) for r in range(rows)],
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
