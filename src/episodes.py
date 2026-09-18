"""Append one row per command to a CSV so runs can be compared afterwards.

A CSV because nothing reads this back except the tests. An earlier version wrote
parquet in a LeRobot-compatible layout -- per-episode observation tables, image
arrays, the whole convention -- which only makes sense if something downstream
trains on it. Nothing here does.
"""
from __future__ import annotations

import csv
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config

log = logging.getLogger(__name__)

LOG_FILE = config.EPISODES / "episodes.csv"


@dataclass
class Episode:
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    raw_utterance: str = ""
    asr_multilingual: str = ""
    asr_tamil: str = ""
    asr_source: str = "text"           # "text" when typed rather than spoken
    cleaned_utterance: str = ""
    detected_language: str = ""
    plan_json: str = "[]"
    plan_source: str = ""              # "llm" | "fallback"
    target_object: str = ""
    detected_xyz: str = ""
    gt_xyz: str = ""
    detect_error_m: float = float("nan")
    success: bool = False
    duration_s: float = 0.0
    llm_latency_s: float = 0.0
    asr_latency_s: float = 0.0


def _fmt_xyz(v: np.ndarray | None) -> str:
    return "" if v is None else ",".join(f"{x:.4f}" for x in v)


def record(ep: Episode) -> str:
    """Append `ep` to the CSV, writing the header on first use."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(ep)
    new_file = not LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    log.info("recorded episode %s (success=%s)", ep.episode_id, ep.success)
    return ep.episode_id


def load() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with LOG_FILE.open(newline="") as f:
        return list(csv.DictReader(f))
