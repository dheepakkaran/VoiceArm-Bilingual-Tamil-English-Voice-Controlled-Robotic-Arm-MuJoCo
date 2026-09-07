"""Episode recording in a LeRobot-compatible parquet layout.

Written directly with pyarrow rather than through the `lerobot` package, which
pulls a heavy dependency tree this project does not otherwise need. The column
names and per-episode directory structure follow the LeRobot v2 convention so
the data can be loaded by it, but nothing here is produced by LeRobot itself.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

INDEX = config.EPISODES / "episodes.parquet"


@dataclass
class Episode:
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
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
    num_sim_steps: int = 0
    llm_latency_s: float = 0.0
    asr_latency_s: float = 0.0


def _fmt_xyz(v: np.ndarray | None) -> str:
    return "" if v is None else json.dumps([round(float(x), 4) for x in v])


def record(ep: Episode, frames: list[np.ndarray] | None = None,
           states: list[np.ndarray] | None = None,
           actions: list[np.ndarray] | None = None) -> str:
    """Append `ep` to the index and write its per-episode observation table."""
    row = pd.DataFrame([asdict(ep)])
    if INDEX.exists():
        row = pd.concat([pd.read_parquet(INDEX), row], ignore_index=True)
    row.to_parquet(INDEX, index=False)

    if states is not None:
        ep_dir = config.EPISODES / ep.episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        n = len(states)
        obs = pd.DataFrame({
            "frame_index": np.arange(n),
            "timestamp": np.arange(n, dtype=float) * 0.02,
            "observation.state": [np.asarray(s, dtype=np.float32) for s in states],
            "action": [np.asarray(a, dtype=np.float32) for a in (actions or states)],
        })
        obs.to_parquet(ep_dir / "data.parquet", index=False)
        if frames:
            np.save(ep_dir / "observation.image.npy",
                    np.asarray(frames[:: max(1, len(frames) // 24)], dtype=np.uint8))
    log.info("recorded episode %s (success=%s)", ep.episode_id, ep.success)
    return ep.episode_id


def load_index() -> pd.DataFrame:
    if not INDEX.exists():
        return pd.DataFrame(columns=[f.name for f in Episode.__dataclass_fields__.values()])
    return pd.read_parquet(INDEX)


def summary() -> dict[str, Any]:
    df = load_index()
    if df.empty:
        return {"episodes": 0, "success_rate": 0.0, "mean_detect_error_cm": float("nan"),
                "mean_duration_s": 0.0, "mean_llm_latency_s": 0.0, "tamil_specialist_share": 0.0}
    spoken = df[df["asr_source"] != "text"]
    return {
        "episodes": int(len(df)),
        "success_rate": float(df["success"].mean()),
        "mean_detect_error_cm": float(np.nanmean(df["detect_error_m"]) * 100),
        "mean_duration_s": float(df["duration_s"].mean()),
        "mean_llm_latency_s": float(df["llm_latency_s"].mean()),
        "tamil_specialist_share": float((spoken["asr_source"] == "tamil_specialist").mean())
        if len(spoken) else 0.0,
    }
