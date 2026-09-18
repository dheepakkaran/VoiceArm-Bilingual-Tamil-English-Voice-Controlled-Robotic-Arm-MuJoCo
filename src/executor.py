"""Run one instruction: plan it, find the object, move the arm."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np

from . import config, perception, planner
from .grasp import MotionFailure, pick, place
from .sim import SimEnv

log = logging.getLogger(__name__)

PLACE_Z = 0.46            # release height inside the container

# Only used to score the detector against the simulator.
BODIES = {
    "a red cube": "red_block", "a green cube": "green_block",
    "a blue cube": "blue_block", "a white bowl": config.CONTAINER,
}


@dataclass
class Result:
    utterance: str
    plan: list[dict]
    plan_source: str
    success: bool
    duration_s: float
    detect_error_m: float = float("nan")
    frames: list = None


def execute(env: SimEnv, utterance: str, on_frame=None) -> Result:
    if not utterance.strip():
        raise ValueError("empty utterance -- refusing to plan from nothing")

    t0 = time.perf_counter()
    steps, source = planner.plan(utterance)
    if not steps:
        log.warning("no plan for %r", utterance)
        return Result(utterance, [], source, False, time.perf_counter() - t0)

    env.reset()
    env.park()
    if on_frame is not None:
        env.start_recording(config.SCENE_CAM, every=25, on_frame=on_frame)

    success, error, held = False, float("nan"), None
    try:
        for step in steps:
            action, target = step["action"], step["target"]
            if action == "say":
                continue

            resting = "bowl" not in target.lower()
            xyz, _ = perception.locate(env, target, resting=resting)
            if xyz is None:
                raise MotionFailure(f"could not see {target!r}")

            if action == "pick":
                body = BODIES.get(target)
                if body:
                    error = float(np.linalg.norm(xyz - env.body_pos(body)))
                held = body
                success = pick(env, xyz, body=body)
            else:
                place(env, np.array([xyz[0], xyz[1], PLACE_Z]))
                if held:
                    final = env.body_pos(held)
                    success = bool(np.linalg.norm(final[:2] - xyz[:2]) < 0.075
                                   and final[2] > 0.40)
        env.park()
    except MotionFailure as exc:
        log.warning("failed: %s", exc)
        success = False

    frames = env.stop_recording() if on_frame is not None else None
    return Result(utterance, steps, source, success,
                  time.perf_counter() - t0, error, frames)
