"""Ties the pipeline together: utterance -> plan -> perception -> motion."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from collections.abc import Callable

from . import config, episodes, perception, planner
from .grasp import MotionFailure, pick, place
from .sim import SimEnv

log = logging.getLogger(__name__)

PLACE_Z = 0.46            # release height inside the container
GT_BODIES = {
    "a red cube": "red_block", "a green cube": "green_block",
    "a blue cube": "blue_block", "a white bowl": config.CONTAINER,
}


def _ground_truth(env: SimEnv, target: str) -> np.ndarray | None:
    """Simulator pose for a target description, used only to score detection."""
    for phrase, body in GT_BODIES.items():
        if phrase.split()[1] in target.lower():        # match on the colour word
            return env.body_pos(body)
    return None


def episode_video(episode_id: str) -> Path:
    return config.EPISODES / episode_id / "video.mp4"


def execute(env: SimEnv, utterance: str, *, cleaned: str | None = None,
            transcript: "object | None" = None, record: bool = True,
            capture_video: bool = False,
            on_frame: Callable[[np.ndarray, int], None] | None = None,
            frame_every: int = 25) -> episodes.Episode:
    """Run one instruction end to end and return the recorded episode."""
    t0 = time.perf_counter()
    if not utterance.strip():
        raise ValueError("empty utterance -- refusing to plan from nothing")
    ep = episodes.Episode(raw_utterance=utterance,
                          cleaned_utterance=cleaned or utterance)

    if transcript is not None:
        ep.asr_multilingual = transcript.candidates.get("multilingual", "")
        ep.asr_tamil = transcript.candidates.get("tamil_specialist", "")
        ep.asr_source = transcript.source
        ep.detected_language = transcript.lang
        ep.asr_latency_s = transcript.latency_s

    steps, source, latency = planner.plan(ep.cleaned_utterance)
    ep.plan_json, ep.plan_source, ep.llm_latency_s = json.dumps(steps), source, latency
    if not steps:
        log.warning("empty plan for %r", utterance)
        ep.duration_s = time.perf_counter() - t0
        if record:
            episodes.record(ep)
        return ep

    env.reset()
    env.park()
    if capture_video or on_frame is not None:
        env.start_recording(config.SCENE_CAM, every=frame_every, on_frame=on_frame)
    steps_before = int(env.data.time / env.model.opt.timestep)
    held: str | None = None

    try:
        for step in steps:
            action, target = step["action"], step["target"]
            if action == "say":
                continue

            resting = "bowl" not in target.lower()
            xyz, det = perception.locate(env, target, resting=resting)
            if xyz is None:
                raise MotionFailure(f"could not see {target!r}")

            if action == "pick":
                ep.target_object = target
                ep.detected_xyz = episodes._fmt_xyz(xyz)
                gt = _ground_truth(env, target)
                if gt is not None:
                    ep.gt_xyz = episodes._fmt_xyz(gt)
                    ep.detect_error_m = float(np.linalg.norm(xyz - gt))
                body = GT_BODIES.get(target)
                held = body
                ep.success = pick(env, xyz, body=body)
            elif action in ("place", "move_to"):
                place(env, np.array([xyz[0], xyz[1], PLACE_Z]))
                if held is not None:
                    final = env.body_pos(held)
                    ep.success = bool(np.linalg.norm(final[:2] - xyz[:2]) < 0.075
                                      and final[2] > 0.40)
        env.park()
    except MotionFailure as exc:
        log.warning("execution failed: %s", exc)
        ep.success = False

    ep.num_sim_steps = int(env.data.time / env.model.opt.timestep) - steps_before
    ep.duration_s = time.perf_counter() - t0

    frames = env.stop_recording() if (capture_video or on_frame is not None) else None
    if record:
        episodes.record(ep, frames=frames, states=[env.get_joints()])
    if frames and capture_video:
        from .video import write_video

        if not write_video(frames, episode_video(ep.episode_id)):
            log.warning("no video encoder available for episode %s", ep.episode_id)
    return ep
