"""Run the pipeline and print what happened.

    python scripts/demo.py                      # a few example commands
    python scripts/demo.py --text "..."         # one command
    python scripts/demo.py --mic                # speak it
    python scripts/demo.py --wav clip.wav       # from a recording
    python scripts/demo.py --check              # arm, IK and grasping only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, speech
from src.executor import execute
from src.grasp import pick_and_place
from src.kinematics import GRASP_DOWN_MAT, fk, ik
from src.sim import SimEnv

EXAMPLES = [
    "drop the red cube into the bowl",
    "green block-ah edu",
    "நீலக் கட்டையை கிண்ணத்தில் போடு",
]


def check(env: SimEnv) -> bool:
    """Arm, IK and grasping, without any models loaded."""
    rng = np.random.default_rng(0)
    scratch = mujoco.MjData(env.model)
    errors = []

    print("inverse kinematics, 10 random targets")
    for _ in range(10):
        target = rng.uniform([0.35, -0.25, 0.45], [0.60, 0.25, 0.65])
        q, _, converged = ik(env.model, scratch, target,
                             target_mat=GRASP_DOWN_MAT, q_init=env.get_joints())
        env.teleport_joints(q)
        reached, _ = fk(env.model, env.data, config.GRIPPER_SITE)
        errors.append(float(np.linalg.norm(reached - target)))
        if not converged:
            print(f"  did not converge at {np.round(target, 3)}")
    print(f"  mean {np.mean(errors) * 1000:.2f} mm, worst "
          f"{np.max(errors) * 1000:.2f} mm, {sum(e < 0.005 for e in errors)}/10 "
          f"within 5 mm\n")

    print("pick and place, using the simulator's own positions")
    placed = 0
    bowl = env.body_pos(config.CONTAINER)
    for block in config.OBJECTS:
        env.reset()
        pick_and_place(env, env.body_pos(block),
                       np.array([bowl[0], bowl[1], 0.46]), body=block)
        final = env.body_pos(block)
        ok = np.linalg.norm(final[:2] - bowl[:2]) < 0.075 and final[2] > 0.40
        placed += ok
        print(f"  {block:<13}{'in the bowl' if ok else 'MISSED'}")
    print(f"  {placed}/{len(config.OBJECTS)} placed")

    return bool(np.max(errors) < 0.005 and placed == len(config.OBJECTS))


def run(env: SimEnv, utterance: str) -> bool:
    result = execute(env, utterance)
    print(f"  plan ({result.plan_source}): "
          f"{[s['action'] + ' ' + s['target'] for s in result.plan]}")
    line = f"  {'ok' if result.success else 'failed'} in {result.duration_s:.1f} s"
    if result.detect_error_m == result.detect_error_m:
        line += f", detector was {result.detect_error_m * 100:.1f} cm off"
    print(line)
    return result.success


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="run this one command")
    ap.add_argument("--mic", action="store_true", help="record and run it")
    ap.add_argument("--wav", help="transcribe this file and run it")
    ap.add_argument("--check", action="store_true",
                    help="arm, IK and grasping only, no models")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    env = SimEnv()

    if args.check:
        ok = check(env)
        print("\ncheck", "PASSED" if ok else "FAILED")
        env.close()
        return 0 if ok else 1

    if args.mic or args.wav:
        audio = speech.record(args.seconds) if args.mic else speech.load_wav(args.wav)
        print(f"audio level rms {speech.rms(audio):.4f}")
        try:
            transcript = speech.transcribe(audio)
        except speech.NoSpeechDetected as exc:
            print(f"no speech: {exc}")
            return 2
        print(f"heard [{transcript.language}]: {transcript.text!r}")
        commands = [transcript.text]
    else:
        commands = [args.text] if args.text else EXAMPLES

    passed = 0
    for utterance in commands:
        print(f"\n{utterance}")
        passed += run(env, utterance)

    print(f"\n{passed}/{len(commands)} succeeded")
    env.close()
    return 0 if passed == len(commands) else 1


if __name__ == "__main__":
    raise SystemExit(main())
