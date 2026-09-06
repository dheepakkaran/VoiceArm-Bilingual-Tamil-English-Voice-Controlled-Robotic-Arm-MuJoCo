"""M1 acceptance: drive the arm through three hardcoded joint configurations.

Renders one frame per configuration, writes a 3-panel image, and asserts the
gripper site actually moved between configurations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.kinematics import interpolate
from src.sim import SimEnv

# Collision-free above the table -- verified with a contact-count sweep.
POSES = {
    "home":       np.array([0.00, 0.00, 0.00, -1.571, 0.00, 1.571, -0.785]),
    "swing_left": np.array([-0.70, 0.00, 0.00, -1.571, 0.00, 1.571, -0.785]),
    "lift_up":    np.array([0.00, -0.40, 0.00, -1.300, 0.00, 1.300, -0.785]),
}


def main() -> int:
    env = SimEnv()
    frames, positions, errors = [], [], []

    q_prev = env.get_joints()
    for name, q in POSES.items():
        env.follow(interpolate(q_prev, q, config.WAYPOINT_STEPS))
        env.settle(300)
        q_prev = env.get_joints()

        track_err = float(np.max(np.abs(q_prev - q)))
        pos = env.gripper_pos()
        positions.append(pos)
        frames.append(env.render(config.TOP_CAM))
        print(f"{name:11s} target={np.round(q, 3)}")
        print(f"{'':11s} reached={np.round(q_prev, 3)}  gripper={np.round(pos, 4)}"
              f"  max_joint_err={track_err:.4f} rad")
        errors.append(track_err)

    panel = np.concatenate(frames, axis=1)
    out = config.OUT / "m1_frames.png"
    Image.fromarray(panel).save(out)

    deltas = [float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]
    print("\ngripper displacement between poses (m):", [round(d, 4) for d in deltas])
    print("saved", out)

    print("max joint tracking error (rad):", [round(e, 4) for e in errors])
    ok = all(d > 0.05 for d in deltas) and max(errors) < 0.05
    print("M1", "PASS" if ok else "FAIL")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
