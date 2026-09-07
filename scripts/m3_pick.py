"""M3 acceptance: pick each block by its ground-truth pose and drop it in the bowl.

Perception is not involved yet -- positions come straight from the simulator.
Records the red-block run to out/m3_pickplace.mp4 (with a frame-grid PNG
fallback when no video encoder is available).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.grasp import pick_and_place
from src.video import frame_grid, write_video
from src.sim import SimEnv

BOWL_XY_TOL = 0.075   # inside the container walls
DROP_Z_MIN = 0.40     # must still be off the floor


def main() -> int:
    env = SimEnv()
    bowl = env.body_pos(config.CONTAINER)
    dst = np.array([bowl[0], bowl[1], 0.46])

    print(f"{'block':<13}{'grasped':<9}{'final xyz':<26}{'xy→bowl':>9}  result")
    results, frames = [], []

    for i, blk in enumerate(config.OBJECTS):
        env.reset()
        if i == 0:
            env.start_recording(config.SCENE_CAM, every=25)

        src = env.body_pos(blk)
        grasped = pick_and_place(env, src, dst, body=blk)

        if i == 0:
            frames = env.stop_recording()

        final = env.body_pos(blk)
        xy = float(np.linalg.norm(final[:2] - bowl[:2]))
        ok = grasped and xy < BOWL_XY_TOL and final[2] > DROP_Z_MIN
        results.append(ok)
        print(f"{blk:<13}{str(grasped):<9}{np.round(final, 3)!s:<26}{xy * 100:>7.1f}cm  "
              f"{'PASS' if ok else 'FAIL'}")

    if frames:
        mp4 = config.OUT / "m3_pickplace.mp4"
        if write_video(frames, mp4):
            print(f"\nsaved {mp4} ({len(frames)} frames)")
        else:
            print("\nno video encoder available, writing frame grid instead")
        grid = config.OUT / "m3_pickplace.png"
        frame_grid(frames, grid)
        print(f"saved {grid}")

    n = sum(results)
    print(f"\nM3 {n}/{len(results)} blocks placed in the container")
    print("M3", "PASS" if n == len(results) else "FAIL")
    env.close()
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
