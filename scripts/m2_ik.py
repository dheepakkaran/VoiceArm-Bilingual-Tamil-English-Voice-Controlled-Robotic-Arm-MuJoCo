"""M2 acceptance: solve IK for 10 random reachable targets and verify with FK.

Each target is solved on a scratch MjData so the live simulation state is not
disturbed, then the solution is applied to the real environment and the achieved
gripper position is measured with forward kinematics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.kinematics import HAS_MINK, fk, ik
from src.sim import SimEnv

N_TARGETS = 10
TOL_M = 0.005  # 5 mm acceptance
# Workspace box above the table (table top is at z = 0.40).
LOW = np.array([0.35, -0.25, 0.45])
HIGH = np.array([0.65, 0.25, 0.68])


def main() -> int:
    rng = np.random.default_rng(0)
    env = SimEnv(render=False)
    scratch = mujoco.MjData(env.model)

    print(f"IK backend: {'mink' if HAS_MINK else 'damped least squares (built-in)'}")
    print(f"{'#':>2}  {'target (x,y,z)':<26} {'achieved':<26} {'err_mm':>7} {'iters_ms':>9}  ok")

    errors, times, passes = [], [], 0
    q_seed = env.get_joints()

    for i in range(N_TARGETS):
        target = rng.uniform(LOW, HIGH)

        t0 = time.perf_counter()
        q, solver_err, converged = ik(env.model, scratch, target, q_init=q_seed)
        dt_ms = (time.perf_counter() - t0) * 1000
        times.append(dt_ms)

        # Apply to the real env and measure independently with FK.
        env.teleport_joints(q)
        achieved, _ = fk(env.model, env.data, config.GRIPPER_SITE)
        err = float(np.linalg.norm(achieved - target))
        errors.append(err)

        ok = err < TOL_M and converged
        passes += ok
        print(f"{i:>2}  {np.round(target, 3)!s:<26} {np.round(achieved, 3)!s:<26} "
              f"{err * 1000:>7.2f} {dt_ms:>9.1f}  {'PASS' if ok else 'FAIL'}")
        q_seed = q  # warm-start the next solve

    print(f"\nmean error  {np.mean(errors) * 1000:.2f} mm   "
          f"max error {np.max(errors) * 1000:.2f} mm   "
          f"mean solve {np.mean(times):.1f} ms")
    print(f"M2 {passes}/{N_TARGETS} targets within {TOL_M * 1000:.0f} mm")
    print("M2", "PASS" if passes == N_TARGETS else "FAIL")
    return 0 if passes == N_TARGETS else 1


if __name__ == "__main__":
    raise SystemExit(main())
