"""Top-down pick-and-place primitives built from Cartesian waypoints.

Each waypoint is solved with 6-DoF IK constrained to a downward gripper, then
executed as a cosine-eased joint-space path so the position actuators can track
it without overshooting.
"""
from __future__ import annotations

import logging

import mujoco
import numpy as np

from . import config
from .kinematics import GRASP_DOWN_MAT, ik
from .sim import SimEnv

log = logging.getLogger(__name__)

GRASP_Z_OFFSET = -0.005   # TCP sits slightly below the block centre
LIFT_HEIGHT = 0.14
RELEASE_HEIGHT = 0.06     # above the container rim before opening
# Approach and lift waypoints only need to be roughly right; the grasp waypoint
# is the one that has to be accurate. Near the edge of the workspace the 2 mm
# grasp tolerance is unreachable for transit poses, so they get their own.
TRANSIT_TOL = 0.006
SETTLE_AFTER_CLOSE = 400
SETTLE_AFTER_OPEN = 200


class MotionFailure(RuntimeError):
    """Raised when a waypoint is unreachable with the grasp orientation."""


def _move_to(env: SimEnv, pos: np.ndarray, steps_per_wp: int = 6,
             waypoints: int = config.WAYPOINT_STEPS,
             tol: float = TRANSIT_TOL) -> None:
    """Solve IK for `pos` with a top-down gripper and drive the arm there."""
    scratch = mujoco.MjData(env.model)
    q_now = env.get_joints()
    q, err, converged = ik(env.model, scratch, pos, target_mat=GRASP_DOWN_MAT,
                           q_init=q_now, tol=tol)
    if not converged:
        raise MotionFailure(f"unreachable top-down: {np.round(pos, 3)} (err {err * 1000:.1f} mm)")
    env.follow(np.asarray([q]) if waypoints <= 1 else _path(q_now, q, waypoints), steps_per_wp)


def _path(q_from: np.ndarray, q_to: np.ndarray, steps: int) -> np.ndarray:
    from .kinematics import interpolate

    return interpolate(q_from, q_to, steps)


def pick(env: SimEnv, xyz: np.ndarray, body: str | None = None) -> bool:
    """Grasp an object whose centre is at `xyz`. Returns whether it lifted."""
    xyz = np.asarray(xyz, dtype=float)
    above = xyz + np.array([0.0, 0.0, LIFT_HEIGHT])
    grasp = xyz + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    z_before = env.body_pos(body)[2] if body else None

    env.open_gripper()
    _move_to(env, above)
    _move_to(env, grasp, steps_per_wp=8, tol=config.IK_POS_TOL)
    env.settle(100)

    env.close_gripper()
    env.settle(SETTLE_AFTER_CLOSE)

    _move_to(env, above, steps_per_wp=8)
    env.settle(200)

    if body is None:
        return True
    lifted = float(env.body_pos(body)[2] - z_before)
    log.info("pick(%s): lifted %.3f m", body, lifted)
    return lifted > 0.05


def place(env: SimEnv, xyz: np.ndarray) -> None:
    """Release whatever is held above `xyz`."""
    xyz = np.asarray(xyz, dtype=float)
    above = xyz + np.array([0.0, 0.0, LIFT_HEIGHT])
    release = xyz + np.array([0.0, 0.0, RELEASE_HEIGHT])

    _move_to(env, above)
    _move_to(env, release, steps_per_wp=8)
    env.open_gripper()
    env.settle(SETTLE_AFTER_OPEN)
    _move_to(env, above, steps_per_wp=8)
    env.settle(200)


def pick_and_place(env: SimEnv, src: np.ndarray, dst: np.ndarray,
                   body: str | None = None) -> bool:
    grasped = pick(env, src, body=body)
    place(env, dst)
    env.park()
    return grasped
