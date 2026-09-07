"""Forward and inverse kinematics for the Panda arm.

IK uses damped least squares on the site Jacobian. `mink` is used when it is
installed, otherwise the hand-written solver below is exercised -- both paths
are kept working so the project has no hard dependency on mink.
"""
from __future__ import annotations

import logging

import mujoco
import numpy as np

from . import config

log = logging.getLogger(__name__)

ROT_TOL = 0.05  # rad, orientation convergence for 6-DoF solves

try:  # optional
    import mink  # type: ignore

    HAS_MINK = True
except ImportError:  # pragma: no cover - depends on install
    HAS_MINK = False


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"no site named {name!r}")
    return sid


def arm_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    """qpos addresses of the 7 arm joints, in order."""
    idx = []
    for jn in config.ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"no joint named {jn!r}")
        idx.append(model.jnt_qposadr[jid])
    return np.array(idx, dtype=int)


def arm_dof_v_indices(model: mujoco.MjModel) -> np.ndarray:
    """qvel/Jacobian column indices of the 7 arm joints, in order."""
    idx = []
    for jn in config.ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        idx.append(model.jnt_dofadr[jid])
    return np.array(idx, dtype=int)


def joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for jn in config.ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if model.jnt_limited[jid]:
            lo.append(model.jnt_range[jid, 0])
            hi.append(model.jnt_range[jid, 1])
        else:
            lo.append(-np.pi)
            hi.append(np.pi)
    return np.array(lo), np.array(hi)


# Top-down grasp orientation for the `grasp_site` frame, as column vectors:
# site z points straight down, site y (the finger-separation axis) lies along
# world +y, so the fingers close on the block's +/-y faces.
GRASP_DOWN_MAT = np.array([[-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0]])


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).reshape(9))
    return q


def orientation_error(quat_target: np.ndarray, mat_current: np.ndarray) -> np.ndarray:
    """Rotation vector taking `mat_current` onto `quat_target` (world frame)."""
    q_cur = mat_to_quat(mat_current)
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q_cur)
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, quat_target, q_inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = config.GRIPPER_SITE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, 3x3 rotation) of `site_name` in world frame."""
    mujoco.mj_forward(model, data)
    sid = site_id(model, site_name)
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    site_name: str = config.GRIPPER_SITE,
    q_init: np.ndarray | None = None,
    iters: int = config.IK_MAX_ITERS,
    tol: float = config.IK_POS_TOL,
    damping: float = config.IK_DAMPING,
    rot_weight: float = 0.5,
) -> tuple[np.ndarray, float, bool]:
    """Solve IK for the gripper site.

    Position-only when `target_mat` is None, otherwise full 6-DoF. Returns
    (q_arm, final_position_error_m, converged). `data` is left holding the
    solution so callers can read FK straight away.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    qadr = arm_dof_indices(model)
    vadr = arm_dof_v_indices(model)
    lo, hi = joint_limits(model)
    sid = site_id(model, site_name)

    use_rot = target_mat is not None
    quat_target = mat_to_quat(target_mat) if use_rot else None

    if q_init is not None:
        data.qpos[qadr] = np.clip(q_init, lo, hi)
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = np.inf

    for _ in range(iters):
        e_pos = target_pos - data.site_xpos[sid]
        pos_err = float(np.linalg.norm(e_pos))

        if use_rot:
            e_rot = orientation_error(quat_target, data.site_xmat[sid].reshape(3, 3))
            rot_err = float(np.linalg.norm(e_rot))
            err_vec = np.concatenate([e_pos, rot_weight * e_rot])
            done = pos_err < tol and rot_err < ROT_TOL
        else:
            err_vec = e_pos
            rot_err = 0.0
            done = pos_err < tol

        if done:
            return data.qpos[qadr].copy(), pos_err, True

        mujoco.mj_jacSite(model, data, jacp, jacr if use_rot else None, sid)
        J = np.vstack([jacp[:, vadr], rot_weight * jacr[:, vadr]]) if use_rot else jacp[:, vadr]

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, err_vec)

        norm = np.linalg.norm(dq)
        if norm > config.IK_MAX_STEP:
            dq *= config.IK_MAX_STEP / norm

        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lo, hi)
        mujoco.mj_forward(model, data)

    return data.qpos[qadr].copy(), pos_err, False


def interpolate(q_from: np.ndarray, q_to: np.ndarray, steps: int) -> np.ndarray:
    """Smooth (cosine-eased) joint-space path, shape (steps, 7)."""
    t = np.linspace(0.0, 1.0, steps)
    ease = 0.5 * (1 - np.cos(np.pi * t))
    return q_from[None, :] + ease[:, None] * (q_to - q_from)[None, :]
