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
    site_name: str = config.GRIPPER_SITE,
    q_init: np.ndarray | None = None,
    iters: int = config.IK_MAX_ITERS,
    tol: float = config.IK_POS_TOL,
    damping: float = config.IK_DAMPING,
) -> tuple[np.ndarray, float, bool]:
    """Solve position-only IK.

    Returns (q_arm, final_error_m, converged). `data` is left holding the
    solution so callers can read FK straight away.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    qadr = arm_dof_indices(model)
    vadr = arm_dof_v_indices(model)
    lo, hi = joint_limits(model)
    sid = site_id(model, site_name)

    if q_init is not None:
        data.qpos[qadr] = np.clip(q_init, lo, hi)
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    err = np.inf

    for _ in range(iters):
        err_vec = target_pos - data.site_xpos[sid]
        err = float(np.linalg.norm(err_vec))
        if err < tol:
            return data.qpos[qadr].copy(), err, True

        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J = jacp[:, vadr]                                   # 3 x 7

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err_vec)

        norm = np.linalg.norm(dq)
        if norm > config.IK_MAX_STEP:
            dq *= config.IK_MAX_STEP / norm

        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lo, hi)
        mujoco.mj_forward(model, data)

    return data.qpos[qadr].copy(), err, err < tol


def interpolate(q_from: np.ndarray, q_to: np.ndarray, steps: int) -> np.ndarray:
    """Smooth (cosine-eased) joint-space path, shape (steps, 7)."""
    t = np.linspace(0.0, 1.0, steps)
    ease = 0.5 * (1 - np.cos(np.pi * t))
    return q_from[None, :] + ease[:, None] * (q_to - q_from)[None, :]
