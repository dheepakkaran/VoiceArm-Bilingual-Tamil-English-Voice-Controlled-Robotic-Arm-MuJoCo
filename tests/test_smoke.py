"""Smoke tests for the milestones completed so far (M1, M2)."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.grasp import pick_and_place
from src.kinematics import GRASP_DOWN_MAT, fk, ik, interpolate, orientation_error, mat_to_quat
from src.sim import PARK_ARM, SimEnv


@pytest.fixture(scope="module")
def env() -> SimEnv:
    e = SimEnv(render=False)
    yield e
    e.close()


def test_scene_compiles(env: SimEnv) -> None:
    assert env.model.nu == 8          # 7 arm actuators + gripper
    assert env.model.nq == 30         # 9 robot + 3 free bodies x 7
    for name in [*config.OBJECTS, config.CONTAINER]:
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def test_injected_site_and_camera(env: SimEnv) -> None:
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, config.GRIPPER_SITE) >= 0
    for cam in (config.TOP_CAM, config.WRIST_CAM):
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0


def test_blocks_rest_on_table(env: SimEnv) -> None:
    env.reset()
    for name in config.OBJECTS:
        z = env.body_pos(name)[2]
        assert 0.41 < z < 0.43, f"{name} at z={z:.3f} is not resting on the table"


def test_ik_converges(env: SimEnv) -> None:
    env.reset()
    scratch = mujoco.MjData(env.model)
    target = np.array([0.50, -0.10, 0.55])
    q, err, converged = ik(env.model, scratch, target, q_init=env.get_joints())
    assert converged, f"IK did not converge (err={err:.4f} m)"

    env.teleport_joints(q)
    achieved, _ = fk(env.model, env.data, config.GRIPPER_SITE)
    assert np.linalg.norm(achieved - target) < config.IK_POS_TOL * 2.5


def test_ik_holds_grasp_orientation(env: SimEnv) -> None:
    env.reset()
    scratch = mujoco.MjData(env.model)
    target = np.array([0.45, 0.05, 0.50])
    q, _, converged = ik(env.model, scratch, target, target_mat=GRASP_DOWN_MAT,
                         q_init=env.get_joints())
    assert converged

    env.teleport_joints(q)
    _, R = fk(env.model, env.data, config.GRIPPER_SITE)
    rot_err = np.linalg.norm(orientation_error(mat_to_quat(GRASP_DOWN_MAT), R))
    assert rot_err < 0.05, f"gripper is {rot_err:.3f} rad off vertical"
    assert R[2, 2] < -0.99, "gripper z axis is not pointing down"


def test_pick_and_place_into_container(env: SimEnv) -> None:
    env.reset()
    bowl = env.body_pos(config.CONTAINER)
    src = env.body_pos("red_block")
    grasped = pick_and_place(env, src, np.array([bowl[0], bowl[1], 0.46]), body="red_block")
    assert grasped, "block was never lifted"

    final = env.body_pos("red_block")
    assert np.linalg.norm(final[:2] - bowl[:2]) < 0.075
    assert final[2] > 0.40


def test_interpolate_endpoints() -> None:
    a, b = np.zeros(7), np.ones(7)
    path = interpolate(a, b, 20)
    assert path.shape == (20, 7)
    np.testing.assert_allclose(path[0], a, atol=1e-9)
    np.testing.assert_allclose(path[-1], b, atol=1e-9)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert steps.max() < 0.3, "path is not smooth enough for the controller"


def test_park_clears_camera(env: SimEnv) -> None:
    env.reset()
    env.park()
    x = env.gripper_pos()[0]
    assert x < 0.20, f"parked gripper at x={x:.3f} still overlaps the worktop"
    np.testing.assert_allclose(env.get_joints(), PARK_ARM, atol=0.05)
