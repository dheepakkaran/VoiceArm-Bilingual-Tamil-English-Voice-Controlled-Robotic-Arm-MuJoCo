"""Smoke tests for the milestones completed so far (M1, M2)."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.kinematics import fk, ik, interpolate
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
