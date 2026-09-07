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


# --- perception -------------------------------------------------------------
@pytest.fixture(scope="module")
def render_env() -> SimEnv:
    """Separate env because the shared fixture is built with rendering off."""
    e = SimEnv(render=True)
    yield e
    e.close()


def test_camera_roundtrip_matches_ground_truth(render_env: SimEnv) -> None:
    """Project each object into the image and deproject it back.

    Exercises the pinhole model and the resting-object centre correction without
    loading the detector.
    """
    from src.perception import TABLE_TOP_Z, camera_pose, capture, deproject, intrinsics

    env = render_env
    env.reset()
    env.park()
    _, depth = capture(env, config.TOP_CAM)
    fx, fy, cx, cy = intrinsics(env.model, config.TOP_CAM)
    cam_pos, cam_mat = camera_pose(env, config.TOP_CAM)

    for name in config.OBJECTS:
        gt = env.body_pos(name)
        p_cam = cam_mat.T @ (gt - cam_pos)
        d = -p_cam[2]
        u, v = p_cam[0] * fx / d + cx, -p_cam[1] * fy / d + cy

        est = deproject(u, v, float(depth[int(round(v)), int(round(u))]), env)
        est[2] = (est[2] + TABLE_TOP_Z) / 2.0
        assert np.linalg.norm(est - gt) < 0.01, f"{name}: {np.round(est - gt, 4)}"


def test_perception_cameras_exist(env: SimEnv) -> None:
    for cam in config.PERCEPTION_CAMERAS:
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0


# --- planner ----------------------------------------------------------------
@pytest.mark.parametrize(("utterance", "target", "n_steps"), [
    ("put the red block in the bowl", "a red cube", 2),
    ("pick up the green cube", "a green cube", 1),
    ("sivappu block-ah bowl-la vai", "a red cube", 2),
    ("pachai block-ah edu", "a green cube", 1),
    ("நீல கட்டையை கிண்ணத்தில் வை", "a blue cube", 2),
    ("சிவப்பு கட்டையை எடு", "a red cube", 1),
])
def test_plan_fallback(utterance: str, target: str, n_steps: int) -> None:
    from src.planner import plan_fallback

    steps = plan_fallback(utterance)
    assert len(steps) == n_steps, steps
    assert steps[0] == {"action": "pick", "target": target}
    if n_steps == 2:
        assert steps[1]["action"] == "place"


def test_plan_fallback_ignores_unknown_objects() -> None:
    from src.planner import plan_fallback

    assert plan_fallback("bring me a coffee") == []


def test_extract_json_strips_fences_and_thinking() -> None:
    from src.planner import _extract_json

    raw = ('<think>the user wants the red one</think>\n```json\n'
           '[{"action":"pick","target":"a red cube"}]\n```')
    assert _extract_json(raw) == [{"action": "pick", "target": "a red cube"}]
    assert _extract_json("sorry, I cannot do that") is None
    assert _extract_json('[{"action":"teleport","target":"x"}]') is None


# --- speech router ----------------------------------------------------------
@pytest.mark.parametrize(("text", "expect_specialist"), [
    ("சிவப்பு பொருளை எடு", True),
    ("put the red block in the bowl", False),
    ("red block-ah edu", False),
    ("சிவப்பு block-ஐ bowl-ல வை", False),      # code-switched stays multilingual
])
def test_script_router(text: str, expect_specialist: bool) -> None:
    from src.speech import TAMIL_SCRIPT_THRESHOLD, tamil_ratio

    ratio = tamil_ratio(text)
    assert 0.0 <= ratio <= 1.0, "combining marks must not inflate the ratio"
    assert (ratio > TAMIL_SCRIPT_THRESHOLD) is expect_specialist


# --- episodes ---------------------------------------------------------------
def test_episode_roundtrip(tmp_path, monkeypatch) -> None:
    from src import episodes

    monkeypatch.setattr(episodes, "INDEX", tmp_path / "episodes.parquet")
    monkeypatch.setattr(config, "EPISODES", tmp_path)

    ep = episodes.Episode(raw_utterance="sivappu block-ah bowl-la vai",
                          plan_source="fallback", success=True, detect_error_m=0.012)
    episodes.record(ep, states=[np.zeros(7), np.ones(7)])

    df = episodes.load_index()
    assert len(df) == 1
    assert df.iloc[0]["raw_utterance"] == "sivappu block-ah bowl-la vai"
    assert (tmp_path / ep.episode_id / "data.parquet").exists()
