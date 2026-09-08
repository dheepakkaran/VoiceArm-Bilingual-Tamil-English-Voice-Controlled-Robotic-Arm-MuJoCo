"""MuJoCo simulation wrapper for the VoiceArm scene.

The Franka model shipped with `mujoco_menagerie` has no end-effector site and no
wrist camera, and MJCF `<include>` cannot inject children into a body defined by
the included file. Both are therefore added programmatically with `MjSpec`
before compiling, which keeps the vendored asset untouched.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from . import config

log = logging.getLogger(__name__)

# Panda TCP: 0.1034 m along +z of the `hand` frame, between the fingertips.
_TCP_OFFSET = (0.0, 0.0, 0.1034)
_HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Retracted pose that pulls the whole arm behind the base and out of the
# overhead camera frustum, so `topcam` sees an unoccluded worktop. Found with a
# joint sweep constrained to gripper x < 0.22 m and no new contacts.
PARK_ARM = np.array([0.0, -1.6, 0.0, -1.6, 0.0, 2.4, -0.7853])

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def build_model() -> mujoco.MjModel:
    """Compile the scene with the grasp site and wrist camera injected."""
    spec = mujoco.MjSpec.from_file(str(config.SCENE_XML))
    hand = spec.body("hand")
    hand.add_site(name=config.GRIPPER_SITE, pos=_TCP_OFFSET, size=[0.006], rgba=[1, 0, 0, 0])
    hand.add_camera(
        name=config.WRIST_CAM,
        pos=[0.05, 0.0, 0.02],
        quat=[0.7071, 0.0, 0.7071, 0.0],
        fovy=58,
    )
    model = spec.compile()
    log.info("model compiled: nq=%d nv=%d nu=%d", model.nq, model.nv, model.nu)
    return model


class SimEnv:
    """Thin, stateful wrapper around one MuJoCo model/data pair."""

    def __init__(self, render: bool = True) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._qadr = self._arm_qpos_adr()
        # A MuJoCo Renderer owns an OpenGL context bound to the thread that
        # created it, and any web framework will call in from a different worker
        # thread per request. On macOS both reusing a context across threads and
        # creating a second one from another thread deadlock rather than raise.
        # All rendering is therefore funnelled through one worker thread that
        # owns every context; callers block on the result.
        self._renderers: dict[bool, mujoco.Renderer] = {}
        self._gl: ThreadPoolExecutor | None = None
        self._render_enabled = render
        self._rec_frames: list[np.ndarray] = []
        self._rec_every = 0
        self._rec_cam = config.SCENE_CAM
        self._rec_tick = 0
        self._rec_cb: Callable[[np.ndarray, int], None] | None = None
        self.reset()

    # -- setup ---------------------------------------------------------------
    def _arm_qpos_adr(self) -> np.ndarray:
        adr = []
        for jn in config.ARM_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            adr.append(self.model.jnt_qposadr[jid])
        return np.array(adr, dtype=int)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qadr] = _HOME_ARM
        self.data.ctrl[:7] = _HOME_ARM
        self.data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.settle(200)

    # -- state ---------------------------------------------------------------
    def get_joints(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    def set_joints(self, q: np.ndarray) -> None:
        """Command the position actuators (does not teleport the arm)."""
        self.data.ctrl[:7] = np.asarray(q, dtype=float)[:7]

    def teleport_joints(self, q: np.ndarray) -> None:
        """Hard-set joint state. Used by IK probing, not by execution."""
        self.data.qpos[self._qadr] = np.asarray(q, dtype=float)[:7]
        self.data.ctrl[:7] = self.data.qpos[self._qadr]
        mujoco.mj_forward(self.model, self.data)

    def park(self, steps_per_wp: int = 8) -> None:
        """Retract the arm clear of the overhead camera, then settle."""
        from .kinematics import interpolate

        self.follow(interpolate(self.get_joints(), PARK_ARM, config.WAYPOINT_STEPS),
                    steps_per_wp)
        self.settle(200)

    def open_gripper(self) -> None:
        self.data.ctrl[7] = GRIPPER_OPEN

    def close_gripper(self) -> None:
        self.data.ctrl[7] = GRIPPER_CLOSED

    def gripper_pos(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, config.GRIPPER_SITE)
        return self.data.site_xpos[sid].copy()

    def body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"no body named {name!r}")
        return self.data.xpos[bid].copy()

    def object_positions(self) -> dict[str, np.ndarray]:
        names = [*config.OBJECTS, config.CONTAINER]
        return {n: self.body_pos(n) for n in names}

    # -- stepping ------------------------------------------------------------
    def start_recording(self, camera: str = config.SCENE_CAM, every: int = 25,
                        on_frame: Callable[[np.ndarray, int], None] | None = None) -> None:
        """Capture a frame every `every` physics steps until `stop_recording`.

        `on_frame(frame, index)` is invoked as each frame is taken, which lets a
        UI stream the motion live instead of waiting for the run to finish.
        """
        self._rec_frames, self._rec_every, self._rec_cam, self._rec_tick = [], every, camera, 0
        self._rec_cb = on_frame

    def stop_recording(self) -> list[np.ndarray]:
        frames, self._rec_frames, self._rec_every = self._rec_frames, [], 0
        self._rec_cb = None
        return frames

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self._rec_every:
                self._rec_tick += 1
                if self._rec_tick % self._rec_every == 0:
                    frame = self.render(self._rec_cam)
                    self._rec_frames.append(frame)
                    if self._rec_cb is not None:
                        self._rec_cb(frame, len(self._rec_frames) - 1)

    def settle(self, n: int = 100) -> None:
        """Step without changing the command, letting the controller converge."""
        self.step(n)

    def follow(self, path: np.ndarray, steps_per_wp: int = 8) -> None:
        """Execute a joint-space path of shape (N, 7)."""
        for q in path:
            self.set_joints(q)
            self.step(steps_per_wp)

    # -- rendering -----------------------------------------------------------
    def _gl_pool(self) -> ThreadPoolExecutor:
        if self._gl is None:
            self._gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-gl")
        return self._gl

    def _render_on_worker(self, camera: str, depth: bool) -> np.ndarray:
        """Runs only on the GL worker thread."""
        r = self._renderers.get(depth)
        if r is None:
            r = mujoco.Renderer(self.model, height=config.RENDER_H, width=config.RENDER_W)
            if depth:
                r.enable_depth_rendering()
            self._renderers[depth] = r
        r.update_scene(self.data, camera=camera)
        return r.render()

    def render(self, camera: str = config.TOP_CAM, depth: bool = False) -> np.ndarray:
        if not self._render_enabled:
            raise RuntimeError("SimEnv was constructed with render=False")
        return self._gl_pool().submit(self._render_on_worker, camera, depth).result()

    def close(self) -> None:
        """Tear down the GL contexts on the thread that owns them."""
        if self._gl is None:
            return

        def _close() -> None:
            while self._renderers:
                self._renderers.popitem()[1].close()

        self._gl.submit(_close).result()
        self._gl.shutdown(wait=True)
        self._gl = None


if __name__ == "__main__":
    env = SimEnv(render=False)
    print("gripper:", env.gripper_pos().round(4))
    for name, pos in env.object_positions().items():
        print(f"  {name:12s} {pos.round(4)}")
