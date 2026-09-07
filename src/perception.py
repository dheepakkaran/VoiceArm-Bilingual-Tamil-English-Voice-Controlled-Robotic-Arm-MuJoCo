"""Open-vocabulary perception: RGB-D capture, OWLv2 detection, 3D grounding.

Detection queries are free-form text that comes straight from the task planner,
so no fixed object vocabulary is baked in anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from . import backend, config
from .memory import release_caches
from .sim import SimEnv

log = logging.getLogger(__name__)

TABLE_TOP_Z = 0.40
DEPTH_CROP = 0.5        # use the middle 50% of a box when sampling depth
MIN_SCORE = 0.15

_model = None
_processor = None
_device = None


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 in pixels
    xyz: np.ndarray | None = None             # world position, filled by ground()
    camera: str = ""                          # view the detection came from


# --- camera model -----------------------------------------------------------
def intrinsics(model: mujoco.MjModel, camera: str,
               width: int = config.RENDER_W, height: int = config.RENDER_H
               ) -> tuple[float, float, float, float]:
    """Pinhole (fx, fy, cx, cy) derived from the MuJoCo camera's vertical fov."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f"no camera named {camera!r}")
    fovy = np.deg2rad(model.cam_fovy[cid])
    fy = (height / 2.0) / np.tan(fovy / 2.0)
    return fy, fy, width / 2.0, height / 2.0     # square pixels


def camera_pose(env: SimEnv, camera: str) -> tuple[np.ndarray, np.ndarray]:
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return env.data.cam_xpos[cid].copy(), env.data.cam_xmat[cid].reshape(3, 3).copy()


def capture(env: SimEnv, camera: str = config.TOP_CAM) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 HxWx3, depth float HxW in metres)."""
    return env.render(camera), env.render(camera, depth=True)


def deproject(u: float, v: float, depth: float, env: SimEnv,
              camera: str = config.TOP_CAM) -> np.ndarray:
    """Pixel + depth -> world point.

    MuJoCo cameras look down their own -z, image rows increase downward, so the
    camera-frame point is (x, -y, -depth) before the world transform.
    """
    fx, fy, cx, cy = intrinsics(env.model, camera)
    cam_pos, cam_mat = camera_pose(env, camera)
    p_cam = np.array([(u - cx) * depth / fx,
                      -(v - cy) * depth / fy,
                      -depth])
    return cam_pos + cam_mat @ p_cam


def _box_depth(depth: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """Median depth over the central crop of a box, ignoring invalid samples."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * DEPTH_CROP / 2, (y1 - y0) * DEPTH_CROP / 2
    xs = slice(max(int(cx - hw), 0), max(int(cx + hw) + 1, int(cx - hw) + 1))
    ys = slice(max(int(cy - hh), 0), max(int(cy + hh) + 1, int(cy - hh) + 1))
    patch = depth[ys, xs]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        raise ValueError(f"no valid depth inside {bbox}")
    return float(np.median(patch))


# --- detector ---------------------------------------------------------------
def _load():
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    _device = backend.torch_device()
    log.info("loading %s on %s", config.DETECTOR, _device)
    _processor = AutoProcessor.from_pretrained(config.DETECTOR)
    _model = (AutoModelForZeroShotObjectDetection
              .from_pretrained(config.DETECTOR).to(_device).eval())
    release_caches()
    return _model, _processor, _device


def detect(rgb: np.ndarray, queries: list[str],
           threshold: float = MIN_SCORE) -> list[Detection]:
    """Open-vocabulary detection. `queries` are arbitrary text descriptions."""
    import torch
    from PIL import Image

    model, processor, device = _load()
    image = Image.fromarray(rgb)
    inputs = processor(text=[queries], images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    release_caches()

    target = torch.tensor([[image.height, image.width]])
    post = getattr(processor, "post_process_grounded_object_detection", None) \
        or processor.post_process_object_detection
    res = post(outputs=outputs, threshold=threshold, target_sizes=target)[0]

    dets = [
        Detection(label=queries[int(lbl)], score=float(sc), bbox=tuple(float(v) for v in bx))
        for sc, lbl, bx in zip(res["scores"], res["labels"], res["boxes"])
    ]
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def best_match(dets: list[Detection], query: str) -> Detection | None:
    for d in dets:
        if d.label == query:
            return d
    return None


def ground(det: Detection, depth: np.ndarray, env: SimEnv,
           camera: str = config.TOP_CAM, resting: bool = True) -> np.ndarray:
    """Attach a world position to a detection.

    The overhead view only ever sees an object's top face, so the deprojected
    point sits half an object-height above its centre. For objects resting on
    the table the centre is the midpoint between that surface and the tabletop,
    which needs no prior knowledge of the object's size.
    """
    x0, y0, x1, y1 = det.bbox
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    surface = deproject(u, v, _box_depth(depth, det.bbox), env, camera)
    if resting:
        surface[2] = (surface[2] + TABLE_TOP_Z) / 2.0
    det.xyz = surface
    return surface


def locate(env: SimEnv, query: str, cameras: tuple[str, ...] = config.PERCEPTION_CAMERAS,
           resting: bool = True) -> tuple[np.ndarray, Detection] | tuple[None, None]:
    """Capture, detect `query`, and return its world position.

    Views are tried in order and the first confident detection wins. This is a
    confidence cascade rather than a per-object rule: nothing here knows which
    objects need which camera, so an unseen noun benefits from the fallback the
    same way a known one does.
    """
    for camera in cameras:
        rgb, depth = capture(env, camera)
        dets = detect(rgb, [query])
        if dets:
            det = dets[0]
            det.camera = camera
            return ground(det, depth, env, camera, resting), det
        log.debug("%r not found in %s", query, camera)
    log.warning("no detection for %r in any of %s", query, cameras)
    return None, None
