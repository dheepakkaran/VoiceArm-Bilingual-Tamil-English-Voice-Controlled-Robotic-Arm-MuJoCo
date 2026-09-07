"""Central configuration: paths, model ids, and tuning constants."""
from __future__ import annotations

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MENAGERIE = ASSETS / "mujoco_menagerie"
PANDA_DIR = MENAGERIE / "franka_emika_panda"
SCENE_XML = ASSETS / "scene.xml"
OUT = ROOT / "out"
DATA = ROOT / "data"
EPISODES = DATA / "episodes"

for _d in (OUT, EPISODES):
    _d.mkdir(parents=True, exist_ok=True)

# --- simulation -------------------------------------------------------------
GRIPPER_SITE = "grasp_site"
TOP_CAM = "topcam"
WRIST_CAM = "wristcam"
SCENE_CAM = "scenecam"
RENDER_W, RENDER_H = 640, 480

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
OBJECTS = ["red_block", "blue_block", "green_block"]
CONTAINER = "bowl"

# --- inverse kinematics -----------------------------------------------------
IK_DAMPING = 0.05          # lambda in damped least squares
IK_MAX_ITERS = 200
IK_POS_TOL = 0.002         # 2 mm
IK_MAX_STEP = 0.30         # rad, per-iteration clamp on |dq|

# --- grasping ---------------------------------------------------------------
APPROACH_HEIGHT = 0.12     # m above target before descending
WAYPOINT_STEPS = 60
GRASP_SNAP_DIST = 0.03     # m, weld fallback threshold

# --- models (later milestones) ---------------------------------------------
LLM_MODEL = "mlx-community/Qwen3-8B-4bit"
ASR_MULTILINGUAL = "mlx-community/whisper-large-v3-mlx"
ASR_TAMIL = "vasista22/whisper-tamil-medium"
DETECTOR = "google/owlvit-base-patch32"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
