"""Paths, model names, and tuning constants."""
from __future__ import annotations

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PANDA_DIR = ASSETS / "mujoco_menagerie" / "franka_emika_panda"
SCENE_XML = ASSETS / "scene.xml"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# --- scene ------------------------------------------------------------------
GRIPPER_SITE = "grasp_site"
TOP_CAM = "topcam"
PERCEPT_CAM = "perceptcam"
SCENE_CAM = "scenecam"      # angled view, used for the demo video
PERCEPTION_CAMERAS = (TOP_CAM, PERCEPT_CAM)
RENDER_W, RENDER_H = 640, 480

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
OBJECTS = ["red_block", "green_block", "blue_block"]
CONTAINER = "bowl"

# --- inverse kinematics -----------------------------------------------------
IK_DAMPING = 0.05
IK_MAX_ITERS = 200
IK_POS_TOL = 0.002
IK_MAX_STEP = 0.30

# --- grasping ---------------------------------------------------------------
WAYPOINT_STEPS = 60

# --- models -----------------------------------------------------------------
ASR_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DETECTOR = "google/owlv2-base-patch16-ensemble"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
