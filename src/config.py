"""Central configuration: paths, model ids, and tuning constants."""
from __future__ import annotations

import logging
import os
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
PERCEPT_CAM = "perceptcam"
# locate() tries these in order and takes the first confident detection.
PERCEPTION_CAMERAS = (TOP_CAM, PERCEPT_CAM)
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
LLM_MODEL = "mlx-community/Qwen3-4B-4bit"
ASR_MULTILINGUAL = "mlx-community/whisper-large-v3-mlx"
ASR_TAMIL = "vasista22/whisper-tamil-medium"
DETECTOR = "google/owlv2-base-patch16-ensemble"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# The Hub clients emit one INFO line per HTTP request and probe a dozen optional
# config files per model, which buries our own output under hundreds of lines.
for _noisy in ("httpx", "httpcore", "urllib3", "filelock",
               "huggingface_hub", "huggingface_hub.utils._http"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
