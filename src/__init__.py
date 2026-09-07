"""VoiceArm — bilingual voice-controlled robotic manipulation in MuJoCo.

MuJoCo reads MUJOCO_GL at import time and rejects values its platform does not
support, so the rendering backend has to be chosen here, before any submodule
imports mujoco. macOS only has CGL and errors on anything else; headless Linux
needs an explicit choice, and EGL is the hardware-accelerated one. Set
MUJOCO_GL yourself to override -- CPU-only Linux hosts want `osmesa`.
"""
from __future__ import annotations

import os
import platform

if "MUJOCO_GL" not in os.environ and platform.system() == "Linux":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
