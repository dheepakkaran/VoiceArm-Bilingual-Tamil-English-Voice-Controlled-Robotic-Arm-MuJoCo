---
title: VoiceArm
emoji: 🦾
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Tamil/English voice-controlled robot arm in MuJoCo
suggested_hardware: zero-a10g
---

# VoiceArm

**Code-switched speech grounding for open-vocabulary robotic manipulation.**

Speak or type an instruction in Tamil, English, or romanized Tanglish. A
dual-ASR router transcribes it, a local Qwen3-4B converts it into a JSON task
plan, OWLv2 grounds the named object in the camera image, and a
damped-least-squares IK controller executes the pick-and-place in MuJoCo.

Simulation only — no sim-to-real transfer is claimed.

## How it differs from the local version

The project is developed on Apple Silicon, where the models run through MLX.
Spaces is x86 Linux with an NVIDIA slice and MLX does not exist there, so
`src/backend.py` selects the model stack at import time:

| | local (Apple Silicon) | this Space |
|---|---|---|
| planner | `mlx-community/Qwen3-4B-4bit` | `Qwen/Qwen3-4B-Instruct-2507` |
| multilingual ASR | `mlx-community/whisper-large-v3-mlx` | `openai/whisper-large-v3-turbo` |
| Tamil ASR | `vasista22/whisper-tamil-medium` | same |
| detector | `google/owlv2-base-patch16-ensemble` | same |
| rendering | GLFW | `MUJOCO_GL=egl` |

Everything above the model-loading layer — the router, the planner prompt, IK,
grasp waypoints, 3D grounding, episode logging — is the same code.

## Quota

Free ZeroGPU accounts get a few minutes of GPU per day. When that is exhausted
the Space keeps working on CPU for text instructions, which is slower but does
not fail. That is deliberate: a demo link that shows an error is worse than a
slow one.

## Source

Full project, benchmarks, and the milestone scripts:
<https://github.com/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo>
