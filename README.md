# VoiceArm

**Code-Switched Speech Grounding for Open-Vocabulary Robotic Manipulation**

A bilingual (Tamil / English / Tanglish) voice-controlled Franka Panda arm in
MuJoCo. Spoken instructions are transcribed, converted into a structured task
plan by a local language model, grounded to 3D object positions with
open-vocabulary detection, and executed through an inverse-kinematics
controller. Everything runs on-device on Apple Silicon — no cloud API.

> Simulation only. No sim-to-real transfer is claimed.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Environment, dependencies, Panda assets | ✅ done |
| **M1** | Scene, arm actuation, offscreen rendering | ✅ done |
| **M2** | Forward + inverse kinematics | ✅ done |
| M3 | Grasp primitives (pick & place) | ⬜ next |
| M4 | Open-vocabulary perception (OWL-ViT + depth) | ⬜ |
| M5 | Local LLM task planner | ⬜ |
| M6 | Dual-ASR Tamil/English speech router | ⬜ |
| M7 | Episode logging + Streamlit dashboard | ⬜ |

## Pipeline (target)

```
  mic ──► Whisper large-v3 (MLX) ──┐
                                   ├─► script-based router ──► cleaned utterance
          whisper-tamil-medium ────┘
                                            │
                                            ▼
                                   Qwen3-8B-4bit planner
                                            │  JSON task plan
                                            ▼
   topcam RGB+depth ──► OWL-ViT ──► 3D grounding ──► IK ──► MuJoCo execution
                                                              │
                                                              ▼
                                                   episode log ──► Streamlit
```

## Results so far

**M1 — arm actuation** (`out/m1_frames.png`)

| Pose | Commanded → reached (max joint error) | Gripper displacement |
|---|---|---|
| home | 0.0066 rad | — |
| swing_left | 0.0066 rad | 0.380 m |
| lift_up | 0.0043 rad | 0.459 m |

**M2 — inverse kinematics**, 10 random reachable targets, damped least squares:

| Metric | Value |
|---|---|
| Targets within 5 mm | **10 / 10** |
| Mean position error | **0.44 mm** |
| Max position error | 1.79 mm |
| Mean solve time | **0.1 ms** |

Reproduce with `./run.sh m1` and `./run.sh m2`.

## Setup

```bash
git clone <this repo> && cd VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo
./setup.sh
```

`setup.sh` creates `.venv`, installs dependencies, and downloads only the
`franka_emika_panda` directory from `mujoco_menagerie` (36 MB) rather than
cloning the full repository.

## Running

```bash
./run.sh m1      # three-pose actuation demo, writes out/m1_frames.png
./run.sh m2      # IK accuracy table over 10 random targets
./run.sh test    # pytest smoke suite
```

## Design notes

**The grasp site and wrist camera are injected with `MjSpec`.** The vendored
Panda model has no end-effector site, and MJCF `<include>` cannot add children
to a body defined inside the included file. `src/sim.py` loads the scene as a
spec, adds a `grasp_site` at the Panda TCP (0.1034 m along `+z` of the `hand`
frame) plus a wrist camera, then compiles. The vendored asset stays untouched.

**IK is damped least squares on the site Jacobian.** `mink` is used when it is
installed; the built-in solver is the default path and is what the numbers above
were measured with. Step size is clamped and joint limits are enforced each
iteration, so the solver stays stable near singularities.

**The overhead camera is rotated, not just raised.** `topcam` uses
`xyaxes="0 1 0 -1 0 0"` so the image's wide axis covers the table's wide axis
(world *y*). At a fixed height this fits the whole worktop in frame, which
matters for M4 detection.

**`SimEnv.park()` exists because the arm occludes the worktop.** At the home
pose the Panda sits directly under the overhead camera. `park()` retracts it
behind the base (gripper at x ≈ −0.33 m), found with a joint sweep constrained
to no new contacts, giving perception an unoccluded view.

## Layout

```
src/config.py       paths, model ids, tuning constants
src/sim.py          SimEnv — model build, actuation, rendering
src/kinematics.py   FK, damped-least-squares IK, joint-space interpolation
assets/scene.xml    table, three blocks, container, overhead camera
scripts/m*.py       one runnable acceptance demo per milestone
tests/test_smoke.py smoke suite
```

## Stack

| Layer | Choice |
|---|---|
| Physics | MuJoCo 3.12 |
| Robot | Franka Emika Panda (`mujoco_menagerie`) |
| IK | Damped least squares (`mink` optional) |
| Perception | OWL-ViT (`google/owlvit-base-patch32`) — M4 |
| Planner | `mlx-community/Qwen3-8B-4bit` — M5 |
| ASR | Whisper large-v3 (MLX) + `vasista22/whisper-tamil-medium` — M6 |
| UI | Streamlit — M7 |

## Model selection

`sarvam-m` (24 B) is the strongest open model for Tamil, but at 4-bit it needs
roughly 13 GB — it cannot co-reside with the detector and both ASR models on a
16 GB machine. `Qwen3-8B-4bit` (~4.7 GB) was chosen instead so the full stack
stays under ~8 GB resident. This is a hardware constraint, not a quality claim.

## License

Panda model © Franka Emika, redistributed from `mujoco_menagerie` under Apache
2.0 (see `assets/mujoco_menagerie/franka_emika_panda/LICENSE`).
