# VoiceArm

**Code-Switched Speech Grounding for Open-Vocabulary Robotic Manipulation**

A bilingual (Tamil / English / Tanglish) voice-controlled Franka Panda arm in
MuJoCo. Spoken instructions are transcribed, converted into a structured task
plan by a local language model, grounded to 3D object positions with
open-vocabulary detection, and executed through an inverse-kinematics
controller. Everything runs on-device on Apple Silicon — no cloud API.

> Simulation only. No sim-to-real transfer is claimed.

![scene](docs/scene.png)

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Environment, dependencies, Panda assets | ✅ done |
| **M1** | Scene, arm actuation, offscreen rendering | ✅ done |
| **M2** | Forward + inverse kinematics | ✅ done |
| **M3** | Grasp primitives (pick & place) | ✅ done |
| **M4** | Open-vocabulary perception (OWLv2 + depth) | ✅ done |
| M5 | Local LLM task planner | ⬜ next |
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

**M3 — pick and place**, each block grasped from its ground-truth pose and
dropped in the container:

![pick and place](docs/m3_pickplace.png)

| Metric | Value |
|---|---|
| Blocks placed in the container | **3 / 3** |
| Lift height on grasp | 0.146 m (all three) |
| Mean final offset from container centre | **1.7 mm** |
| Grasp strategy | friction only — no weld constraint needed |

**M4 — open-vocabulary grounding**, four free-text queries against simulator
ground truth:

![detections](docs/m4_detections.png)

| Query | Camera | Score | Error |
|---|---|---|---|
| "a red cube" | topcam | 0.58 | 0.5 cm |
| "a green cube" | topcam | 0.56 | 0.2 cm |
| "a blue cube" | topcam | 0.65 | 0.5 cm |
| "a white bowl" | perceptcam | 0.21 | 1.5 cm |

**4 / 4 within 3 cm**, mean error **0.66 cm**.

Reproduce with `./run.sh m1` … `./run.sh m4`.

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
./run.sh m3      # pick and place all three blocks, writes out/m3_pickplace.mp4
./run.sh m4      # open-vocabulary detection + 3D grounding vs ground truth
./run.sh test    # pytest smoke suite
```

## Design notes

**The grasp site and wrist camera are injected with `MjSpec`.** The vendored
Panda model has no end-effector site, and MJCF `<include>` cannot add children
to a body defined inside the included file. `src/sim.py` loads the scene as a
spec, adds a `grasp_site` at the Panda TCP (0.1034 m along `+z` of the `hand`
frame) plus a wrist camera, then compiles. The vendored asset stays untouched.

**Grasping needed 6-DoF IK, not position-only.** The Panda's fingers slide along
the hand frame's *y* axis, so a top-down grasp has to constrain orientation as
well as position. `ik()` stacks the positional and rotational site Jacobians and
solves both together; `GRASP_DOWN_MAT` points the site *z* axis at the table and
aligns the finger-separation axis with world *y*.

**Transit waypoints use a looser tolerance than the grasp.** Near the edge of the
workspace the 2 mm grasp tolerance is unreachable for approach and lift poses,
which made `place()` fail on a target that was in fact fine to reach. Approach
and lift now solve to 6 mm; only the grasp waypoint holds 2 mm.

**Perception uses a camera cascade, not a per-object rule.** Straight down, a
bowl is only a white ring and OWLv2 scored nothing against any bowl-like query;
from an angle it scores 0.29. Cubes ground far more accurately from overhead
(0.2–0.5 cm vs 1.4–2.0 cm) because their top face is what the depth sample hits.
`locate()` therefore tries `topcam` first and falls back to the angled
`perceptcam`, taking the first confident detection. Nothing in that path knows
which objects need which view, so an unseen noun gets the same benefit.

**Depth lands on the top face, not the centre.** An overhead view only ever sees
an object's top surface, so the deprojected point sits half an object-height too
high. For anything resting on the table the centre is the midpoint between that
surface and the tabletop — which needs no prior knowledge of the object's size.

**The bowl is a ring of 16 boxes with quaternion rotations.** MuJoCo takes its
angle unit from the compiler, and the vendored `panda.xml` sets
`angle="radian"`; degree-valued `euler` attributes were silently reinterpreted
and produced a starburst instead of a bowl.

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
src/kinematics.py   FK, damped-least-squares IK (3- and 6-DoF), interpolation
src/grasp.py        top-down pick / place primitives from Cartesian waypoints
src/perception.py   RGB-D capture, OWLv2 detection, pinhole 3D grounding
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
| Perception | OWLv2 (`google/owlv2-base-patch16-ensemble`) |
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
