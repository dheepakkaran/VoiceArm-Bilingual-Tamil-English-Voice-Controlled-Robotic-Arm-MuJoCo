# Notes

What I measured, and the bugs I hit getting there. The [README](README.md) is
the short version.

## Numbers

**Moving the arm.** Three joint targets, checking the arm actually reaches them:

| Pose | Joint error | How far the gripper moved |
|---|---|---|
| home | 0.0066 rad | — |
| swing_left | 0.0066 rad | 0.380 m |
| lift_up | 0.0043 rad | 0.459 m |

**Inverse kinematics.** 10 random reachable points, seeded so it reruns the same:

| | |
|---|---|
| Within 5 mm | **10 / 10** |
| Mean error | **0.44 mm** |
| Worst | 1.79 mm |
| Solve time | **0.1 ms** |

**Pick and place**, using the simulator's own object positions (no camera yet):

![pick and place](docs/media/m3_pickplace.png)

| | |
|---|---|
| Blocks that landed in the bowl | **3 / 3** |
| Lift height | 0.146 m, all three |
| Distance from bowl centre | **1.7 mm** |

Plain friction grasp. I expected to need a weld constraint to fake it and
didn't.

**Finding objects from text.** I ask OWLv2 for "a red cube" and compare what it
finds against the simulator. Three trials, blocks shuffled between each -- one
fixed layout would only prove it works on that layout:

| | |
|---|---|
| Within 3 cm | **12 / 12** |
| Mean error | **0.67 cm** |
| Worst | 1.44 cm |

![detections](docs/media/m4_detections.png)

The bowl is the worst every single trial, and also the lowest confidence (about
0.20, where the blocks get 0.6). It's the one object where the depth reading
hits the inside floor instead of a top surface, so my height correction doesn't
apply and I only score its *x* and *y*.

**Turning sentences into plans.** Qwen3-4B, running locally:

| Language | Commands | Ran |
|---|---|---|
| English | 2 | 2 |
| Tanglish | 3 | 3 |
| Tamil script | 3 | 3 |
| **Total** | **8** | **8** |

Median 1.17 s per plan once warm. All eight came from the model; the regex
fallback never had to kick in.

These eight are **not in the prompt.** My first version accidentally reused four
of the prompt's own examples as test cases, so half the score was just the model
repeating what I'd shown it. I swapped them for phrasings it hadn't seen --
"drop the red cube into the bowl", "neela cube-ah bowl-ukkulla vai", a Tamil
pick command. Still 8/8, which is what I thought I was measuring the first time.

**Which speech model is better at Tamil.** 6 sentences, spoken by the macOS
Tamil voice and saved as wav files so room noise isn't in the way. 3 runs each,
because Whisper doesn't give the same answer twice:

| model | mean error | median | worst |
|---|---|---|---|
| whisper-large-v3 | 16.7% | **0.0%** | **100.0%** |
| whisper-tamil-medium | **4.8%** | **0.0%** | 14.3% |
| what I ship (routed) | **4.8%** | **0.0%** | 14.3% |

The "worst" column is why I bothered with two models. The big model is usually
perfect and then occasionally outputs the whole sentence in Devanagari -- 100%
wrong. A mean of 16.7% makes that look like general sloppiness and a median of
0% hides it completely. The Tamil model never fails that badly; its small errors
are just spelling conventions (`நீலக்` instead of `நீல`).

Across those 18 runs, Whisper thought the audio was Tamil 15 times and Hindi 3
times. My router picked the Tamil model all 18.

These are synthesised voices, one speaker. Real speech is harder, and Tanglish
especially -- I haven't tested that properly.

**Everything together**, over 9 logged runs (8 typed, 1 spoken):

| | |
|---|---|
| Succeeded | **9 / 9** |
| Mean detection error | **0.37 cm** (worst 0.50 cm) |
| Mean time per command | 5.6 s |
| Median plan time | 1.86 s |
| Speech, warm | 13-20 s |

Rerun any of it with `./run.sh m1` through `./run.sh m5`.

## Bugs I hit

**Silence got turned into a task.** The first time I tried the microphone it
recorded nothing -- my mac's input volume was at 27 out of 100. Whisper
correctly gave me an empty string. Then the planner made up an instruction from
that empty string, the arm executed it, and the script printed PASS. So it
didn't fail, it succeeded at something I never said.

Now `route()` raises `NoSpeechDetected` if the clip is silent or the transcript
is empty, and `execute()` won't plan from an empty string. My first fix was a
volume cutoff at 0.01 rms, which rejected speech that was perfectly audible --
Whisper is better at deciding whether something is speech than a number I picked
is, so the cutoff now only catches actual silence.

**Whisper guessed the wrong language, at random.** Running the same audio file
several times, it said `ta` most of the time but sometimes `hi` or `kn`, and
transcribed Tamil into Devanagari or Kannada script. My router was checking "is
this Tamil script?" to decide whether to use the Tamil model -- which meant it
skipped the Tamil model exactly when the other one had gone wrong. Now any
Indian language guess triggers it.

**A top-down grasp needs orientation, not just position.** The Panda's fingers
slide along the hand's *y* axis, so pointing the gripper at the right spot
isn't enough -- it also has to be facing down with the fingers across the block.
`ik()` stacks the position and rotation Jacobians and solves both at once.

**The depth camera sees the top of the block, not the middle.** Looking straight
down, the depth reading is the top face, so my 3D point was half a block too
high. Since the blocks sit on a table I know the height of, the centre is just
the midpoint between the top face and the tabletop -- I don't need to know how
big the object is.

**Rendering from two threads hung the app, silently.** MuJoCo's renderer owns an
OpenGL context tied to the thread that made it, and the web framework calls in
from a different thread each request. On macOS, both reusing a context across
threads and making a second one just deadlock instead of raising an error, so
the app froze on its first command with no traceback. `SimEnv` now sends every
render to one dedicated worker thread and waits for the result.

**The arm was standing in front of the camera.** At the home pose the Panda sits
right under the overhead camera, so the detector couldn't see the blocks.
`park()` folds it behind the base (gripper at x ≈ -0.33 m). I found that pose by
sweeping joint angles and keeping the ones with no new contacts.

## Why I picked these models

**Qwen3-4B instead of 8B.** I tried both on the same 8 commands:

| planner | plans correct | generation | memory |
|---|---|---|---|
| **Qwen3-4B-4bit** | **8/8** | **0.87 s** | **2.26 GB** |
| Qwen3-8B-4bit | 8/8 | 1.55 s | 4.61 GB |

Same accuracy, half the memory, nearly twice as fast. Easy call.

**I wanted `sarvam-m` (24B) and couldn't fit it.** It's the best open model for
Tamil, but at 4-bit it needs about 13 GB and I already have the detector and two
speech models loaded. That's a memory problem, not a quality judgement -- on a
bigger machine I'd test it, since the planner does have to read Tanglish.

**Both speech models, with a router.** Table above. The Tamil model is better at
Tamil but only speaks Tamil, so neither one is safe on its own.

## Speed

Same command, run four times in a row, all models loaded:

| run | speech | plan | detect | total |
|---|---|---|---|---|
| cold | 16.63 s | 4.33 s | 4.04 s | 25.00 s |
| 2nd | 8.15 s | 4.87 s | 1.11 s | 14.13 s |
| 3rd | 6.44 s | 1.73 s | 0.87 s | 9.04 s |
| 4th | 4.94 s | 1.42 s | 0.94 s | **7.30 s** |

MLX memory 5.35 GB, peak 5.93 GB.

**It was slow because of memory, not compute.** With the 8B planner this loop
took 45 s and got *worse* every iteration, up to 63 s. I assumed the model was
just too big. It wasn't -- MLX and PyTorch each hold onto an allocator pool
after a forward pass, and with four models loaded those pools were enough to
push the machine into swap. One `mx.clear_cache()` call took a single plan from
37.5 s to 1.7 s. Same model, 22x faster.

`src/memory.py` clears both caches between stages. I first tried clearing after
every single call and it got *worse* -- 60 s -- because the next operation had to
reallocate from scratch. The cache is there for a reason. Only the handoff
between models is worth clearing.

**Benchmarking it wrong made it look fine.** If I ran all the speech calls, then
all the planner calls, then all the detector calls, everything looked fast.
Running one full command at a time -- speech, plan, detect -- is 5-10x slower,
because each switch pages the last model out. That's the real workload, so
that's what the table shows.

**Some of this is my laptop.** `vm.swapusage` showed 20.1 GB of 21.5 GB swap
already in use before I started, from ordinary apps. On a machine with free swap
these would be better. I'm reporting what I measured.

## Running on two machines

MLX only exists on Apple Silicon, and Colab is Linux with an NVIDIA GPU. Rather
than keep two copies of the project, `src/backend.py` checks what's available at
import and picks the model ids:

| | my Mac (MLX) | Colab (PyTorch) |
|---|---|---|
| planner | `mlx-community/Qwen3-4B-4bit` | `Qwen/Qwen3-4B-Instruct-2507` |
| multilingual speech | `mlx-community/whisper-large-v3-mlx` | `openai/whisper-large-v3-turbo` |
| Tamil speech | `vasista22/whisper-tamil-medium` | same |
| detector | `google/owlv2-base-patch16-ensemble` | same |
| rendering | CGL, the macOS default | `MUJOCO_GL=egl` |

Everything above that -- the router, the prompt, IK, grasping, 3D grounding,
logging -- is the same code both ways.

To test the Colab path on a Mac without deploying:

```bash
VOICEARM_BACKEND=torch ./run.sh m5
```

Same command end to end: **7.0 s** on MLX, **27.5 s** through PyTorch on MPS.
Colab's T4 sits somewhere in between.

## Why there's a web app

A notebook cell could call `execute()` in three lines, so the Gradio app needs a
reason: **the microphone.** `sounddevice` needs a local input device and Colab
doesn't have one, so without a browser UI the voice half of a voice-controlled
arm is unreachable in the only demo someone else can run. It also avoids the
input-volume problem that made my first mic test record silence.

One file covers all three cases: `localhost` while I'm working, `--share` for a
link, and the notebook's launch cell on Colab.
