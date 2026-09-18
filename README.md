# VoiceArm

**Telling a robot arm what to do in the language I actually speak.**

![pick and place](docs/media/demo.gif)

> Simulation only. I have not tested this on a real arm.

---

## 1. The problem

I speak Tamil and English in the same sentence. Not as a translation exercise --
"sivappu block-ah bowl-la vai" is one thought, and switching mid-sentence is
normal where I grew up. Almost every voice interface I use assumes I will pick a
language and stay in it.

For robots specifically, the research that lets you talk to one is
English-only. [SayCan](https://arxiv.org/pdf/2204.01691) pairs a language model
with a robot's affordances and takes English instructions. So does most work
after it. If you only speak Tamil, or you speak the mix, none of it reaches you.

**Why I picked this.** I wanted a project where the interesting part was
wiring real models together under a real constraint, not training something. And
I wanted the constraint to be one I could actually feel: if the pipeline doesn't
understand how I talk, I notice immediately. A benchmark score would not have
told me that.

## 2. What I chose to build

**In scope**

| | |
|---|---|
| Input | Tamil script, English, or the two mixed (romanized Tamil) |
| Commands | pick up an object; pick it up and put it in the bowl |
| Scene | a table, three coloured blocks, one bowl, a Franka Panda arm |
| Objects | named in free text, never from a fixed list in the code |
| Runs on | my laptop, or a free Colab GPU |

**Out of scope**

| | why |
|---|---|
| A real robot arm | I don't have one, and sim-to-real is its own project |
| Training or fine-tuning anything | every model here is off the shelf; the work is the integration |
| Multi-step or conditional instructions | "if the red one is gone, use the blue one" needs planning I didn't build |
| Objects outside the scene | the detector would find them, the grasp controller assumes small cubes |
| Speaker-independent accuracy claims | I tested with synthesised speech and my own voice, not a speaker set |

## 3. Where the ideas come from

Nothing here is a new method. It is four published ideas connected together, and
knowing which is which matters:

| Idea | From | What I took |
|---|---|---|
| A language model as the task planner | [SayCan](https://arxiv.org/pdf/2204.01691) (Ahn et al., 2022) | Ask an LLM for a plan in a fixed vocabulary of robot actions, then execute it. SayCan used GPT-3; I use a local 1.5B model. |
| Finding objects by text description | [OWLv2](https://arxiv.org/pdf/2306.09683) (Minderer et al., NeurIPS 2023) | Query a detector with a phrase instead of a class index. SayCan used ViLD for the same job. |
| Multilingual speech recognition | [Whisper](https://arxiv.org/abs/2212.04356) (Radford et al., 2022) | One model that transcribes Tamil without me training anything. |
| Damped least squares inverse kinematics | Wampler (1986); [Buss (2004)](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf) | Invert the Jacobian with a damping term so the solver stays stable near singularities. Textbook, and I wrote it out rather than calling a library. |
| The robot model and physics | [MuJoCo](https://mujoco.org) (Todorov et al., 2012) and `mujoco_menagerie` | The Franka Panda, unmodified. |

**What is mine:** the wiring, and the decisions it forced. Which model handles
Tamil, what the planner is allowed to output, how a 2D detection becomes a 3D
grasp target, and what to do when a stage fails.

## 4. The solution

Four modules. Each one solves a problem the previous one creates.

```
  mic --> Whisper --> text --> Qwen2.5 plans --> OWLv2 locates --> IK moves
```

### Speech → text

| | |
|---|---|
| Problem | I want to speak Tamil, English, or a mix, without choosing first |
| Solution | `whisper-large-v3-turbo`, no language hint, let it decide |
| Result | Tamil script and English both transcribe. Mixed speech is the weak case |

### Text → plan

| | |
|---|---|
| Problem | "sivappu block-ah bowl-la vai" is not a robot command |
| Solution | Qwen2.5-1.5B with a prompt that fixes the output to a JSON action list, plus a romanized-Tamil glossary. A regex parser as fallback if the model returns nothing valid |
| Result | 7 of 8 held-out commands planned correctly |

### Plan → 3D position

| | |
|---|---|
| Problem | The plan says `"a red cube"`. The arm needs *x, y, z* |
| Solution | Pass that exact string to OWLv2 as a query, take the box, read the depth buffer inside it, unproject through the camera's pinhole model |
| Result | Within 3 cm of the simulator's own position, typically 0.5 cm |

### Position → motion

| | |
|---|---|
| Problem | A gripper pose is 6 numbers; the arm has 7 joints and limits |
| Solution | Damped least squares on the site Jacobian, solving position and orientation together, then waypoints for approach, grasp, lift, place |
| Result | 10 of 10 random targets within 5 mm; 3 of 3 blocks land in the bowl |

The handoff between modules two and three is the part I would point at. The
planner emits free text and the detector consumes free text, so there is no
object list anywhere in the code. Ask for a yellow duck and it looks for a
yellow duck.

## 5. How I built it

In order, with what convinced me each step worked.

| # | Step | How I knew it worked |
|---|---|---|
| 1 | Load the Panda into a MuJoCo scene with a table, blocks and a bowl | It compiles and the blocks settle on the table instead of falling through |
| 2 | Drive the arm to hardcoded joint angles | Rendered three poses; the gripper moved 0.38 m and 0.46 m between them |
| 3 | Forward kinematics for the gripper | Site position matched where the render showed the hand |
| 4 | Inverse kinematics, position only | 10 random targets, all within 5 mm measured by FK, not by the solver's own tolerance |
| 5 | Extend IK to orientation | Gripper points down; the finger axis lines up across the block |
| 6 | Grasp waypoints: approach, descend, close, lift | The block's *z* rises 0.146 m and stays up |
| 7 | Place waypoints into the bowl | All three blocks end within 2 mm of the bowl centre |
| 8 | OWLv2 on the camera image, queried with text | Boxes land on the right blocks; scores around 0.6 |
| 9 | Unproject the box to a world position | Compared against the simulator's own coordinates -- 0.2 to 0.5 cm |
| 10 | LLM planner with a fixed JSON schema | Valid JSON on every call; a held-out command set to check it isn't echoing its prompt |
| 11 | Whisper in front of the planner | Spoke a Tamil command and the arm did it |

Steps 4, 9 and 10 each needed a rewrite after I checked them properly. Those are
in [NOTES.md](NOTES.md).

## 6. Result: accuracy

Every number below is checked against the simulator's own state, which is the
one oracle I have -- the detector's guess is compared to where MuJoCo actually
put the object, not to a label I wrote. Reproduce the first two rows with
`./run.sh check` and the rest with `./run.sh demo`.

| | |
|---|---|
| IK, 10 random reachable targets | **10 / 10 within 5 mm** (mean 0.82 mm, worst 1.82 mm) |
| Pick and place, each block | **3 / 3 landed in the bowl** |
| Object position from text query | **within 3 cm**, observed 0.2-0.5 cm |
| Commands executed, held out from the prompt | **7 / 8** |

The eight commands span English, romanized Tanglish, and Tamil script. They are
deliberately *not* the examples in the planner prompt -- an earlier version
reused four of them, so half the score was the model repeating what I had shown
it.

**The one failure:** `சிவப்புப் பொருளை எடு` ("pick up the red thing"). The
planner gets the colour right and then adds a step to put it in the bowl, which
I did not ask for. A 1.5B model wants to finish the task.

## 7. Result: speed

Same command, run repeatedly, on an M5 MacBook with 16 GB:

| | |
|---|---|
| First command after start | 10 - 12 s |
| After that | **1.3 - 2.9 s** |

Ranges across repeated runs, not a single best case. The first command pays for
loading three models. After that, a pick-only command is about 1.3 s and a
pick-and-place about 2.7 s, since the second one plans two steps and runs twice
as much motion.

This number was 569 s before I found the memory bug in
[NOTES.md](NOTES.md#it-was-slow-because-of-memory-not-compute). It was not
compute.

## 8. Result: running it

| | |
|---|---|
| Models loaded at once | 3 -- Whisper 1.6 GB, Qwen2.5-1.5B 3.0 GB, OWLv2 1.5 GB |
| Machine it was built on | M5 MacBook, 16 GB, no discrete GPU |
| Also runs on | a free Colab T4, from the notebook below |
| Dependencies | 10 |
| Code | 1,581 lines of Python |
| External services | none -- no API keys anywhere |

**Free GPU, one click:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo/blob/main/notebooks/voicearm_demo.ipynb)

**Locally:**

```bash
git clone https://github.com/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo.git
cd VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo
./setup.sh
```

```bash
./run.sh demo                    # three example commands
./run.sh demo --text "..."       # your own
./run.sh demo --mic              # say it out loud
./run.sh check                   # arm and grasping only, no models
./run.sh test                    # 12 tests
```

`setup.sh` builds a virtualenv and fetches only the Franka Panda from
`mujoco_menagerie` (36 MB). The microphone needs a real input device, so in
Colab you type the command or pass a wav file -- same pipeline either way.

## 9. What broke

Short version; the full account is in [NOTES.md](NOTES.md).

| What happened | What it actually was |
|---|---|
| A run printed PASS having executed a command I never gave | My mic volume was at 27/100. Whisper returned an empty string and the planner invented an instruction from it |
| The gripper closed above the block every time | Depth from overhead reads the top face, so my 3D point was half a block too high |
| The app froze on its first command, no error | MuJoCo's GL context is thread-bound and macOS deadlocks instead of raising |
| One command took over nine minutes | Not model size -- PyTorch's allocator pools pushed a 16 GB machine into swap |
| The detector saw no blocks | The arm was parked directly under the camera |
| 8/8 on the command test | Four of my eight test cases were copied from the planner's own prompt |

The last one is the one I would want to be asked about. The score was real and
the test was not measuring what I thought.

## 10. What I would do next

In the order I would actually do them:

1. **Fix the over-planning bug properly.** The planner adds a place step to
   pick-only commands. A prompt line fixed 2 of 3 cases and I stopped there
   rather than keep tuning against my own test set. The right fix is probably
   constrained decoding, so the model cannot emit a step the grammar disallows.
2. **Test with real speakers.** Everything I measured used synthesised Tamil or
   my own voice. Accuracy on other speakers, and on genuine code-switching, is
   unknown and I would assume it is worse.
3. **Try the bigger Tamil model.** `sarvam-m` is 24B and the best open model for
   Tamil; at 4-bit it needs about 13 GB and does not fit alongside the other
   three. On a 24 GB GPU I would measure whether it actually plans better.
4. **More than three cubes.** The grasp controller assumes a small box it can
   pinch. Anything else needs grasp pose estimation, not waypoints.
5. **A real arm.** Everything above is simulation. The gap is not small and I
   would not claim otherwise.

## Built with

Python · PyTorch · MuJoCo · Franka Emika Panda · OWLv2 · Qwen2.5 · Whisper

## License

Apache 2.0. The Panda model is Franka Emika's, redistributed from
`mujoco_menagerie`.
