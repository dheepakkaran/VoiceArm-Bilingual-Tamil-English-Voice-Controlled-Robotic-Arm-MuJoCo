# VoiceArm

**A robot arm you can talk to in Tamil or English.**

I say something like "sivappu block-ah bowl-la vai" and a simulated Franka Panda
arm picks up the red block and puts it in the bowl. It works in Tamil, English,
or the two mixed together, which is how I actually talk.

Everything runs locally -- no API keys, no cloud.

> Simulation only. I haven't tried it on a real arm.

![pick and place](docs/media/demo.gif)

## How it works

```
  mic --> Whisper --> text --> Qwen2.5 makes a plan
                                     |  JSON steps
                                     v
  camera RGB+depth --> OWLv2 finds the object --> IK moves the arm --> MuJoCo
```

Four stages, wired together. The interesting part is the handoff: the planner
returns `{"action": "pick", "target": "a red cube"}`, and that `"a red cube"`
string goes straight to the object detector as a search query. So I never wrote
a list of objects anywhere -- if the planner says "a yellow duck", the detector
looks for a yellow duck.

## Try it

**Free GPU, one click:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo/blob/main/notebooks/voicearm_demo.ipynb)

**On your own machine:**

```bash
git clone https://github.com/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo.git
cd VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo
./setup.sh
```

```bash
./run.sh demo                    # three example commands
./run.sh demo --text "..."       # your own
./run.sh demo --mic              # say it out loud
./run.sh demo --wav clip.wav     # from a recording
./run.sh check                   # arm and grasping only, no models
./run.sh test                    # 12 tests
```

`setup.sh` makes a virtualenv and downloads just the Franka Panda from
`mujoco_menagerie` (36 MB) instead of the whole repo.

The microphone only works locally -- `sounddevice` needs a real input device and
Colab has none. In the notebook you type the command or pass a wav file, which
runs the same pipeline.

## What works

| | |
|---|---|
| Arm reaching a point | 10/10 random targets within 5 mm |
| Picking and placing | 3/3 blocks into the bowl |
| Finding an object from text | within 3 cm, typically 0.5 cm |
| Running commands | 7/8, across English, Tanglish, Tamil script |
| Time per command | 1.4-2.7 s warm |

The one failure is a Tamil pick-only command where the planner adds a "put it in
the bowl" step I didn't ask for. [NOTES.md](NOTES.md) has that and the other
bugs I hit.

## Files

```
src/            the pipeline -- simulation, kinematics, grasping,
                object detection, planning, speech
scripts/demo.py   run it from the terminal
notebooks/        the Colab notebook
tests/            12 tests
assets/           the MuJoCo scene
```

## Built with

Python · PyTorch · MuJoCo · Franka Emika Panda · OWLv2 · Qwen2.5 · Whisper

## License

Apache 2.0. The Panda model is Franka Emika's, from `mujoco_menagerie`.
