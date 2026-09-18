# VoiceArm

**A robot arm you can talk to in Tamil or English.**

I say something like "sivappu block-ah bowl-la vai" and a simulated Franka Panda
arm picks up the red block and puts it in the bowl. It works in Tamil, English,
or the two mixed together, which is how I actually talk.

Everything runs on my laptop. No API keys, no cloud.

> This is simulation only. I haven't tried it on a real arm.

![pick and place](docs/media/demo.gif)

*"sivappu block-ah bowl-la vai" -- spoken, took 4.9 seconds end to end.*

## How it works

```
  mic --> whisper-large-v3 -----+
                                +--> which model won? --> transcript
          whisper-tamil-medium -+                            |
                                                             v
                                                    Qwen3-4B makes a plan
                                                             |  JSON steps
                                                             v
  camera RGB+depth --> OWLv2 finds the object --> IK moves the arm
                                                             |
                                                             v
                                                  MuJoCo runs it --> CSV log
```

Two speech models run, and I pick whichever one did better on Tamil. The plan
comes back as JSON like `{"action": "pick", "target": "a red cube"}`. That
`"a red cube"` string goes straight to the object detector as a search query, so
I never had to write a list of objects anywhere.

## Try it

**Free GPU, one click:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo/blob/main/notebooks/voicearm_demo.ipynb)

The notebook clones this repo, installs everything, runs each stage, and starts
the web app with a public link. Around 30-60 seconds per command on a Colab T4.

**On your own machine:**

```bash
git clone https://github.com/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo.git
cd VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo
./setup.sh
```

```bash
./run.sh app     # web app on localhost:7860
./run.sh m3      # watch it pick and place
./run.sh test    # 29 tests
```

`setup.sh` makes a virtualenv and downloads just the Franka Panda from
`mujoco_menagerie` (36 MB) instead of the whole repo. On a Mac it uses MLX and
takes about 4.9 seconds per command; on anything else it uses PyTorch. I didn't
want to maintain two versions, so `src/backend.py` just checks which one is
available.

`./run.sh app --share` gives you a public `*.gradio.live` link that works while
the process is running.

## What I measured

| | |
|---|---|
| Inverse kinematics | 10/10 random targets within 5 mm (mean 0.44 mm) |
| Pick and place | 3/3 blocks landed in the bowl, 1.7 mm off centre |
| Finding objects | 12/12 within 3 cm across 3 shuffled layouts, mean 0.67 cm |
| Running commands | 8/8, English + Tanglish + Tamil script |
| Tamil transcription | 4.8% character error over 18 runs |
| Speed once warm | 7.3 seconds end to end |

Small numbers, and I say so where it matters. Full tables and the bugs I hit are
in [NOTES.md](NOTES.md) -- that file is more interesting than this one.

## Files

```
src/            the actual pipeline -- simulation, kinematics, grasping,
                the object detector, the planner, speech, logging
scripts/        one script per stage so I could test them separately
webapp/         the Gradio app (local, --share, and Colab all use this file)
notebooks/      the Colab notebook
tests/          29 tests
assets/         the MuJoCo scene
docs/media/     the GIF and screenshots
```

## Built with

MuJoCo · Franka Emika Panda · OWLv2 · Qwen3-4B · Whisper · MLX · PyTorch · Gradio

## License

Apache 2.0. The Panda model is Franka Emika's, from `mujoco_menagerie`.
