# Notes

The bugs I hit and what I did about them. The [README](README.md) is the short
version.

## Silence got turned into a task

The first time I tried the microphone it recorded nothing -- my mac's input
volume was at 27 out of 100. Whisper correctly returned an empty string. Then
the planner made up an instruction from that empty string, the arm executed it,
and the script printed PASS.

So nothing failed. It succeeded at something I never said, which took me a while
to even notice.

`speech.transcribe()` now raises `NoSpeechDetected` if the clip is silent or
Whisper returns nothing, and `execute()` refuses to plan from an empty string.
My first fix was a volume cutoff at 0.01 rms, which rejected speech that was
perfectly audible -- Whisper is better at deciding what counts as speech than a
number I picked, so the cutoff only catches actual silence now.

## A top-down grasp needs orientation, not just position

The Panda's fingers slide along the hand's *y* axis. Pointing the gripper at the
right coordinates isn't enough -- it also has to be facing down, with the
fingers across the block rather than along it. `ik()` stacks the position and
rotation Jacobians and solves both at once.

## The depth camera sees the top of the block, not the middle

Looking straight down, the depth reading is the top face, so my 3D point was
half a block too high and the gripper closed above the cube. The blocks sit on a
table whose height I know, so the centre is just the midpoint between the top
face and the tabletop. No need to know how big the object is.

## Rendering from two threads hung the app with no error

MuJoCo's renderer owns an OpenGL context tied to the thread that created it, and
Gradio calls in from a different worker thread each request. On macOS, both
reusing a context across threads and creating a second one deadlock instead of
raising -- so the app just froze on its first command, no traceback. `SimEnv`
sends every render to one dedicated thread and waits for the result.

## The arm was standing in front of the camera

At the home pose the Panda sits right under the overhead camera, so the detector
couldn't see the blocks it was being asked about. `park()` folds it behind the
base. I found that pose by sweeping joint angles and keeping ones that added no
new contacts.

## It was slow because of memory, not compute

Running the whole thing took over nine minutes per command. I assumed the models
were too big for the machine.

They were, but not the way I thought. PyTorch keeps an allocator pool alive after
a forward pass, and with three models loaded those pools were enough to push a
16 GB machine into swap. Clearing them between stages -- `src/memory.py`, about
20 lines -- brought it to under 3 seconds.

I also tried clearing after *every* call, which made it worse, because the next
operation had to reallocate from scratch. Only the handoff between models is
worth clearing.

The other half of the fix was the model itself. Qwen3-4B in bfloat16 is about
8 GB. Qwen2.5-1.5B is 3 GB and plans 7 of my 8 test commands correctly, so I use
that. A bigger model that swaps is slower than a smaller one that fits.

## The planner over-plans on pick-only commands

Ask a small model to "pick up the green block" and it helpfully adds a step to
put it in the bowl. Adding one line to the prompt -- only add a place step if the
user said where to put it -- fixed 2 of the 3 cases. The one that still fails is
`சிவப்புப் பொருளை எடு`, where "பொருள்" means thing rather than block. I left it
failing rather than keep tuning the prompt against my own test set.

## My test set was leaking

My first version of the command test reused four of the planner prompt's own
examples as test cases, so half the score was the model repeating what I'd shown
it. I swapped them for phrasings it hadn't seen -- "drop the red cube into the
bowl", "neela cube-ah bowl-ukkulla vai", a Tamil pick command.

That's also how I found the over-planning bug above: it was there all along, but
the leaked examples were hiding it.
