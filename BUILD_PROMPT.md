# VoiceArm — Full Build Prompt

Copy everything below the line into a fresh Claude Code session opened in
`~/Desktop/1/voicearm`. It is self-contained.

---

## ROLE

You are building a complete, working project end-to-end in one session. Do not
stop to ask design questions — every decision is already made below. Only stop
if a command fails in a way you cannot work around, and then report exactly what
failed with the error output.

Work milestone by milestone (M1 → M7). After each milestone, RUN it and prove it
works before moving on. Do not write all files first and test at the end.

## PROJECT

**VoiceArm** — a bilingual (Tamil + English) voice-controlled robot arm running
in MuJoCo simulation.

The user speaks or types a command in Tamil, English, or Tanglish
(e.g. "sivappu block-ah bowl-la vai" / "put the red block in the bowl").
The system transcribes it, converts it to a structured task plan with the Claude
API, locates the named object with open-vocabulary vision, solves inverse
kinematics, and executes a pick-and-place with a simulated Franka Panda arm.
Every run is recorded as an episode and shown on a Streamlit dashboard.

## HARD ENVIRONMENT FACTS (already verified — do not re-check)

- macOS on Apple **M5**, arm64, 16 GB RAM, ~320 GB free disk
- `python3` = 3.10.1 at `/usr/local/bin/python3`, universal2, **running arm64**
- `git` present. **No CUDA** — use Apple **MPS** for torch, **MLX** for Whisper
- Homebrew, ffmpeg, cmake may or may not be installed — check and report if missing;
  do NOT try to install Homebrew yourself (it needs the user's password).
  If ffmpeg is missing, use the sounddevice-numpy audio path which does not need it.
- Node is off-PATH and Docker belongs to another user account — do not use either.
- Create and use a virtualenv at `.venv` inside the project. Never install into
  system python.

## TECH STACK — PINNED, DO NOT SUBSTITUTE

| Layer | Choice |
|---|---|
| Physics sim | `mujoco` (python bindings) |
| Robot model | `mujoco_menagerie` → `franka_emika_panda` |
| IK | `mink` if it installs cleanly; else hand-written damped least squares |
| Perception | `transformers` → `google/owlvit-base-patch32` on MPS |
| Planner LLM | `mlx-lm` + **`mlx-community/Qwen3-8B-4bit`** (local, ~4.7 GB) |
| ASR (multilingual) | `mlx-whisper` + **`mlx-community/whisper-large-v3-mlx`** (~1.6 GB) |
| ASR (Tamil specialist) | `transformers` + **`vasista22/whisper-tamil-medium`** on MPS (fp16) |
| Mic capture | `sounddevice` |
| UI | `streamlit` |
| Episode storage | LeRobot-**compatible** parquet (see M7 — do NOT pip install lerobot) |

## SETUP (M0)

```bash
cd ~/Desktop/1/voicearm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install mujoco numpy scipy opencv-python torch torchvision transformers \
            streamlit sounddevice pandas pyarrow pillow
pip install mlx-lm mlx-whisper || echo "MLX_FAILED"
pip install mink || echo "MINK_FAILED — use DLS fallback"
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
```

Write a `requirements.txt` reflecting what actually installed.
**No cloud API is used anywhere in this project.** Every model runs locally on
Apple Silicon. Total resident memory across all four models must stay under
~8 GB so the machine does not swap:

| Model | Approx RAM |
|---|---|
| MuJoCo scene | 0.5 GB |
| OWL-ViT base | 0.6 GB |
| whisper-large-v3-mlx | 1.6 GB |
| whisper-tamil-medium (fp16, lazy) | 1.5 GB |
| Qwen3-8B-4bit | 4.7 GB |

Load `dtype=float16` for the Tamil specialist and unload it if RSS approaches
12 GB. `whisper-tamil-large-v2` is more accurate but ~3 GB — only substitute it
if you verify total RSS stays under 12 GB.

Load the LLM **lazily** (first use) and keep a single module-level singleton for
each model — never reload per request. Do NOT use `sarvam-m` (24 B): at 4-bit it
needs ~13 GB and will not co-exist with the other models on 16 GB.

## REPO STRUCTURE

```
voicearm/
├── README.md
├── requirements.txt
├── run.sh                      # convenience launcher
├── assets/
│   ├── mujoco_menagerie/       # cloned
│   └── scene.xml               # your scene: panda + table + blocks + bowl
├── src/
│   ├── __init__.py
│   ├── sim.py                  # M1  scene load, step loop, render
│   ├── kinematics.py           # M2  FK + IK
│   ├── grasp.py                # M3  pick & place primitives
│   ├── perception.py           # M4  camera → OWL-ViT → 3D coords
│   ├── planner.py              # M5  NL → task plan JSON
│   ├── speech.py               # M6  mic → Whisper → text
│   ├── episodes.py             # M7  episode recording
│   └── config.py               # constants, paths, model ids
├── scripts/
│   ├── m1_move.py ... m6_voice.py   # one runnable demo per milestone
├── app.py                      # M7  Streamlit dashboard
└── tests/test_smoke.py
```

---

# MILESTONES

## M1 — Scene + arm moves
Build `assets/scene.xml`: include the Panda from menagerie, a table plane, a
**red cube**, a **blue cube**, a **green cube** (4 cm), and a **bowl** (use a
cylinder/mesh primitive), plus a fixed overhead camera named `topcam` and a
wrist camera named `wristcam`. Add reasonable lighting.

`src/sim.py` — a `SimEnv` class: `reset()`, `step(n)`, `render(cam, depth=False)`,
`set_joints(q)`, `get_joints()`, `open_gripper()`, `close_gripper()`.
Use `mujoco.Renderer` for offscreen RGB and depth.

**Acceptance:** `python scripts/m1_move.py` drives the arm through 3 hardcoded
joint configurations and saves `out/m1_frames.png` (a 3-panel image). Assert the
gripper site actually moved between frames.

## M2 — Kinematics
`src/kinematics.py`:
- `fk(model, data, site_name) -> (pos, rotmat)` using `mj_forward` + site xpos.
- `ik(model, data, site_name, target_pos, target_quat=None, iters=200) -> q`
  Use `mink` if available. Otherwise implement **damped least squares**:
  compute the site Jacobian with `mujoco.mj_jacSite`, then
  `dq = J.T @ inv(J @ J.T + λ²I) @ err` with `λ=0.05`, clamp `|dq|`, respect
  joint limits, iterate until position error < 2 mm.

**Acceptance:** `python scripts/m2_ik.py` — for 10 random reachable targets,
solve IK, apply, and verify FK position error < 5 mm. Print a pass/fail table.

## M3 — Grasp primitives
`src/grasp.py`: `pick(env, xyz)` and `place(env, xyz)` built from waypoints —
approach 12 cm above → descend → close gripper → lift → move → descend →
open → retreat. Interpolate between waypoints in joint space over ~60 sim steps
each so motion is smooth. Weld/friction tuning is allowed to make grasps stick.

**Timebox this milestone to 45 minutes of physics tuning.** If a friction-based
grasp is still unreliable after that, stop tuning and switch to the deterministic
approach: on gripper close, if the target body is within 3 cm of the gripper
site, create a `mjOPTION`-level equality **weld** between the gripper body and
the cube (`mujoco.mj_setConstraint` / an `<equality><weld>` element toggled via
`data.eq_active`), and release it on gripper open. Document this clearly in the
README as a simplification. Do NOT spend hours on contact parameters — the
grasp is not the point of this project, the language pipeline is.

**Acceptance:** `python scripts/m3_pick.py` picks the red cube by its **known
ground-truth position** and drops it in the bowl. Verify by checking the cube's
final z and xy distance to the bowl centre. Save `out/m3_pickplace.mp4` (or a
frame grid PNG if video encoding is unavailable).

## M4 — Perception
`src/perception.py`:
- `capture(env, cam) -> rgb, depth`
- `detect(rgb, queries: list[str]) -> [{label, score, bbox}]` using OWL-ViT on
  MPS (fall back to CPU). Queries are free-form text like `"a red cube"`.
- `deproject(bbox, depth, cam_intrinsics, cam_pose) -> xyz_world` — take the
  median depth inside the middle 50 % of the box, unproject with the pinhole
  model, transform to world frame. Derive intrinsics from the MuJoCo camera
  fovy and the render resolution.

**Acceptance:** `python scripts/m4_detect.py` detects all three cubes and the
bowl from `topcam` and prints estimated vs ground-truth world positions.
Estimated error must be < 3 cm. Save an annotated `out/m4_detections.png`.

## M5 — Planner (local LLM, no cloud API)
`src/planner.py`:
- `_get_llm()` — lazy singleton loading `mlx-community/Qwen3-8B-4bit` via
  `mlx_lm.load`. Cache it at module level. Log load time once.
- `plan(utterance: str) -> list[dict]` — build a chat prompt with
  `tokenizer.apply_chat_template`. System prompt must: state that input may be
  Tamil script, English, or romanized Tanglish; describe the scene (red / blue /
  green cubes, one bowl, a table); and demand **JSON only, no prose**, matching
  `[{"action":"pick","target":"<free-text object description>"},
    {"action":"place","target":"<free-text object description>"}]`.
  Allowed actions: `pick`, `place`, `move_to`, `say`.
  Qwen3 supports a thinking mode — **disable it** (append `/no_think` or set
  `enable_thinking=False` in the chat template) so output is clean JSON.
  Generate with `mlx_lm.generate`, temperature 0.2, max_tokens 300.
  Strip any ```json fences, parse, validate the schema, retry once with a
  "return only valid JSON" nudge on failure.
- `plan_fallback(utterance)` — deterministic keyword/regex parser used when the
  LLM is unavailable or both parse attempts fail. Must cover Tamil colour words
  (சிவப்பு/sivappu/red, நீலம்/neelam/blue, பச்சை/pachai/green), the container
  (கிண்ணம்/bowl), and verbs (எடு/edu/pick/take, வை/vai/put/place).
- The `target` string from the plan is passed **verbatim** to OWL-ViT as the
  detection query — that is the point of open-vocabulary perception. Do not map
  it through a hardcoded object dictionary.

**Acceptance:** `python scripts/m5_plan.py` runs 8 utterances (mixed Tamil
script, English, Tanglish) end-to-end through the sim and prints a per-command
success table plus LLM latency. At least 6/8 must fully execute.

## M6 — Tamil voice (dual-ASR language router)
`src/speech.py`:
- `record(seconds=5) -> np.ndarray` via sounddevice, 16 kHz mono float32.
- `transcribe_multilingual(audio) -> (text, lang)` — `mlx_whisper.transcribe`
  with `path_or_hf_repo="mlx-community/whisper-large-v3-mlx"`, language
  auto-detect. Returns the detected language code too.
- `transcribe_tamil(audio) -> text` — `vasista22/whisper-tamil-medium` through
  `transformers.pipeline("automatic-speech-recognition", device="mps")`, forced
  to Tamil. Load lazily; this model is **Tamil-only** and will produce garbage on
  English audio, so it must never be called on its own.
- `route(audio) -> {"text", "source", "lang", "candidates"}` — the router:
  1. Always run `transcribe_multilingual` first.
  2. Compute the fraction of Tamil-script characters (Unicode block U+0B80–U+0BFF)
     in the result.
  3. If that fraction > 0.6 **or** the detected language is `ta`, also run
     `transcribe_tamil` and prefer its output.
  4. Otherwise keep the multilingual output (this is the Tanglish / English path).
  Record both candidates in the return value — the dashboard shows them side by
  side, and the episode log stores them.
- `clean_transcript(text) -> text` — pass the chosen transcript through the
  **local Qwen3 planner LLM** from M5 with an instruction saying the text is
  likely code-switched Tanglish from a Tamil speaker and asking for one cleaned
  intent sentence. This step is required: neither ASR model handles
  code-switching well, and the router cannot fix it.
- If `mlx-whisper` failed to install, `route` raises `SpeechUnavailable` and the
  UI degrades to text input only. If only the Tamil specialist fails to load,
  log a warning and continue with the multilingual model alone.

**Acceptance:** `python scripts/m6_voice.py --wav <file>` transcribes a Tamil
audio file, prints both ASR candidates and which one the router chose, and
executes the command. Also support a `--text` flag that skips ASR entirely. If
no Tamil audio file exists, generate one with `say -v <available voice>` and note
in the output that it is synthetic speech, not a real recording.

## M7 — Episodes + Dashboard
`src/episodes.py`: after every command, append a row to
`data/episodes/episodes.parquet` with columns
`episode_id, timestamp, raw_utterance, asr_multilingual, asr_tamil, asr_source,
cleaned_utterance, detected_language, plan_json, plan_source, target_object,
detected_xyz, gt_xyz, detect_error_m, success, duration_s, num_sim_steps,
llm_latency_s, asr_latency_s`.
`asr_source` is `multilingual`|`tamil_specialist`; `plan_source` is `llm`|`fallback`. Also write per-episode frames under
`data/episodes/<id>/` in a LeRobot-style layout (`observation.image`,
`observation.state`, `action` columns in a per-episode parquet). Document in the
README that this is LeRobot-*compatible*, not produced by the lerobot package.

`app.py` — Streamlit with:
1. Text input + a "🎤 Record 5s" button (disabled if speech unavailable)
2. Live render of the last executed episode (frame strip or video)
3. The parsed plan JSON, plus both ASR candidates side by side with the router's choice highlighted
4. Detection overlay image
5. Metrics row: total episodes, success rate, mean detection error, mean duration,
   mean LLM latency, and how often the Tamil specialist ASR won over the
   multilingual model
6. A table of all past episodes, filterable by language and success

**Acceptance:** `streamlit run app.py` starts, a typed Tamil command executes and
appears in the table, and the metrics update.

---

## QUALITY BAR

- Every module has a `if __name__ == "__main__":` smoke path.
- `tests/test_smoke.py` covers: scene loads, IK converges on one target,
  planner fallback parses one Tamil sentence, episode parquet round-trips.
- No bare `except:`. Log with the `logging` module, not prints, inside `src/`.
- Constants (paths, model ids, thresholds) live only in `src/config.py`.
- Type hints on public functions. Short docstrings. No over-commenting.

## README.md — required contents

Architecture diagram (ASCII), the full pipeline, setup commands, how to run each
milestone, a results table (success rate over the 8 test utterances, mean
detection error, mean episode duration, ASR router split, mean LLM latency), a
model-selection section explaining why each local model was chosen and what was
rejected, known limitations (Tanglish ASR
accuracy, sim-only, no sim-to-real transfer, Qwen3-8B chosen over the stronger
Tamil-native sarvam-m purely because of the 16 GB memory ceiling), and the tech stack table.
Be accurate — do not claim results you did not measure.

## WHEN YOU FINISH

Report: which milestones passed, which acceptance checks you actually ran with
their real output, anything that fell back (mink → DLS, MLX → text-only,
LLM unavailable → rule parser, Tamil ASR unavailable → multilingual only), and
the measured numbers that went into the README results table — including how
often the ASR router preferred the Tamil specialist.
