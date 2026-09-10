"""Gradio front-end, launched by the Colab notebook.

Why a web UI at all, when a notebook cell can call `execute()` in three lines:

* Gradio's microphone records in the browser, which is the whole reason this
  exists rather than a notebook cell: `sounddevice` needs a local input device
  and there is none on a Colab runtime. It also sidesteps the input-volume trap
  that made the first local microphone test record silence.
* Models load once at import, not per request.
* The handlers are generators, so the arm is visible while it moves rather than
  only in a video afterwards.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The rendering backend is chosen in src/__init__.py, which runs before any
# module imports mujoco -- setting MUJOCO_GL here would be too late on Linux and
# outright invalid on macOS, where only CGL exists.
#
# The model backend is deliberately NOT set here. An earlier version forced
# VOICEARM_BACKEND=torch, written when this file was destined for an x86 Space,
# and it survived the move: `./run.sh app` on Apple Silicon then took 27.5 s per
# command through torch instead of 4.9 s through MLX. src/backend.py already
# picks MLX where it exists and torch everywhere else, which is the right answer
# in both places.

from src import backend, config, episodes, perception, planner, speech  # noqa: E402
from src.executor import execute  # noqa: E402
from src.sim import SimEnv  # noqa: E402
from src.video import write_video  # noqa: E402

log = logging.getLogger("voicearm.webapp")

LIVE_EVERY = 2          # stream every other captured frame

EXAMPLES = [
    ["put the red block in the bowl"],
    ["sivappu block-ah bowl-la vai"],
    ["pachai block-ah edu"],
    ["நீல கட்டையை கிண்ணத்தில் வை"],
    ["சிவப்பு கட்டையை எடுத்து கிண்ணத்தில் வை"],
]

_env: SimEnv | None = None


def get_env() -> SimEnv:
    global _env
    if _env is None:
        _env = SimEnv()
    return _env


def warm_models() -> str:
    """Load every model once at startup so GPU time is spent on inference only."""
    t0 = time.perf_counter()
    planner._load()
    perception._load()
    return f"{backend.describe()} · models warm in {time.perf_counter() - t0:.0f}s"


def run_voice(audio, progress=gr.Progress()):
    """Transcribe, then stream the execution. A generator, so Gradio updates live."""
    if audio is None:
        yield None, None, "", "Record something first.", ""
        return

    rate, samples = audio
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples /= max(np.abs(samples).max(), 1e-6)

    if rate != speech.SAMPLE_RATE:
        import scipy.signal as sps

        samples = sps.resample(
            samples, int(len(samples) * speech.SAMPLE_RATE / rate)
        ).astype(np.float32)

    progress(0.2, desc="Transcribing")
    try:
        transcript = speech.route(samples)
    except speech.NoSpeechDetected as exc:
        yield None, None, "", f"No speech detected: {exc}", ""
        return

    candidates = "\n".join(
        f"{'* ' if k == transcript.source else '  '}{k}: {v}"
        for k, v in transcript.candidates.items()
    )
    progress(0.4, desc="Reading the instruction")
    cleaned = speech.clean_transcript(transcript.text)

    for frame, video, plan, status in _execute(cleaned, transcript):
        yield frame, video, plan, status, candidates


def run_text(text: str, progress=gr.Progress()):
    if not (text or "").strip():
        yield None, None, "", "Type an instruction first.", ""
        return
    for frame, video, plan, status in _execute(text.strip(), None):
        yield frame, video, plan, status, "(text input, ASR skipped)"


def _execute(utterance: str, transcript):
    """Run one instruction, yielding (frame, video, plan, status) as it goes.

    `execute()` is synchronous and hands frames to a callback, so it runs on a
    worker thread and the frames come back through a queue. That is safe because
    SimEnv already funnels all rendering onto a single GL thread -- the physics
    and the OpenGL context never move.
    """
    env = get_env()
    frames: list[np.ndarray] = []
    stream: queue.Queue = queue.Queue()
    outcome: dict = {}

    def on_frame(frame, index):
        frames.append(frame)
        if index % LIVE_EVERY == 0:          # ~10 fps to the browser
            stream.put(frame)

    def work():
        try:
            outcome["ep"] = execute(env, utterance, cleaned=utterance,
                                    transcript=transcript, capture_video=False,
                                    on_frame=on_frame)
        except Exception as exc:             # surfaced below, not swallowed
            outcome["error"] = exc
        finally:
            stream.put(None)

    worker = threading.Thread(target=work, name="voicearm-execute", daemon=True)
    worker.start()

    yield None, None, "", "Planning…"
    while True:
        frame = stream.get()
        if frame is None:
            break
        yield frame, None, "", "Executing…"
    worker.join()

    if "error" in outcome:
        raise gr.Error(f"{type(outcome['error']).__name__}: {outcome['error']}")

    ep = outcome["ep"]
    video_path = None
    if frames:
        out = config.OUT / f"webapp_{ep.episode_id}.mp4"
        if write_video(frames, out):
            video_path = str(out)

    plan = json.dumps(json.loads(ep.plan_json), indent=2, ensure_ascii=False)
    status = (f"{'success' if ep.success else 'failed'} · plan from {ep.plan_source} "
              f"· {ep.duration_s:.1f}s")
    if ep.detect_error_m == ep.detect_error_m:      # not NaN
        status += f" · detection error {ep.detect_error_m * 100:.1f} cm"

    yield frames[-1] if frames else None, video_path, plan, status


def build() -> gr.Blocks:
    with gr.Blocks(title="VoiceArm") as demo:
        gr.Markdown(
            "# VoiceArm\n"
            "### Code-switched speech grounding for open-vocabulary robotic manipulation\n"
            "Speak or type in **Tamil, English, or Tanglish**. A dual-ASR router "
            "transcribes it, a local Qwen3-4B turns it into a task plan, OWLv2 finds "
            "the named object in the camera image, and an IK controller executes the "
            "pick-and-place in MuJoCo.\n\n"
            "*Simulation only. No sim-to-real transfer is claimed.*"
        )
        status_md = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Speak"):
                    mic = gr.Audio(sources=["microphone"], type="numpy",
                                   label="Say it in Tamil, English, or Tanglish")
                    mic_btn = gr.Button("Transcribe and execute", variant="primary")
                with gr.Tab("Type"):
                    text = gr.Textbox(label="Instruction",
                                      placeholder="sivappu block-ah bowl-la vai")
                    gr.Examples(EXAMPLES, inputs=text)
                    text_btn = gr.Button("Execute", variant="primary")
                asr_box = gr.Textbox(label="ASR candidates (* = router's choice)",
                                     lines=3, interactive=False)
            with gr.Column(scale=3):
                # Live frames land here as the arm moves; the video appears when
                # the episode finishes, so the wait is not a blank panel.
                live = gr.Image(label="Live", type="numpy", height=380,
                                interactive=False)
                video = gr.Video(label="Replay", autoplay=True, loop=True)
                plan_box = gr.Code(label="Task plan", language="json")
                result = gr.Markdown()

        outputs = [live, video, plan_box, result, asr_box]
        mic_btn.click(run_voice, [mic], outputs)
        text_btn.click(run_text, [text], outputs)
        demo.load(lambda: warm_models(), None, status_md)
    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true",
                    help="expose a public gradio.live tunnel URL (expires in 72h)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    build().queue(max_size=8).launch(
        theme=gr.themes.Soft(), share=args.share, server_port=args.port,
    )
