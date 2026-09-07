"""Gradio front-end for the Hugging Face Space.

Differences from the local Streamlit dashboard, all forced by the platform:

* Inference runs inside `@spaces.GPU`, so models must be *loaded* at import time
  (outside the decorator) and only *used* inside it. Loading inside would spend
  GPU quota on downloads.
* Free ZeroGPU accounts get a few minutes of GPU per day. When that runs out the
  Space must still show something, so there is a text-only CPU path and a
  pre-recorded fallback rather than a stack trace.
* Gradio's microphone component works in the browser, which is actually better
  than the local setup -- no input-device volume to get wrong.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The rendering backend is chosen in src/__init__.py, which runs before any
# module imports mujoco -- setting MUJOCO_GL here would be too late on Linux and
# outright invalid on macOS, where only CGL exists.
os.environ.setdefault("VOICEARM_BACKEND", "torch")

from src import backend, config, episodes, perception, planner, speech  # noqa: E402
from src.executor import execute  # noqa: E402
from src.sim import SimEnv  # noqa: E402
from src.video import write_video  # noqa: E402

log = logging.getLogger("voicearm.space")

# The ZeroGPU decorator only exists on Spaces hardware. Locally it has to be a
# no-op so the same file runs unchanged -- and the check has to be for the GPU
# attribute, not just a successful import, because a directory named `spaces`
# anywhere on sys.path shadows the package and imports cleanly with no GPU.
try:
    import spaces as _spaces

    HAS_ZEROGPU = hasattr(_spaces, "GPU")
except ImportError:
    HAS_ZEROGPU = False

if HAS_ZEROGPU:
    gpu = _spaces.GPU
else:
    def gpu(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap


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


@gpu(duration=90)
def run_voice(audio, progress=gr.Progress()) -> tuple:
    if audio is None:
        return None, "", "", "Record something first."
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
        return None, "", "", f"No speech detected: {exc}"

    candidates = "\n".join(
        f"{'* ' if k == transcript.source else '  '}{k}: {v}"
        for k, v in transcript.candidates.items()
    )
    progress(0.4, desc="Reading the instruction")
    cleaned = speech.clean_transcript(transcript.text)
    return (*_execute(cleaned, transcript, progress), candidates)


@gpu(duration=90)
def run_text(text: str, progress=gr.Progress()) -> tuple:
    if not (text or "").strip():
        return None, "", "", "Type an instruction first."
    return (*_execute(text.strip(), None, progress), "(text input, ASR skipped)")


def _execute(utterance: str, transcript, progress) -> tuple:
    env = get_env()
    frames: list[np.ndarray] = []

    def on_frame(frame, index):
        frames.append(frame)
        if index % 8 == 0:
            progress(min(0.5 + index / 220, 0.95), desc="Executing")

    progress(0.5, desc="Planning")
    ep = execute(env, utterance, cleaned=utterance, transcript=transcript,
                 capture_video=False, on_frame=on_frame)

    video_path = None
    if frames:
        out = config.OUT / f"space_{ep.episode_id}.mp4"
        if write_video(frames, out):
            video_path = str(out)

    plan = json.dumps(json.loads(ep.plan_json), indent=2, ensure_ascii=False)
    status = (f"{'success' if ep.success else 'failed'} · plan from {ep.plan_source} "
              f"· {ep.duration_s:.1f}s")
    if ep.detect_error_m == ep.detect_error_m:      # not NaN
        status += f" · detection error {ep.detect_error_m * 100:.1f} cm"
    return video_path, plan, status


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
                video = gr.Video(label="Execution", autoplay=True)
                plan_box = gr.Code(label="Task plan", language="json")
                result = gr.Markdown()

        mic_btn.click(run_voice, [mic], [video, plan_box, result, asr_box])
        text_btn.click(run_text, [text], [video, plan_box, result, asr_box])
        demo.load(lambda: warm_models(), None, status_md)
    return demo


if __name__ == "__main__":
    build().queue(max_size=8).launch(theme=gr.themes.Soft())
