"""Gradio app: type or speak a command and watch the arm do it.

A web UI rather than a notebook cell because of the microphone -- sounddevice
needs a local input device and Colab has none, so without this the voice half
of the project is unreachable in the only demo someone else can run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, speech
from src.executor import execute
from src.sim import SimEnv

EXAMPLES = [
    ["put the red block in the bowl"],
    ["sivappu block-ah bowl-la vai"],
    ["pachai block-ah edu"],
    ["நீல கட்டையை கிண்ணத்தில் வை"],
]

_env: SimEnv | None = None


def get_env() -> SimEnv:
    global _env
    if _env is None:
        _env = SimEnv()
    return _env


def run_text(text: str, progress=gr.Progress()):
    if not (text or "").strip():
        return None, "", "Type something first."
    return _run(text.strip(), progress)


def run_voice(audio, progress=gr.Progress()):
    if audio is None:
        return None, "", "Record something first."

    rate, samples = audio
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples /= max(np.abs(samples).max(), 1e-6)
    if rate != speech.SAMPLE_RATE:
        import scipy.signal as sps
        samples = sps.resample(
            samples, int(len(samples) * speech.SAMPLE_RATE / rate)).astype(np.float32)

    progress(0.2, desc="Transcribing")
    try:
        transcript = speech.transcribe(samples)
    except speech.NoSpeechDetected as exc:
        return None, "", f"No speech: {exc}"

    return _run(transcript.text, progress, heard=transcript.text)


def _run(utterance: str, progress, heard: str | None = None):
    progress(0.5, desc="Planning and moving")
    result = execute(get_env(), utterance)

    frame = get_env().render(config.SCENE_CAM)
    plan = json.dumps(result.plan, indent=2, ensure_ascii=False)
    status = f"{'success' if result.success else 'failed'} · {result.duration_s:.1f} s"
    if result.detect_error_m == result.detect_error_m:
        status += f" · detector {result.detect_error_m * 100:.1f} cm off"
    if heard:
        status = f"heard: *{heard}*\n\n{status}"
    return frame, plan, status


def build() -> gr.Blocks:
    with gr.Blocks(title="VoiceArm") as demo:
        gr.Markdown(
            "# VoiceArm\n"
            "Type or say something in **Tamil, English, or both mixed**, and the "
            "arm does it. Simulation only.\n"
        )
        with gr.Row():
            with gr.Column():
                with gr.Tab("Type"):
                    text = gr.Textbox(label="Command",
                                      placeholder="sivappu block-ah bowl-la vai")
                    gr.Examples(EXAMPLES, inputs=text)
                    text_btn = gr.Button("Run", variant="primary")
                with gr.Tab("Speak"):
                    mic = gr.Audio(sources=["microphone"], type="numpy",
                                   label="Tamil, English, or mixed")
                    mic_btn = gr.Button("Transcribe and run", variant="primary")
            with gr.Column():
                view = gr.Image(label="Result", type="numpy", interactive=False)
                plan_box = gr.Code(label="Plan", language="json")
                status = gr.Markdown()

        outputs = [view, plan_box, status]
        text_btn.click(run_text, [text], outputs)
        mic_btn.click(run_voice, [mic], outputs)
    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    build().queue(max_size=8).launch(share=args.share, server_port=args.port)
