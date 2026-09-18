"""Record audio and transcribe it with Whisper."""
from __future__ import annotations

import logging
import wave
from dataclasses import dataclass

import numpy as np
import scipy.signal as sps

from . import config
from .memory import release_caches

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SILENCE_RMS = 0.002

_pipe = None


class NoSpeechDetected(RuntimeError):
    """The clip was silent, or Whisper returned nothing.

    This has to stop the pipeline. An empty transcript passed on to the planner
    does not fail -- the model invents a plausible instruction from nothing and
    the arm runs it, which looks exactly like success.
    """


@dataclass
class Transcript:
    text: str
    language: str


def record(seconds: float = 5.0) -> np.ndarray:
    import sounddevice as sd

    log.info("recording %.1f s", seconds)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def load_wav(path: str) -> np.ndarray:
    """Read a wav as mono float32 at 16 kHz, without needing ffmpeg."""
    with wave.open(path, "rb") as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
        width, channels = w.getsampwidth(), w.getnchannels()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = sps.resample(audio, int(len(audio) * SAMPLE_RATE / rate)).astype(np.float32)
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        log.info("loading %s on %s", config.ASR_MODEL, device)
        _pipe = pipeline("automatic-speech-recognition", model=config.ASR_MODEL,
                         device=device,
                         dtype=torch.float32 if device == "cpu" else torch.float16)
    return _pipe


def transcribe(audio: np.ndarray) -> Transcript:
    level = rms(audio)
    if level < SILENCE_RMS:
        raise NoSpeechDetected(
            f"clip is silent (rms {level:.4f}); the mic is muted or permission "
            "is denied"
        )

    release_caches()
    result = _load()(audio.copy(), return_language=True,
                     generate_kwargs={"task": "transcribe"})
    text = result["text"].strip()
    if not text:
        raise NoSpeechDetected(f"Whisper returned nothing (rms {level:.4f})")

    chunks = result.get("chunks") or []
    language = str((chunks[0].get("language") if chunks else None) or "unknown")
    return Transcript(text=text, language=language)
