"""Bilingual speech front-end with a script-based dual-ASR router.

Whisper large-v3 handles English and code-switched Tanglish well but degrades on
sustained Tamil; `vasista22/whisper-tamil-medium` is far better on Tamil but is
Tamil-only and produces nonsense on English audio. Neither is a safe default on
its own, so the router runs the multilingual model first and re-runs the Tamil
specialist only when the transcript is actually Tamil script.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import config

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
TAMIL_BLOCK = re.compile(r"[஀-௿]")
TAMIL_SCRIPT_THRESHOLD = 0.6

_tamil_pipe = None


class SpeechUnavailable(RuntimeError):
    """Raised when no ASR backend could be loaded."""


@dataclass
class Transcript:
    text: str
    source: str                       # "multilingual" | "tamil_specialist"
    lang: str
    latency_s: float
    candidates: dict[str, str] = field(default_factory=dict)


def tamil_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Tamil.

    Combining vowel signs live in the Tamil block but are not alphabetic, so
    they are excluded from both sides -- counting them in the numerator only
    pushes the ratio above 1.0.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    tamil = sum(1 for c in letters if TAMIL_BLOCK.match(c))
    return tamil / len(letters)


# --- capture ----------------------------------------------------------------
def record(seconds: float = 5.0) -> np.ndarray:
    """Record mono float32 audio from the default input device."""
    import sounddevice as sd

    log.info("recording %.1f s", seconds)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def load_wav(path: str) -> np.ndarray:
    """Read a wav file as mono float32 at 16 kHz without needing ffmpeg."""
    import wave

    import scipy.signal as sps

    with wave.open(path, "rb") as w:
        rate, frames, width, channels = (w.getframerate(), w.readframes(w.getnframes()),
                                         w.getsampwidth(), w.getnchannels())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = sps.resample(audio, int(len(audio) * SAMPLE_RATE / rate)).astype(np.float32)
    return audio


# --- backends ---------------------------------------------------------------
def transcribe_multilingual(audio: np.ndarray) -> tuple[str, str]:
    import mlx_whisper

    res = mlx_whisper.transcribe(audio, path_or_hf_repo=config.ASR_MULTILINGUAL)
    return res["text"].strip(), res.get("language", "unknown")


def transcribe_tamil(audio: np.ndarray) -> str:
    """Tamil-only specialist. Never call this without checking the script first."""
    global _tamil_pipe
    if _tamil_pipe is None:
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info("loading %s on %s", config.ASR_TAMIL, device)
        _tamil_pipe = pipeline("automatic-speech-recognition", model=config.ASR_TAMIL,
                               device=device, torch_dtype=torch.float16)
        _tamil_pipe.model.config.forced_decoder_ids = (
            _tamil_pipe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")
        )
    return _tamil_pipe(audio.copy())["text"].strip()


def route(audio: np.ndarray) -> Transcript:
    """Transcribe, dispatching to the Tamil specialist when the script warrants."""
    t0 = time.perf_counter()
    try:
        text, lang = transcribe_multilingual(audio)
    except Exception as exc:
        raise SpeechUnavailable(f"multilingual ASR failed: {exc}") from exc

    candidates = {"multilingual": text}
    source = "multilingual"

    if tamil_ratio(text) > TAMIL_SCRIPT_THRESHOLD or lang == "ta":
        try:
            ta_text = transcribe_tamil(audio)
            candidates["tamil_specialist"] = ta_text
            if ta_text:
                text, source = ta_text, "tamil_specialist"
        except Exception as exc:
            log.warning("Tamil specialist unavailable (%s), keeping multilingual output",
                        type(exc).__name__)

    return Transcript(text=text, source=source, lang=lang,
                      latency_s=time.perf_counter() - t0, candidates=candidates)


def clean_transcript(text: str) -> str:
    """Normalise a code-switched transcript into one clear intent sentence.

    Neither ASR model handles Tamil-English code-switching well and the router
    cannot repair it -- it only picks between two imperfect transcripts. The
    planner LLM is far better at reading Tanglish, so it gets a pass at the raw
    text before planning.
    """
    from .planner import _generate

    try:
        out = _generate(
            text,
            "\nThe user text above is a speech transcript, likely code-switched "
            "Tamil-English from a Tamil speaker, and may contain recognition "
            "errors. Instead of planning, reply with one corrected English "
            "sentence stating what the user wants. No JSON, no explanation.",
        )
        cleaned = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        return cleaned.splitlines()[0].strip() if cleaned else text
    except Exception as exc:
        log.warning("transcript cleanup unavailable (%s), using raw text",
                    type(exc).__name__)
        return text
