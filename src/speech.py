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

# Whisper's language ID is unstable on short Tamil clips and regularly reports a
# neighbouring Indic language, transcribing Tamil speech into Kannada, Malayalam
# or Telugu script. A Tamil-script test alone therefore misses the specialist
# exactly when it is most needed, so any Indic guess also triggers it.
INDIC_LANGS = frozenset({"ta", "kn", "ml", "te", "hi", "bn", "mr",
                         "gu", "pa", "or", "si", "ne", "sa"})

_tamil_pipe = None


# Only digital silence / a muted or unpermitted mic is gated on level. Quiet
# speech still transcribes fine, and Whisper judges that far better than a fixed
# amplitude cutoff does -- the empty-transcript check below is the real guard.
SILENCE_RMS = 0.002
QUIET_RMS = 0.012


class SpeechUnavailable(RuntimeError):
    """Raised when no ASR backend could be loaded."""


class NoSpeechDetected(RuntimeError):
    """Raised when the clip is silent or every backend returned an empty string.

    This has to stop the pipeline rather than pass an empty string downstream:
    the planner LLM will happily invent a plausible instruction from nothing,
    which looks like a successful run but is pure hallucination.
    """


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


def trim_silence(audio: np.ndarray, frame_ms: int = 10, pad_ms: int = 200,
                 rel_threshold: float = 0.08) -> np.ndarray:
    """Cut leading and trailing silence with energy-based endpointing.

    Whisper hallucinates fluent continuations over trailing silence, so a fixed
    recording window that ends well after the speaker stops reliably produces
    invented text appended to an otherwise correct transcript. Trimming to the
    speech region removes the padding that triggers it.
    """
    n = int(SAMPLE_RATE * frame_ms / 1000)
    if audio.size < n * 3:
        return audio
    frames = audio[: audio.size // n * n].reshape(-1, n)
    energy = np.sqrt(np.mean(np.square(frames), axis=1))
    if energy.max() <= 0:
        return audio

    voiced = np.flatnonzero(energy > rel_threshold * energy.max())
    if voiced.size == 0:
        return audio

    pad = int(pad_ms / frame_ms)
    start = max(int(voiced[0]) - pad, 0) * n
    end = min(int(voiced[-1]) + pad + 1, frames.shape[0]) * n
    return audio[start:end]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def route(audio: np.ndarray) -> Transcript:
    """Transcribe, dispatching to the Tamil specialist when the script warrants."""
    t0 = time.perf_counter()
    audio = trim_silence(audio)
    level = rms(audio)
    if level < SILENCE_RMS:
        raise NoSpeechDetected(
            f"clip is digitally silent (rms {level:.4f}); the mic is muted or "
            "input permission is denied"
        )
    if level < QUIET_RMS:
        log.warning("quiet clip (rms %.4f) -- transcription may be unreliable", level)
    try:
        text, lang = transcribe_multilingual(audio)
    except Exception as exc:
        raise SpeechUnavailable(f"multilingual ASR failed: {exc}") from exc

    candidates = {"multilingual": text}
    source = "multilingual"

    base_ratio = tamil_ratio(text)
    if base_ratio > TAMIL_SCRIPT_THRESHOLD or lang in INDIC_LANGS:
        try:
            ta_text = transcribe_tamil(audio)
            candidates["tamil_specialist"] = ta_text
            # Prefer the specialist whenever it produced Tamil script. Measured
            # on 4 synthesised Tamil sentences the specialist averaged 9.7% CER
            # against 85.8% for the multilingual model, which fails two ways on
            # Tamil: it loops and emits the sentence twice, and it sometimes
            # decodes into a neighbouring script. Both failures still look like
            # confident Tamil-script output, so the choice cannot be made on
            # script alone -- it has to default to the specialist.
            if ta_text and tamil_ratio(ta_text) > TAMIL_SCRIPT_THRESHOLD:
                text, source = ta_text, "tamil_specialist"
        except Exception as exc:
            log.warning("Tamil specialist unavailable (%s), keeping multilingual output",
                        type(exc).__name__)

    if not text.strip():
        raise NoSpeechDetected(
            f"every ASR backend returned an empty transcript (audio rms {level:.4f})"
        )

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

    if not text.strip():
        raise NoSpeechDetected("refusing to clean an empty transcript")

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
