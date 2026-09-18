"""Transcribe a spoken command and execute it.

    python scripts/m6_voice.py --mic              # record 5 s from the microphone
    python scripts/m6_voice.py --wav clip.wav     # use a recording
    python scripts/m6_voice.py --text "..."       # skip speech entirely

Prints both ASR transcripts and which one the router chose, then runs the plan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import speech
from src.executor import execute
from src.sim import SimEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="record from the microphone")
    ap.add_argument("--wav", help="transcribe this wav file instead")
    ap.add_argument("--text", help="skip speech and plan this text")
    ap.add_argument("--seconds", type=float, default=5.0, help="mic recording length")
    args = ap.parse_args()

    if not (args.mic or args.wav or args.text):
        ap.error("pass one of --mic, --wav or --text")

    env = SimEnv()
    transcript = None

    if args.text:
        utterance = args.text
        print(f"text input, ASR skipped: {utterance!r}")
    else:
        audio = speech.record(args.seconds) if args.mic else speech.load_wav(args.wav)
        print(f"audio level: rms {speech.rms(audio):.4f}")

        try:
            transcript = speech.route(audio)
        except speech.NoSpeechDetected as exc:
            print(f"\nno speech: {exc}")
            print("M6 FAIL (nothing was transcribed, so this is not a pipeline result)")
            return 2
        except speech.SpeechUnavailable as exc:
            print(f"ASR unavailable: {exc}")
            return 2

        print(f"\n{'model':<20}transcript")
        for model, text in transcript.candidates.items():
            mark = "  <- chosen" if model == transcript.source else ""
            print(f"{model:<20}{text}{mark}")
        print(f"\nlanguage {transcript.lang}   "
              f"tamil script {speech.tamil_ratio(transcript.text):.2f}   "
              f"asr {transcript.latency_s:.2f} s")

        utterance = speech.clean_transcript(transcript.text)
        print(f"cleaned: {utterance!r}")

    ep = execute(env, utterance, cleaned=utterance, transcript=transcript)
    print(f"\nplan ({ep.plan_source}): {ep.plan_json}")
    print("M6", "PASS" if ep.success else "FAIL")
    env.close()
    return 0 if ep.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
