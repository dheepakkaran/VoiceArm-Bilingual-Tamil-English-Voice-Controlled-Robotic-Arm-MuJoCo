"""M6 acceptance: transcribe an utterance and execute it.

    python scripts/m6_voice.py --wav sample.wav      # dual-ASR router
    python scripts/m6_voice.py --mic                 # record 5 s live
    python scripts/m6_voice.py --text "..."          # skip ASR entirely

With no arguments a short Tamil clip is synthesised with macOS `say` so the
router has something to run on. Synthetic speech is not a substitute for a real
recording and is reported as such.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import speech
from src.executor import execute
from src.sim import SimEnv

TAMIL_SAMPLE = "சிவப்பு கட்டையை கிண்ணத்தில் வை"


def tamil_voice() -> str | None:
    """Find an installed macOS voice that can speak Tamil."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "ta_IN" in line:
            return line.split()[0]
    return None


def synth_wav(text: str, voice: str, path: Path) -> bool:
    aiff = path.with_suffix(".aiff")
    try:
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return path.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text")
    ap.add_argument("--mic", action="store_true")
    args = ap.parse_args()

    env = SimEnv()
    transcript = None
    synthetic = False

    if args.text:
        utterance = args.text
        print(f"text input (ASR skipped): {utterance!r}")
    else:
        wav = args.wav
        if not wav and not args.mic:
            voice = tamil_voice()
            if voice is None:
                print("No Tamil system voice installed and no --wav given.\n"
                      "Install one in System Settings > Accessibility > Spoken Content,\n"
                      "or rerun with --wav <file> or --text \"...\".")
                return 2
            tmp = Path(tempfile.mkdtemp()) / "sample.wav"
            if not synth_wav(TAMIL_SAMPLE, voice, tmp):
                print("Could not synthesise a sample; rerun with --wav or --text.")
                return 2
            wav, synthetic = str(tmp), True
            print(f"NOTE: using synthetic speech from `say -v {voice}` "
                  f"({TAMIL_SAMPLE!r}), not a real recording.")

        audio = speech.record(5.0) if args.mic else speech.load_wav(wav)
        try:
            transcript = speech.route(audio)
        except speech.SpeechUnavailable as exc:
            print(f"ASR unavailable: {exc}")
            return 2

        print(f"\n{'model':<20}{'transcript'}")
        for model, text in transcript.candidates.items():
            mark = " <- chosen" if model == transcript.source else ""
            print(f"{model:<20}{text}{mark}")
        print(f"\ndetected language: {transcript.lang}   "
              f"tamil script ratio: {speech.tamil_ratio(transcript.candidates['multilingual']):.2f}   "
              f"asr latency: {transcript.latency_s:.2f} s")

        utterance = speech.clean_transcript(transcript.text)
        print(f"cleaned: {utterance!r}")

    ep = execute(env, utterance, cleaned=utterance, transcript=transcript)
    print(f"\nplan ({ep.plan_source}): {ep.plan_json}")
    print(f"M6 {'PASS' if ep.success else 'FAIL'}"
          + ("  (synthetic speech)" if synthetic else ""))
    env.close()
    return 0 if ep.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
