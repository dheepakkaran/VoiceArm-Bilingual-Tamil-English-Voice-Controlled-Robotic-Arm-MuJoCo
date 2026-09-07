"""Character-error-rate benchmark for the two ASR backends on Tamil.

Sentences are synthesised with the macOS Tamil voice and read back from wav
files rather than through the speakers, so the measurement isolates the models
from room acoustics. Synthetic speech is easier than real speech -- these
numbers are a floor on error, not an estimate of real-world accuracy.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import speech

SENTENCES = [
    "சிவப்பு கட்டையை கிண்ணத்தில் வை",
    "பச்சை கட்டையை எடு",
    "நீல கட்டையை கிண்ணத்தில் போடு",
    "சிவப்பு பொருளை எடுத்து வை",
    "பச்சை கட்டையை கிண்ணத்தில் வை",
    "நீல பொருளை எடு",
]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if c.isalnum())


def cer(ref: str, hyp: str) -> float:
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return float("nan")
    row = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(h) + 1):
            cur = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return float(row[len(h)]) / len(r)


def tamil_voice() -> str | None:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    return next((l.split()[0] for l in out.splitlines() if "ta_IN" in l), None)


def synth(text: str, voice: str, path: Path) -> bool:
    aiff = path.with_suffix(".aiff")
    try:
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return path.exists()


def main() -> int:
    voice = tamil_voice()
    if voice is None:
        print("No Tamil system voice installed; cannot run this benchmark.")
        return 2
    print(f"synthesising with `say -v {voice}` -- synthetic speech, not human\n")

    tmp = Path(tempfile.mkdtemp())
    ml_errs, ta_errs, routed_errs, chose_specialist = [], [], [], 0

    for i, ref in enumerate(SENTENCES):
        wav = tmp / f"{i}.wav"
        if not synth(ref, voice, wav):
            print(f"skip {ref!r}: synthesis failed")
            continue
        audio = speech.load_wav(str(wav))

        ml_text, lang = speech.transcribe_multilingual(audio)
        ta_text = speech.transcribe_tamil(audio)
        routed = speech.route(audio)
        chose_specialist += routed.source == "tamil_specialist"

        ml_errs.append(cer(ref, ml_text))
        ta_errs.append(cer(ref, ta_text))
        routed_errs.append(cer(ref, routed.text))

        print(f"ref                 {ref}")
        print(f"  multilingual [{lang}] {ml_errs[-1] * 100:6.1f}%  {ml_text}")
        print(f"  specialist         {ta_errs[-1] * 100:6.1f}%  {ta_text}")
        print(f"  routed -> {routed.source:<18}{routed_errs[-1] * 100:6.1f}%\n")

    n = len(routed_errs)
    print(f"{'backend':<22}{'mean CER':>10}{'median':>10}")
    for name, errs in (("multilingual", ml_errs), ("tamil_specialist", ta_errs),
                       ("routed (shipped)", routed_errs)):
        print(f"{name:<22}{np.mean(errs) * 100:>9.1f}%{np.median(errs) * 100:>9.1f}%")
    print(f"\nn={n}  router chose the specialist {chose_specialist}/{n} times")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
