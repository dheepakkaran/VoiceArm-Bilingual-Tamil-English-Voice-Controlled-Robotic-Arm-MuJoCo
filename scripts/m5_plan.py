"""M5 acceptance: run mixed Tamil / English / Tanglish utterances end to end."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.executor import execute
from src.sim import SimEnv

UTTERANCES = [
    ("english",  "put the red block in the bowl"),
    ("english",  "pick up the green cube"),
    ("tanglish", "sivappu block-ah bowl-la vai"),
    ("tanglish", "pachai block-ah edu"),
    ("tanglish", "neela block-ah bowl-la podu"),
    ("tamil",    "நீல கட்டையை கிண்ணத்தில் வை"),
    ("tamil",    "பச்சை கட்டையை எடு"),
    ("tamil",    "சிவப்பு கட்டையை கிண்ணத்தில் வை"),
]


def main() -> int:
    env = SimEnv()
    print(f"{'lang':<10}{'utterance':<36}{'plan':<8}{'steps':>6}{'lat_s':>7}{'ok':>5}")
    passes = 0

    for lang, text in UTTERANCES:
        ep = execute(env, text, record=True)
        n = len(__import__("json").loads(ep.plan_json))
        passes += ep.success
        print(f"{lang:<10}{text:<36}{ep.plan_source:<8}{n:>6}{ep.llm_latency_s:>7.2f}"
              f"{'PASS' if ep.success else 'FAIL':>6}")

    print(f"\nM5 {passes}/{len(UTTERANCES)} utterances executed successfully")
    ok = passes >= 6
    print("M5", "PASS" if ok else "FAIL", "(threshold 6/8)")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
