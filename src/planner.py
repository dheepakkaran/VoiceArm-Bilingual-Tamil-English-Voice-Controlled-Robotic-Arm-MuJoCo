"""Natural-language task planner.

Runs a local quantized LLM through MLX -- no cloud API. Accepts Tamil script,
English, or romanized Tanglish and emits a JSON task plan whose `target` fields
are free-form object descriptions handed straight to open-vocabulary detection.

A deterministic keyword parser stands in whenever the model is unavailable, so
the pipeline never hard-fails on a missing dependency or a bad generation.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from . import config

log = logging.getLogger(__name__)

ACTIONS = {"pick", "place", "move_to", "say"}

_llm = None
_tokenizer = None

SYSTEM_PROMPT = """You convert a spoken robot instruction into a JSON task plan.

The instruction may be in Tamil script, in English, or in romanized Tanglish
(Tamil written with English letters, often mixed with English words).

The robot sees a table holding a red cube, a green cube, a blue cube, and one
white bowl.

Return ONLY a JSON array. No prose, no markdown fences, no explanation.
Each element is {"action": ..., "target": ...}.
Allowed actions: "pick", "place", "move_to", "say".
"target" is a short English noun phrase describing the object, for example
"a red cube" or "a white bowl". It is passed to a vision model, so describe the
object rather than naming a variable.

Romanized Tamil words you will see, with their meanings:
  sivappu / sivapu = red        pachai / pachchai = green
  neelam / neela = blue         manjal = yellow
  kattai / block / cube = cube  kinnam / bowl = bowl
  edu / eduthu = pick up        vai / vei / podu = put, place
  -ah, -ai, -ya = object marker (ignore it)
  -la, -le, -il = "in" or "on"  (ignore it)

So "sivappu block-ah bowl-la vai" means "put the red cube in the bowl".

Examples:
"put the red block in the bowl"
[{"action":"pick","target":"a red cube"},{"action":"place","target":"a white bowl"}]

"pachai block-ah edu"
[{"action":"pick","target":"a green cube"}]

"sivappu block-ah bowl-la vai"
[{"action":"pick","target":"a red cube"},{"action":"place","target":"a white bowl"}]

"neela kattai-ah edu"
[{"action":"pick","target":"a blue cube"}]

"நீல கட்டையை கிண்ணத்தில் வை"
[{"action":"pick","target":"a blue cube"},{"action":"place","target":"a white bowl"}]
"""

# --- deterministic fallback -------------------------------------------------
_COLOURS = {
    "a red cube":   [r"red", r"sivapp?u", r"சிவப்ப"],
    "a green cube": [r"green", r"pach?ai", r"பச்ச"],
    "a blue cube":  [r"blue", r"neel[a-z]*", r"நீல"],
}
_CONTAINER = [r"bowl", r"kinn?am", r"கிண்ண", r"basket", r"container"]
_PLACE_VERBS = [r"\bput\b", r"\bplace\b", r"\bdrop\b", r"\bvai\b", r"\bvei\b", r"வை", r"போடு", r"podu"]
_PICK_VERBS = [r"\bpick\b", r"\btake\b", r"\bgrab\b", r"\bedu\b", r"எடு"]


def _hits(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def plan_fallback(utterance: str) -> list[dict[str, Any]]:
    """Regex parser over Tamil, Tanglish, and English colour and verb words."""
    text = utterance.strip()
    target = next((canon for canon, pats in _COLOURS.items() if _hits(text, pats)), None)
    if target is None:
        return []

    steps: list[dict[str, Any]] = [{"action": "pick", "target": target}]
    if _hits(text, _CONTAINER) or (_hits(text, _PLACE_VERBS) and not _hits(text, _PICK_VERBS)):
        steps.append({"action": "place", "target": "a white bowl"})
    return steps


# --- local LLM --------------------------------------------------------------
def _load():
    global _llm, _tokenizer
    if _llm is not None:
        return _llm, _tokenizer
    from mlx_lm import load

    t0 = time.perf_counter()
    log.info("loading %s", config.LLM_MODEL)
    _llm, _tokenizer = load(config.LLM_MODEL)
    log.info("planner ready in %.1f s", time.perf_counter() - t0)
    return _llm, _tokenizer


def _extract_json(text: str) -> list[dict[str, Any]] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?|```", "", text)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    steps = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        action, target = item.get("action"), item.get("target")
        if action not in ACTIONS or not isinstance(target, str) or not target.strip():
            return None
        steps.append({"action": action, "target": target.strip()})
    return steps or None


def _generate(utterance: str, nudge: str = "") -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + nudge},
        {"role": "user", "content": utterance},
    ]
    try:  # Qwen3 exposes a thinking mode; JSON-only output needs it off
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    return generate(model, tokenizer, prompt=prompt, max_tokens=300,
                    sampler=make_sampler(temp=0.2), verbose=False)


def plan(utterance: str) -> tuple[list[dict[str, Any]], str, float]:
    """Return (steps, source, latency_s). `source` is "llm" or "fallback"."""
    t0 = time.perf_counter()
    try:
        raw = _generate(utterance)
        steps = _extract_json(raw)
        if steps is None:
            log.warning("unparseable plan, retrying: %s", raw[:160].replace("\n", " "))
            raw = _generate(utterance, "\nYour previous reply was not valid JSON. "
                                       "Reply with the JSON array only.")
            steps = _extract_json(raw)
        if steps is not None:
            return steps, "llm", time.perf_counter() - t0
        log.warning("LLM produced no valid plan, using fallback parser")
    except Exception as exc:  # model missing, OOM, download failure
        log.warning("planner LLM unavailable (%s), using fallback parser", type(exc).__name__)

    return plan_fallback(utterance), "fallback", time.perf_counter() - t0


if __name__ == "__main__":
    for u in ["put the red block in the bowl", "pachai block-ah edu",
              "நீல கட்டையை கிண்ணத்தில் வை", "sivappu block-ah bowl-la vai"]:
        print(f"{u!r:45s} -> {plan_fallback(u)}")
