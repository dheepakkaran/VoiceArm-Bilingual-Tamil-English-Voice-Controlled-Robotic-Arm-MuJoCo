"""Compare planner models on the eight reference utterances.

Exists because the README says `sarvam-m` was rejected on footprint rather than
on quality, and that is a claim worth being able to check. A 24B model at 4-bit
needs roughly 13 GB and cannot co-reside with the ASR models and the detector on
a 16 GB laptop -- but it fits on a 32 GB cloud session, so the comparison can be
made there.

    python scripts/bench_planner.py
    python scripts/bench_planner.py --models Qwen/Qwen3-4B-Instruct-2507 sarvamai/sarvam-m --load-4bit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import planner

# (utterance, expected object phrase, expects a place step)
CASES = [
    ("put the red block in the bowl", "a red cube", True),
    ("pick up the green cube", "a green cube", False),
    ("sivappu block-ah bowl-la vai", "a red cube", True),
    ("pachai block-ah edu", "a green cube", False),
    ("neela block-ah bowl-la podu", "a blue cube", True),
    ("நீல கட்டையை கிண்ணத்தில் வை", "a blue cube", True),
    ("பச்சை கட்டையை எடு", "a green cube", False),
    ("சிவப்பு கட்டையை கிண்ணத்தில் வை", "a red cube", True),
]
DEFAULT_MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"]
# The 24B Tamil-native model the local machine could not hold. The community
# 4-bit upload is 14 GB against 47 GB for the official fp16 weights.
SARVAM_4BIT = "neuralnets/sarvam-m-4bit-q"


def best_dtype():
    """bfloat16 where the GPU supports it, float16 otherwise.

    A T4 is compute capability 7.5 and has no bfloat16 units -- asking for it
    silently falls back to a slow emulated path, which matters a lot when the
    point of the run is a latency comparison. Kaggle's free tier is T4 x2, so
    this is the common case rather than an edge one.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def already_quantized(repo: str) -> bool:
    """True when the checkpoint ships its own quantization config.

    Community 4-bit uploads carry a quantization_config, and handing those a
    fresh BitsAndBytesConfig makes transformers try to quantize an already
    quantized tensor. Worth checking rather than requiring the caller to know:
    the pre-quantized sarvam-m is 14 GB against 47 GB for the official weights,
    so it is the one you actually want on a free GPU.
    """
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(repo)
    except Exception:
        return False
    return getattr(cfg, "quantization_config", None) is not None


def load(repo: str, four_bit: bool):
    """Load a planner model directly, bypassing the module-level singleton."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = best_dtype()
    tok = AutoTokenizer.from_pretrained(repo)
    kwargs: dict = {"low_cpu_mem_usage": True, "device_map": "auto"}

    if four_bit and already_quantized(repo):
        print("  checkpoint is pre-quantized; not re-quantizing")
        four_bit = False
        kwargs.pop("dtype", None)
    elif four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    elif not already_quantized(repo):
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    return model.eval(), tok


def generate(model, tok, utterance: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": planner.SYSTEM_PROMPT},
        {"role": "user", "content": utterance},
    ]
    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=planner.MAX_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def score(raw: str, want_obj: str, want_place: bool) -> bool:
    steps = planner._extract_json(raw)
    if not steps:
        return False
    first_ok = steps[0].get("target", "").strip().lower() == want_obj
    place_ok = any(s["action"] == "place" for s in steps) == want_place
    return first_ok and place_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--load-4bit", action="store_true",
                    help="quantize with bitsandbytes; needed for 20B+ models")
    args = ap.parse_args()

    rows = []
    for repo in args.models:
        print(f"\n=== {repo} ===")
        t0 = time.perf_counter()
        try:
            model, tok = load(repo, args.load_4bit)
        except Exception as exc:
            print(f"  load failed: {type(exc).__name__}: {exc}")
            continue
        load_s = time.perf_counter() - t0

        correct, lats = 0, []
        for utterance, want_obj, want_place in CASES:
            t0 = time.perf_counter()
            raw = generate(model, tok, utterance)
            lats.append(time.perf_counter() - t0)
            ok = score(raw, want_obj, want_place)
            correct += ok
            print(f"  {'PASS' if ok else 'FAIL'}  {utterance[:38]:<40}"
                  f"{lats[-1]:5.2f}s  {raw.strip()[:60]}")

        try:
            import torch

            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            vram = float("nan")

        rows.append((repo, correct, float(np.mean(lats)), load_s, vram))
        del model, tok
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    if not rows:
        print("\nno model loaded successfully")
        return 1

    print(f"\n{'model':<40}{'correct':>9}{'mean gen':>10}{'load':>8}{'peak vram':>11}")
    for repo, correct, lat, load_s, vram in rows:
        print(f"{repo:<40}{correct:>5}/{len(CASES)}{lat:>9.2f}s{load_s:>7.0f}s{vram:>10.1f}G")

    out = Path("out/planner_bench.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        [{"model": r, "correct": c, "of": len(CASES), "mean_gen_s": round(l, 3),
          "load_s": round(s, 1), "peak_vram_gb": round(v, 2)}
         for r, c, l, s, v in rows], indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
