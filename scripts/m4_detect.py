"""M4 acceptance: locate every object from the cameras with OWLv2.

Compares the open-vocabulary estimate against simulator ground truth over
several trials, shuffling the blocks between them. Queries are free-form text,
not a fixed class list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.perception import capture, locate
from src.sim import SimEnv

TOL_M = 0.03  # 3 cm acceptance

QUERIES = {
    "a red cube": "red_block",
    "a green cube": "green_block",
    "a blue cube": "blue_block",
    "a white bowl": config.CONTAINER,
}
COLORS = [(255, 80, 80), (80, 220, 100), (90, 140, 255), (250, 220, 90)]

# Blocks are shuffled between trials. Measuring one fixed layout shows the
# detector works on that layout; it says nothing about whether the grounding
# generalises, which is what the result actually claims.
TRIALS = 3
BLOCK_X = (0.38, 0.50)
BLOCK_Y = (-0.22, 0.22)
MIN_SEPARATION = 0.09


def shuffle_blocks(env: SimEnv, rng: np.random.Generator) -> None:
    """Scatter the blocks on the tabletop, keeping them apart and off the bowl."""
    placed: list[np.ndarray] = []
    bowl_xy = env.body_pos(config.CONTAINER)[:2]

    for name in config.OBJECTS:
        for _ in range(200):
            xy = np.array([rng.uniform(*BLOCK_X), rng.uniform(*BLOCK_Y)])
            if np.linalg.norm(xy - bowl_xy) < 0.14:
                continue
            if all(np.linalg.norm(xy - p) > MIN_SEPARATION for p in placed):
                placed.append(xy)
                env.place_object(name, xy)
                break
        else:
            raise RuntimeError(f"could not find a spot for {name}")
    env.settle(300)


def run_trial(env: SimEnv, queries: list[str], label: str):
    canvases = {cam: Image.fromarray(capture(env, cam)[0])
                for cam in config.PERCEPTION_CAMERAS}
    draws = {cam: ImageDraw.Draw(c) for cam, c in canvases.items()}
    passes, errors = 0, []

    for q in queries:
        body = QUERIES[q]
        resting = body in config.OBJECTS
        est, det = locate(env, q, resting=resting)
        if est is None:
            print(f"  {label}{q:<15}{'--':<12}{'--':>6}  not detected"
                  f"{'':<24}{'':>8}  FAIL")
            continue

        gt = env.body_pos(body)
        err = (float(np.linalg.norm(est - gt)) if resting
               else float(np.linalg.norm(est[:2] - gt[:2])))
        errors.append(err)
        ok = err < TOL_M
        passes += ok

        colour = COLORS[queries.index(q) % len(COLORS)]
        d = draws[det.camera]
        d.rectangle(det.bbox, outline=colour, width=3)
        d.text((det.bbox[0] + 4, max(det.bbox[1] - 12, 2)),
               f"{q} {det.score:.2f}", fill=colour)

        print(f"  {label}{q:<15}{det.camera:<12}{det.score:>6.2f}  "
              f"{np.round(est, 3)!s:<26}{err * 100:>6.1f}cm  {'PASS' if ok else 'FAIL'}")

    return passes, errors, canvases


def main() -> int:
    rng = np.random.default_rng(0)
    env = SimEnv()
    env.park()                      # clear the arm out of the overhead view
    queries = list(QUERIES)

    print(f"{TRIALS} trials, blocks shuffled between each\n")
    print(f"  {'trial':<7}{'query':<15}{'camera':<12}{'score':>6}  "
          f"{'estimated xyz':<26}{'err':>8}  ok")

    total_passes, all_errors, saved = 0, [], None
    for t in range(TRIALS):
        if t:
            shuffle_blocks(env, rng)
        passes, errors, canvases = run_trial(env, queries, f"{t + 1}/{TRIALS}    ")
        saved = saved or canvases
        total_passes += passes
        all_errors += errors
        print()

    panel = np.concatenate([np.asarray(saved[c])
                            for c in config.PERCEPTION_CAMERAS], axis=1)
    out = config.OUT / "m4_detections.png"
    Image.fromarray(panel).save(out)
    print(f"saved {out}")

    expected = TRIALS * len(queries)
    if all_errors:
        print(f"mean error {np.mean(all_errors) * 100:.2f} cm   "
              f"max {np.max(all_errors) * 100:.2f} cm")
    print(f"M4 {total_passes}/{expected} localizations within {TOL_M * 100:.0f} cm")
    ok_all = total_passes == expected
    print("M4", "PASS" if ok_all else "FAIL")
    env.close()
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
