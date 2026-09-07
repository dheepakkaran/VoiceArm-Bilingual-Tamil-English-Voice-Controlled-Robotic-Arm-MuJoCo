"""M4 acceptance: locate every object from the overhead camera with OWL-ViT.

Compares the open-vocabulary estimate against simulator ground truth and writes
an annotated image. Queries are free-form text, not a fixed class list.
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
    "a white bowl": "bowl",
}
COLORS = [(255, 80, 80), (80, 220, 100), (90, 140, 255), (250, 220, 90)]


def main() -> int:
    env = SimEnv()
    env.park()                      # clear the arm out of the overhead view
    queries = list(QUERIES)
    canvases = {cam: Image.fromarray(capture(env, cam)[0]) for cam in config.PERCEPTION_CAMERAS}
    draws = {cam: ImageDraw.Draw(c) for cam, c in canvases.items()}

    print(f"{'query':<15}{'camera':<12}{'score':>6}  {'estimated xyz':<26}"
          f"{'ground truth':<26}{'err':>8}  ok")
    passes, errors = 0, []

    for q in queries:
        body = QUERIES[q]
        resting = body in config.OBJECTS
        est, det = locate(env, q, resting=resting)
        if est is None:
            print(f"{q:<15}{'--':<12}{'--':>6}  {'not detected':<26}{'':<26}{'':>8}  FAIL")
            continue

        gt = env.body_pos(body)
        err = float(np.linalg.norm(est - gt)) if resting else float(np.linalg.norm(est[:2] - gt[:2]))
        errors.append(err)
        ok = err < TOL_M
        passes += ok

        colour = COLORS[queries.index(q) % len(COLORS)]
        d = draws[det.camera]
        d.rectangle(det.bbox, outline=colour, width=3)
        d.text((det.bbox[0] + 4, max(det.bbox[1] - 12, 2)), f"{q} {det.score:.2f}", fill=colour)

        print(f"{q:<15}{det.camera:<12}{det.score:>6.2f}  {np.round(est, 3)!s:<26}"
              f"{np.round(gt, 3)!s:<26}{err * 100:>6.1f}cm  {'PASS' if ok else 'FAIL'}")

    panel = np.concatenate([np.asarray(canvases[c]) for c in config.PERCEPTION_CAMERAS], axis=1)
    out = config.OUT / "m4_detections.png"
    Image.fromarray(panel).save(out)
    print(f"\nsaved {out}")
    if errors:
        print(f"mean error {np.mean(errors) * 100:.2f} cm   max {np.max(errors) * 100:.2f} cm")

    print(f"M4 {passes}/{len(queries)} objects localized within {TOL_M * 100:.0f} cm")
    ok_all = passes == len(queries)
    print("M4", "PASS" if ok_all else "FAIL")
    env.close()
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
