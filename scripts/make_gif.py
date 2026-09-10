"""Render a pick-and-place episode straight to an animated GIF.

GitHub renders a relative `.mp4` in a README as a link, not a player, so the one
format that actually animates inline is a GIF. Written with PIL rather than
ffmpeg, which is not installed on the development machine and would be one more
thing a contributor has to have.

    python scripts/make_gif.py
    python scripts/make_gif.py --width 560 --fps 14 --out docs/media/demo.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.grasp import pick_and_place
from src.sim import SimEnv

DEFAULT_OUT = config.ROOT / "docs" / "media" / "demo.gif"


def render_episode(block: str, capture_every: int) -> list[np.ndarray]:
    env = SimEnv()
    env.park()
    bowl = env.body_pos(config.CONTAINER)
    env.start_recording(config.SCENE_CAM, every=capture_every)
    pick_and_place(env, env.body_pos(block),
                   np.array([bowl[0], bowl[1], 0.46]), body=block)
    frames = env.stop_recording()
    env.close()
    return frames


def write_gif(frames: list[np.ndarray], out: Path, width: int, fps: int,
              colors: int = 128) -> None:
    """Resize, quantize against one shared palette, and save.

    The quantization method matters more than the palette size here. PIL's
    default MEDIANCUT splits colour space by pixel population, and the blocks are
    about 0.3% of each frame against a large wood gradient and a checkerboard
    floor -- so it spends the palette on those and drops the blocks to grey. At
    256 colours it still lost green entirely. Since the demo is "pick up the
    *red* cube", a GIF with grey blocks is broken, not merely small.

    FASTOCTREE subdivides colour space geometrically instead, so small saturated
    clusters survive. Measured on one frame, source blocks at rgb(155,29,28),
    (27,137,41) and (25,58,155):

        MEDIANCUT  128   red lost,  green lost,  blue lost
        MEDIANCUT  256   red kept,  green lost,  blue kept
        FASTOCTREE 128   all three kept

    The palette is still built from frames sampled across the episode so the
    blocks are seen in every position they occupy.
    """
    h, w = frames[0].shape[:2]
    size = (width, round(h * width / w))
    resized = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in frames]

    sample_idx = np.linspace(0, len(resized) - 1, min(12, len(resized))).astype(int)
    montage = Image.new("RGB", (size[0], size[1] * len(sample_idx)))
    for row, i in enumerate(sample_idx):
        montage.paste(resized[i], (0, row * size[1]))
    palette = montage.quantize(colors=colors, method=Image.FASTOCTREE)

    images = [im.quantize(palette=palette, dither=Image.NONE) for im in resized]
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:],
        duration=round(1000 / fps), loop=0, optimize=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="red_block", choices=config.OBJECTS)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--capture-every", type=int, default=36,
                    help="physics steps between captured frames; higher is a "
                         "smaller file and a faster-looking arm")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    frames = render_episode(args.block, args.capture_every)
    write_gif(frames, args.out, args.width, args.fps, args.colors)

    kb = args.out.stat().st_size / 1024
    print(f"{len(frames)} frames -> {args.out} ({kb:.0f} KB, "
          f"{args.width}px, {args.fps} fps)")
    if kb > 5120:
        print("over 5 MB; raise --capture-every or lower --width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
