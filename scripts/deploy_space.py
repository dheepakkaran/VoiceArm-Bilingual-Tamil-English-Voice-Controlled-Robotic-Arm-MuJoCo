"""Assemble a Hugging Face Space directory from this repo.

Gradio Spaces need `app.py` at the repo root, and the Space cannot reach the
vendored Panda assets because they are gitignored here -- so this stages a
self-contained tree rather than pushing the repo as-is.

    python scripts/deploy_space.py            # stage into build/space
    python scripts/deploy_space.py --push USER/SPACE

Pushing needs `huggingface_hub` and a token with write access; it is a separate
step so the staged tree can be inspected first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "build" / "space"

# (source, destination) relative to ROOT / STAGE
FILES = [
    ("hf_space/app.py", "app.py"),
    ("hf_space/requirements.txt", "requirements.txt"),
    ("hf_space/packages.txt", "packages.txt"),
    ("hf_space/README.md", "README.md"),
    ("assets/scene.xml", "assets/scene.xml"),
]
TREES = [
    ("src", "src"),
    ("assets/mujoco_menagerie/franka_emika_panda", "assets/mujoco_menagerie/franka_emika_panda"),
]


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for src, dst in FILES:
        target = STAGE / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, target)

    for src, dst in TREES:
        source = ROOT / src
        if not source.exists():
            sys.exit(f"missing {src} -- run ./setup.sh first")
        shutil.copytree(source, STAGE / dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))

    # A demo video so the Space has something to show before the first run.
    demo = ROOT / "docs" / "m3_pickplace.mp4"
    if demo.exists():
        (STAGE / "docs").mkdir(exist_ok=True)
        shutil.copy2(demo, STAGE / "docs" / demo.name)

    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    n = sum(1 for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {n} files, {size / 1e6:.1f} MB -> {STAGE}")
    return STAGE


def push(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo_id, repo_type="space")
    print(f"pushed -> https://huggingface.co/spaces/{repo_id}")
    print("Now set the hardware to ZeroGPU in the Space settings.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/SPACE",
                    help="upload the staged tree to this Space")
    args = ap.parse_args()

    stage()
    if args.push:
        push(args.push)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
