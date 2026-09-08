"""Assemble a Hugging Face Space directory from this repo.

Two kinds of Space, because hosting a Gradio Space needs a paid tier while a
static one is free:

    --kind static   a showcase page: results, benchmarks, recorded episodes,
                    and buttons that open the live notebook on Colab or Kaggle
    --kind gradio   the live app; requires HF PRO to actually run

Either way this stages a self-contained tree rather than pushing the repo as-is,
because Spaces need their entry point at the root and the vendored Panda assets
are gitignored here.

    python scripts/deploy_space.py --kind static
    python scripts/deploy_space.py --kind static --push USER/SPACE

Pushing needs a token with write access, kept a separate step so the staged tree
can be inspected first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {"gradio": ROOT / "build" / "space", "static": ROOT / "build" / "static"}

# docs/ is the single source for the showcase page: GitHub Pages serves it
# directly from there, and the HF Space gets a copy of the same tree. Keeping one
# copy means the two surfaces cannot drift apart.
STATIC_SOURCE = "docs"

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


def stage_static() -> Path:
    stage = STAGES["static"]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    src = ROOT / STATIC_SOURCE
    if not (src / "index.html").exists():
        sys.exit(f"missing {STATIC_SOURCE}/index.html")
    shutil.copytree(src, stage, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))

    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    n = sum(1 for f in stage.rglob("*") if f.is_file())
    print(f"staged {n} files, {size / 1e6:.1f} MB -> {stage}")
    return stage


def stage_gradio() -> Path:
    STAGE = STAGES["gradio"]
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


def push(repo_id: str, kind: str, stage: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk=kind, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="space")
    print(f"pushed -> https://huggingface.co/spaces/{repo_id}")
    if kind == "gradio":
        print("Now set the hardware to ZeroGPU in the Space settings.")
        print("Note: hosting a Gradio Space requires an HF PRO subscription.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["static", "gradio"], default="static",
                    help="static showcase page (free) or the live Gradio app (needs PRO)")
    ap.add_argument("--push", metavar="USER/SPACE",
                    help="upload the staged tree to this Space")
    args = ap.parse_args()

    stage = stage_static() if args.kind == "static" else stage_gradio()
    if args.push:
        push(args.push, args.kind, stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
