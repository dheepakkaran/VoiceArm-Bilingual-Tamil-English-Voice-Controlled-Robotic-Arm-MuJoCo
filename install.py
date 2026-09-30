"""One-time setup: a virtualenv, the dependencies, and the Franka Panda model.

    python install.py

Written in Python rather than bash so it runs the same on macOS, Linux and
Windows. The previous setup.sh needed curl, xargs and a POSIX shell, none of
which Windows has by default.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PANDA = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda"
BASE = ("https://raw.githubusercontent.com/google-deepmind/"
        "mujoco_menagerie/main/franka_emika_panda")
API = ("https://api.github.com/repos/google-deepmind/mujoco_menagerie/"
       "contents/franka_emika_panda/assets")

MIN_PYTHON = (3, 10)


def venv_python() -> Path:
    """Where pip and python live inside the venv, per platform."""
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer, "
                 f"found {sys.version.split()[0]}")


def make_venv() -> Path:
    py = venv_python()
    if not py.exists():
        print("creating .venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"],
                   check=True)
    print("installing dependencies")
    subprocess.run([str(py), "-m", "pip", "install", "-q", "-r",
                    str(ROOT / "requirements.txt")], check=True)
    return py


def fetch_panda() -> int:
    """Download one robot, not all of mujoco_menagerie.

    The full repository is hundreds of megabytes of meshes for robots this
    project never loads, so the files come over the raw endpoint instead.
    """
    (PANDA / "assets").mkdir(parents=True, exist_ok=True)

    for name in ["panda.xml", "hand.xml", "scene.xml", "LICENSE"]:
        urllib.request.urlretrieve(f"{BASE}/{name}", PANDA / name)

    with urllib.request.urlopen(API) as response:
        listing = json.load(response)

    meshes = [f for f in listing if f["type"] == "file"]
    for i, mesh in enumerate(meshes, 1):
        urllib.request.urlretrieve(mesh["download_url"],
                                   PANDA / "assets" / mesh["name"])
        print(f"\r  meshes {i}/{len(meshes)}", end="", flush=True)
    print()
    return len(meshes)


def linux_note() -> None:
    if platform.system() != "Linux":
        return
    print("\nOn Linux you may also need system packages:")
    print("  sudo apt install libegl1 libosmesa6   # MuJoCo renders offscreen")
    print("  sudo apt install libportaudio2        # only for --mic")


def main() -> int:
    check_python()
    py = make_venv()
    count = fetch_panda()

    print(f"\ndone: {count} mesh files")
    print("\nTry it:")
    print(f"  {py} scripts/demo.py --check      # arm and grasping, no models")
    print(f"  {py} scripts/demo.py              # three example commands")
    print(f"  {py} -m pytest tests/ -q          # tests")
    linux_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
