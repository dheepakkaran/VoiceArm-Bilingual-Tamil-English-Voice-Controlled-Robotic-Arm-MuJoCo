"""Generate the Colab/Kaggle notebooks from source held in this file.

Notebooks are generated rather than hand-edited so the repo URL, model ids and
prose live in one reviewable place instead of inside JSON blobs.

Note on the `source` field: nbformat expects a list of lines that each *end*
with a newline. Splitting on "\\n" without putting them back produces a cell
whose lines concatenate into one -- `import torch` followed by `print(...)`
becomes `import torchprint(...)` and fails with a SyntaxError that points at a
line you never wrote.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GH = "https://github.com/dheepakkaran/VoiceArm-Bilingual-Tamil-English-Voice-Controlled-Robotic-Arm-MuJoCo"
REPO = f"{GH}.git"


def _lines(src: str) -> list[str]:
    return src.strip().splitlines(keepends=True) or [""]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": _lines(src)}


def code(src: str) -> dict:
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": _lines(src)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, nb: dict) -> None:
    """Validate first, then write.

    Ordering matters: an earlier version wrote the file and validated after, so
    a failing build still left a broken notebook on disk -- and the next run
    read that file back as its own input.
    """
    validate_nb(nb, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  wrote {path} ({len(nb['cells'])} cells)")


def validate_nb(nb: dict, name: str) -> None:
    """Compile every code cell, which is what catches the newline bug."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        stripped = "\n".join(
            "" if ln.lstrip().startswith("!") else ln for ln in src.split("\n")
        )
        try:
            compile(stripped, f"cell {i}", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"{name} cell {i}: {exc}\n---\n{src}\n---") from exc


# --- demo notebook ----------------------------------------------------------
def demo_notebook() -> dict:
    return notebook([
md(f"""
# VoiceArm — run the full pipeline on a free GPU

**Code-switched speech grounding for open-vocabulary robotic manipulation.**

Speak or type in Tamil, English, or Tanglish. A dual-ASR router transcribes it, a
local Qwen3-4B turns it into a JSON task plan, OWLv2 finds the named object in the
camera image, and an IK controller executes the pick-and-place in MuJoCo.

Source: [GitHub]({GH})

**Before you run:** turn on the GPU.
* Kaggle — Settings, Accelerator, **`GPU T4 x2`** (the default P100 is sm_60 and
  this stack needs sm_75+), and switch **Internet** on (needs a phone-verified
  account, or the model downloads fail)
* Colab — Runtime, Change runtime type, `T4 GPU`

Simulation only. No sim-to-real transfer is claimed.
"""),
md("## 1. System libraries\n\nMuJoCo renders offscreen, so it needs an EGL driver. This is the only apt step."),
code("""
import subprocess
subprocess.run("apt-get -qq update", shell=True)
subprocess.run(
    "apt-get -qq install -y libegl1 libgles2 libosmesa6 libglfw3 > /dev/null",
    shell=True,
)
print("egl libraries installed")
"""),
md("## 2. Project and dependencies"),
code(f"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
PROJ = ROOT / "voicearm"

if not PROJ.exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(PROJ)], check=True)

os.chdir(PROJ)
sys.path.insert(0, str(PROJ))
print("project at", PROJ)
"""),
code("""
# requirements.txt pins the Apple/MLX stack; on an NVIDIA box we want the
# portable one that the Space uses.
!pip install -q -r webapp/requirements.txt 2>&1 | tail -3
print("deps installed")
"""),
code("""
# Only the Franka Panda, not the whole menagerie.
import json
import urllib.request
from pathlib import Path

DIR = Path("assets/mujoco_menagerie/franka_emika_panda")
BASE = ("https://raw.githubusercontent.com/google-deepmind/"
        "mujoco_menagerie/main/franka_emika_panda")
(DIR / "assets").mkdir(parents=True, exist_ok=True)

for name in ["panda.xml", "hand.xml", "scene.xml", "LICENSE"]:
    urllib.request.urlretrieve(f"{BASE}/{name}", DIR / name)

api = ("https://api.github.com/repos/google-deepmind/mujoco_menagerie/"
       "contents/franka_emika_panda/assets")
for mesh in json.load(urllib.request.urlopen(api)):
    if mesh["type"] == "file":
        urllib.request.urlretrieve(mesh["download_url"], DIR / "assets" / mesh["name"])

print(len(list((DIR / "assets").iterdir())), "mesh assets")
"""),
md("""
## 3. Backend check

`src/backend.py` picks the model stack at import. MLX does not exist here, so it
selects the transformers path and CUDA. `MUJOCO_GL` is set in `src/__init__.py`,
before anything imports mujoco.
"""),
code("""
import os

os.environ["VOICEARM_BACKEND"] = "torch"

import torch

from src import backend

print(backend.describe())
print("gpu :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("vram:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB"
      if torch.cuda.is_available() else "-")
"""),
md("## 4. Simulation and kinematics\n\nPhysics, rendering and IK only -- no models yet."),
code("!python scripts/m1_move.py"),
code("!python scripts/m2_ik.py"),
code("""
from PIL import Image

Image.open("out/m1_frames.png")
"""),
md("## 5. Pick and place\n\nGrasps each block from its ground-truth pose. Perception comes next."),
code("!python scripts/m3_pick.py"),
code("""
from IPython.display import Video

Video("out/m3_pickplace.mp4", embed=True, width=640)
"""),
md("""
## 6. Open-vocabulary perception

OWLv2 locates each object from a free-text description, then the box is
unprojected through the depth buffer. Downloads about 1.5 GB.
"""),
code("!python scripts/m4_detect.py"),
code("""
from PIL import Image

Image.open("out/m4_detections.png")
"""),
md("""
## 7. Language

Downloads Qwen3-4B (about 8 GB). Runs eight utterances -- English, Tanglish and
Tamil script -- end to end through plan, perception and motion.
"""),
code("!python scripts/m5_plan.py"),
md("""
## 8. Live demo

Launches the Gradio app with a public share link, valid while this session
lives. Record yourself in Tamil, English or Tanglish, or type an instruction.

The **Live** panel updates while the arm moves -- the handlers are generators,
so frames reach the browser as they are rendered rather than only as a video at
the end. Expect roughly 30-60 s per command on a T4, against about 5 s locally
on Apple Silicon through MLX. If the live view feels sluggish over the tunnel,
raise `LIVE_EVERY` in `webapp/app.py`.

The share URL dies when the session ends, which is why the project's permanent
link is a GitHub Pages page rather than this.
"""),
code("""
import importlib.util
import sys

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("voicearm_app", "webapp/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

app.warm_models()
app.build().queue(max_size=8).launch(share=True)
"""),
md(f"""
## 9. Optional: does a Tamil-native 24B planner do better?

`sarvam-m` is the strongest open model for Tamil, and this project rejected it
for one reason only -- at 4-bit it needs about 13 GB and would not co-reside
with the other three models on a 16 GB laptop. A T4 x2 session has 32 GB, so the
comparison the README could not make is possible here.

`scripts/bench_planner.py` runs that comparison:

```
!python scripts/bench_planner.py \
    --models Qwen/Qwen3-4B-Instruct-2507 neuralnets/sarvam-m-4bit-q --load-4bit
```

It needs sm_75 or newer, which a Colab T4 is, and about 22 GB of downloads. The
pre-quantized sarvam-m upload is 14 GB against 47 GB for the official weights.
"""),
    ])


if __name__ == "__main__":
    print("building notebooks:")
    write(ROOT / "notebooks" / "voicearm_demo.ipynb", demo_notebook())
