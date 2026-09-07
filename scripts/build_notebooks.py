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
* Kaggle — Settings, Accelerator, `GPU T4 x2`, and switch **Internet** on
  (needs a phone-verified account, or the model downloads fail)
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
!pip install -q -r hf_space/requirements.txt 2>&1 | tail -3
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

The share URL dies when the session ends, which is why the project's permanent
link is a static page rather than this.
"""),
code("""
import importlib.util
import sys

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("space_app", "hf_space/app.py")
space_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space_app)

space_app.warm_models()
space_app.build().queue(max_size=8).launch(share=True)
"""),
md(f"""
## 9. Optional: does a Tamil-native 24B planner do better?

`sarvam-m` is the strongest open model for Tamil, and this project rejected it
for one reason only -- at 4-bit it needs about 13 GB and would not co-reside
with the other three models on a 16 GB laptop. A T4 x2 session has 32 GB, so the
comparison the README could not make is possible here.

There is a dedicated notebook for it:
[voicearm_planner_ablation.ipynb]({GH}/blob/main/notebooks/voicearm_planner_ablation.ipynb)
"""),
    ])


# --- planner ablation notebook ---------------------------------------------
def ablation_notebook() -> dict:
    return notebook([
md(f"""
# VoiceArm — planner ablation: does a Tamil-native 24B plan better?

[VoiceArm]({GH}) ships `Qwen3-4B` as its task planner. `sarvam-m` (24B) is the
strongest open model for Tamil and was rejected for one reason only: at 4-bit it
needs about 13 GB and will not co-reside with two ASR models and a detector on a
16 GB laptop.

That is a hardware constraint, not a quality claim. This notebook makes the
comparison the laptop could not, on eight reference utterances spanning English,
Tanglish and Tamil script.

No MuJoCo here -- the planner is a pure text-in, JSON-out stage, so this needs
nothing but transformers.

**Accelerator:** GPU. `T4 x2` (32 GB) is comfortable; a single 16 GB P100 works
but offloads a few layers to CPU and runs slower.
"""),
code("""
import torch

print("gpu  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("count:", torch.cuda.device_count())
print("vram :", ", ".join(
    f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB"
    for i in range(torch.cuda.device_count())) or "-")
print("bf16 :", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
"""),
code("""
!pip install -q -U transformers accelerate bitsandbytes 2>&1 | tail -2

import bitsandbytes
import transformers

print("transformers", transformers.__version__, "| bitsandbytes", bitsandbytes.__version__)
"""),
code(f"""
import os
import subprocess
import sys
from pathlib import Path

PROJ = Path("/kaggle/working") / "voicearm"
if not PROJ.exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(PROJ)], check=True)

os.chdir(PROJ)
sys.path.insert(0, str(PROJ))
os.environ["VOICEARM_BACKEND"] = "torch"
print("project at", PROJ)
"""),
md("""
## The comparison

`scripts/bench_planner.py` scores each model on whether the first step targets
the right object and whether a `place` step appears exactly when the instruction
asks for one. It detects that the community `sarvam-m` upload is already
quantized and does not re-quantize it, and it picks float16 over bfloat16 on GPUs
without bf16 units -- a T4 is one of those.

Downloads about 22 GB: 8 GB for Qwen3-4B and 14 GB for the pre-quantized
sarvam-m, against 47 GB for the official fp16 weights.
"""),
code("""
!python scripts/bench_planner.py \
    --models Qwen/Qwen3-4B-Instruct-2507 neuralnets/sarvam-m-4bit-q \
    --load-4bit
"""),
code("""
import json
from pathlib import Path

results = Path("out/planner_bench.json")
if results.exists():
    rows = json.loads(results.read_text())
    print(f"{'model':<40}{'correct':>9}{'mean gen':>10}{'peak vram':>11}")
    for row in rows:
        print(f"{row['model']:<40}{row['correct']:>5}/{row['of']}"
              f"{row['mean_gen_s']:>9.2f}s{row.get('peak_vram_gb', float('nan')):>10.1f}G")
else:
    print("no results -- check the cell above for a load failure")
"""),
md("""
## Reading the result

If the 24B model does not score higher, the README's model choice is confirmed on
evidence rather than on footprint, which is a stronger claim than the one it
makes now.

If it does score higher, that is a real finding: the shipped planner is limited
by the laptop, and the constraint is worth stating as a measured cost.

Either way the latency and VRAM columns matter. A model that is more accurate but
six times slower is a different trade, not a free win.
"""),
    ])


if __name__ == "__main__":
    print("building notebooks:")
    write(ROOT / "notebooks" / "voicearm_demo.ipynb", demo_notebook())
    write(ROOT / "notebooks" / "voicearm_planner_ablation.ipynb", ablation_notebook())
