#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and the Franka Panda assets.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Only the Panda is needed. Cloning all of mujoco_menagerie pulls hundreds of
# megabytes of meshes for robots this project never loads, so fetch the one
# model directory over the raw endpoint instead.
DIR=assets/mujoco_menagerie/franka_emika_panda
BASE=https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda
mkdir -p "$DIR/assets"
( cd "$DIR" && for f in panda.xml hand.xml scene.xml LICENSE; do curl -sfLO "$BASE/$f"; done )
curl -s https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/franka_emika_panda/assets \
  | python3 -c "import json,sys; print('\n'.join(f['download_url'] for f in json.load(sys.stdin) if f['type']=='file'))" \
  | ( cd "$DIR/assets" && xargs -P 8 -n 1 curl -sfLO )

echo "setup complete: $(ls "$DIR/assets" | wc -l | tr -d ' ') mesh assets"
