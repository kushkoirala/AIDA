#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1) Create/activate venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2) Install deps (without pybullet to avoid local build failures)
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# Optional: try to install pybullet with binary wheels only; skip on failure
if ! pip install --prefer-binary pybullet==3.2.6; then
  echo "Warning: pybullet install failed. Install manually or use a wheel for your platform." >&2
fi

# 3) Sync assets
python scripts/prepare_assets.py

echo "Ready:"
echo "- To run rollout stub: python scripts/rollout.py"
echo "- To run PPO stub:    python scripts/train_ppo.py"
echo "- Telemetry WS:      python -m aida_sim.io.telemetry (wire state_fn later)"
echo "Viewer assets live in viewer/public/Udaan-Product4.gltf"
