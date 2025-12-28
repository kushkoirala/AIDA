#!/bin/bash
# Full BC Training Pipeline with GPU Physics
# Run this after CuPy is installed

set -e  # Exit on error

cd /home/AIDA
source .venv-linux/bin/activate
export PYTHONPATH=/home/AIDA:$PYTHONPATH

echo "============================================================"
echo "  AIDA BC Training Pipeline with GPU Physics"
echo "============================================================"
echo ""

# Step 1: Test GPU Physics
echo "[1/5] Testing GPU Physics Simulator..."
cd gpu-flight-dynamics/python
python demo_udaan.py
cd /home/AIDA
echo "✓ GPU Physics test complete!"
echo ""

# Step 2: Generate BC Dataset with GPU
echo "[2/5] Generating BC Dataset with GPU Physics..."
echo "  This will take ~5-10 minutes"
python scripts/generate_bc_dataset_gpu.py \
  --num-episodes 1000 \
  --episode-length 500 \
  --batch-size 100 \
  --output checkpoints/bc_dataset_gpu_new.npz

echo "✓ BC dataset generated!"
echo ""

# Step 3: Convert GPU dataset to AIDA format
echo "[3/5] Converting GPU dataset to AIDA format..."
python scripts/gpu_to_aida_adapter.py \
  --input checkpoints/bc_dataset_gpu_new.npz \
  --output checkpoints/bc_dataset_aida_from_gpu.npz \
  --target-altitude 30.0 \
  --target-heading 0.0

echo "✓ Dataset converted!"
echo ""

# Step 4: Train BC Policy
echo "[4/5] Training BC Policy on GPU-generated data..."
python scripts/train_bc_policy.py \
  --dataset checkpoints/bc_dataset_aida_from_gpu.npz \
  --output checkpoints/bc_policy_from_gpu.pt \
  --epochs 20 \
  --batch-size 256 \
  --device cpu \
  --hidden-dim 256

echo "✓ BC policy trained!"
echo ""

# Step 5: Train PPO with BC Warmstart
echo "[5/5] Training PPO with BC warmstart..."
echo "  This will take ~15-20 minutes"
python scripts/train_ppo_flight.py \
  --policy-init checkpoints/bc_policy_from_gpu.pt \
  --task cruise \
  --timesteps 500000 \
  --device cpu \
  --num-envs 4

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo ""
echo "New checkpoints created:"
echo "  - checkpoints/bc_dataset_gpu_new.npz       (GPU physics data)"
echo "  - checkpoints/bc_dataset_aida_from_gpu.npz (Converted to AIDA)"
echo "  - checkpoints/bc_policy_from_gpu.pt        (BC policy)"
echo "  - checkpoints/ppo_flight_final.pt          (PPO with BC warmstart)"
echo ""
echo "To visualize:"
echo "  ./run_all_cruise.sh"
echo "  Then open: http://127.0.0.1:8080"
echo ""
