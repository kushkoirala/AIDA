#!/bin/bash
# =============================================================================
# GPU Flight Dynamics - Setup Script for Dell 7920 + RTX 4060
# =============================================================================
#
# This script sets up the complete development environment for the
# GPU-accelerated flight dynamics simulator on your Dell 7920.
#
# Prerequisites:
#   - Ubuntu 22.04 or 24.04 (recommended) OR Windows 11 with WSL2
#   - NVIDIA Driver 535+ (for RTX 4060)
#
# Author: Kushal Koirala
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   GPU Flight Dynamics Simulator - RTX 4060 Setup                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# Step 1: Check NVIDIA Driver
# =============================================================================
echo "Step 1: Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
    echo "✓ NVIDIA driver found"
else
    echo "✗ NVIDIA driver not found!"
    echo ""
    echo "Install the driver first:"
    echo "  Ubuntu: sudo apt install nvidia-driver-535"
    echo "  Then reboot and run this script again."
    exit 1
fi
echo ""

# =============================================================================
# Step 2: Check/Install CUDA Toolkit
# =============================================================================
echo "Step 2: Checking CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✓ CUDA Toolkit found"
else
    echo "✗ CUDA Toolkit not found. Installing..."
    echo ""
    echo "Option A - Quick install via apt (Ubuntu 22.04+):"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo ""
    echo "Option B - Full install from NVIDIA (recommended):"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run"
    echo "  sudo sh cuda_12.3.1_545.23.08_linux.run"
    echo ""
    echo "After installing CUDA, add to ~/.bashrc:"
    echo '  export PATH=/usr/local/cuda/bin:$PATH'
    echo '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    echo ""
    read -p "Press Enter after installing CUDA, or Ctrl+C to exit..."
fi
echo ""

# =============================================================================
# Step 3: Create project directory
# =============================================================================
echo "Step 3: Setting up project directory..."
PROJECT_DIR="$HOME/projects/gpu-flight-dynamics"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo "Working directory: $PROJECT_DIR"
echo ""

# =============================================================================
# Step 4: Extract or clone project
# =============================================================================
echo "Step 4: Project files..."
if [ -f "cuda/flight_dynamics.h" ]; then
    echo "✓ Project files already present"
else
    echo "Extract gpu-flight-dynamics.tar.gz here, or clone from GitHub:"
    echo "  tar -xzvf gpu-flight-dynamics.tar.gz"
    echo "  # OR"
    echo "  git clone https://github.com/kushkoirala/gpu-flight-dynamics.git ."
fi
echo ""

# =============================================================================
# Step 5: Build CUDA library
# =============================================================================
echo "Step 5: Building CUDA library..."
if [ -d "cuda" ]; then
    cd cuda
    
    # Clean previous build
    make clean 2>/dev/null || true
    
    # Build for RTX 4060 (sm_89 = Ada Lovelace)
    echo "Building for RTX 4060 (sm_89)..."
    make CUDA_ARCH=sm_89
    
    echo "✓ CUDA library built successfully"
    cd ..
else
    echo "✗ cuda/ directory not found. Extract project files first."
fi
echo ""

# =============================================================================
# Step 6: Set up Python environment
# =============================================================================
echo "Step 6: Setting up Python environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python packages..."
pip install numpy matplotlib

# Install CuPy for CUDA 12.x
echo "Installing CuPy (CUDA 12.x)..."
pip install cupy-cuda12x

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install additional packages for RL
pip install gymnasium stable-baselines3

# Install project
cd python
pip install -e .
cd ..

echo "✓ Python environment ready"
echo ""

# =============================================================================
# Step 7: Run tests
# =============================================================================
echo "Step 7: Running tests..."
cd cuda
./test_main && echo "✓ CUDA tests passed" || echo "✗ CUDA tests failed"
cd ..

echo ""
echo "Testing Python bindings..."
python -c "
from flight_dynamics import FlightSimulator, benchmark
import torch

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Quick simulation test
sim = FlightSimulator(n_instances=100, use_gpu=True)
sim.step_n(100)
print('✓ Python simulation working')
"
echo ""

# =============================================================================
# Step 8: Run benchmarks
# =============================================================================
echo "Step 8: Running benchmarks..."
echo ""
cd cuda
./benchmark
cd ..
echo ""

# =============================================================================
# Done!
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Quick Start:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python python/flight_dynamics.py"
echo ""
echo "Run benchmarks:"
echo "  cd cuda && ./benchmark"
echo ""
echo "Jupyter notebook:"
echo "  pip install jupyterlab"
echo "  jupyter lab"
echo ""
