# =============================================================================
# GPU Flight Dynamics - Windows PowerShell Setup
# =============================================================================
# 
# For Dell 7920 with RTX 4060
# Run in PowerShell (as Administrator for some steps)
#
# Prerequisites:
#   1. NVIDIA Driver (already installed with GPU)
#   2. CUDA Toolkit 12.x from: https://developer.nvidia.com/cuda-downloads
#   3. Visual Studio 2022 with C++ tools
#   4. Python 3.10+
#
# Author: Kushal Koirala
# =============================================================================

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   GPU Flight Dynamics Simulator - Windows Setup                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# Step 1: Check NVIDIA Driver
# -----------------------------------------------------------------------------
Write-Host "Step 1: Checking NVIDIA driver..." -ForegroundColor Yellow

try {
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
    Write-Host "✓ NVIDIA driver found" -ForegroundColor Green
} catch {
    Write-Host "✗ nvidia-smi not found. Install NVIDIA driver first." -ForegroundColor Red
    exit 1
}
Write-Host ""

# -----------------------------------------------------------------------------
# Step 2: Check CUDA
# -----------------------------------------------------------------------------
Write-Host "Step 2: Checking CUDA Toolkit..." -ForegroundColor Yellow

try {
    nvcc --version
    Write-Host "✓ CUDA Toolkit found" -ForegroundColor Green
} catch {
    Write-Host "✗ CUDA not found." -ForegroundColor Red
    Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "Select: Windows > x86_64 > 11 > exe (local)" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# -----------------------------------------------------------------------------
# Step 3: Set up project directory
# -----------------------------------------------------------------------------
Write-Host "Step 3: Setting up project directory..." -ForegroundColor Yellow

$ProjectDir = "$HOME\projects\gpu-flight-dynamics"
New-Item -ItemType Directory -Force -Path $ProjectDir | Out-Null
Set-Location $ProjectDir
Write-Host "Working directory: $ProjectDir"
Write-Host ""

# -----------------------------------------------------------------------------
# Step 4: Extract project (if tar.gz exists)
# -----------------------------------------------------------------------------
Write-Host "Step 4: Extract project files..." -ForegroundColor Yellow
Write-Host "Copy gpu-flight-dynamics.tar.gz to: $ProjectDir"
Write-Host "Then run: tar -xzvf gpu-flight-dynamics.tar.gz"
Write-Host ""

# -----------------------------------------------------------------------------
# Step 5: Build instructions
# -----------------------------------------------------------------------------
Write-Host "Step 5: Build CUDA library..." -ForegroundColor Yellow
Write-Host @"

For Windows native build, you need Visual Studio with CUDA integration.

Option A - Use the provided Makefile with MinGW:
    cd cuda
    mingw32-make CUDA_ARCH=sm_89

Option B - Use Visual Studio Developer Command Prompt:
    cd cuda
    nvcc -arch=sm_89 -O3 --shared -o flight_dynamics.dll *.cu

Option C - (Recommended) Use WSL2 instead - much easier!

"@ -ForegroundColor Gray

# -----------------------------------------------------------------------------
# Step 6: Python setup
# -----------------------------------------------------------------------------
Write-Host "Step 6: Setting up Python..." -ForegroundColor Yellow

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install --upgrade pip
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Note: CuPy on Windows
Write-Host ""
Write-Host "Installing CuPy for Windows..." -ForegroundColor Yellow
pip install cupy-cuda12x

Write-Host ""
Write-Host "✓ Python environment ready" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Step 7: Test Python (CPU mode works without CUDA build)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 7: Testing Python (CPU mode)..." -ForegroundColor Yellow

python -c @"
from flight_dynamics import FlightSimulator
import torch

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test CPU mode
sim = FlightSimulator(n_instances=100, use_gpu=False)
sim.step_n(100)
print('✓ CPU simulation working')

# Test GPU mode (if available)
try:
    sim_gpu = FlightSimulator(n_instances=100, use_gpu=True)
    sim_gpu.step_n(100)
    print('✓ GPU simulation working')
except Exception as e:
    print(f'GPU mode not available: {e}')
"@

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Setup Complete!                                                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "To activate environment in future sessions:" -ForegroundColor Yellow
Write-Host "  cd $ProjectDir"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
