# Windows Setup Guide (Dell 7920 + RTX 4060)

## Recommended: Use WSL2

WSL2 gives you a full Linux environment with direct GPU access. It's the easiest way to do CUDA development on Windows.

---

## WSL2 Setup (5 minutes)

### 1. Install WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted.

### 2. Open Ubuntu and Install CUDA

Click **Ubuntu** in Start Menu, then run:

```bash
# Update
sudo apt update && sudo apt upgrade -y

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit build-essential -y

# Verify GPU
nvidia-smi
nvcc --version
```

### 3. Set Up Project

```bash
# Create project folder
mkdir -p ~/projects && cd ~/projects

# Copy file from Windows Downloads folder
# (Your C: drive is at /mnt/c/ in WSL)
cp /mnt/c/Users/YOUR_USERNAME/Downloads/gpu-flight-dynamics.tar.gz .

# Extract
tar -xzvf gpu-flight-dynamics.tar.gz
cd gpu-flight-dynamics

# Build
cd cuda
make
make test    # Should pass
make bench   # See speedup numbers!
```

### 4. Python Setup

```bash
cd ~/projects/gpu-flight-dynamics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install numpy cupy-cuda12x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Run demo
python python/flight_dynamics.py
```

---

## Native Windows Setup (Alternative)

If you prefer native Windows:

### Prerequisites

1. **NVIDIA Driver** - Already installed with your RTX 4060
2. **CUDA Toolkit 12.x** - Download from [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. **Visual Studio 2022** - With "Desktop development with C++" workload
4. **Python 3.10+** - From [python.org](https://python.org)

### Build Steps

Open **x64 Native Tools Command Prompt for VS 2022**:

```cmd
cd %USERPROFILE%\projects\gpu-flight-dynamics\cuda

:: Compile CUDA code
nvcc -arch=sm_89 -O3 -Xcompiler "/EHsc /W3" ^
     -c atmosphere.cu -o atmosphere.obj
nvcc -arch=sm_89 -O3 -Xcompiler "/EHsc /W3" ^
     -c aerodynamics.cu -o aerodynamics.obj
nvcc -arch=sm_89 -O3 -Xcompiler "/EHsc /W3" ^
     -c equations_of_motion.cu -o equations_of_motion.obj
nvcc -arch=sm_89 -O3 -Xcompiler "/EHsc /W3" ^
     -c flight_dynamics_api.cu -o flight_dynamics_api.obj

:: Link to DLL
nvcc -arch=sm_89 --shared -o flight_dynamics.dll ^
     atmosphere.obj aerodynamics.obj equations_of_motion.obj flight_dynamics_api.obj

:: Build benchmark
nvcc -arch=sm_89 -O3 -o benchmark.exe benchmark.cu ^
     atmosphere.obj aerodynamics.obj equations_of_motion.obj flight_dynamics_api.obj
```

### Python Setup (PowerShell)

```powershell
cd $HOME\projects\gpu-flight-dynamics

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install numpy cupy-cuda12x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Test (CPU mode works without building CUDA)
python python\flight_dynamics.py
```

---

## Quick Comparison

| Task | WSL2 (Linux) | Native Windows |
|------|--------------|----------------|
| Install CUDA | `apt install nvidia-cuda-toolkit` | Download 3GB installer |
| Build project | `make` | Visual Studio + manual nvcc |
| Python packages | Just works | Usually works |
| Complexity | ⭐ Easy | ⭐⭐⭐ Complex |

**WSL2 is strongly recommended** unless you have a specific reason to use native Windows.

---

## Troubleshooting

### WSL2: "nvidia-smi not found"

Make sure you have the latest NVIDIA driver on Windows (not in WSL). WSL2 shares the Windows driver.

```powershell
# In PowerShell, check driver version
nvidia-smi
```

If this works in PowerShell but not WSL, update WSL:

```powershell
wsl --update
```

### WSL2: "CUDA out of memory"

Your RTX 4060 has 8GB. WSL2 can access all of it. If you see memory errors:

```bash
# Check GPU memory
nvidia-smi

# Reduce instance count
python -c "from flight_dynamics import FlightSimulator; sim = FlightSimulator(1000)"
```

### Windows: "nvcc not found"

Add CUDA to your PATH:

```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

Or add permanently via System Properties > Environment Variables.

---

## Expected Performance (RTX 4060)

| Instances | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|--------:|
| 100       | ~50      | ~1       | 50x     |
| 1,000     | ~500     | ~4       | 125x    |
| 10,000    | ~5,000   | ~30      | 165x    |

You should see **100x+ speedup** on your RTX 4060!
